"""
Head-to-head: AtlasInfer vs the standard 4-bit quantization stack on the same
model and the same WikiText-2 eval.

Methods compared:
  * fp16                  - dense baseline
  * AtlasInfer int8/int4  - uniform block-wise + sparse outliers (eager path)
  * AtlasInfer gptq-nf4   - GPTQ error-compensated NF4 (AtlasInfer's best 4-bit)
  * AtlasInfer mixed      - sensitivity-allocated per-layer precision (the point)
  * bnb int8 (LLM.int8()) - bitsandbytes 8-bit
  * bnb nf4               - bitsandbytes 4-bit NormalFloat
  * gptq (auto-gptq)      - the reference GPTQ-INT4 (via transformers GPTQConfig)
  * awq   (autoawq)       - Activation-aware Weight Quantization INT4

The last two are the *real* SOTA-tier 4-bit baselines (not just the accessible
bitsandbytes path); they're optional and skipped with a clear note if their
libraries aren't installed, so the script still runs a full comparison against
whatever baselines are present.

Memory is measured method-agnostically as the total bytes of all parameters +
buffers actually resident on the model, so every method is counted the same way.
NOTE: memory is not perfectly apples-to-apples across *methods* - e.g. bnb and
gptq/awq pack scales/zeros differently and bnb double-quantizes its scales - so
read the memory column as "same accounting rule, method-specific packing", and
the perplexity column (identical eval for all) as the primary axis.

Run under Linux/WSL2 + CUDA (bitsandbytes/auto-gptq/awq are Linux-only):
    python compare_baselines.py --model EleutherAI/pythia-410m
    python compare_baselines.py --model Qwen/Qwen2.5-0.5B --skip awq   # skip a slow/absent one
Install the external baselines with:
    pip install -e ".[baselines]"
"""
import argparse
import gc
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

# Windows/CUDA stability: init the CUDA context before importing transformers
# (see benchmark.py for the full note — avoids a 0xC0000005 access violation on
# some Windows torch builds). No-op on CPU.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from atlasinfer.patcher import quantize_model, quantize_model_mixed
from atlasinfer.sensitivity import SensitivityProfiler
from atlasinfer.allocator import allocate_optimal
from atlasinfer.gptq import quantize_model_gptq
from atlasinfer.awq import quantize_model_awq
from benchmark import evaluate_perplexity, load_wikitext

hf_logging.set_verbosity_error()


def resident_bytes(model) -> int:
    """Total bytes of all params + buffers resident on the model (any method)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total


def gptq_baseline(model_name, tokenizer, calib, device_map):
    """Reference GPTQ-INT4 via transformers' GPTQConfig (optimum + auto-gptq).

    Quantizes on load using the same WikiText calibration docs AtlasInfer's GPTQ
    path uses, so it's a like-for-like 4-bit comparison of the error-compensation
    machinery rather than a different calibration set.
    """
    from transformers import GPTQConfig
    calib_docs = [t for t in calib if t.strip()][:128]
    qc = GPTQConfig(bits=4, dataset=calib_docs, tokenizer=tokenizer,
                    group_size=128, desc_act=False)
    return AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=qc, dtype=torch.float16,
        device_map=("auto" if device_map else {"": 0}))


def awq_baseline(model_name, tokenizer):
    """Activation-aware Weight Quantization INT4 via autoawq.

    Returns the underlying transformers model (``.model``) so perplexity and
    resident-bytes accounting go through the exact same code path as every other
    method.
    """
    from awq import AutoAWQForCausalLM
    # autoawq's kwarg is torch_dtype (not dtype); load straight onto the GPU so
    # calibration and the returned model share one device (else eval hits a
    # cuda/cpu mismatch). NOTE: autoawq 0.2.9 is deprecated and last tested on
    # transformers 4.51 — it may fail to import/quantize on newer transformers,
    # in which case the caller skips it.
    m = AutoAWQForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda:0")
    m.quantize(tokenizer, quant_config={
        "w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"})
    return m.model.to("cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="EleutherAI/pythia-410m")
    ap.add_argument("--eval-tokens", type=int, default=40000)
    ap.add_argument("--mixed-bits", type=float, default=5.0)
    ap.add_argument("--device-map", action="store_true",
                    help="shard across all GPUs (device_map=auto) for models too big "
                         "for one card, e.g. 7-13B on Kaggle T4x2")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    ap.add_argument("--double-quant", action="store_true",
                    help="also measure AtlasInfer nf4 / gptq-nf4 with double-quantized "
                         "scales (QLoRA-style), which closes most of the 4-bit memory "
                         "gap to bnb's NF4 at ~unchanged perplexity")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="method keys to skip, e.g. --skip awq gptq. Keys: gptq, awq "
                         "(external baselines), gptq-nf4, awq-nf4, nf4-dq, gptq-nf4-dq "
                         "(AtlasInfer). A prefix skips its variants, so `--skip awq` "
                         "drops both the external AWQ and AtlasInfer's awq-nf4.")
    args = ap.parse_args()

    from atlasinfer import seed_everything
    seed_everything(args.seed)
    print(f"seed={args.seed}")

    assert torch.cuda.is_available(), "needs CUDA"
    dev = torch.device("cuda")
    DMAP = "auto" if args.device_map else None       # None -> single-GPU (load on CPU, move later)
    BNB_MAP = "auto" if args.device_map else {"": 0}
    tok = AutoTokenizer.from_pretrained(args.model)
    calib, eval_text = load_wikitext()
    cfg = AutoConfig.from_pretrained(args.model)
    max_len = min(getattr(cfg, "max_position_embeddings", 1024) or 1024, 1024)

    rows = []

    def record(label, model, bits):
        # device_map / bnb models are already dispatched on GPU(s) and must not be
        # .to()'d; single-GPU fp16/AtlasInfer models come back on CPU and need moving.
        on_cuda = next(model.parameters()).device.type == "cuda"
        model = model.eval() if on_cuda else model.to(dev).eval()
        ppl = evaluate_perplexity(model, tok, eval_text, dev,
                                  max_len=max_len, stride=max_len // 2,
                                  max_tokens=args.eval_tokens)
        mb = resident_bytes(model) / 1024 ** 2
        print(f"  {label:<22} {mb:8.1f} MB  ppl={ppl:8.3f}  (~{bits} bits)")
        rows.append({"method": label, "mb": mb, "ppl": ppl, "bits": bits})
        if not args.device_map:
            model.to("cpu")
        del model; gc.collect(); torch.cuda.empty_cache()

    def fp16():
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map=DMAP)

    def place(model):  # put on GPU for in-place work (no-op if already device-mapped)
        return model if args.device_map else model.to(dev)

    def guarded(key, label, fn, bits=4):
        """Measure one method, but never let its failure discard the whole run.

        The results table is only written at the very end, so an unguarded
        exception here (an OOM quantizing a 7B+ model is the common one) throws
        away every row already measured -- potentially hours of a cloud run. Skip
        keys are matched loosely so `--skip awq` also skips `awq-nf4`.
        """
        if any(k == key or key.startswith(k + "-") for k in args.skip):
            print(f"  {label:<22} SKIPPED (--skip)")
            return
        try:
            record(label, fn(), bits)
        except Exception as exc:  # noqa: BLE001 - one method must not sink the run
            print(f"  {label:<22} FAILED ({type(exc).__name__}: {exc}) -- continuing")
            gc.collect(); torch.cuda.empty_cache()

    print(f"\nComparing on {args.model} (WikiText-2, {args.eval_tokens} eval tokens)\n")

    # FP16 baseline.
    record("fp16", fp16(), 16)

    # AtlasInfer uniform (symmetric int4 vs NF4 to show the codebook's effect).
    record("AtlasInfer int8", quantize_model(fp16(), precision="int8", verbose=False), 8)
    record("AtlasInfer int4 (sym)",
           quantize_model(fp16(), precision="int4", quant_4bit="int4", verbose=False), 4)
    record("AtlasInfer nf4",
           quantize_model(fp16(), precision="int4", quant_4bit="nf4", verbose=False), 4)
    if args.double_quant:
        guarded("nf4-dq", "AtlasInfer nf4+dq",
                lambda: quantize_model(fp16(), precision="int4", quant_4bit="nf4",
                                       double_quant=True, verbose=False))

    # NF4 + GPTQ error compensation (needs the model on-device for the Hessian pass).
    def _gptq(double_quant=False):
        gm = place(fp16()).eval()
        quantize_model_gptq(gm, tokenizer=tok, calibration_texts=calib,
                            double_quant=double_quant, verbose=False)
        return gm
    guarded("gptq-nf4", "AtlasInfer gptq-nf4", _gptq)
    if args.double_quant:
        guarded("gptq-nf4-dq", "AtlasInfer gptq-nf4+dq", lambda: _gptq(double_quant=True))

    # AWQ: activation-aware scaling — a Hessian-free route to the same 4-bit tier.
    def _awq():
        am = place(fp16()).eval()
        quantize_model_awq(am, tokenizer=tok, calibration_texts=calib, verbose=False)
        return am
    guarded("awq-nf4", "AtlasInfer awq-nf4", _awq)

    # AtlasInfer mixed (profile once, allocate at target bits).
    base = place(fp16()).eval()
    profiles = SensitivityProfiler().profile_end_to_end(base, tokenizer=tok, calibration_texts=calib)
    if not args.device_map:
        base.to("cpu")
    del base; gc.collect(); torch.cuda.empty_cache()
    n_params = sum(p.param_count for p in profiles.values())
    alloc = allocate_optimal(profiles, budget_bytes=int(n_params * args.mixed_bits / 8))
    record(f"AtlasInfer mixed-{args.mixed_bits:g}bit",
           quantize_model_mixed(fp16(), allocation=alloc.allocations, verbose=False),
           round(alloc.avg_bits, 1))

    # bitsandbytes int8 (LLM.int8()) and nf4.
    bnb8 = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=BNB_MAP, dtype=torch.float16)
    record("bnb int8 (LLM.int8)", bnb8, 8)

    bnb4 = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=BNB_MAP, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16))
    record("bnb nf4", bnb4, 4)

    # External SOTA-tier 4-bit baselines (optional). Each is guarded: a missing
    # library or a version/runtime failure prints a SKIPPED note and the rest of
    # the comparison continues, so you always get whatever baselines are present.
    def external(key, label, fn):
        if key in args.skip:
            print(f"  {label:<22} SKIPPED (--skip {key})")
            return
        try:
            record(label, fn(), 4)
        except ImportError as exc:
            print(f"  {label:<22} SKIPPED (not installed: {exc}. "
                  f"`pip install -e \".[baselines]\"`)")
        except Exception as exc:  # noqa: BLE001 - baseline libs are version-fragile
            print(f"  {label:<22} SKIPPED (failed: {type(exc).__name__}: {exc})")
            gc.collect(); torch.cuda.empty_cache()

    external("gptq", "gptq (auto-gptq)",
             lambda: gptq_baseline(args.model, tok, calib, args.device_map))
    external("awq", "awq (autoawq)",
             lambda: awq_baseline(args.model, tok))

    # Markdown table (stdout + a reproducible file under results/).
    fp16_ppl = next(r["ppl"] for r in rows if r["method"] == "fp16")
    header = (f"### {args.model}  (WikiText-2, {args.eval_tokens} eval tokens, seed {args.seed})\n\n"
              "| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |\n"
              "| --- | ---: | ---: | ---: | ---: |")
    body = "\n".join(
        f"| {r['method']} | {r['bits']} | {r['mb']:.1f} | "
        f"{r['ppl']:.3f} | {r['ppl'] - fp16_ppl:+.3f} |"
        for r in rows
    )
    table = header + "\n" + body + "\n"
    print("\n" + table)

    os.makedirs("results", exist_ok=True)
    safe = args.model.replace("/", "_")
    out = os.path.join("results", f"comparison_{safe}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

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
import json
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
from atlasinfer.allocator import allocate_greedy, allocate_optimal
from atlasinfer.gptq import quantize_model_gptq
from atlasinfer.awq import quantize_model_awq
from atlasinfer.evaluation import evaluate_perplexity, load_wikitext, resident_bytes

hf_logging.set_verbosity_error()


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


class Ctx:
    """Model/tokenizer plumbing plus memoized profiling for the method builders.

    Profiling is the single most expensive step and three arms need it, so it is
    computed at most once and only if one of those arms actually runs -- with
    ``--only bnb-nf4`` it is never computed at all.
    """

    def __init__(self, args, tok, calib, dev, dmap, bnb_map):
        self.args, self.tok, self.calib = args, tok, calib
        self.dev, self.dmap, self.bnb_map = dev, dmap, bnb_map
        self._cache = {}

    def fp16(self):
        return AutoModelForCausalLM.from_pretrained(
            self.args.model, dtype=torch.float16, device_map=self.dmap)

    def place(self, model):
        """Put on GPU for in-place work (no-op if already device-mapped)."""
        return model if self.args.device_map else model.to(self.dev)

    def profiles(self):
        if "profiles" not in self._cache:
            base = self.place(self.fp16()).eval()
            prof = SensitivityProfiler(max_samples=self.args.calib_samples,
                                       seq_len=self.args.calib_seqlen)
            self._cache["profiles"] = prof.profile_end_to_end(
                base, tokenizer=self.tok,
                calibration_texts=self.calib[self.args.calib_offset:])
            if not self.args.device_map:
                base.to("cpu")
            del base
            gc.collect()
            torch.cuda.empty_cache()
        return self._cache["profiles"]

    def budget(self):
        p = self.profiles()
        return int(sum(x.param_count for x in p.values()) * self.args.mixed_bits / 8)

    def knapsack(self):
        if "knapsack" not in self._cache:
            self._cache["knapsack"] = allocate_optimal(
                self.profiles(), budget_bytes=self.budget())
        return self._cache["knapsack"]

    def greedy(self):
        if "greedy" not in self._cache:
            p = self.profiles()
            self._cache["greedy"] = allocate_greedy(
                {n: x.sensitivity("int4") for n, x in p.items()},
                {n: x.param_count for n, x in p.items()},
                self.budget(), profiles=p)
        return self._cache["greedy"]

    def gptq(self, double_quant=False, allocation=None):
        gm = self.place(self.fp16()).eval()
        quantize_model_gptq(gm, tokenizer=self.tok, calibration_texts=self.calib,
                            double_quant=double_quant, allocation=allocation,
                            verbose=False)
        return gm

    def awq(self):
        am = self.place(self.fp16()).eval()
        quantize_model_awq(am, tokenizer=self.tok, calibration_texts=self.calib,
                           verbose=False)
        return am


def build_methods(ctx) -> list:
    """The method registry: ``(key, label, builder, bits)`` per measured arm.

    Kept out of ``main()`` so the set of methods can be inspected and unit-tested
    without running a single forward pass. ``bits`` may be a zero-arg callable
    when it depends on an allocation that only exists after the builder runs.
    """
    a = ctx.args
    mb = a.mixed_bits
    methods = [
        ("fp16", "fp16", ctx.fp16, 16),
        ("int8", "AtlasInfer int8",
         lambda: quantize_model(ctx.fp16(), precision="int8", verbose=False), 8),
        ("int4-sym", "AtlasInfer int4 (sym)",
         lambda: quantize_model(ctx.fp16(), precision="int4", quant_4bit="int4",
                                verbose=False), 4),
        ("nf4", "AtlasInfer nf4",
         lambda: quantize_model(ctx.fp16(), precision="int4", quant_4bit="nf4",
                                verbose=False), 4),
    ]
    if a.double_quant:
        methods.append(
            ("nf4-dq", "AtlasInfer nf4+dq",
             lambda: quantize_model(ctx.fp16(), precision="int4", quant_4bit="nf4",
                                    double_quant=True, verbose=False), 4))
    methods.append(("gptq-nf4", "AtlasInfer gptq-nf4", ctx.gptq, 4))
    if a.double_quant:
        methods.append(("gptq-nf4-dq", "AtlasInfer gptq-nf4+dq",
                        lambda: ctx.gptq(double_quant=True), 4))
    methods += [
        ("awq-nf4", "AtlasInfer awq-nf4", ctx.awq, 4),
        # Sensitivity-allocated per-layer precision, and its ablation.
        ("mixed", f"AtlasInfer mixed-{mb:g}bit",
         lambda: quantize_model_mixed(ctx.fp16(),
                                      allocation=ctx.knapsack().allocations,
                                      verbose=False),
         lambda: round(ctx.knapsack().avg_bits, 1)),
        # Same budget, benefit-per-byte greedy instead of the exact DP. Reuses the
        # one profiling pass, so it is nearly free -- and it is what shows whether
        # the knapsack solve is doing any work (per PAPER/01_go_nogo.md 2f: it
        # isn't; the two land within noise of each other).
        ("greedy", f"AtlasInfer greedy-{mb:g}bit",
         lambda: quantize_model_mixed(ctx.fp16(),
                                      allocation=ctx.greedy().allocations,
                                      verbose=False),
         lambda: round(ctx.greedy().avg_bits, 1)),
        # THE composition experiment: GPTQ error compensation applied *within* a
        # mixed allocation. Orthogonal mechanisms -- the allocator picks each
        # layer's width, GPTQ compensates the residual at that width -- so they
        # should stack. Plain GPTQ-NF4 Pareto-dominates mixed precision on 3 of 4
        # small models, so if this does not beat GPTQ alone the allocator adds
        # nothing over a method that costs no extra memory. See K4.
        ("gptq-mixed", f"AtlasInfer gptq-mixed-{mb:g}bit",
         lambda: ctx.gptq(allocation=ctx.knapsack().allocations),
         lambda: round(ctx.knapsack().avg_bits, 1)),
        # bitsandbytes. Guarded like everything else: it is a native extension and
        # the single most likely thing to fail on an unfamiliar cloud image.
        ("bnb-int8", "bnb int8 (LLM.int8)",
         lambda: AutoModelForCausalLM.from_pretrained(
             a.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
             device_map=ctx.bnb_map, dtype=torch.float16), 8),
        ("bnb-nf4", "bnb nf4",
         lambda: AutoModelForCausalLM.from_pretrained(
             a.model, device_map=ctx.bnb_map, dtype=torch.float16,
             quantization_config=BitsAndBytesConfig(
                 load_in_4bit=True, bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.float16)), 4),
        # External SOTA-tier 4-bit references (optional, version-fragile).
        ("gptq", "gptq (auto-gptq)",
         lambda: gptq_baseline(a.model, ctx.tok, ctx.calib, a.device_map), 4),
        ("awq", "awq (autoawq)", lambda: awq_baseline(a.model, ctx.tok), 4),
    ]
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="EleutherAI/pythia-410m")
    ap.add_argument("--eval-tokens", type=int, default=40000)
    ap.add_argument("--mixed-bits", type=float, default=5.0)
    ap.add_argument("--device-map", action="store_true",
                    help="shard across all GPUs (device_map=auto) for models too big "
                         "for one card, e.g. 7-13B on Kaggle T4x2")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    ap.add_argument("--out", default="results",
                    help="output directory. Point a smoke/preflight run somewhere "
                         "else (e.g. results/_smoke) so a short-eval run cannot "
                         "overwrite a committed full-length result for the same model.")
    # Calibration knobs. Previously hardcoded in SensitivityProfiler's defaults,
    # which made "how robust is the allocation to the calibration set?" -- the
    # first question anyone asks about a calibration-driven method -- unanswerable
    # without editing the library. Sweep these to answer it.
    ap.add_argument("--calib-samples", type=int, default=8,
                    help="calibration sequences used by the sensitivity profiler")
    ap.add_argument("--calib-seqlen", type=int, default=128,
                    help="tokens per calibration sequence for the profiler")
    ap.add_argument("--calib-offset", type=int, default=0,
                    help="skip the first N calibration documents, so a different "
                         "--calib-offset gives a disjoint calibration set at the "
                         "same size (use to test allocation stability)")
    ap.add_argument("--double-quant", action="store_true",
                    help="also measure AtlasInfer nf4 / gptq-nf4 with double-quantized "
                         "scales (QLoRA-style), which closes most of the 4-bit memory "
                         "gap to bnb's NF4 at ~unchanged perplexity")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="method keys to skip, e.g. --skip awq gptq. Keys: fp16, "
                         "int8, int4-sym, nf4, gptq-nf4, awq-nf4, mixed, greedy, "
                         "gptq-mixed, bnb-int8, bnb-nf4, gptq, awq (+ nf4-dq and "
                         "gptq-nf4-dq under --double-quant). A prefix skips its "
                         "variants, so `--skip awq` drops both the external AWQ "
                         "and AtlasInfer's awq-nf4.")
    ap.add_argument("--only", nargs="*", default=[],
                    help="run ONLY these method keys (exact match). Use to re-run "
                         "one arm cheaply after a failure instead of paying for the "
                         "whole comparison again -- e.g. --only fp16 gptq-mixed. "
                         "Note the JSON is rewritten with just these rows, so point "
                         "--out somewhere else if you want to keep the full table.")
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
    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    out_json = os.path.join(args.out, f"comparison_{safe}.json")

    def _flush():
        """Persist everything measured so far, after every single row.

        A 7B comparison is hours of paid GPU across a dozen methods. Writing only
        at the end means an OOM, a driver hiccup, or a preempted spot instance in
        the last method discards all of it. The file is a few KB; rewriting it
        each row is free insurance.
        """
        fp16_row = next((r for r in rows if r["method"] == "fp16"), None)
        for r in rows:
            r["delta_vs_fp16"] = (r["ppl"] - fp16_row["ppl"]) if fp16_row else None
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "eval_tokens": args.eval_tokens,
                "seed": args.seed,
                "mixed_bits": args.mixed_bits,
                "max_len": max_len,
                "double_quant": args.double_quant,
                "skipped": list(args.skip),
                "device_map": args.device_map,
                "calibration": {"samples": args.calib_samples,
                                "seq_len": args.calib_seqlen,
                                "offset": args.calib_offset},
                # `bits` is the NOMINAL target the allocator budgeted against; it
                # excludes per-block scales and sparse-outlier storage. `mb` is
                # the measured resident footprint and includes all of it. Compare
                # methods on `mb`, never on `bits`.
                "bits_are_nominal": True,
                "complete": False,   # set True once the run finishes cleanly
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__,
                "rows": rows,
            }, f, indent=2)

    def record(label, model, bits):
        # device_map / bnb models are already dispatched on GPU(s) and must not be
        # .to()'d; single-GPU fp16/AtlasInfer models come back on CPU and need moving.
        on_cuda = next(model.parameters()).device.type == "cuda"
        model = model.eval() if on_cuda else model.to(dev).eval()
        ppl = evaluate_perplexity(model, tok, eval_text, dev,
                                  max_len=max_len, stride=max_len // 2,
                                  max_tokens=args.eval_tokens)
        mb = resident_bytes(model) / 1024 ** 2
        print(f"  {label:<22} {mb:8.1f} MB  ppl={ppl:8.3f}  (~{bits} bits)", flush=True)
        rows.append({"method": label, "mb": mb, "ppl": ppl, "bits": bits})
        _flush()
        if not args.device_map:
            model.to("cpu")
        del model; gc.collect(); torch.cuda.empty_cache()

    def guarded(key, label, fn, bits=4):
        """Measure one method, but never let its failure discard the whole run.

        An unguarded exception here (an OOM quantizing a 7B+ model is the common
        one, and a missing native library the other) would abort before the later
        methods run -- potentially hours of paid GPU for nothing. Skip keys are
        matched loosely so `--skip awq` also skips `awq-nf4`.

        ``bits`` may be a zero-arg callable when it depends on an allocation that
        is only computed inside ``fn``; it is resolved after the build succeeds.
        """
        if any(k == key or key.startswith(k + "-") for k in args.skip):
            print(f"  {label:<22} SKIPPED (--skip)")
            return
        try:
            model = fn()
            record(label, model, bits() if callable(bits) else bits)
        except ImportError as exc:
            print(f"  {label:<22} SKIPPED (not installed: {exc}. "
                  f"`pip install -e \".[baselines]\"`)")
        except Exception as exc:  # noqa: BLE001 - one method must not sink the run
            print(f"  {label:<22} FAILED ({type(exc).__name__}: {exc}) -- continuing")
            gc.collect(); torch.cuda.empty_cache()

    print(f"\nComparing on {args.model} (WikiText-2, {args.eval_tokens} eval tokens)\n")

    for key, label, builder, bits in build_methods(
            Ctx(args, tok, calib, dev, DMAP, BNB_MAP)):
        if args.only and key not in args.only:
            continue
        guarded(key, label, builder, bits=bits)

    # Markdown table (stdout + a reproducible file under results/).
    fp16_row = next((r for r in rows if r["method"] == "fp16"), None)
    if fp16_row is None:
        print("\nERROR: the FP16 reference row is missing, so no delta column can "
              "be computed. Partial rows are still in "
              f"{out_json} -- inspect them there.")
        return
    fp16_ppl = fp16_row["ppl"]
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

    out = os.path.join(args.out, f"comparison_{safe}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table)

    # Final flush, now marked complete. The markdown is for humans; the JSON is
    # the source of truth every table and figure is generated from, so no number
    # is ever re-typed by hand. `complete: true` distinguishes a finished run
    # from the partial file left behind by a crash or a preempted instance.
    _flush()
    with open(out_json, encoding="utf-8") as f:
        payload = json.load(f)
    payload["complete"] = True
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out} and {out_json}")


if __name__ == "__main__":
    main()

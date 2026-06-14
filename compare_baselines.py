"""
Head-to-head: AtlasInfer vs bitsandbytes (the de-facto accessible-quantization
baseline) on the same model and the same WikiText-2 eval.

Methods compared:
  * fp16                  - dense baseline
  * AtlasInfer int8/int4  - uniform block-wise + sparse outliers (eager path)
  * AtlasInfer mixed      - sensitivity-allocated per-layer precision (the point)
  * bnb int8 (LLM.int8()) - bitsandbytes 8-bit
  * bnb nf4               - bitsandbytes 4-bit NormalFloat

Memory is measured method-agnostically as the total bytes of all parameters +
buffers actually resident on the model, so every method is counted the same way.

Run under Linux/WSL2 + CUDA (bitsandbytes is Linux-only):
    ~/atlasvenv/bin/python compare_baselines.py --model EleutherAI/pythia-410m
"""
import argparse
import gc
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from atlasinfer.patcher import quantize_model, quantize_model_mixed
from atlasinfer.sensitivity import SensitivityProfiler
from atlasinfer.allocator import allocate_optimal
from benchmark import evaluate_perplexity, load_wikitext

hf_logging.set_verbosity_error()


def resident_bytes(model) -> int:
    """Total bytes of all params + buffers resident on the model (any method)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="EleutherAI/pythia-410m")
    ap.add_argument("--eval-tokens", type=int, default=40000)
    ap.add_argument("--mixed-bits", type=float, default=5.0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    calib, eval_text = load_wikitext()
    cfg = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).config
    max_len = min(getattr(cfg, "max_position_embeddings", 1024) or 1024, 1024)
    del cfg; gc.collect()

    rows = []

    def record(label, model, bits):
        # bnb models are pre-placed on CUDA (device_map) and must not be .to()'d;
        # AtlasInfer/fp16 models come back on CPU and need moving.
        on_cuda = next(model.parameters()).device.type == "cuda"
        model = model.eval() if on_cuda else model.to(dev).eval()
        ppl = evaluate_perplexity(model, tok, eval_text, dev,
                                  max_len=max_len, stride=max_len // 2,
                                  max_tokens=args.eval_tokens)
        mb = resident_bytes(model) / 1024 ** 2
        print(f"  {label:<22} {mb:8.1f} MB  ppl={ppl:8.3f}  (~{bits} bits)")
        rows.append({"method": label, "mb": mb, "ppl": ppl, "bits": bits})
        model.to("cpu"); del model; gc.collect(); torch.cuda.empty_cache()

    def fp16():
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)

    print(f"\nComparing on {args.model} (WikiText-2, {args.eval_tokens} eval tokens)\n")

    # FP16 baseline.
    record("fp16", fp16(), 16)

    # AtlasInfer uniform (symmetric int4 vs NF4 to show the codebook's effect).
    record("AtlasInfer int8", quantize_model(fp16(), precision="int8", verbose=False), 8)
    record("AtlasInfer int4 (sym)",
           quantize_model(fp16(), precision="int4", quant_4bit="int4", verbose=False), 4)
    record("AtlasInfer nf4",
           quantize_model(fp16(), precision="int4", quant_4bit="nf4", verbose=False), 4)

    # AtlasInfer mixed (profile once, allocate at target bits).
    base = fp16().to(dev).eval()
    profiles = SensitivityProfiler().profile_end_to_end(base, tokenizer=tok, calibration_texts=calib)
    base.to("cpu"); del base; gc.collect(); torch.cuda.empty_cache()
    n_params = sum(p.param_count for p in profiles.values())
    alloc = allocate_optimal(profiles, budget_bytes=int(n_params * args.mixed_bits / 8))
    record(f"AtlasInfer mixed-{args.mixed_bits:g}bit",
           quantize_model_mixed(fp16(), allocation=alloc.allocations, verbose=False),
           round(alloc.avg_bits, 1))

    # bitsandbytes int8 (LLM.int8()) and nf4.
    bnb8 = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0}, dtype=torch.float16)
    record("bnb int8 (LLM.int8)", bnb8, 8)

    bnb4 = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16))
    record("bnb nf4", bnb4, 4)

    # Markdown table.
    fp16_ppl = next(r["ppl"] for r in rows if r["method"] == "fp16")
    print("\n| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['method']} | {r['bits']} | {r['mb']:.1f} | "
              f"{r['ppl']:.3f} | {r['ppl'] - fp16_ppl:+.3f} |")


if __name__ == "__main__":
    main()

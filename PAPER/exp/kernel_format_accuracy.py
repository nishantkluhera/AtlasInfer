"""
Closes the repo's biggest unmeasured gap: what does the KERNEL format cost in
accuracy?

AtlasInfer ships two different quantizers (see docs/tech_debt.md #6):

  eager  -- per-block (64/128) scales, sparse FP16 outliers, NF4 codebook.
            Every perplexity number in results/ comes from this.
  kernel -- per-output-channel symmetric int8/int4, no blocks, no outliers,
            no codebook. Every speed number comes from this.

Nothing measured both on the same model, so "what is the perplexity of the thing
you benchmarked for speed?" had no answer. This script answers it.

It does NOT need Triton: W8A16Linear/W4A16Linear carry an eager dequant fallback
that is numerically identical to the fused kernel (the kernel test asserts they
agree to 3e-4). So the *format's* accuracy is measurable on any CUDA box --
Windows included -- even though the *speed* claim needs WSL2.

    python PAPER/exp/kernel_format_accuracy.py --model Qwen/Qwen2.5-0.5B
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from atlasinfer import seed_everything  # noqa: E402
from atlasinfer.evaluation import evaluate_perplexity, load_wikitext, resident_bytes  # noqa: E402
from atlasinfer.patcher import quantize_model  # noqa: E402
from atlasinfer.triton_kernels import HAS_TRITON  # noqa: E402

hf_logging.set_verbosity_error()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--eval-tokens", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="PAPER/exp/results")
    args = ap.parse_args()

    seed_everything(args.seed)
    assert torch.cuda.is_available(), "needs CUDA"
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    _calib, eval_text = load_wikitext()
    cfg = AutoConfig.from_pretrained(args.model)
    max_len = min(getattr(cfg, "max_position_embeddings", 1024) or 1024, 1024)

    print(f"model={args.model}  eval_tokens={args.eval_tokens}  seed={args.seed}")
    print(f"Triton available: {HAS_TRITON} "
          f"(irrelevant to accuracy -- the eager fallback is numerically identical)\n")

    def load():
        return AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, low_cpu_mem_usage=True)

    rows = []

    def measure(label, model, fmt, bits):
        model = model.to(dev).eval()
        t0 = time.time()
        ppl = evaluate_perplexity(model, tok, eval_text, dev, max_len=max_len,
                                  stride=max_len // 2, max_tokens=args.eval_tokens)
        mb = resident_bytes(model) / 1024 ** 2
        rows.append({"config": label, "format": fmt, "nominal_bits": bits,
                     "ppl": ppl, "mb": mb})
        print(f"  {label:<28} ppl={ppl:8.4f}  {mb:8.1f} MB  ({time.time()-t0:.0f}s)")
        model.to("cpu"); del model; gc.collect(); torch.cuda.empty_cache()

    measure("fp16", load(), "dense", 16)

    # Eager: block-wise + sparse outliers (+ NF4 at 4-bit). What results/ reports.
    measure("eager int8 (block+outlier)",
            quantize_model(load(), precision="int8", verbose=False), "eager", 8)
    measure("eager nf4 (block+outlier)",
            quantize_model(load(), precision="int4", quant_4bit="nf4", verbose=False),
            "eager", 4)

    # Kernel: per-output-channel symmetric, no blocks/outliers/codebook. What
    # bench_triton_kernel.py reports speed for.
    measure("kernel W8A16 (per-channel)",
            quantize_model(load(), precision="int8", use_kernel=True, verbose=False),
            "kernel", 8)
    measure("kernel W4A16 (per-channel)",
            quantize_model(load(), precision="int4", use_kernel=True, verbose=False),
            "kernel", 4)

    fp16_ppl = rows[0]["ppl"]
    for r in rows:
        r["delta_vs_fp16"] = r["ppl"] - fp16_ppl

    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    payload = {
        "model": args.model, "eval_tokens": args.eval_tokens, "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "has_triton": HAS_TRITON,
        "note": ("Accuracy of the two quantization FORMATS. The kernel rows use "
                 "the eager dequant fallback, which is numerically identical to "
                 "the fused Triton kernel (tests/test_triton_kernels.py asserts "
                 "3e-4 agreement), so these numbers are valid without Triton."),
        "rows": rows,
    }
    path = os.path.join(args.out, f"kernel_format_accuracy_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n| Config | format | ~bits | MB | Perplexity | delta vs FP16 |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['config']} | {r['format']} | {r['nominal_bits']} | "
              f"{r['mb']:.1f} | {r['ppl']:.4f} | {r['delta_vs_fp16']:+.4f} |")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

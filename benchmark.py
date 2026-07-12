"""
AtlasInfer benchmark - measure the accuracy/memory trade-off on real models.

For a given model this script evaluates WikiText-2 perplexity and weight-memory
footprint for several quantization configurations:

  * fp16            - dense baseline (no quantization)
  * uniform int8    - every layer 8-bit
  * uniform int4    - every layer 4-bit
  * mixed @ N bits  - AtlasInfer's sensitivity-driven allocation at a target
                      average bit-width (a sweep)

The point of the sweep is to show that mixed precision dominates the uniform
options: at a memory footprint between INT4 and INT8 it stays far closer to the
FP16 perplexity than uniform quantization can.

Usage:
    python benchmark.py --model gpt2
    python benchmark.py --model facebook/opt-125m --eval-tokens 40000
    python benchmark.py --model gpt2 --bits 4.5 5 6 7
"""
import argparse
import gc
import json
import os
import time
from typing import List, Optional

import torch

# Quieten HF noise so the benchmark output stays readable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from atlasinfer.linear import QuantizedLinear, QuantizedLinear4bit  # noqa: E402
from atlasinfer.patcher import quantize_model, quantize_model_mixed  # noqa: E402
from atlasinfer.sensitivity import SensitivityProfiler  # noqa: E402
from atlasinfer.allocator import allocate_optimal  # noqa: E402

hf_logging.set_verbosity_error()


# --------------------------------------------------------------------------- #
# Memory accounting
# --------------------------------------------------------------------------- #
def model_weight_bytes(model: torch.nn.Module) -> int:
    """Resident weight footprint: quantized buffers + remaining dense params."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedLinear4bit)):
            total += module.quantized_weights.memory_bytes()
            if module.bias is not None:
                total += module.bias.numel() * module.bias.element_size()
    return total


def linear_param_count(model: torch.nn.Module, exclude=("embed", "lm_head", "norm", "ln_")) -> int:
    """Total parameters in the quantizable linear layers (drives the budget)."""
    from atlasinfer.allocator import get_layer_sizes
    sizes = get_layer_sizes(model)
    return sum(
        n for name, n in sizes.items()
        if not any(p in name.lower() for p in exclude)
    )


# --------------------------------------------------------------------------- #
# Perplexity (standard sliding-window negative log-likelihood)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_perplexity(
    model, tokenizer, text: str, device, max_len: int = 1024, stride: int = 512,
    max_tokens: Optional[int] = None,
) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    input_ids = input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: List[torch.Tensor] = []
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100  # only score the new tokens in this window
        out = model(ids, labels=target)
        # out.loss is the mean NLL over (trg_len - 1) scored positions.
        nlls.append(out.loss.float() * trg_len)
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break
    return float(torch.exp(torch.stack(nlls).sum() / n_tokens))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_wikitext():
    """Return (calibration_texts, eval_text) from WikiText-2."""
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(t for t in test["text"] if t.strip())

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Plenty of calibration docs: the sensitivity profiler caps at its own
    # max_samples (8), while GPTQ consumes many more for a well-conditioned Hessian.
    calib = [t for t in train["text"] if len(t.strip()) > 200][:256]
    return calib, eval_text


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def fresh_model(model_name: str, dtype=torch.float16, device_map=None):
    return AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True, device_map=device_map)


def run(model_name: str, eval_tokens: int, bit_targets: List[float], device: torch.device,
        device_map: bool = False):
    print(f"\n{'='*72}\nBenchmarking {model_name} on {device}\n{'='*72}")
    dmap = "auto" if device_map else None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    calib_texts, eval_text = load_wikitext()

    # Context window: respect the model's max positions, capped for speed.
    cfg = AutoConfig.from_pretrained(model_name)
    max_len = min(getattr(cfg, "max_position_embeddings", 1024) or 1024, 1024)

    results = []

    def measure(label: str, model, avg_bits: Optional[float]):
        model = model.eval() if device_map else model.to(device).eval()
        t0 = time.time()
        ppl = evaluate_perplexity(
            model, tokenizer, eval_text, device,
            max_len=max_len, stride=max_len // 2, max_tokens=eval_tokens,
        )
        dt = time.time() - t0
        mb = model_weight_bytes(model) / 1024 ** 2
        bits = avg_bits if avg_bits is not None else 16.0
        print(f"  {label:<18} ppl={ppl:8.3f}  weights={mb:8.1f} MB  "
              f"avg_bits={bits:4.1f}  ({dt:.1f}s)")
        results.append({"config": label, "ppl": ppl, "mb": mb, "avg_bits": bits})
        if not device_map:
            model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def load():
        return fresh_model(model_name, device_map=dmap)

    # 1) FP16 baseline
    measure("fp16", load(), 16.0)

    # 2) Uniform INT8 / INT4
    measure("uniform-int8", quantize_model(load(), precision="int8", verbose=False), 8.0)
    # 4-bit defaults to NF4 (the better codebook); see compare_baselines.py.
    measure("uniform-nf4", quantize_model(load(), precision="int4", verbose=False), 4.0)

    # 3) Mixed precision sweep. Profile ONCE, reuse across budgets.
    print("  profiling layer sensitivities end-to-end (once)...")
    base = load().eval() if device_map else load().to(device).eval()
    profiler = SensitivityProfiler()
    profiles = profiler.profile_end_to_end(
        base, tokenizer=tokenizer, calibration_texts=calib_texts
    )
    if not device_map:
        base.to("cpu")
    del base
    gc.collect()
    torch.cuda.empty_cache()

    n_params = sum(p.param_count for p in profiles.values())
    for target_bits in bit_targets:
        budget_bytes = int(n_params * target_bits / 8)
        alloc = allocate_optimal(profiles, budget_bytes=budget_bytes)
        actual_bits = alloc.avg_bits
        m = quantize_model_mixed(load(), allocation=alloc.allocations, verbose=False)
        measure(f"mixed-{target_bits:g}bit", m, actual_bits)

    return results


def write_outputs(model_name: str, results: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.replace("/", "_")

    # JSON
    with open(os.path.join(out_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    fp16_ppl = next(r["ppl"] for r in results if r["config"] == "fp16")
    lines = [
        f"### {model_name}",
        "",
        "| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        delta = r["ppl"] - fp16_ppl
        lines.append(
            f"| {r['config']} | {r['avg_bits']:.1f} | {r['mb']:.1f} | "
            f"{r['ppl']:.3f} | {delta:+.3f} |"
        )
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, f"{safe}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)

    # Pareto figure: perplexity vs memory, uniform vs mixed.
    try:
        _plot(model_name, results, os.path.join(out_dir, f"{safe}.png"))
    except Exception as exc:  # pragma: no cover
        print(f"(skipped plot: {exc})")


def _plot(model_name: str, results: List[dict], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uni = [r for r in results if r["config"].startswith(("fp16", "uniform"))]
    mix = [r for r in results if r["config"].startswith("mixed")]
    uni.sort(key=lambda r: r["mb"])
    mix.sort(key=lambda r: r["mb"])

    fig, ax = plt.subplots(figsize=(7, 5))
    if uni:
        ax.plot([r["mb"] for r in uni], [r["ppl"] for r in uni],
                "o--", color="#888", label="uniform / fp16")
        for r in uni:
            ax.annotate(r["config"], (r["mb"], r["ppl"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
    if mix:
        ax.plot([r["mb"] for r in mix], [r["ppl"] for r in mix],
                "s-", color="#1f77b4", label="AtlasInfer mixed")
    ax.set_xlabel("Weight memory (MB)")
    ax.set_ylabel("WikiText-2 perplexity (lower is better)")
    ax.set_title(f"Accuracy vs memory - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description="AtlasInfer benchmark")
    ap.add_argument("--model", "-m", default="gpt2", help="HF model name")
    ap.add_argument("--eval-tokens", type=int, default=30000,
                    help="Cap on eval tokens (lower = faster, less precise)")
    ap.add_argument("--bits", type=float, nargs="+", default=[4.5, 5.0, 6.0],
                    help="Target average bit-widths for the mixed sweep")
    ap.add_argument("--out", default="results", help="Output directory")
    ap.add_argument("--device-map", action="store_true",
                    help="shard across all GPUs (device_map=auto) for big models, e.g. on Kaggle T4x2")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    args = ap.parse_args()

    from atlasinfer import seed_everything
    seed_everything(args.seed)
    print(f"seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(args.model, args.eval_tokens, args.bits, device, device_map=args.device_map)
    write_outputs(args.model, results, args.out)


if __name__ == "__main__":
    main()

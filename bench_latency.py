"""
AtlasInfer latency / runtime-memory benchmark.

Where `benchmark.py` measures *accuracy* (perplexity) vs *stored* weight size,
this script measures the **runtime** story during generation:

  * peak GPU memory actually used (the practical "does it fit?" number)
  * decode throughput in tokens/sec

It makes the honest tradeoff explicit: weight-only quantization shrinks the
resident footprint (only one layer is dequantized to FP16 at a time), but because
that dequant happens on every matmul with no fused low-bit kernel, it costs some
latency rather than saving it. See the README's "what this does and doesn't do".

    python bench_latency.py --model gpt2
    python bench_latency.py --model EleutherAI/pythia-410m --gen-tokens 64
"""
import argparse
import gc
import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from atlasinfer.patcher import quantize_model, quantize_model_mixed
from atlasinfer.sensitivity import SensitivityProfiler
from atlasinfer.allocator import allocate_optimal

hf_logging.set_verbosity_error()


def fresh_model(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, low_cpu_mem_usage=True)


@torch.no_grad()
def measure(model, tokenizer, device, prompt, gen_tokens, repeats=3):
    """Peak GPU memory and decode throughput, best of ``repeats`` timed runs.

    Best-of-N rather than mean: on a laptop GPU the distribution is
    right-skewed by thermal/clock noise and background work, so the minimum time
    is the most stable estimate of the machine's actual capability. All samples
    are returned so the spread can be reported rather than hidden.
    """
    model = model.to(device).eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup (also triggers any lazy kernel autotuning before the clock starts).
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        new_tokens = out.shape[1] - ids.shape[1]
        samples.append(new_tokens / dt)

    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return peak_mb, max(samples), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="gpt2")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="The history of computing began")
    ap.add_argument("--mixed-bits", type=float, default=5.0)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed generate() runs per config; the best is reported")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # deterministic=False on purpose: this measures speed, and pinning cuDNN to
    # deterministic kernels would benchmark the slow path (see reproducibility.py).
    from atlasinfer import seed_everything
    seed_everything(args.seed, deterministic=False)

    assert torch.cuda.is_available(), "this benchmark needs a CUDA GPU"
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = []

    def run(label, model):
        peak, tps, samples = measure(model, tok, device, args.prompt,
                                     args.gen_tokens, repeats=args.repeats)
        spread = (max(samples) - min(samples)) / max(samples) * 100 if samples else 0.0
        print(f"  {label:<14} peak GPU {peak:7.1f} MB   {tps:6.1f} tok/s "
              f"(spread {spread:.1f}% over {len(samples)} runs)")
        rows.append((label, peak, tps, samples))

    print(f"\nLatency / memory benchmark - {args.model} (decode {args.gen_tokens} tokens)\n")
    run("fp16", fresh_model(args.model))
    run("uniform-int8", quantize_model(fresh_model(args.model), precision="int8", verbose=False))
    run("uniform-nf4", quantize_model(fresh_model(args.model), precision="int4", verbose=False))

    # Mixed precision at the requested average bit-width.
    base = fresh_model(args.model).to(device).eval()
    profiles = SensitivityProfiler().profile_end_to_end(base, tokenizer=tok)
    base.to("cpu"); del base; gc.collect(); torch.cuda.empty_cache()
    n_params = sum(p.param_count for p in profiles.values())
    alloc = allocate_optimal(profiles, budget_bytes=int(n_params * args.mixed_bits / 8))
    run(f"mixed-{args.mixed_bits:g}bit",
        quantize_model_mixed(fresh_model(args.model), allocation=alloc.allocations, verbose=False))

    fp16_peak = rows[0][1]
    fp16_tps = rows[0][2]
    lines = ["| Config | Peak GPU (MB) | vs FP16 | tok/s | rel. speed |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, peak, tps, _s in rows:
        lines.append(f"| {label} | {peak:.1f} | {peak/fp16_peak:.2f}x | "
                     f"{tps:.1f} | {tps/fp16_tps:.2f}x |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)

    # Persist. This table was previously printed to stdout only, so the README's
    # runtime numbers had no committed source and could not be re-checked -- the
    # exact failure mode tests/test_readme_consistency.py exists to prevent.
    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    payload = {
        "model": args.model,
        "gen_tokens": args.gen_tokens,
        "prompt": args.prompt,
        "mixed_bits": args.mixed_bits,
        "seed": args.seed,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "rows": [
            {"config": label, "peak_mb": peak, "tok_s": tps,
             "peak_vs_fp16": peak / fp16_peak, "speed_vs_fp16": tps / fp16_tps,
             # All timed samples, not just the reported best: a reader can see
             # the spread instead of taking a single number on faith.
             "tok_s_samples": samples}
            for label, peak, tps, samples in rows
        ],
    }
    with open(os.path.join(args.out, f"latency_{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(args.out, f"latency_{safe}.md"), "w", encoding="utf-8") as f:
        f.write(f"### {args.model} - decode latency / peak memory "
                f"({args.gen_tokens} tokens, {torch.cuda.get_device_name(0)}, "
                f"best of {args.repeats})\n\n" + table)
    print(f"Wrote {args.out}/latency_{safe}.{{md,json}}")


if __name__ == "__main__":
    main()

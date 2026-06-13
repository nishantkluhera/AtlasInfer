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
def measure(model, tokenizer, device, prompt, gen_tokens):
    model = model.to(device).eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warmup.
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    t0 = time.time()
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0

    new_tokens = out.shape[1] - ids.shape[1]
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    tok_s = new_tokens / dt
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return peak_mb, tok_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="gpt2")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="The history of computing began")
    ap.add_argument("--mixed-bits", type=float, default=5.0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this benchmark needs a CUDA GPU"
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = []

    def run(label, model):
        peak, tps = measure(model, tok, device, args.prompt, args.gen_tokens)
        print(f"  {label:<14} peak GPU {peak:7.1f} MB   {tps:6.1f} tok/s")
        rows.append((label, peak, tps))

    print(f"\nLatency / memory benchmark - {args.model} (decode {args.gen_tokens} tokens)\n")
    run("fp16", fresh_model(args.model))
    run("uniform-int8", quantize_model(fresh_model(args.model), precision="int8", verbose=False))
    run("uniform-int4", quantize_model(fresh_model(args.model), precision="int4", verbose=False))

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
    print("\n| Config | Peak GPU (MB) | vs FP16 | tok/s | rel. speed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for label, peak, tps in rows:
        print(f"| {label} | {peak:.1f} | {peak/fp16_peak:.2f}x | {tps:.1f} | {tps/fp16_tps:.2f}x |")


if __name__ == "__main__":
    main()

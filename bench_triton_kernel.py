"""
Microbenchmark for the fused W8A16 Triton kernel (run under Linux/WSL2 + CUDA).

Compares three ways to compute one quantized linear at decode-ish shapes:

  1. fp16            - torch F.linear on FP16 weights (the speed ceiling)
  2. dequant+matmul  - AtlasInfer's current path: rebuild the FP16 weight, F.linear
  3. fused (triton)  - read int8 weights directly, dequantize in-register, matmul

The point: (3) should beat (2) decisively (no full-weight materialization) and,
at batch-1 decode, approach or beat (1) because it streams half the weight bytes.

    ~/atlasvenv/bin/python /mnt/c/.../AtlasInfer/bench_triton_kernel.py
"""
import importlib.util
import json
import os
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "atlas_triton_kernels", os.path.join(_HERE, "atlasinfer", "triton_kernels.py")
)
tk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tk)


def bench(fn, iters=300, warmup=50, windows=5):
    """Best-of-N timing windows; returns (best_ms, all_window_ms).

    Best-of rather than mean because the distribution is right-skewed by clock
    and thermal noise on a laptop GPU. All windows are returned so the spread can
    be REPORTED rather than hidden: at small shapes this benchmark has been seen
    to vary ~50% between invocations, which makes a single number to two decimal
    places actively misleading.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(windows):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.time() - t0) / iters * 1e3)
    return min(samples), samples  # ms


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--peak-bw", type=float, default=None,
                    help="theoretical peak HBM bandwidth in GB/s; enables the "
                         "bandwidth-efficiency line (RTX 3060 Laptop = 336)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    assert tk.HAS_TRITON, "Triton not available"
    dev = "cuda"
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch {torch.__version__}\n")

    # (M, K, N): M=1/4 are decode; larger M are prefill-ish.
    shapes = [
        (1, 4096, 4096),
        (1, 4096, 11008),
        (1, 5120, 5120),
        (4, 4096, 4096),
        (16, 4096, 4096),
    ]

    # Speedup alone flatters a low-bit kernel: at batch-1 the matmul is bound by
    # weight streaming, so int8 gets 2x and int4 gets 4x essentially for free.
    # What actually says whether the kernel is any good is the fraction of that
    # headroom it captures -- reported here as achieved GB/s and as a percentage
    # of the ideal speedup. See PAPER/01_go_nogo.md 2c.
    rows = []
    print(f"{'shape (M,K,N)':>20} | {'fp16':>8} | {'w8a16':>8} | {'w4a16':>8} | "
          f"{'w8/fp16':>8} | {'w4/fp16':>8} | {'w8 eff':>7} | {'w4 eff':>7} | "
          f"{'w8 err':>7} | {'w4 err':>7} | {'fp16 sd':>7}")
    print("-" * 130)
    for (M, K, N) in shapes:
        x = torch.randn(M, K, device=dev, dtype=torch.float16)
        W = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.05
        qW, s8 = tk.quantize_w8a16(W)
        pW, s4 = tk.quantize_w4a16(W)

        def f_fp16():
            return torch.nn.functional.linear(x, W)

        def f_w8():
            return tk.w8a16_linear(x, qW, s8)

        def f_w4():
            return tk.w4a16_linear(x, pW, s4)

        ref = f_fp16().float()
        e8 = ((ref - f_w8().float()).norm() / ref.norm()).item()
        e4 = ((ref - f_w4().float()).norm() / ref.norm()).item()

        t_fp16, s_fp16 = bench(f_fp16)
        t_w8, s_w8 = bench(f_w8)
        t_w4, s_w4 = bench(f_w4)
        # Spread of the FP16 baseline is the honesty check: if the reference is
        # unstable, every speedup derived from it is unstable too.
        spread16 = (max(s_fp16) - min(s_fp16)) / max(s_fp16) * 100

        # Weight bytes streamed per call, and the bandwidth that implies.
        wb16 = K * N * 2
        bw16, bw8, bw4 = (wb16 / (t_fp16 * 1e-3) / 1e9,
                          (wb16 / 2) / (t_w8 * 1e-3) / 1e9,
                          (wb16 / 4) / (t_w4 * 1e-3) / 1e9)
        # Fraction of the ideal (bandwidth-bound) speedup actually captured.
        eff8, eff4 = (t_fp16 / t_w8) / 2.0, (t_fp16 / t_w4) / 4.0

        flag = " !" if spread16 > 10 else ""
        print(f"{str((M,K,N)):>20} | {t_fp16:8.4f} | {t_w8:8.4f} | {t_w4:8.4f} | "
              f"{t_fp16/t_w8:7.2f}x | {t_fp16/t_w4:7.2f}x | {eff8*100:6.1f}% | "
              f"{eff4*100:6.1f}% | {e8:7.4f} | {e4:7.4f} | {spread16:5.1f}%{flag}")
        rows.append({
            "M": M, "K": K, "N": N,
            "ms_fp16": t_fp16, "ms_w8a16": t_w8, "ms_w4a16": t_w4,
            "speedup_w8": t_fp16 / t_w8, "speedup_w4": t_fp16 / t_w4,
            "ideal_speedup_w8": 2.0, "ideal_speedup_w4": 4.0,
            "efficiency_w8": eff8, "efficiency_w4": eff4,
            "gbps_fp16": bw16, "gbps_w8a16": bw8, "gbps_w4a16": bw4,
            "rel_err_w8": e8, "rel_err_w4": e4,
            "fp16_spread_pct": spread16,
            "ms_fp16_samples": s_fp16, "ms_w8a16_samples": s_w8,
            "ms_w4a16_samples": s_w4,
        })

    if args.peak_bw:
        best = max(r["gbps_fp16"] for r in rows)
        pct = best / args.peak_bw * 100
        if pct >= 75:
            print(f"\nfp16 baseline peaks at {best:.1f} GB/s = {pct:.0f}% of the "
                  f"stated {args.peak_bw:.0f} GB/s -- a competent baseline.")
        else:
            print(f"\nWARNING: fp16 baseline only reached {best:.1f} GB/s = "
                  f"{pct:.0f}% of the stated {args.peak_bw:.0f} GB/s. The GPU was "
                  f"probably contended or throttled -- every speedup above is "
                  f"inflated against a hobbled baseline. Re-run on an idle GPU.")

    os.makedirs(args.out, exist_ok=True)
    safe = gpu.replace(" ", "_").replace("/", "_")
    payload = {
        "gpu": gpu,
        "torch": torch.__version__,
        "triton": getattr(getattr(tk, "triton", None), "__version__", "unknown"),
        "peak_bw_gbps": args.peak_bw,
        "iters": 300, "warmup": 50, "timing": "best-of-5 windows",
        "note": ("Synthetic torch.randn matrices, single GEMM. This is NOT "
                 "end-to-end decode -- see results/latency_*.json for that."),
        "rows": rows,
    }
    path = os.path.join(args.out, f"triton_kernel_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Generated markdown alongside the JSON. The previous kernel table in
    # results/ was transcribed by hand from stdout, which is exactly the
    # provenance gap the rest of the harnesses avoid.
    md = [
        f"### Fused Triton kernels — {gpu}",
        "",
        f"`bench_triton_kernel.py`, best-of-5 windows of {300} iters, torch "
        f"{torch.__version__} / Triton {payload['triton']}. Latency in ms.",
        "",
        "**Read the efficiency columns, not the speedups.** At batch-1 the matmul is "
        "bound by weight streaming, so int8 gets ~2x and int4 ~4x for free; what "
        "matters is how much of that headroom the kernel captures.",
        "",
        "**The `fp16 spread` column is WITHIN-run only** (across the 5 timing "
        "windows of a single invocation) and understates the real uncertainty. "
        "Across separate invocations on a laptop GPU, the (1, 4096, 4096) row has "
        "been observed anywhere from 1.01x to 1.66x — Triton autotunes once per "
        "shape, so whichever tile it picks depends on the clock state at that "
        "moment. Quote the largest shape (1, 4096, 11008), which is consistently "
        "~2.0x for W8A16 across runs, or quote a range. Do not quote a small-shape "
        "number to two decimal places.",
        "",
        "| shape (M, K, N) | fp16 | W8A16 | W4A16 | W8 vs fp16 (of 2.0x) | W4 vs fp16 (of 4.0x) | W8 err | W4 err | fp16 spread |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| ({r['M']}, {r['K']}, {r['N']}) | {r['ms_fp16']:.4f} | "
            f"{r['ms_w8a16']:.4f} | {r['ms_w4a16']:.4f} | "
            f"{r['speedup_w8']:.2f}x ({r['efficiency_w8']*100:.0f}%) | "
            f"{r['speedup_w4']:.2f}x ({r['efficiency_w4']*100:.0f}%) | "
            f"{r['rel_err_w8']:.4f} | {r['rel_err_w4']:.4f} | "
            f"{r['fp16_spread_pct']:.1f}%{' ⚠' if r['fp16_spread_pct'] > 10 else ''} |")
    over = [r for r in rows if max(r["efficiency_w8"], r["efficiency_w4"]) > 1.0]
    if over:
        md += ["", "> ⚠ One or more rows show **efficiency above 100%**, which is "
                   "impossible for a purely bandwidth-bound comparison: the kernel "
                   "cannot beat the ceiling set by reading half (or a quarter) of "
                   "the bytes. It means the FP16 reference ran slow in this "
                   "invocation, so the speedups here are optimistic. Shapes "
                   "affected: " + ", ".join(f"({r['M']}, {r['K']}, {r['N']})"
                                             for r in over) + "."]
    if args.peak_bw:
        best = max(r["gbps_fp16"] for r in rows)
        pct = best / args.peak_bw * 100
        # Don't assert "competent baseline" unconditionally -- if the FP16 row is
        # far off peak the run is contended or thermally throttled, and every
        # speedup in the table is inflated against a baseline that was hobbled.
        verdict = ("a competent baseline, not a straw man"
                   if pct >= 75 else
                   "**well below peak — treat this run as unreliable**: the GPU was "
                   "likely contended or throttled, which inflates every speedup here")
        md += ["", f"fp16 baseline peaks at {best:.1f} GB/s = {pct:.0f}% of this "
                   f"card's {args.peak_bw:.0f} GB/s — {verdict}."]
    md += ["",
           "The `W4 err` column (~0.156) is the *int4 quantization* error vs FP16, not "
           "a kernel bug — the W4A16 kernel reconstructs its own packed weights to "
           "~3e-4 (`tests/test_triton_kernels.py::TestFusedKernel`). But note that "
           "per-channel int4 costs **+14.39 perplexity** end-to-end, so this format "
           "is not deployable; see `docs/tech_debt.md` #6.",
           "",
           "Generated by `python bench_triton_kernel.py --peak-bw <peak>`; "
           f"source of truth is `{os.path.basename(path)}`.", ""]
    md_path = os.path.join(args.out, f"triton_kernel_{safe}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nWrote {path} and {md_path}")


if __name__ == "__main__":
    main()

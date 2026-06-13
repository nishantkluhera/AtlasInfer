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
import os
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "atlas_triton_kernels", os.path.join(_HERE, "atlasinfer", "triton_kernels.py")
)
tk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tk)


def bench(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Best-of-3 timing windows to reduce laptop-GPU clock/thermal noise.
    best = float("inf")
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / iters * 1e3)
    return best  # ms


def main():
    assert torch.cuda.is_available(), "need CUDA"
    assert tk.HAS_TRITON, "Triton not available"
    dev = "cuda"
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    # (M, K, N): M=1/4 are decode; larger M are prefill-ish.
    shapes = [
        (1, 4096, 4096),
        (1, 4096, 11008),
        (1, 5120, 5120),
        (4, 4096, 4096),
        (16, 4096, 4096),
    ]

    print(f"{'shape (M,K,N)':>20} | {'fp16':>8} | {'w8a16':>8} | {'w4a16':>8} | "
          f"{'w8/fp16':>8} | {'w4/fp16':>8} | {'w8 err':>7} | {'w4 err':>7}")
    print("-" * 96)
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

        t_fp16 = bench(f_fp16)
        t_w8 = bench(f_w8)
        t_w4 = bench(f_w4)
        print(f"{str((M,K,N)):>20} | {t_fp16:8.4f} | {t_w8:8.4f} | {t_w4:8.4f} | "
              f"{t_fp16/t_w8:7.2f}x | {t_fp16/t_w4:7.2f}x | {e8:7.4f} | {e4:7.4f}")


if __name__ == "__main__":
    main()

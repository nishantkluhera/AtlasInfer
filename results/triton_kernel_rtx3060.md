### Fused Triton kernel — RTX 3060 Laptop (Ampere), WSL2

`bench_triton_kernel.py`, best-of-3, torch 2.5.1+cu124 / Triton 3.1.0, run under
WSL2 (Ubuntu 24.04) on the host RTX 3060. Latency in ms; speedups are fp16/kernel.

| shape (M, K, N) | fp16 | W8A16 | W4A16 | W8 vs fp16 | W4 vs fp16 | W8 err | W4 err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (1, 4096, 4096)  | 0.156 | 0.099 | 0.088 | **1.58×** | **1.78×** | 0.009 | 0.156 |
| (1, 4096, 11008) | 0.337 | 0.173 | 0.123 | **1.95×** | **2.74×** | 0.009 | 0.156 |
| (1, 5120, 5120)  | 0.198 | 0.112 | 0.109 | **1.78×** | **1.82×** | 0.009 | 0.156 |
| (4, 4096, 4096)  | 0.131 | 0.082 | 0.088 | 1.60× | 1.49× | 0.009 | 0.156 |
| (16, 4096, 4096) | 0.134 | 0.088 | 0.129 | 1.52× | 1.04× | 0.009 | 0.158 |

At batch-1 (memory-bandwidth bound) both fused kernels beat FP16 by streaming
half/a-quarter of the weight bytes; the W8A16 edge holds to ~1.5× even at M=16
while W4A16 falls to ~1.0× as the matmul becomes compute-bound. W8A16 is
near-lossless (<1% error). The `W4 err` column (~0.156) is the *int4 quantization*
error vs FP16, not a kernel bug — the W4A16 kernel *math* reconstructs its own
packed weights to 0.0003 rel error (see the correctness checks in
`tests/test_triton_kernels.py::TestFusedKernel`). Reproduce per
[docs/wsl_triton.md](../docs/wsl_triton.md):
`~/atlasvenv/bin/python bench_triton_kernel.py`.

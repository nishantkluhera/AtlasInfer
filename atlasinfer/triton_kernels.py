"""
Fused dequantize + matmul Triton kernel (W8A16).

The rest of AtlasInfer dequantizes a whole weight matrix to FP16 and then calls
``F.linear`` — correct, but it materializes the full FP16 weight every forward and
reads/writes more memory than a plain FP16 matmul, so it's a memory win at a
latency cost. This kernel instead reads the **int8** weights directly (half the
bytes of FP16), dequantizes them in-register with a per-output-channel scale, and
does the matmul in one pass. At batch-1 decode — where the matmul is bound by how
fast weights stream from HBM — reading int8 instead of FP16 is the speedup.

Triton is Linux+GPU only (incl. WSL2). This module imports cleanly without it;
``HAS_TRITON`` is False and the helpers raise if called. Quantization here is
symmetric per-output-channel int8 (no per-block / outlier handling) — a clean,
self-contained kernel demo rather than a drop-in for the full AtlasInfer format.

The GEMM tiling (block sizes, warps, pipeline stages) is chosen by
``@triton.autotune`` per ``(M, N, K)`` so the kernel ports across GPU
architectures: a single fixed tile that's near-optimal on Ampere (cp.async
software pipelining) stalls badly on Turing (no cp.async), so we let Triton pick
the schedule for whatever card it runs on.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - platform dependent
    HAS_TRITON = False


def kernel_available() -> bool:
    """True if the fused kernel can actually run (Triton present + CUDA GPU)."""
    return HAS_TRITON and torch.cuda.is_available()


def quantize_w8a16(weight: torch.Tensor):
    """Symmetric per-output-channel int8 quantization, stored **transposed** for the kernel.

    Args:
        weight: dense weight of shape (out_features N, in_features K).
    Returns:
        (qweight int8 [K, N], scale fp16 [N]). The int8 weight is returned
        transposed to (K, N) so the fused GEMM's weight tile load is **coalesced**
        — consecutive output channels (the kernel's fast tile axis) are contiguous
        in memory. With the natural (N, K) layout that load is strided by K, which
        Ampere hides via cp.async but Turing (T4) can't, costing ~7x. The eager
        fallback transposes back.
    """
    scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.t().contiguous(), scale.to(torch.float16)   # (K, N), coalesced for the kernel


def quantize_w4a16(weight: torch.Tensor):
    """Symmetric per-output-channel int4 quantization, packed 2-per-byte along K.

    Each packed byte holds two consecutive K values for one output channel: low
    nibble = even K, high nibble = odd K, each stored as ``value + 8`` so the
    nibble is unsigned [1,15]. The packed tensor is returned **transposed** to
    (K//2, N) so the fused GEMM's weight tile load is coalesced (see
    :func:`quantize_w8a16`); the eager fallback transposes back.

    Args:
        weight: dense weight (out_features N, in_features K). K must be even
            (always true for transformer layers).
    Returns:
        (packed uint8 [K//2, N], scale fp16 [N]).
    """
    N, K = weight.shape
    if K % 2 != 0:
        raise ValueError(f"W4A16 needs an even in_features, got K={K}")
    scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 7.0
    q = (weight / scale[:, None]).round().clamp(-7, 7).to(torch.int8)  # (N, K)
    low_u = (q[:, 0::2] + 8).to(torch.uint8) & 0xF   # even K -> low nibble
    high_u = (q[:, 1::2] + 8).to(torch.uint8) & 0xF  # odd  K -> high nibble
    packed = ((high_u << 4) | low_u)                 # (N, K//2) uint8
    return packed.t().contiguous(), scale.to(torch.float16)   # (K//2, N), coalesced


def _unpack_w4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack kernel-layout (K//2, N) nibble-packed weights to (N, K) int8 in [-7,7]."""
    p = packed.t().contiguous()                       # (K//2, N) -> (N, K//2)
    pu = p.to(torch.int16) & 0xFF
    low = (pu & 0xF) - 8
    high = ((pu >> 4) & 0xF) - 8
    N, K2 = p.shape
    q = torch.empty((N, K2 * 2), dtype=torch.int16, device=p.device)
    q[:, 0::2] = low
    q[:, 1::2] = high
    return q.to(torch.int8)


if HAS_TRITON:

    # Autotune the tile schedule per (M, N, K). Tiling alone did NOT recover the
    # Turing T4 (every config stayed ~7x slower than FP16) — the real penalty there
    # was the strided/uncoalesced weight load, now fixed by storing the weights
    # transposed (K, N) in quantize_w8a16/quantize_w4a16. Autotuning still earns its
    # keep: it lets Triton pick an Ampere-friendly tile on the 3060 and a
    # higher-occupancy one elsewhere. Configs that overflow shared memory are pruned.
    _W8_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    ]
    _W4_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32,  "BLOCK_K2": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K2": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K2": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K2": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K2": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K2": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K2": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K2": 32}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=_W8_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _w8a16_gemm_kernel(
        x_ptr, qw_ptr, scale_ptr, bias_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_qn, stride_qk,
        stride_ym, stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        # Weight tile (BLOCK_K, BLOCK_N): element [k, n] == qweight_T[k, n] (the
        # int8 weight is stored transposed (K, N), so the fast axis n is contiguous
        # -> stride_qn == 1 -> coalesced loads).
        qw_ptrs = qw_ptr + offs_k[:, None] * stride_qk + offs_n[None, :] * stride_qn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            w = tl.load(qw_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
            acc += tl.dot(x, w.to(tl.float16))
            x_ptrs += BLOCK_K * stride_xk
            qw_ptrs += BLOCK_K * stride_qk

        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc * scale[None, :]
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += b[None, :]

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def w8a16_linear(
        x: torch.Tensor,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """y = (x @ dequant(qweight).T) using the fused int8 kernel.

        Tiling is chosen by ``@triton.autotune`` per (M, N, K); the first call for
        a new shape pays a one-time tuning sweep, then it's cached.

        Args:
            x: (..., K) FP16 activations.
            qweight: (K, N) int8 weights, transposed for coalesced loads.
            scale: (N,) FP16 per-output-channel scales.
            bias: optional (N,) bias.
        """
        *batch, K = x.shape
        N = qweight.shape[1]                  # qweight is (K, N)
        xf = x.reshape(-1, K).to(torch.float16).contiguous()
        M = xf.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.float16)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _w8a16_gemm_kernel[grid](
            xf, qweight, scale,
            bias if bias is not None else scale,  # placeholder ptr when no bias
            y,
            M, N, K,
            xf.stride(0), xf.stride(1),
            qweight.stride(1), qweight.stride(0),  # (stride_qn, stride_qk) for (K, N)
            y.stride(0), y.stride(1),
            HAS_BIAS=bias is not None,
        )
        return y.reshape(*batch, N)

    @triton.autotune(configs=_W4_CONFIGS, key=["M", "N", "K2"])
    @triton.jit
    def _w4a16_gemm_kernel(
        x_ptr, pw_ptr, scale_ptr, bias_ptr, y_ptr,
        M, N, K2,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_ym, stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K2: tl.constexpr,
    ):
        # K2 = K // 2 packed rows (the weight is stored transposed (K2, N) so the
        # fast axis n is contiguous -> stride_pn == 1 -> coalesced loads). Each
        # packed byte holds two int4 weights: low nibble = even-K, high nibble =
        # odd-K; we avoid nibble interleaving by contracting the halves separately.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k2 = tl.arange(0, BLOCK_K2)

        xe_ptrs = x_ptr + offs_m[:, None] * stride_xm + (2 * offs_k2)[None, :] * stride_xk
        xo_ptrs = x_ptr + offs_m[:, None] * stride_xm + (2 * offs_k2 + 1)[None, :] * stride_xk
        pw_ptrs = pw_ptr + offs_k2[:, None] * stride_pk + offs_n[None, :] * stride_pn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for p0 in range(0, K2, BLOCK_K2):
            kmask = (p0 + offs_k2) < K2
            xe = tl.load(xe_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
            xo = tl.load(xo_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
            b = tl.load(pw_ptrs, mask=kmask[:, None] & (offs_n[None, :] < N), other=0)
            # b is uint8 in [0,255]; nibbles are unsigned, so >> is a logical shift.
            # Cast back to fp16 (the `- 8.0` would otherwise promote to fp32 and
            # break tl.dot's same-dtype requirement).
            w_low = ((b & 0xF).to(tl.float16) - 8.0).to(tl.float16)     # even-K
            w_high = ((b >> 4).to(tl.float16) - 8.0).to(tl.float16)     # odd-K
            acc += tl.dot(xe, w_low)
            acc += tl.dot(xo, w_high)
            xe_ptrs += BLOCK_K2 * 2 * stride_xk
            xo_ptrs += BLOCK_K2 * 2 * stride_xk
            pw_ptrs += BLOCK_K2 * stride_pk

        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc * scale[None, :]
        if HAS_BIAS:
            bb = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += bb[None, :]
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def w4a16_linear(x, packed, scale, bias=None):
        """y = (x @ dequant(packed).T) using the fused int4 kernel.

        packed: (K//2, N) uint8 nibble-packed weights (transposed for coalesced
        loads); scale: (N,) fp16. Tiling is autotuned per (M, N, K2); the first
        call for a new shape pays a one-time tuning sweep, then it's cached.
        """
        *batch, K = x.shape
        K2, N = packed.shape                 # packed is (K//2, N)
        xf = x.reshape(-1, K).to(torch.float16).contiguous()
        M = xf.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _w4a16_gemm_kernel[grid](
            xf, packed, scale,
            bias if bias is not None else scale,
            y,
            M, N, K2,
            xf.stride(0), xf.stride(1),
            packed.stride(1), packed.stride(0),  # (stride_pn, stride_pk) for (K2, N)
            y.stride(0), y.stride(1),
            HAS_BIAS=bias is not None,
        )
        return y.reshape(*batch, N)

else:  # pragma: no cover - exercised only where triton is unavailable

    def w8a16_linear(*_args, **_kwargs):
        raise RuntimeError(
            "w8a16_linear requires Triton + a CUDA GPU (Linux/WSL2). "
            "HAS_TRITON is False in this environment."
        )

    def w4a16_linear(*_args, **_kwargs):
        raise RuntimeError(
            "w4a16_linear requires Triton + a CUDA GPU (Linux/WSL2). "
            "HAS_TRITON is False in this environment."
        )


class W8A16Linear(nn.Module):
    """Linear with per-output-channel INT8 weights and FP16 activations.

    Uses the fused Triton kernel on CUDA when available; otherwise falls back to
    an eager per-channel dequant + ``F.linear`` so the layer is correct on CPU and
    on machines without Triton (just without the speedup). Drop-in for the layers
    the allocator marks INT8 when the kernel path is enabled.
    """

    def __init__(self, qweight, scale, bias=None, in_features=None, out_features=None):
        super().__init__()
        self.precision = "int8-kernel"
        self.register_buffer("qweight", qweight)  # (K, N) int8 (transposed)
        self.register_buffer("scale", scale)      # (N,) fp16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        k, n = qweight.shape
        self.out_features = out_features if out_features is not None else n
        self.in_features = in_features if in_features is not None else k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda:
            return w8a16_linear(x, self.qweight, self.scale, self.bias)
        # Eager fallback (CPU / no Triton): correct, just not accelerated.
        # qweight is stored transposed (K, N); F.linear wants (N, K).
        in_dtype = x.dtype
        w = self.qweight.t().to(torch.float16) * self.scale.to(torch.float16)[:, None]
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(x.to(torch.float16), w, bias).to(in_dtype)

    def memory_bytes(self) -> int:
        b = (self.qweight.numel() * self.qweight.element_size()
             + self.scale.numel() * self.scale.element_size())
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return b

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, precision={self.precision}")

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "W8A16Linear":
        q, s = quantize_w8a16(linear.weight.data.cpu())
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(q, s, bias, linear.in_features, linear.out_features)


class W4A16Linear(nn.Module):
    """Linear with packed per-output-channel INT4 weights and FP16 activations.

    Same design as :class:`W8A16Linear` (fused Triton kernel on CUDA, eager
    fallback elsewhere) but 4-bit: half the bytes again, used for the layers the
    allocator marks INT4 when the kernel path is enabled.
    """

    def __init__(self, packed, scale, bias=None, in_features=None, out_features=None):
        super().__init__()
        self.precision = "int4-kernel"
        self.register_buffer("packed", packed)  # (K//2, N) uint8 (transposed)
        self.register_buffer("scale", scale)    # (N,) fp16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        k2, n = packed.shape
        self.out_features = out_features if out_features is not None else n
        self.in_features = in_features if in_features is not None else k2 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and x.is_cuda:
            return w4a16_linear(x, self.packed, self.scale, self.bias)
        # Eager fallback: unpack nibbles, dequant, F.linear.
        in_dtype = x.dtype
        q = _unpack_w4(self.packed).to(torch.float16)
        w = q * self.scale.to(torch.float16)[:, None]
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(x.to(torch.float16), w, bias).to(in_dtype)

    def memory_bytes(self) -> int:
        b = (self.packed.numel() * self.packed.element_size()
             + self.scale.numel() * self.scale.element_size())
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return b

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, precision={self.precision}")

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "W4A16Linear":
        packed, s = quantize_w4a16(linear.weight.data.cpu())
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(packed, s, bias, linear.in_features, linear.out_features)

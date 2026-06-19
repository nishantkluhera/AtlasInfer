"""
Tests for the fused W8A16 Triton kernel.

The quantization helper is CPU-only and always tested. The kernel itself needs
Triton + a CUDA GPU (Linux/WSL2), so that test skips elsewhere (Windows, CI).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

from atlasinfer.triton_kernels import (
    HAS_TRITON, quantize_w8a16, quantize_w4a16, W8A16Linear, W4A16Linear, _unpack_w4,
)

_GPU = HAS_TRITON and torch.cuda.is_available()


class TestQuantizeW8A16:
    def test_roundtrip_shapes_and_error(self):
        torch.manual_seed(0)
        W = torch.randn(128, 256, dtype=torch.float16) * 0.05
        q, scale = quantize_w8a16(W)

        # Stored transposed (K, N) so the kernel's weight load is coalesced.
        assert q.shape == (W.shape[1], W.shape[0]) and q.dtype == torch.int8
        assert scale.shape == (128,) and scale.dtype == torch.float16
        assert q.abs().max() <= 127

        # Per-channel int8 should reconstruct the weight closely (transpose back to (N, K)).
        dq = q.t().to(torch.float32) * scale.to(torch.float32)[:, None]
        rel = (W.float() - dq).norm() / W.float().norm()
        assert rel < 0.02, f"reconstruction error too high: {rel}"


class TestW8A16LinearEager:
    """W8A16Linear must be correct on CPU too (eager dequant fallback)."""

    def test_from_linear_matches(self):
        torch.manual_seed(0)
        lin = nn.Linear(256, 128)
        lin.weight.data *= 0.05
        layer = W8A16Linear.from_linear(lin)

        assert layer.qweight.dtype == torch.int8
        assert layer.qweight.shape == (256, 128)  # (K, N), transposed for the kernel

        x = torch.randn(4, 256)
        ref = lin(x)
        got = layer(x)  # CPU -> eager fallback path
        assert got.shape == (4, 128)
        rel = (ref - got).norm() / ref.norm()
        assert rel < 0.03, f"eager W8A16 rel err {rel}"

    def test_memory_smaller_than_fp16(self):
        lin = nn.Linear(512, 512)
        layer = W8A16Linear.from_linear(lin)
        fp16_bytes = 512 * 512 * 2
        assert layer.memory_bytes() < fp16_bytes


class TestW4A16LinearEager:
    """W4A16: packing round-trip and eager (CPU) forward correctness."""

    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(0)
        W = torch.randn(64, 128, dtype=torch.float16) * 0.05
        packed, scale = quantize_w4a16(W)
        assert packed.shape == (64, 64) and packed.dtype == torch.uint8
        q = _unpack_w4(packed)
        assert q.shape == (64, 128)
        assert int(q.min()) >= -7 and int(q.max()) <= 7

    def test_from_linear_matches(self):
        torch.manual_seed(0)
        lin = nn.Linear(256, 128)
        lin.weight.data *= 0.05
        layer = W4A16Linear.from_linear(lin)
        x = torch.randn(4, 256)
        rel = (lin(x) - layer(x)).norm() / lin(x).norm()
        assert rel < 0.12, f"eager W4A16 rel err {rel}"

    def test_memory_roughly_quarter_fp16(self):
        layer = W4A16Linear.from_linear(nn.Linear(512, 512))
        assert layer.memory_bytes() < 512 * 512 * 2 * 0.4  # <40% of fp16

    def test_odd_in_features_rejected(self):
        with pytest.raises(ValueError):
            quantize_w4a16(torch.randn(8, 7))


@pytest.mark.skipif(not _GPU, reason="needs Triton + CUDA (Linux/WSL2)")
class TestFusedKernel:
    def test_matches_fp16_linear(self):
        from atlasinfer.triton_kernels import w8a16_linear

        torch.manual_seed(0)
        K, N = 512, 384
        W = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        qW, scale = quantize_w8a16(W)
        for M in (1, 4, 32):
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            ref = torch.nn.functional.linear(x, W).float()
            got = w8a16_linear(x, qW, scale).float()
            rel = (ref - got).norm() / ref.norm()
            assert rel < 0.05, f"M={M} rel err {rel}"

    def test_with_bias(self):
        from atlasinfer.triton_kernels import w8a16_linear

        torch.manual_seed(1)
        K, N = 256, 128
        W = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        qW, scale = quantize_w8a16(W)
        x = torch.randn(3, K, device="cuda", dtype=torch.float16)
        ref = torch.nn.functional.linear(x, W, bias).float()
        got = w8a16_linear(x, qW, scale, bias).float()
        assert (ref - got).norm() / ref.norm() < 0.05

    def test_w4a16_matches_reference(self):
        # Kernel correctness: compare to the dequantized reference computed from
        # the *same* packed weights (isolates kernel math from int4 quant error).
        from atlasinfer.triton_kernels import w4a16_linear

        torch.manual_seed(0)
        K, N = 512, 384
        W = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        packed, scale = quantize_w4a16(W)
        ref_w = (_unpack_w4(packed).to(torch.float16) * scale[:, None]).cuda()
        for M in (1, 4, 32):
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            ref = torch.nn.functional.linear(x, ref_w).float()
            got = w4a16_linear(x, packed, scale).float()
            rel = (ref - got).norm() / ref.norm()
            assert rel < 0.02, f"M={M} W4A16 kernel rel err {rel}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

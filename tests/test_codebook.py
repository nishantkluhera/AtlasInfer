"""
Tests for the EXPERIMENTAL sub-4-bit vector-quantized codebook.

These check correctness and the bit-rate arithmetic only — NOT that VQ is
competitive with AQLM/QuIP#/QTIP at 2 bit (that needs 7B-scale eval this repo
can't run; the module says so up front). What must hold: round-trip works, the
rate is ~8/vector_dim bits/weight, more bits reconstructs better, and it beats a
naive scalar 2-bit grid on real-shaped weights.
"""
import torch

from atlasinfer.codebook import (
    quantize_tensor_vq, dequantize_tensor_vq, VectorQuantizedTensor, CODEBOOK_SIZE,
)


def _rel(w, r):
    return ((w.float() - r.float()).norm() / w.float().norm()).item()


class TestVectorQuant:
    def test_roundtrip_shape_dtype(self):
        torch.manual_seed(0)
        w = torch.randn(512, 512, dtype=torch.float16)
        qt = quantize_tensor_vq(w, vector_dim=4)
        r = dequantize_tensor_vq(qt)
        assert r.shape == w.shape and r.dtype == torch.float16
        assert torch.isfinite(r).all()
        assert isinstance(qt, VectorQuantizedTensor)

    def test_rate_is_about_two_bits_at_dim4(self):
        torch.manual_seed(0)
        w = torch.randn(1024, 1024, dtype=torch.float16)
        qt = quantize_tensor_vq(w, vector_dim=4, block_size=256)
        # 8 bits per 4-vector = 2 bits/weight, plus a small amortized codebook +
        # per-block scale overhead. Should sit just above 2.
        assert 2.0 <= qt.bits_per_weight() < 2.6

    def test_more_bits_reconstructs_better(self):
        torch.manual_seed(0)
        w = torch.randn(512, 512, dtype=torch.float16)
        err_2bit = _rel(w, dequantize_tensor_vq(quantize_tensor_vq(w, vector_dim=4)))
        err_4bit = _rel(w, dequantize_tensor_vq(quantize_tensor_vq(w, vector_dim=2)))
        assert err_4bit < err_2bit  # d=2 (4-bit) must beat d=4 (2-bit)

    def test_beats_naive_scalar_2bit(self):
        # Scalar 2-bit = a 4-level symmetric grid per weight. VQ at the same 2-bit
        # rate should reconstruct Gaussian weights at least as well.
        torch.manual_seed(0)
        w = torch.randn(512, 512, dtype=torch.float16)
        # naive scalar 2-bit, per-block absmax (levels {-1,-1/3,1/3,1} * scale)
        blocks = w.float().reshape(-1, 256)
        scale = blocks.abs().amax(1, keepdim=True).clamp(min=1e-8)
        q = ((blocks / scale) * 1.5).round().clamp(-2, 1) / 1.5 * scale
        scalar_err = (blocks - q).norm() / blocks.norm()
        vq_err = _rel(w, dequantize_tensor_vq(quantize_tensor_vq(w, vector_dim=4)))
        assert vq_err <= scalar_err.item() + 1e-3

    def test_memory_smaller_than_fp16(self):
        w = torch.randn(1024, 1024, dtype=torch.float16)
        qt = quantize_tensor_vq(w, vector_dim=4)
        assert qt.memory_bytes() < w.numel() * 2 / 3  # well under FP16

    def test_codebook_size(self):
        qt = quantize_tensor_vq(torch.randn(256, 256), vector_dim=4)
        assert qt.codebook.shape == (CODEBOOK_SIZE, 4)

    def test_device_move(self):
        qt = quantize_tensor_vq(torch.randn(128, 128), vector_dim=4)
        moved = qt.to(torch.device("cpu"))
        assert moved.codes.device == torch.device("cpu")
        assert dequantize_tensor_vq(moved).shape == (128, 128)

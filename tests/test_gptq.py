"""
Tests for GPTQ-style error-compensated NF4 quantization (CPU-only, no downloads).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.quantizer import quantize_tensor_nf4, dequantize_tensor_nf4
from atlasinfer.gptq import gptq_quantize_nf4


def _out_err(W, Wq, X):
    """Layer output error ||X Wᵀ - X Wqᵀ|| (what GPTQ minimizes)."""
    return (X @ W.t() - X @ Wq.t()).norm().item()


class TestGPTQ:
    def test_reduces_output_error_vs_plain_nf4(self):
        torch.manual_seed(0)
        out, in_, tokens = 64, 256, 512
        W = torch.randn(out, in_) * 0.1
        X = torch.randn(tokens, in_)
        H = X.t() @ X

        W_plain = dequantize_tensor_nf4(quantize_tensor_nf4(W, block_size=64)).float()
        W_gptq = dequantize_tensor_nf4(gptq_quantize_nf4(W, H, group_size=64)).float()

        err_plain = _out_err(W, W_plain, X)
        err_gptq = _out_err(W, W_gptq, X)
        # GPTQ compensation must lower the output error it optimizes for.
        assert err_gptq < err_plain, f"gptq {err_gptq} !< plain {err_plain}"

    def test_output_shape_and_finiteness(self):
        torch.manual_seed(1)
        W = torch.randn(32, 128) * 0.1
        X = torch.randn(400, 128)
        H = X.t() @ X  # a real (PSD) Hessian
        qt = gptq_quantize_nf4(W, H, group_size=64)
        Wq = dequantize_tensor_nf4(qt)
        assert Wq.shape == W.shape
        assert torch.isfinite(Wq.float()).all()

    def test_rejects_unaligned_in_features(self):
        W = torch.randn(8, 130)          # 130 not divisible by 64
        H = torch.eye(130)
        with pytest.raises(ValueError):
            gptq_quantize_nf4(W, H, group_size=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

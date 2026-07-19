"""
Tests for double-quantized block scales (the QLoRA / bitsandbytes memory trick).

The point of double-quant is to shrink the per-block FP32 scale overhead ~4x
*without* moving perplexity — so these tests assert both the memory reduction and
that reconstruction stays tight, at the tensor level and through the real
QuantizedLinear4bit layer (honest resident-memory accounting).
"""
import torch
import torch.nn as nn

from atlasinfer.double_quant import (
    double_quantize, dq_relative_error, DoubleQuantScales, DEFAULT_DQ_GROUP,
)
from atlasinfer.quantizer import (
    quantize_tensor_nf4, dequantize_tensor_nf4, QuantizedTensor4bit,
)
from atlasinfer.linear import QuantizedLinear4bit


class TestDoubleQuantScales:
    def test_roundtrip_tight_on_realistic_scales(self):
        # Block absmax scales are positive and smoothly varying; DQ should
        # reconstruct them to well under 1% relative error.
        torch.manual_seed(0)
        scales = torch.rand(4096).abs() * 0.05 + 1e-3
        assert dq_relative_error(scales) < 0.01

    def test_reconstruct_shape_and_length(self):
        scales = torch.rand(300)  # not a multiple of the group size (padding path)
        dq = double_quantize(scales, group_size=DEFAULT_DQ_GROUP)
        recon = dq.reconstruct()
        assert recon.shape == scales.shape
        assert dq.num_scales == 300

    def test_memory_is_about_a_quarter(self):
        scales = torch.rand(2560)  # 10 groups of 256
        dq = double_quantize(scales)
        fp32_bytes = scales.numel() * 4
        # 1 byte/scale + (4+4) per 256-group ~= 1.03 bytes/scale -> ~4x smaller.
        assert dq.memory_bytes() < fp32_bytes / 3.5

    def test_device_move(self):
        dq = double_quantize(torch.rand(512))
        moved = dq.to(torch.device("cpu"))
        assert moved.codes.device == torch.device("cpu")
        assert moved.reconstruct().numel() == 512


class TestNF4DoubleQuant:
    def test_smaller_and_near_identical_accuracy(self):
        torch.manual_seed(0)
        w = torch.randn(512, 512, dtype=torch.float16)

        q_plain = quantize_tensor_nf4(w, double_quant=False)
        q_dq = quantize_tensor_nf4(w, double_quant=True)

        # Double-quant must actually reduce the stored footprint...
        assert q_dq.memory_bytes() < q_plain.memory_bytes()
        # ...and the FP32 scales are no longer resident.
        assert q_dq.scales.numel() == 0 and q_dq.scales_dq is not None

        # ...while barely moving reconstruction error vs plain NF4.
        r_plain = dequantize_tensor_nf4(q_plain)
        r_dq = dequantize_tensor_nf4(q_dq)
        e_plain = (w.float() - r_plain.float()).norm() / w.float().norm()
        e_dq = (w.float() - r_dq.float()).norm() / w.float().norm()
        assert e_dq < e_plain * 1.05 + 1e-4, f"dq {e_dq:.4f} vs plain {e_plain:.4f}"

    def test_scale_overhead_actually_shrinks(self):
        # The saving is in the *scale* portion specifically. At block_size 64 the
        # FP32 scales are ~0.5 bit/weight; DQ should cut that materially.
        torch.manual_seed(0)
        w = torch.randn(1024, 1024, dtype=torch.float16)
        q_plain = quantize_tensor_nf4(w, double_quant=False)
        q_dq = quantize_tensor_nf4(w, double_quant=True)
        plain_scale_bytes = q_plain.scales.numel() * q_plain.scales.element_size()
        dq_scale_bytes = q_dq.scales_dq.memory_bytes()
        assert dq_scale_bytes < plain_scale_bytes / 3.0


class TestQuantizedLinear4bitDoubleQuant:
    def test_layer_forward_and_memory(self):
        torch.manual_seed(0)
        lin = nn.Linear(256, 128)
        x = torch.randn(4, 256)

        plain = QuantizedLinear4bit.from_linear(lin, scheme="nf4", double_quant=False)
        dq = QuantizedLinear4bit.from_linear(lin, scheme="nf4", double_quant=True)

        # Resident memory drops (scales compressed, FP32 scales not held).
        assert dq.quantized_weights.memory_bytes() < plain.quantized_weights.memory_bytes()

        # Outputs are nearly identical (only the scales' second rounding differs).
        y_plain, y_dq = plain(x), dq(x)
        rel = (y_plain - y_dq).norm() / (y_plain.norm() + 1e-8)
        assert rel < 0.02, f"double-quant changed the layer output too much: {rel:.4f}"

    def test_roundtrip_through_buffers(self):
        # The DoubleQuantScales must survive being registered as buffers and
        # reconstructed via the .quantized_weights property.
        torch.manual_seed(0)
        lin = nn.Linear(320, 64)
        dq = QuantizedLinear4bit.from_linear(lin, scheme="nf4", double_quant=True)
        qt = dq.quantized_weights
        assert isinstance(qt, QuantizedTensor4bit)
        assert qt.scales_dq is not None
        assert qt.block_scales().numel() == qt.scales_dq.num_scales

"""
Tests for AWQ (activation-aware weight quantization).

AWQ's promise: on layers with a few high-activation input channels, scaling those
columns up before NF4 quantization (and dividing the activation back out at run
time) lowers the layer's output error vs plain NF4 — with the exact-cancellation
identity ``(W·diag(s))·(x/s) == W·x`` keeping it correct. These tests exercise the
scale search, the identity, and the end-to-end model pass on a tiny model.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from atlasinfer.awq import search_awq_scale, quantize_model_awq
from atlasinfer.quantizer import quantize_tensor_nf4, dequantize_tensor_nf4
from atlasinfer.linear import QuantizedLinear4bit


def _nf4_out_err(W, x_rows, s=None):
    """Output MSE of NF4-quantizing W (optionally AWQ-scaled by s)."""
    ref = x_rows @ W.t()
    if s is None:
        qW = dequantize_tensor_nf4(quantize_tensor_nf4(W)).float()
        out = x_rows @ qW.t()
    else:
        qWs = dequantize_tensor_nf4(quantize_tensor_nf4(W * s)).float()
        out = (x_rows / s) @ qWs.t()
    return (out - ref).pow(2).mean().item()


class TestScaleSearch:
    def test_never_worse_than_plain_nf4(self):
        # alpha=0 (no scaling) is in the grid, so AWQ can't lose to plain NF4.
        torch.manual_seed(0)
        W = torch.randn(128, 128)
        x = torch.randn(64, 128)
        s = search_awq_scale(W, x.abs().mean(0), x)
        assert _nf4_out_err(W, x, s) <= _nf4_out_err(W, x) + 1e-8

    def test_helps_and_scales_up_the_salient_channel(self):
        # One input channel dominates the activation; AWQ should amplify its
        # weight column (s>1 there) and cut the output error.
        torch.manual_seed(0)
        W = torch.randn(128, 128)
        x = torch.randn(64, 128)
        x[:, 0] *= 40.0                      # channel 0 is highly salient
        s = search_awq_scale(W, x.abs().mean(0), x)
        assert not torch.allclose(s, torch.ones_like(s)), "AWQ picked trivial scaling"
        assert s[0] > 1.0, "salient channel not scaled up"
        assert _nf4_out_err(W, x, s) < _nf4_out_err(W, x), "AWQ did not reduce error"


class TestLayerIdentity:
    def test_in_scale_cancels(self):
        # With no quantization error the in_scale must cancel exactly:
        # QuantizedLinear4bit(W·diag(s), in_scale=s)(x) ~= F.linear(x, W).
        torch.manual_seed(0)
        lin = nn.Linear(64, 32, bias=True)
        s = torch.rand(64) + 0.5
        Ws = lin.weight.data * s
        qt = quantize_tensor_nf4(Ws)
        layer = QuantizedLinear4bit(qt, bias=lin.bias.data.clone(),
                                    in_features=64, out_features=32,
                                    scheme="nf4", in_scale=s.to(torch.float16))
        x = torch.randn(8, 64)
        got = layer(x)
        ref = torch.nn.functional.linear(x, lin.weight.data, lin.bias.data)
        rel = (got - ref).norm() / ref.norm()
        assert rel < 0.15, f"AWQ layer identity broken: {rel:.3f}"


class _Tiny(nn.Module):
    """Minimal LM-shaped model (embedding -> 2 linears) for the AWQ model pass."""
    def __init__(self, d=64, vocab=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.config = SimpleNamespace(vocab_size=vocab)

    def forward(self, input_ids):
        return self.fc2(torch.relu(self.fc1(self.emb(input_ids))))


class TestModelPass:
    def test_quantize_model_awq_smoke(self):
        torch.manual_seed(0)
        model = _Tiny()
        quantize_model_awq(model, tokenizer=None, nsamples=4, seqlen=16, verbose=False)
        # Both linears became AWQ NF4 layers carrying an in_scale.
        assert isinstance(model.fc1, QuantizedLinear4bit)
        assert isinstance(model.fc2, QuantizedLinear4bit)
        assert model.fc1.in_scale is not None and model.fc2.in_scale is not None
        # Still runs and produces finite output.
        out = model(torch.randint(0, 128, (2, 16)))
        assert out.shape == (2, 16, 64) and torch.isfinite(out).all()

    def test_awq_double_quant_composes(self):
        torch.manual_seed(0)
        model = _Tiny()
        quantize_model_awq(model, tokenizer=None, nsamples=4, seqlen=16,
                           double_quant=True, verbose=False)
        qt = model.fc1.quantized_weights
        assert qt.scales_dq is not None            # double-quant active
        assert model.fc1.in_scale is not None       # and AWQ scaling active

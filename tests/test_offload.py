"""
Tests for CPU offload.

`offload.py` implements the "run a model that doesn't fit in VRAM" feature and
had no dedicated tests. The behaviour that actually matters is subtle and easy to
regress: the hook must move the module *and* the activations, or a GPU-resident
block meets CPU-resident inputs and the forward dies with a device mismatch.

Most of this is testable without a GPU by using CPU as both the "host" and the
"target" device — what is being checked is the hook plumbing (does the module get
moved, do nested tensor structures get moved, does the output come back), not
CUDA itself. The one genuinely device-dependent test is marked and skips.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.offload import (
    CPUOffloadHook,
    _move,
    estimate_model_memory,
    setup_cpu_offload,
)
from atlasinfer.patcher import find_decoder_layers

_CUDA = torch.cuda.is_available()


class Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, **kw):
        return self.lin(x)


def _model_with(pattern: str, n=3, dim=8):
    """Build a model exposing its decoder list at ``pattern`` (dotted path)."""
    layers = nn.ModuleList([Block(dim) for _ in range(n)])
    root = nn.Module()
    node = root
    parts = pattern.split(".")
    for p in parts[:-1]:
        child = nn.Module()
        setattr(node, p, child)
        node = child
    setattr(node, parts[-1], layers)
    return root, layers


class TestMove:
    """`_move` must recurse through the containers HF models actually return."""

    def test_moves_bare_tensor(self):
        assert _move(torch.zeros(2), torch.device("cpu")).device.type == "cpu"

    def test_recurses_into_tuples_lists_dicts(self):
        obj = (torch.zeros(2), [torch.zeros(2), {"k": torch.zeros(2)}])
        moved = _move(obj, torch.device("cpu"))
        assert isinstance(moved, tuple) and isinstance(moved[1], list)
        assert isinstance(moved[1][1], dict)
        assert moved[1][1]["k"].device.type == "cpu"

    def test_leaves_non_tensors_untouched(self):
        sentinel = object()
        out = _move({"cache": sentinel, "n": 3, "s": "x"}, torch.device("cpu"))
        assert out["cache"] is sentinel and out["n"] == 3 and out["s"] == "x"

    def test_preserves_container_types(self):
        """A tuple must stay a tuple: HF forwards unpack positionally."""
        assert isinstance(_move((torch.zeros(1),), torch.device("cpu")), tuple)
        assert isinstance(_move([torch.zeros(1)], torch.device("cpu")), list)


class TestLayerDiscovery:
    @pytest.mark.parametrize("pattern", [
        "model.layers",          # Llama, Mistral, Gemma
        "model.decoder.layers",  # OPT
        "transformer.h",         # GPT-2, GPT-J
        "gpt_neox.layers",       # GPT-NeoX, Pythia
    ])
    def test_finds_each_supported_architecture(self, pattern):
        model, layers = _model_with(pattern)
        assert find_decoder_layers(model) is layers

    def test_returns_none_for_unknown_architecture(self):
        model, _ = _model_with("something.unexpected")
        assert find_decoder_layers(model) is None


class TestSetupCPUOffload:
    def test_attaches_a_hook_to_every_decoder_layer(self):
        model, layers = _model_with("model.layers", n=4)
        setup_cpu_offload(model, torch.device("cpu"))
        # accelerate marks hooked modules with _hf_hook.
        hooked = sum(1 for layer in layers if hasattr(layer, "_hf_hook"))
        assert hooked == 4

    def test_forward_still_produces_correct_output(self):
        """Offloading must not change the numbers, only where they're computed."""
        torch.manual_seed(0)
        model, layers = _model_with("model.layers", n=2)
        x = torch.randn(1, 8)
        expected = layers[0](x)

        setup_cpu_offload(model, torch.device("cpu"))
        got = layers[0](x)
        assert torch.allclose(expected, got, atol=1e-6)

    def test_unknown_architecture_fails_loudly(self):
        """Fail fast rather than silently not offloading.

        Quietly no-op'ing would leave the whole model resident and surface later
        as an OOM with no connection to the real cause. The error names the
        patterns tried and points at the `layer_patterns` escape hatch.
        """
        model, _ = _model_with("mystery.blocks")
        with pytest.raises(ValueError, match="Could not find decoder layers"):
            setup_cpu_offload(model, torch.device("cpu"))

    def test_explicit_layer_patterns_override_discovery(self):
        """The escape hatch the error message advertises must actually work."""
        model, layers = _model_with("mystery.blocks", n=2)
        setup_cpu_offload(model, torch.device("cpu"),
                          layer_patterns=["mystery.blocks"])
        assert all(hasattr(layer, "_hf_hook") for layer in layers)

    @pytest.mark.skipif(not _CUDA, reason="needs CUDA to exercise a real transfer")
    def test_layer_returns_to_cpu_after_forward(self):
        """The point of offloading: the block must not stay resident on the GPU."""
        model, layers = _model_with("model.layers", n=2)
        setup_cpu_offload(model, torch.device("cuda"))
        layers[0](torch.randn(1, 8))
        assert next(layers[0].parameters()).device.type == "cpu", (
            "block stayed on the GPU after its forward -- offloading is not "
            "reclaiming memory")


class TestHookDirectly:
    def test_pre_forward_moves_module_and_inputs(self):
        block = Block()
        hook = CPUOffloadHook(torch.device("cpu"))
        args, kwargs = hook.pre_forward(block, torch.zeros(1, 8))
        assert args[0].device.type == "cpu"
        assert isinstance(kwargs, dict)

    def test_post_forward_moves_output_back(self):
        block = Block()
        hook = CPUOffloadHook(torch.device("cpu"))
        out = hook.post_forward(block, (torch.zeros(1, 8), None))
        assert out[0].device.type == "cpu"
        assert out[1] is None  # non-tensors survive


class TestMemoryEstimate:
    def test_reports_a_plausible_total(self):
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
        info = estimate_model_memory(model)
        assert info["total_gb"] > 0
        # 2 x (64*64 + 64) params at fp32 = 33,280 bytes ~ 3.1e-5 GB
        assert info["total_gb"] < 0.01

    def test_scales_with_dtype(self):
        big = estimate_model_memory(nn.Linear(128, 128).float())["total_gb"]
        small = estimate_model_memory(nn.Linear(128, 128).half())["total_gb"]
        assert big > small

    def test_counts_three_bit_layers(self):
        """Regression: the estimator must include the 3-bit (NF3) tier.

        A mixed-precision allocation can assign layers to QuantizedLinear3bit; if
        the estimator skips that type it silently undercounts every 3-bit layer
        (the engine then prints a too-low "Model memory"). model_weight_bytes
        already counts them, so the two accountings must not disagree.
        """
        from atlasinfer.linear import QuantizedLinear3bit, create_quantized_linear
        layer = create_quantized_linear(nn.Linear(256, 256), precision="int3")
        assert isinstance(layer, QuantizedLinear3bit)
        expected = layer.quantized_weights.memory_bytes()
        if layer.bias is not None:
            expected += layer.bias.numel() * layer.bias.element_size()
        info = estimate_model_memory(layer)
        assert info["quantized_bytes"] == expected > 0
        assert info["dense_bytes"] == 0


class TestKernelLayerMemoryAccounting:
    """`model_weight_bytes` must count W8A16/W4A16 kernel layers.

    Regression: those layers store packed weight, scale and bias as *buffers*
    (not params), so `sum(model.parameters())` misses them; before the fix
    `model_weight_bytes` returned 0 for a kernel-quantized block and silently
    disagreed with both `resident_bytes` and `estimate_model_memory`.
    """

    def test_model_weight_bytes_counts_kernel_layers(self):
        from atlasinfer.evaluation import model_weight_bytes, resident_bytes
        from atlasinfer.triton_kernels import W4A16Linear, W8A16Linear
        model = nn.Sequential(
            W8A16Linear.from_linear(nn.Linear(256, 256)),
            W4A16Linear.from_linear(nn.Linear(256, 128)),
        )
        mwb = model_weight_bytes(model)
        assert mwb > 0
        # These layers hold no plain params, so the "params + quantized buffers"
        # rule must match resident_bytes' "params + all buffers".
        assert mwb == resident_bytes(model)

    def test_model_weight_bytes_agrees_with_estimate_on_kernel_model(self):
        from atlasinfer.evaluation import model_weight_bytes
        from atlasinfer.triton_kernels import W8A16Linear
        model = nn.Sequential(W8A16Linear.from_linear(nn.Linear(256, 256)))
        assert estimate_model_memory(model)["quantized_bytes"] == model_weight_bytes(model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Integration tests for the full quantization pipeline on a tiny synthetic LM.

These run entirely on CPU with no model downloads, so they're safe for CI while
still exercising profiler -> allocator -> patcher -> quantized forward end to end.
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.sensitivity import SensitivityProfiler
from atlasinfer.allocator import allocate_optimal
from atlasinfer.patcher import quantize_model_mixed, quantize_model, get_model_info
from atlasinfer.linear import QuantizedLinear
from atlasinfer.offload import _move, CPUOffloadHook, estimate_model_memory


class TinyLM(nn.Module):
    """Minimal causal-LM-shaped model: embed -> 2 linear blocks -> lm_head."""

    def __init__(self, vocab: int = 64, dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab)
        self.config = SimpleNamespace(vocab_size=vocab, max_position_embeddings=128)

    def forward(self, input_ids, labels=None, **_kw):
        x = self.embed(input_ids)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _quantizable(profiles):
    # embed and lm_head are excluded by default; fc1/fc2 remain.
    return set(profiles.keys())


class TestActivationPipeline:
    def test_profile_allocate_quantize_forward(self):
        torch.manual_seed(0)
        model = TinyLM().eval()

        profiler = SensitivityProfiler(max_samples=4, seq_len=32)
        profiles = profiler.profile(model)  # random-id calibration (no tokenizer)

        assert _quantizable(profiles) == {"fc1", "fc2"}
        for prof in profiles.values():
            assert "int8" in prof.errors and "int4" in prof.errors
            assert prof.errors["int8"] >= 0 and prof.errors["int4"] >= 0

        n_params = sum(p.param_count for p in profiles.values())
        alloc = allocate_optimal(profiles, budget_bytes=int(n_params * 6 / 8))
        assert set(alloc.allocations) == {"fc1", "fc2"}

        quantize_model_mixed(model, allocation=alloc.allocations, verbose=False)

        ids = torch.randint(0, 64, (2, 16))
        out = model(ids).logits
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()


class TestEndToEndPipeline:
    def test_end_to_end_profile_drives_allocation(self):
        torch.manual_seed(0)
        model = TinyLM().eval()

        profiler = SensitivityProfiler(max_samples=4, seq_len=32)
        profiles = profiler.profile_end_to_end(model)

        assert _quantizable(profiles) == {"fc1", "fc2"}
        for prof in profiles.values():
            # End-to-end errors are loss increases: non-negative, INT4 >= INT8-ish.
            assert prof.errors["int8"] >= 0 and prof.errors["int4"] >= 0

        n_params = sum(p.param_count for p in profiles.values())
        alloc = allocate_optimal(profiles, budget_bytes=int(n_params * 5 / 8))
        quantize_model_mixed(model, allocation=alloc.allocations, verbose=False)

        ids = torch.randint(0, 64, (1, 16))
        assert not torch.isnan(model(ids).logits).any()


class TestUniformAndInfo:
    def test_uniform_quantize_counts(self):
        model = TinyLM().eval()
        quantize_model(model, precision="int4", verbose=False)
        info = get_model_info(model)
        # fc1 + fc2 quantized to 4-bit; embed/lm_head untouched.
        assert info["quantized_linear_4bit_count"] == 2
        assert isinstance(model.lm_head, nn.Linear)

    def test_estimate_memory_counts_quantized(self):
        model = TinyLM().eval()
        quantize_model(model, precision="int8", verbose=False)
        mem = estimate_model_memory(model)
        assert mem["quantized_bytes"] > 0
        assert isinstance(model.fc1, QuantizedLinear)


class TestQuant4bitScheme:
    def test_nf4_is_default_4bit_scheme(self):
        model = TinyLM().eval()
        quantize_model(model, precision="int4", verbose=False)  # default quant_4bit
        # fc1/fc2 are QuantizedLinear4bit with the NF4 scheme.
        assert model.fc1.scheme == "nf4" and model.fc2.scheme == "nf4"
        assert not torch.isnan(model(torch.randint(0, 64, (1, 16))).logits).any()

    def test_symmetric_int4_still_available(self):
        model = TinyLM().eval()
        quantize_model(model, precision="int4", quant_4bit="int4", verbose=False)
        assert model.fc1.scheme == "int4"
        assert not torch.isnan(model(torch.randint(0, 64, (1, 16))).logits).any()


class TestKernelBackend:
    def test_kernel_backend_swaps_layers(self):
        model = TinyLM().eval()
        x = torch.randint(0, 64, (1, 16))
        ref = model(x).logits.clone()

        # use_kernel=True routes INT8 layers through the kernel-backed module.
        quantize_model(model, precision="int8", verbose=False, use_kernel=True)
        info = get_model_info(model)
        assert info["w8a16_kernel_count"] == 2
        assert info["quantized_linear_count"] == 0

        out = model(x).logits  # CPU -> eager fallback inside W8A16Linear
        assert out.shape == ref.shape
        assert not torch.isnan(out).any()

    def test_mixed_with_kernel_backend(self):
        from atlasinfer.triton_kernels import W8A16Linear, W4A16Linear

        model = TinyLM().eval()
        alloc = {"fc1": "int8", "fc2": "int4"}
        quantize_model_mixed(model, allocation=alloc, verbose=False, use_kernel=True)
        assert isinstance(model.fc1, W8A16Linear)   # int8 -> W8A16 kernel layer
        assert isinstance(model.fc2, W4A16Linear)   # int4 -> W4A16 kernel layer
        info = get_model_info(model)
        assert info["w8a16_kernel_count"] == 1 and info["w4a16_kernel_count"] == 1
        assert not torch.isnan(model(torch.randint(0, 64, (1, 16))).logits).any()


class TestOffloadHelpers:
    def test_move_nested_structures(self):
        cpu = torch.device("cpu")
        obj = {
            "a": torch.randn(3),
            "b": (torch.randn(2), [torch.randn(1)]),
            "c": "not a tensor",
            "d": None,
        }
        moved = _move(obj, cpu)
        assert moved["a"].device == cpu
        assert moved["b"][1][0].device == cpu
        assert moved["c"] == "not a tensor"  # non-tensors pass through untouched
        assert moved["d"] is None

    def test_hook_returns_aligned_io(self):
        hook = CPUOffloadHook(torch.device("cpu"))
        layer = nn.Linear(8, 8)
        args, kwargs = hook.pre_forward(layer, torch.randn(2, 8), foo=torch.randn(2))
        assert args[0].device == torch.device("cpu")
        assert kwargs["foo"].device == torch.device("cpu")
        out = hook.post_forward(layer, (torch.randn(2, 8),))
        assert out[0].device == torch.device("cpu")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

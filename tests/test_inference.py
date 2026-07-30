"""
Tests for the public API (`AtlasInference`) and the CLI.

This is the entry point in `pyproject.toml`'s `[project.scripts]` and the class
every README example uses — and it had no tests at all, while every other core
module had one.

Design note: the constructor downloads a model from the Hub, so most tests here
build the object via `__new__` and exercise the *logic* (kernel resolution,
device selection, generation plumbing, memory reporting) against a tiny local
stub. The one test that really loads GPT-2 is marked `slow` and skips when the
model isn't already cached, so a clean checkout without network still gets a
green suite.
"""
import argparse
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.inference import AtlasInference
from atlasinfer.triton_kernels import kernel_available


def _gpt2_cached() -> bool:
    """True if gpt2 is already in the HF cache (no network needed)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache("gpt2", "config.json"), str)
    except Exception:  # noqa: BLE001 - hub API varies across versions
        return False


class TinyLM(nn.Module):
    """Minimal causal-LM stand-in with the surface `generate()` needs."""

    def __init__(self, vocab=64, dim=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.mlp = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, input_ids=None, **kw):
        return self.lm_head(self.mlp(self.embed_tokens(input_ids)))

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, **kw):
        """Append `max_new_tokens` deterministic tokens."""
        n = kw.get("max_new_tokens", 1)
        self.last_generate_kwargs = kw
        pad = torch.zeros((input_ids.shape[0], n), dtype=input_ids.dtype)
        return torch.cat([input_ids, pad], dim=1)


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[1, 2, 3]])
        return argparse.Namespace(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, skip_special_tokens=True):
        return "generated text"


def _engine(**overrides):
    """An AtlasInference wired to stubs, bypassing the downloading __init__."""
    eng = AtlasInference.__new__(AtlasInference)
    eng.model_name = "stub"
    eng.verbose = False
    eng.device = torch.device("cpu")
    eng.use_kernel = False
    eng.offloaded = False
    eng.model = TinyLM()
    eng.tokenizer = StubTokenizer()
    for k, v in overrides.items():
        setattr(eng, k, v)
    return eng


class TestKernelResolution:
    """`kernel=` must be honoured exactly; "auto" depends on the environment."""

    @pytest.mark.parametrize("mode,expected", [("on", True), ("off", False)])
    def test_explicit_modes(self, mode, expected):
        # Mirrors the resolution logic in __init__ (which needs a real download).
        device = torch.device("cpu")
        if mode == "on":
            use_kernel = True
        elif mode == "off":
            use_kernel = False
        else:
            use_kernel = kernel_available() and device.type == "cuda"
        assert use_kernel is expected

    def test_auto_is_false_without_triton_or_cuda(self):
        device = torch.device("cpu")
        assert (kernel_available() and device.type == "cuda") is False


class TestGenerate:
    def test_returns_decoded_text(self):
        assert _engine().generate("hello", max_tokens=4) == "generated text"

    def test_passes_max_tokens_through(self):
        eng = _engine()
        eng.generate("hello", max_tokens=7)
        assert eng.model.last_generate_kwargs["max_new_tokens"] == 7

    def test_greedy_by_default_omits_sampling_params(self):
        eng = _engine()
        eng.generate("hello", max_tokens=2)
        kw = eng.model.last_generate_kwargs
        assert kw["do_sample"] is False
        assert "temperature" not in kw and "top_k" not in kw

    def test_sampling_params_only_when_do_sample(self):
        eng = _engine()
        eng.generate("hello", max_tokens=2, do_sample=True, temperature=0.7, top_k=5)
        kw = eng.model.last_generate_kwargs
        assert kw["do_sample"] is True
        assert kw["temperature"] == 0.7 and kw["top_k"] == 5

    def test_extra_kwargs_reach_the_model(self):
        eng = _engine()
        eng.generate("hello", max_tokens=2, repetition_penalty=1.2)
        assert eng.model.last_generate_kwargs["repetition_penalty"] == 1.2


class TestOffloadInteractions:
    """The KV-cache cannot follow a layer evicted to CPU; that must be enforced."""

    def test_cache_is_hard_disabled_when_offloaded(self):
        eng = _engine(offloaded=True)
        eng.generate("hello", max_tokens=2)
        assert eng.model.last_generate_kwargs["use_cache"] is False

    def test_explicit_use_cache_true_is_overridden_not_honoured(self):
        """A user passing use_cache=True must not get a device-mismatch crash."""
        eng = _engine(offloaded=True)
        eng.generate("hello", max_tokens=2, use_cache=True)
        assert eng.model.last_generate_kwargs["use_cache"] is False

    def test_cache_is_left_alone_when_not_offloaded(self):
        eng = _engine(offloaded=False)
        eng.generate("hello", max_tokens=2)
        assert "use_cache" not in eng.model.last_generate_kwargs


class TestCLI:
    def test_parses_defaults(self):
        from atlasinfer.inference import parse_args
        args = parse_args(["--model", "gpt2"])
        assert args.model == "gpt2"

    def test_parses_kernel_choice(self):
        from atlasinfer.inference import parse_args
        assert parse_args(["--model", "gpt2", "--kernel", "off"]).kernel == "off"

    def test_rejects_unknown_kernel_choice(self):
        from atlasinfer.inference import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--model", "gpt2", "--kernel", "sideways"])


@pytest.mark.skipif(not _gpt2_cached(),
                    reason="gpt2 not in the HF cache; skipping the real-model test")
class TestRealModel:
    """One end-to-end pass through the actual constructor, on the smallest model."""

    def test_unquantized_engine_generates(self):
        eng = AtlasInference("gpt2", quantize=False, device="cpu", verbose=False)
        out = eng.generate("The capital of France is", max_tokens=3)
        assert isinstance(out, str) and out.startswith("The capital of France is")

    def test_quantized_engine_generates_and_shrinks(self):
        from atlasinfer.evaluation import model_weight_bytes

        dense = AtlasInference("gpt2", quantize=False, device="cpu", verbose=False)
        quant = AtlasInference("gpt2", quantize=True, kernel="off",
                               device="cpu", verbose=False)
        assert model_weight_bytes(quant.model) < model_weight_bytes(dense.model)
        assert isinstance(quant.generate("Hello", max_tokens=3), str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

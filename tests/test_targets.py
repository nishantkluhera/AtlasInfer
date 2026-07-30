"""
Tests for the shared layer-selection module.

This logic used to be duplicated across patcher.py, sensitivity.py and
benchmark.py, and had already drifted (the benchmark copy was missing
"layernorm"). The consequence of drift is silent: the profiler measures one set
of layers, the patcher quantizes another, and unmatched layers quietly take
`default_precision` instead of the allocated one — producing a model that looks
fine and is not the one the allocator designed.

These tests pin (a) that the profiler and the patcher agree by construction, and
(b) that a mismatch is reported rather than swallowed.
"""
import os
import sys

import pytest
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer._targets import (
    DEFAULT_EXCLUDE,
    check_allocation_covers,
    collect_targets,
    dense_bytes,
    is_excluded,
    is_linear_layer,
    param_count,
    target_modules,
)
from atlasinfer.patcher import quantize_model_mixed
from atlasinfer.sensitivity import SensitivityProfiler


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Linear(8, 8)
        self.layernorm = nn.Linear(8, 8)
        self.ln_f = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 8)
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 16, bias=False)

    def forward(self, x):
        return self.mlp(self.attn(x))


class TestSelection:
    def test_excludes_only_the_intended_layers(self):
        assert set(target_modules(Model())) == {"attn", "mlp"}

    def test_exclusion_is_case_insensitive_substring(self):
        assert is_excluded("model.LayerNorm.weight")
        assert is_excluded("transformer.ln_f")
        assert not is_excluded("model.layers.0.self_attn.q_proj")

    def test_is_linear_layer_accepts_conv1d_by_name(self):
        class Conv1D(nn.Module):  # HuggingFace GPT-2 style, duck-typed
            pass

        assert is_linear_layer(nn.Linear(4, 4))
        assert is_linear_layer(Conv1D())
        assert not is_linear_layer(nn.LayerNorm(4))

    def test_param_count_includes_bias_only_when_present(self):
        assert param_count(nn.Linear(8, 4, bias=True)) == 8 * 4 + 4
        assert param_count(nn.Linear(8, 4, bias=False)) == 8 * 4

    def test_dense_bytes_tracks_dtype(self):
        lin = nn.Linear(8, 4, bias=False)
        assert dense_bytes(lin) == 32 * 4          # fp32
        assert dense_bytes(lin.half()) == 32 * 2   # fp16

    def test_collect_targets_returns_usable_parent_attr(self):
        model = Model()
        for parent, attr, module, name in collect_targets(model):
            assert getattr(parent, attr) is module
            assert model.get_submodule(name) is module


class TestProfilerPatcherAgreement:
    """The whole point of the module: these two must not be able to disagree."""

    def test_same_layer_set(self):
        model = Model()
        profiled = set(SensitivityProfiler()._target_layers(model))
        patched = {name for _p, _a, _m, name in collect_targets(model)}
        assert profiled == patched

    def test_same_layer_set_under_custom_exclusions(self):
        model = Model()
        custom = ("embed", "lm_head", "norm", "ln_", "layernorm", "mlp")
        profiled = set(SensitivityProfiler(exclude_patterns=custom)._target_layers(model))
        patched = {name for _p, _a, _m, name in collect_targets(model, custom)}
        assert profiled == patched
        assert "mlp" not in profiled


class TestAllocationCoverage:
    def test_exact_allocation_reports_nothing(self, capsys):
        names = [n for _p, _a, _m, n in collect_targets(Model())]
        out = check_allocation_covers({n: "int8" for n in names}, names)
        assert out == {"unmatched": [], "unallocated": []}
        assert capsys.readouterr().out == ""

    def test_empty_allocation_is_not_flagged(self, capsys):
        """A uniform quantize passes no allocation; that must stay silent."""
        names = [n for _p, _a, _m, n in collect_targets(Model())]
        assert check_allocation_covers(None, names)["unmatched"] == []
        assert check_allocation_covers({}, names)["unallocated"] == []
        assert capsys.readouterr().out == ""

    def test_unmatched_key_is_reported(self):
        out = check_allocation_covers({"attn": "fp16", "ghost": "int4"}, ["attn", "mlp"])
        assert out["unmatched"] == ["ghost"]
        assert out["unallocated"] == ["mlp"]

    def test_warning_reaches_stdout(self, capsys):
        check_allocation_covers({"ghost": "int4"}, ["attn"], context="unit-test")
        printed = capsys.readouterr().out
        assert "WARNING" in printed and "unit-test" in printed
        assert "ghost" in printed

    def test_patcher_warns_on_a_mismatched_allocation(self, capsys):
        quantize_model_mixed(Model(), allocation={"nonexistent": "int4"}, verbose=False)
        assert "WARNING" in capsys.readouterr().out

    def test_patcher_is_silent_on_a_correct_allocation(self, capsys):
        model = Model()
        names = [n for _p, _a, _m, n in collect_targets(model)]
        quantize_model_mixed(model, allocation={n: "int8" for n in names}, verbose=False)
        assert "WARNING" not in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

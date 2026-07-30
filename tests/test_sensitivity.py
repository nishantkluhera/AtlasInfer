"""
Tests for the sensitivity profiler.

This module produces every input the allocator consumes, so a silent regression
here corrupts every mixed-precision result without failing anything else. It had
no dedicated test file until now (see PAPER/00_audit.md section 5).

The properties that actually matter:
  * a layer that is genuinely harder to quantize must score higher than one that
    isn't -- the profiler is only useful if the *ordering* is right;
  * more aggressive precision must score at least as lossy as less aggressive;
  * a layer that cannot be profiled must be pushed toward FP16, never given a
    plausible-looking mid-range error;
  * calibration must reach the layers via real forward activations, not noise.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.allocator import BYTES_PER_PARAM, allocate_optimal
from atlasinfer.sensitivity import (
    DEFAULT_PRECISIONS,
    LayerProfile,
    SensitivityProfiler,
    _failed_profile_errors,
)


class TinyModel(nn.Module):
    """Two stacked linear layers, used to check activation capture.

    ``spiky`` carries a few extreme weights; ``dense`` is plain Gaussian. Note
    this makes ``spiky`` *easier* to quantize, not harder -- see
    ``TestQuantizationDifficulty`` for why.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.spiky = nn.Linear(dim, dim)
        with torch.no_grad():
            self.dense.weight.normal_(0, 0.02)
            self.spiky.weight.normal_(0, 0.02)
            for i in range(0, dim, 8):
                self.spiky.weight[i, i] = 8.0

    def forward(self, x):
        return self.spiky(self.dense(x))


def _linear_with(weight_fn, dim=64):
    layer = nn.Linear(dim, dim)
    with torch.no_grad():
        layer.weight.copy_(weight_fn())
        layer.bias.zero_()
    return layer


@pytest.fixture
def acts():
    torch.manual_seed(0)
    return torch.randn(16, 64)


class TestLayerProfile:
    def test_sensitivity_prefers_the_named_reference(self):
        p = LayerProfile("l", 100, {"int8": 0.01, "int4": 0.4})
        assert p.sensitivity("int4") == 0.4
        assert p.sensitivity("int8") == 0.01

    def test_sensitivity_falls_back_to_worst_when_reference_absent(self):
        p = LayerProfile("l", 100, {"int8": 0.01, "int4": 0.4})
        assert p.sensitivity("int2") == 0.4

    def test_sensitivity_of_empty_profile_is_zero(self):
        assert LayerProfile("l", 100, {}).sensitivity() == 0.0


class TestFailedProfilePenalty:
    """An unprofilable layer must be pushed to FP16, not silently guessed at."""

    def test_penalty_is_ordered_by_aggressiveness(self):
        errs = _failed_profile_errors(("int8", "int4"))
        assert errs["int4"] > errs["int8"] > 0

    def test_ordering_is_independent_of_tuple_order(self):
        a = _failed_profile_errors(("int8", "int4"))
        b = _failed_profile_errors(("int4", "int8"))
        assert a == b

    def test_allocator_keeps_a_failed_layer_at_fp16(self):
        profiles = {
            "broken": LayerProfile("broken", 1000, _failed_profile_errors(DEFAULT_PRECISIONS)),
            "fine": LayerProfile("fine", 1000, {"int8": 0.01, "int4": 0.05}),
        }
        # Budget for one FP16 (2000B) + one INT4 (500B), with slack.
        result = allocate_optimal(profiles, budget_bytes=2600)
        assert result.allocations["broken"] == "fp16"


class TestMeasureLayer:
    def test_more_aggressive_precision_is_never_less_lossy(self, acts):
        layer = nn.Linear(64, 64)
        with torch.no_grad():
            layer.weight.normal_(0, 0.02)
        errors = SensitivityProfiler()._measure_layer(layer, acts)
        assert set(errors) == set(DEFAULT_PRECISIONS)
        assert errors["int4"] >= errors["int8"]
        assert all(e >= 0 for e in errors.values())

    def test_int8_is_near_lossless_on_well_conditioned_weights(self, acts):
        layer = nn.Linear(64, 64)
        with torch.no_grad():
            layer.weight.normal_(0, 0.02)
        errors = SensitivityProfiler()._measure_layer(layer, acts)
        assert errors["int8"] < 0.05, f"INT8 rel error unexpectedly high: {errors['int8']}"

    def test_kernel_scheme_is_measured_separately(self, acts):
        """use_kernel must profile the per-channel kernel format, not the eager one.

        These are different quantizers, so profiling the wrong one hands the
        allocator errors for a model it isn't going to build.
        """
        layer = nn.Linear(64, 64)
        with torch.no_grad():
            layer.weight.normal_(0, 0.02)
            layer.weight[0, 0] = 5.0  # make the two schemes visibly disagree
        eager = SensitivityProfiler(use_kernel=False)._measure_layer(layer, acts)
        kernel = SensitivityProfiler(use_kernel=True)._measure_layer(layer, acts)
        assert eager["int8"] != kernel["int8"]


class TestQuantizationDifficulty:
    """What the profiler scores as hard, and why.

    Recorded here because the intuitive expectation is backwards. A layer with a
    few extreme weights is *not* harder to quantize under this quantizer -- the
    sparse-outlier path keeps those weights exact in FP16 AND excludes them from
    the block scale, so the remaining weights get a tighter grid. Since the
    preserved spikes also dominate the layer's output, relative error collapses:
    measured ~16x LOWER than a plain Gaussian layer.

    What is genuinely hard is *distribution mismatch*: NF4's levels are placed at
    the quantiles of a normal, so uniform or bimodal weights -- where no small
    subset can be peeled off as outliers -- fit the codebook worst.
    """

    def test_outlier_dominated_layer_is_rescued_by_the_sparse_path(self, acts):
        torch.manual_seed(0)
        gaussian = _linear_with(lambda: torch.randn(64, 64) * 0.02)
        spiky = _linear_with(lambda: (torch.randn(64, 64) * 0.02).index_put_(
            (torch.arange(0, 64, 8), torch.arange(0, 64, 8)), torch.full((8,), 8.0)))

        p = SensitivityProfiler()
        e_gauss = p._measure_layer(gaussian, acts)["int4"]
        e_spiky = p._measure_layer(spiky, acts)["int4"]
        assert e_spiky < e_gauss / 4, (
            f"outlier path failed to rescue the spiky layer: {e_spiky:.5f} vs "
            f"gaussian {e_gauss:.5f} -- expected a large reduction")

    def test_distribution_mismatch_is_harder_than_gaussian(self, acts):
        """NF4 is Gaussian-matched, so bimodal weights should score worse."""
        torch.manual_seed(0)
        gaussian = _linear_with(lambda: torch.randn(64, 64) * 0.02)
        bimodal = _linear_with(
            lambda: (torch.randint(0, 2, (64, 64)).float() * 2 - 1) * 0.05
            + torch.randn(64, 64) * 0.002)

        p = SensitivityProfiler()
        assert p._measure_layer(bimodal, acts)["int4"] > p._measure_layer(gaussian, acts)["int4"]

    def test_profiler_discriminates_between_layers(self, acts):
        """The allocator is only useful if scores actually differ across layers."""
        torch.manual_seed(0)
        p = SensitivityProfiler()
        scores = [
            p._measure_layer(_linear_with(fn), acts)["int4"]
            for fn in (lambda: torch.randn(64, 64) * 0.02,
                       lambda: (torch.rand(64, 64) * 2 - 1) * 0.05,
                       lambda: (torch.randn(64, 64) * 0.02).index_put_(
                           (torch.arange(0, 64, 8), torch.arange(0, 64, 8)),
                           torch.full((8,), 8.0)))
        ]
        assert max(scores) > 4 * min(scores), (
            f"profiler barely separates very different layers: {scores}")

    def test_measurement_is_deterministic(self, acts):
        torch.manual_seed(0)
        layer = _linear_with(lambda: torch.randn(64, 64) * 0.02)
        p = SensitivityProfiler()
        assert p._measure_layer(layer, acts) == p._measure_layer(layer, acts)


class TestActivationCapture:
    def test_captured_activations_are_real_and_capped(self):
        torch.manual_seed(0)
        model = TinyModel()
        profiler = SensitivityProfiler(max_rows=10)
        batches = [torch.randn(1, 8, 64) for _ in range(4)]
        captured = profiler._capture_activations(
            model, {"dense": model.dense, "spiky": model.spiky}, batches)

        assert set(captured) == {"dense", "spiky"}
        for name, rows in captured.items():
            assert rows.shape[0] <= 10, f"{name} exceeded max_rows"
            assert rows.shape[1] == 64
            assert torch.isfinite(rows.float()).all()
        # `spiky`'s input is `dense`'s output, so the two must differ -- this is
        # what "profiled on real activations, not noise" actually means.
        assert not torch.allclose(captured["dense"].float(), captured["spiky"].float())

    def test_hooks_are_removed_after_capture(self):
        model = TinyModel()
        profiler = SensitivityProfiler(max_rows=8)
        profiler._capture_activations(
            model, {"dense": model.dense}, [torch.randn(1, 4, 64)])
        assert len(model.dense._forward_hooks) == 0, "forward hook leaked"


class TestTargetSelection:
    def test_excludes_embeddings_norms_and_lm_head(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Linear(8, 8)
                self.layernorm = nn.Linear(8, 8)
                self.lm_head = nn.Linear(8, 8)
                self.mlp = nn.Linear(8, 8)

        targets = SensitivityProfiler()._target_layers(M())
        assert set(targets) == {"mlp"}

    def test_custom_exclude_patterns_are_honoured(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.keep = nn.Linear(8, 8)
                self.drop = nn.Linear(8, 8)

        targets = SensitivityProfiler(exclude_patterns=("drop",))._target_layers(M())
        assert set(targets) == {"keep"}


class TestCalibrationBatches:
    def test_falls_back_to_random_ids_without_a_tokenizer(self):
        profiler = SensitivityProfiler(max_samples=3, seq_len=12)

        class Cfg:
            vocab_size = 50

        class M(nn.Module):
            config = Cfg()

        batches = profiler._build_calibration_batches(M(), None, None)
        assert len(batches) == 3
        for b in batches:
            assert b.shape == (1, 12)
            assert int(b.max()) < 50

    def test_random_fallback_is_seeded(self):
        class Cfg:
            vocab_size = 50

        class M(nn.Module):
            config = Cfg()

        p = SensitivityProfiler(max_samples=2, seq_len=8)
        a = p._build_calibration_batches(M(), None, None)
        b = p._build_calibration_batches(M(), None, None)
        assert all(torch.equal(x, y) for x, y in zip(a, b))

    def test_respects_max_samples(self):
        profiler = SensitivityProfiler(max_samples=2)
        texts = ["one", "two", "three", "four"]

        class Tok:
            def __call__(self, text, **kw):
                return {"input_ids": torch.ones(1, 4, dtype=torch.long)}

        assert len(profiler._build_calibration_batches(None, Tok(), texts)) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

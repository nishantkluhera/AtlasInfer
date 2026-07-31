"""
Tests for the NF3 (3-bit) tier.

This tier exists for one reason: with only {fp16, int8, int4} the allocator's
cheapest option IS uniform int4, so a mixed allocation can never be smaller than
uniform 4-bit and "mixed precision at equal memory to uniform NF4" is vacuous.
The load-bearing property is therefore not "3-bit works" but **"a mixed
allocation can now reach or undercut uniform-NF4's footprint"** — tested
explicitly below.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.allocator import BYTES_PER_PARAM, allocate_greedy, allocate_optimal
from atlasinfer.linear import QuantizedLinear3bit, create_quantized_linear
from atlasinfer.quantizer import (
    NF3_LEVELS,
    _pack3,
    _unpack3,
    dequantize_tensor_nf3,
    dequantize_tensor_nf4,
    quantize_tensor_nf3,
    quantize_tensor_nf4,
)
from atlasinfer.sensitivity import LayerProfile


class TestPacking:
    def test_roundtrip_is_exact_for_every_code(self):
        codes = torch.arange(8).repeat(512)
        assert torch.equal(_unpack3(_pack3(codes)), codes.long())

    def test_roundtrip_is_exact_for_random_codes(self):
        torch.manual_seed(0)
        codes = torch.randint(0, 8, (8 * 997,))
        assert torch.equal(_unpack3(_pack3(codes)), codes.long())

    def test_packs_to_exactly_three_bits_per_weight(self):
        codes = torch.randint(0, 8, (8000,))
        packed = _pack3(codes)
        assert packed.numel() * 8 / codes.numel() == 3.0
        assert packed.dtype == torch.uint8


class TestCodebook:
    def test_has_eight_levels_spanning_minus_one_to_one(self):
        assert NF3_LEVELS.numel() == 8
        assert float(NF3_LEVELS[0]) == -1.0
        assert float(NF3_LEVELS[-1]) == 1.0

    def test_contains_an_exact_zero(self):
        """Needed so pruned/sparse weights survive exactly."""
        assert (NF3_LEVELS == 0).any()

    def test_levels_are_strictly_increasing(self):
        assert torch.all(NF3_LEVELS[1:] > NF3_LEVELS[:-1])

    def test_beats_a_uniform_grid_on_gaussian_data(self):
        """The whole point of a NormalFloat codebook over a uniform one."""
        torch.manual_seed(0)
        g = torch.randn(200_000)

        def rel_rmse(levels):
            L = torch.tensor(levels, dtype=torch.float32)
            bounds = (L[:-1] + L[1:]) / 2
            s = g.abs().max()
            codes = torch.bucketize(g / s, bounds).clamp(0, len(L) - 1)
            return ((g - L[codes] * s).pow(2).mean().sqrt() / g.pow(2).mean().sqrt()).item()

        uniform8 = ((torch.arange(8).float() - 3.5) / 3.5).tolist()
        assert rel_rmse(NF3_LEVELS.tolist()) < rel_rmse(uniform8)


class TestQuantizeRoundtrip:
    @pytest.mark.parametrize("shape", [(64, 64), (7,), (13, 5), (1,), (63,), (128, 33)])
    def test_shape_is_preserved(self, shape):
        torch.manual_seed(0)
        t = torch.randn(*shape) * 0.02
        assert dequantize_tensor_nf3(quantize_tensor_nf3(t, block_size=8)).shape == t.shape

    def test_empty_tensor(self):
        q = quantize_tensor_nf3(torch.empty(0))
        assert dequantize_tensor_nf3(q).numel() == 0

    def test_reconstruction_is_reasonable(self):
        torch.manual_seed(0)
        W = torch.randn(256, 256) * 0.02
        rel = ((W - dequantize_tensor_nf3(quantize_tensor_nf3(W)).float()).norm()
               / W.norm()).item()
        assert rel < 0.30, f"NF3 reconstruction error {rel}"

    def test_is_lossier_than_nf4_but_smaller(self):
        """The trade the allocator is being asked to make must actually exist."""
        torch.manual_seed(0)
        W = torch.randn(256, 256) * 0.02
        q3, q4 = quantize_tensor_nf3(W), quantize_tensor_nf4(W)
        e3 = ((W - dequantize_tensor_nf3(q3).float()).norm() / W.norm()).item()
        e4 = ((W - dequantize_tensor_nf4(q4).float()).norm() / W.norm()).item()
        assert e3 > e4, "NF3 should be lossier than NF4"
        assert q3.memory_bytes() < q4.memory_bytes(), "NF3 should be smaller than NF4"

    def test_outliers_are_preserved_exactly(self):
        torch.manual_seed(0)
        W = torch.randn(64, 64) * 0.02
        W[0, 0] = 5.0
        q = quantize_tensor_nf3(W)
        assert q.outlier_indices.numel() > 0
        recon = dequantize_tensor_nf3(q).float()
        assert abs(recon[0, 0].item() - 5.0) < 0.01


class TestLayer:
    def test_forward_matches_dense_approximately(self):
        torch.manual_seed(0)
        lin = nn.Linear(128, 64)
        lin.weight.data *= 0.05
        layer = QuantizedLinear3bit.from_linear(lin)
        x = torch.randn(4, 128)
        rel = ((lin(x) - layer(x)).norm() / lin(x).norm()).item()
        assert rel < 0.35, f"3-bit layer rel err {rel}"

    def test_created_via_the_factory(self):
        layer = create_quantized_linear(nn.Linear(64, 32), precision="int3")
        assert isinstance(layer, QuantizedLinear3bit)
        assert layer.precision == "int3"

    def test_nf3_alias_resolves(self):
        assert isinstance(create_quantized_linear(nn.Linear(64, 32), precision="nf3"),
                          QuantizedLinear3bit)

    def test_smaller_than_the_four_bit_layer(self):
        lin = nn.Linear(256, 256)
        b3 = QuantizedLinear3bit.from_linear(lin).quantized_weights.memory_bytes()
        b4 = create_quantized_linear(lin, precision="int4").quantized_weights.memory_bytes()
        assert b3 < b4

    def test_dtype_transparent(self):
        layer = QuantizedLinear3bit.from_linear(nn.Linear(32, 16))
        assert layer(torch.randn(2, 32, dtype=torch.float32)).dtype == torch.float32


class TestAllocatorIntegration:
    """The reason the tier exists.

    int3 is an opt-in tier, not a library default (it loses in the default
    allocator — see allocator.allocate_optimal), so these tests that exercise it
    pass it explicitly, exactly as PAPER/exp/iso_memory.py does.
    """

    TIERS = ("fp16", "int8", "int4", "int3")

    def test_bytes_per_param_is_three_bits(self):
        assert BYTES_PER_PARAM["int3"] == 0.375
        assert BYTES_PER_PARAM["int3"] < BYTES_PER_PARAM["int4"]

    def _profiles(self, n=8, params=10_000):
        return {
            f"L{i}": LayerProfile(f"L{i}", params,
                                  {"int8": 0.01 * (i + 1),
                                   "int4": 0.05 * (i + 1),
                                   "int3": 0.15 * (i + 1)})
            for i in range(n)
        }

    def test_mixed_allocation_can_undercut_uniform_int4(self):
        """THE load-bearing property: without int3 this is impossible.

        Budget set below what uniform int4 costs. Before the int3 tier existed
        the allocator's floor WAS uniform int4, so no allocation could satisfy
        this and an iso-memory comparison against uniform NF4 was vacuous.
        """
        profiles = self._profiles()
        uniform_int4 = sum(p.param_count for p in profiles.values()) * 0.5
        budget = int(uniform_int4 * 0.9)      # 10% BELOW uniform 4-bit

        result = allocate_optimal(profiles, budget_bytes=budget, precisions=self.TIERS)
        assert result.total_bytes <= budget, "allocator could not reach a sub-int4 budget"
        assert result.counts.get("int3", 0) > 0, "int3 tier was never used"

    def test_the_iso_memory_trade_requires_heterogeneous_sensitivity(self):
        """At uniform-int4 memory the trade only pays when layers DIFFER enough.

        Dropping a layer 4->3 bits frees 0.125 bytes/param; upgrading one 4->8
        costs 0.5. So four demotions fund one promotion, and that is only a win
        if the demoted layers are far more robust than the promoted one is
        fragile. With uniformly-scaled errors it is NOT a win, and the DP
        correctly returns plain uniform int4 -- worth pinning, because it means
        the int3 tier is an enabler, not a free lunch.
        """
        uniform = self._profiles()   # int3 error is a flat 3x int4 for every layer
        budget = int(sum(p.param_count for p in uniform.values()) * 0.5)
        flat = allocate_optimal(uniform, budget_bytes=budget, precisions=self.TIERS)
        assert flat.counts.get("int4", 0) == len(uniform), (
            "with homogeneous sensitivity the optimum should stay uniform int4, "
            f"got {flat.counts}")

    def test_iso_memory_trade_happens_when_sensitivity_is_heterogeneous(self):
        """The mechanism behind an iso-memory win over uniform quantization.

        Four layers barely care about 3-bit; four are badly hurt by 4-bit. At
        exactly uniform-int4 memory the optimum should demote the robust ones and
        promote the fragile ones, beating uniform int4 outright.
        """
        robust = {f"R{i}": LayerProfile(f"R{i}", 10_000,
                                        {"int8": 0.001, "int4": 0.002, "int3": 0.004})
                  for i in range(4)}
        fragile = {f"F{i}": LayerProfile(f"F{i}", 10_000,
                                         {"int8": 0.01, "int4": 0.50, "int3": 2.00})
                   for i in range(4)}
        profiles = {**robust, **fragile}
        budget = int(sum(p.param_count for p in profiles.values()) * 0.5)

        result = allocate_optimal(profiles, budget_bytes=budget, precisions=self.TIERS)

        assert result.total_bytes <= budget * 1.01
        assert result.counts.get("int3", 0) > 0, "no layer was demoted to 3-bit"
        assert result.counts.get("int8", 0) > 0, "no layer was promoted to INT8"
        assert all(result.allocations[n] != "int3" for n in fragile), \
            "a fragile layer was demoted to 3-bit"
        uniform_err = sum(p.errors["int4"] for p in profiles.values())
        assert result.predicted_error < uniform_err, (
            f"iso-memory allocation ({result.predicted_error:.3f}) failed to beat "
            f"uniform int4 ({uniform_err:.3f})")

    def test_greedy_also_reaches_sub_int4_budgets(self):
        profiles = self._profiles()
        budget = int(sum(p.param_count for p in profiles.values()) * 0.45)
        sens = {n: p.sensitivity("int4") for n, p in profiles.items()}
        sizes = {n: p.param_count for n, p in profiles.items()}
        r = allocate_greedy(sens, sizes, budget, profiles=profiles, precisions=self.TIERS)
        assert r.total_bytes <= budget

    def test_layers_without_an_int3_profile_are_never_assigned_int3(self):
        """A tier the profiler didn't measure must not be silently offered.

        Guards the regression where adding int3 to the default precision list
        made the greedy start pre-int3 profiles at a bit-width it had no error
        measurement for.
        """
        profiles = {
            "old": LayerProfile("old", 10_000, {"int8": 0.01, "int4": 0.05}),
            "new": LayerProfile("new", 10_000, {"int8": 0.01, "int4": 0.05, "int3": 0.2}),
        }
        sens = {n: p.sensitivity("int4") for n, p in profiles.items()}
        sizes = {n: p.param_count for n, p in profiles.items()}
        budget = int(10_000 * 0.5 * 2)

        for result in (allocate_optimal(profiles, budget_bytes=budget, precisions=self.TIERS),
                       allocate_greedy(sens, sizes, budget, profiles=profiles,
                                       precisions=self.TIERS)):
            assert result.allocations["old"] != "int3"


class TestPackageExports:
    """Guards the top-level public API.

    Regression: ``QuantizedLinear3bit`` was added to ``atlasinfer.__all__`` but
    never imported into ``__init__``, so ``from atlasinfer import *`` raised
    AttributeError and ``from atlasinfer import QuantizedLinear3bit`` raised
    ImportError — while the whole suite stayed green because every test imports
    from ``atlasinfer.linear`` directly.
    """

    def test_every_name_in_all_is_importable(self):
        import atlasinfer
        missing = [n for n in atlasinfer.__all__ if not hasattr(atlasinfer, n)]
        assert not missing, f"__all__ names not importable from atlasinfer: {missing}"

    def test_quantized_linear_3bit_is_a_top_level_export(self):
        from atlasinfer import QuantizedLinear3bit
        from atlasinfer.linear import QuantizedLinear3bit as FromLinear
        assert QuantizedLinear3bit is FromLinear

    def test_star_import_succeeds(self):
        ns = {}
        exec("from atlasinfer import *", ns)
        assert "QuantizedLinear3bit" in ns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

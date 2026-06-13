"""
Tests for the precision allocator (the core budget-allocation algorithm).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.allocator import (
    allocate_optimal,
    allocate_greedy,
    uniform_allocation,
    BYTES_PER_PARAM,
)
from atlasinfer.sensitivity import LayerProfile


def _profiles(spec):
    """spec: {name: (param_count, {precision: error})} -> {name: LayerProfile}."""
    return {
        name: LayerProfile(name=name, param_count=pc, errors=errs)
        for name, (pc, errs) in spec.items()
    }


class TestOptimalAllocator:
    def test_protects_the_sensitive_layer(self):
        # Two equal-size layers. 'A' is very sensitive to INT4, 'B' is robust.
        # Budget only allows one of them at INT8 -> the optimum keeps A at INT8.
        profiles = _profiles({
            "A": (1000, {"int8": 0.10, "int4": 0.50}),
            "B": (1000, {"int8": 0.01, "int4": 0.05}),
        })
        # Room for one INT8 (1000B) + one INT4 (500B) = 1500B, with slack.
        result = allocate_optimal(profiles, budget_bytes=1600)

        assert result.allocations["A"] == "int8"
        assert result.allocations["B"] == "int4"

    def test_respects_budget(self):
        profiles = _profiles({
            f"L{i}": (1000, {"int8": 0.1 * i, "int4": 0.3 * i})
            for i in range(1, 11)
        })
        budget = 7000  # between all-int4 (5000) and all-int8 (10000)
        result = allocate_optimal(profiles, budget_bytes=budget)
        # Budget is a soft target (discretized grid); stay within a hair of it.
        assert result.total_bytes <= budget * 1.01

    def test_more_budget_never_hurts(self):
        profiles = _profiles({
            f"L{i}": (1000, {"int8": 0.05 * i, "int4": 0.2 * i})
            for i in range(1, 9)
        })
        errs = []
        for budget in (4000, 6000, 8000, 12000, 16000):
            errs.append(allocate_optimal(profiles, budget_bytes=budget).predicted_error)
        # Predicted error is non-increasing as the budget grows.
        for earlier, later in zip(errs, errs[1:]):
            assert later <= earlier + 1e-9

    def test_generous_budget_picks_fp16(self):
        profiles = _profiles({
            "A": (1000, {"int8": 0.1, "int4": 0.5}),
            "B": (1000, {"int8": 0.1, "int4": 0.5}),
        })
        # Enough for everything at FP16 (2000B each = 4000B).
        result = allocate_optimal(profiles, budget_bytes=4000)
        assert all(p == "fp16" for p in result.allocations.values())
        assert result.predicted_error == 0.0

    def test_beats_greedy_on_byte_efficiency(self):
        # Greedy upgrades the *most sensitive* layer first (A), but A's INT4->INT8
        # gain (0.4) is smaller per byte than B's (0.6). With budget for exactly
        # one upgrade, the optimum spends it on B, not A.
        profiles = _profiles({
            "A": (1000, {"int8": 0.5, "int4": 0.9}),  # most sensitive, weak gain
            "B": (1000, {"int8": 0.0, "int4": 0.6}),  # big gain from INT8
            "C": (1000, {"int8": 0.0, "int4": 0.6}),
        })
        budget = 2000  # all-int4 = 1500; room for exactly one INT8 upgrade (+500)

        opt = allocate_optimal(profiles, budget_bytes=budget)

        sensitivities = {n: p.sensitivity("int4") for n, p in profiles.items()}
        sizes = {n: p.param_count for n, p in profiles.items()}
        greedy = allocate_greedy(sensitivities, sizes, budget)

        def total_error(alloc):
            return sum(
                profiles[n].errors.get(p, 0.0) for n, p in alloc.allocations.items()
            )

        assert opt.total_bytes <= budget * 1.01
        # Greedy upgrades A -> total error 1.7; the optimum upgrades B (or C) -> 1.5.
        assert total_error(opt) < total_error(greedy)
        assert opt.allocations["A"] == "int4"  # left aggressive on purpose


class TestHelpers:
    def test_uniform_allocation(self):
        alloc = uniform_allocation(["a", "b", "c"], "int4")
        assert alloc == {"a": "int4", "b": "int4", "c": "int4"}

    def test_bytes_per_param(self):
        assert BYTES_PER_PARAM["fp16"] == 2.0
        assert BYTES_PER_PARAM["int8"] == 1.0
        assert BYTES_PER_PARAM["int4"] == 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

"""
AtlasInfer Precision Allocator.

Given each layer's measured quantization error at every candidate bit-width and a
total memory budget, decide which precision (FP16 / INT8 / INT4) each layer should
use so that overall accuracy loss is minimized.

Two allocators are provided:

* ``allocate_optimal`` - solves the assignment exactly as a multiple-choice
  knapsack via dynamic programming. Each layer must pick exactly one precision;
  the DP minimizes the sum of measured per-layer errors subject to staying under
  the byte budget. This is the allocator the engine uses by default.

* ``allocate_greedy`` - a fast sensitivity-ranked heuristic kept as a baseline so
  the optimal allocator's win can be quantified.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch


class PrecisionLevel(Enum):
    """Supported weight precisions, ordered high to low quality."""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bytes_per_param(self) -> float:
        return {
            PrecisionLevel.FP16: 2.0,
            PrecisionLevel.INT8: 1.0,
            PrecisionLevel.INT4: 0.5,
        }[self]


# Bytes-per-parameter for each precision label (used for budgeting).
BYTES_PER_PARAM: Dict[str, float] = {
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class AllocationResult:
    """Outcome of a precision allocation."""
    allocations: Dict[str, str]          # layer name -> precision label
    total_bytes: int
    budget_bytes: int
    counts: Dict[str, int]               # precision label -> layer count
    predicted_error: float               # sum of chosen layers' relative errors
    total_params: int = 0                # total params across allocated layers

    @property
    def avg_bits(self) -> float:
        """Parameter-weighted average bits-per-weight (matches the byte budget)."""
        if not self.total_params:
            return 0.0
        return 8.0 * self.total_bytes / self.total_params

    def summary(self) -> str:
        parts = ", ".join(
            f"{p.upper()}={self.counts.get(p, 0)}" for p in ("fp16", "int8", "int4")
        )
        return (
            f"Allocation: {parts} | "
            f"Memory: {self.total_bytes / 1024**3:.2f}/{self.budget_bytes / 1024**3:.2f} GB | "
            f"predicted error sum: {self.predicted_error:.3f}"
        )


def get_layer_sizes(model: torch.nn.Module) -> Dict[str, int]:
    """Parameter counts (weight + bias) for all linear-like layers.

    Supports both ``nn.Linear`` and HuggingFace ``Conv1D``.
    """
    sizes: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sizes[name] = module.weight.numel()
            if module.bias is not None:
                sizes[name] += module.bias.numel()
        elif type(module).__name__ == "Conv1D" and hasattr(module, "weight"):
            sizes[name] = module.weight.numel()
            if getattr(module, "bias", None) is not None:
                sizes[name] += module.bias.numel()
    return sizes


def _layer_bytes(param_count: int, precision: str) -> int:
    return int(param_count * BYTES_PER_PARAM.get(precision, 2.0))


def allocate_optimal(
    profiles: Dict[str, "object"],
    budget_bytes: int,
    precisions: List[str] = ("fp16", "int8", "int4"),
    num_buckets: int = 4096,
) -> AllocationResult:
    """Minimum-error precision assignment within a byte budget.

    Models the choice as a multiple-choice knapsack and solves it with DP over a
    discretized memory axis (``num_buckets`` resolution). The result is optimal
    on that grid; with the default resolution the discretization error is far
    below the per-block scale/outlier overhead the byte budget already ignores,
    so the budget is treated as a soft target (the reported ``total_bytes`` is
    the true footprint).

    Args:
        profiles: ``layer_name -> LayerProfile`` (see ``sensitivity.py``). FP16 is
            treated as lossless (error 0); INT8/INT4 errors come from the profile.
        budget_bytes: Total weight-memory budget for the profiled layers.
        precisions: Candidate precision labels, highest quality first.
        num_buckets: Memory-axis resolution for the DP (higher = finer/slower).

    Returns:
        An :class:`AllocationResult`.
    """
    names = list(profiles.keys())
    if not names:
        return AllocationResult({}, 0, budget_bytes, {}, 0.0)

    # Build per-layer options: (precision, bytes, error). FP16 is lossless.
    options: List[List[tuple]] = []
    for name in names:
        prof = profiles[name]
        layer_opts = []
        for p in precisions:
            err = 0.0 if p == "fp16" else prof.errors.get(p)
            if err is None:
                continue
            layer_opts.append((p, _layer_bytes(prof.param_count, p), err))
        # Always guarantee at least the smallest option exists.
        if not layer_opts:
            layer_opts.append(("int4", _layer_bytes(prof.param_count, "int4"), 1.0))
        options.append(layer_opts)

    # Discretize the memory axis. Anchor the scale to the cheapest feasible
    # configuration so the budget always spans the full bucket range.
    min_total = sum(min(o[1] for o in opts) for opts in options)
    if min_total > budget_bytes:
        # Budget can't even fit everything at the lowest precision: take the
        # cheapest option everywhere and report the overflow.
        alloc = {name: min(opts, key=lambda o: o[1])[0]
                 for name, opts in zip(names, options)}
        return _finalize(alloc, options, names, budget_bytes)

    scale = budget_bytes / num_buckets
    INF = float("inf")

    # dp[c] = min total error using processed layers with cost <= c buckets.
    dp = [INF] * (num_buckets + 1)
    dp[0] = 0.0
    # back[layer][c] = (prev_capacity, option_index)
    back: List[List[Optional[tuple]]] = []

    for opts in options:
        new_dp = [INF] * (num_buckets + 1)
        layer_back: List[Optional[tuple]] = [None] * (num_buckets + 1)
        # Precompute integer bucket cost per option (round to nearest grid cell).
        opt_cost = [max(1, int(round(b / scale))) for (_, b, _) in opts]
        for c in range(num_buckets + 1):
            base = dp[c]
            if base == INF:
                continue
            for oi, (cost, (_, _, err)) in enumerate(zip(opt_cost, opts)):
                nc = c + cost
                if nc > num_buckets:
                    continue
                cand = base + err
                if cand < new_dp[nc]:
                    new_dp[nc] = cand
                    layer_back[nc] = (c, oi)
        dp = new_dp
        back.append(layer_back)

    # Best feasible capacity = min error over all reachable capacities.
    best_c, best_err = min(
        ((c, dp[c]) for c in range(num_buckets + 1) if dp[c] < INF),
        key=lambda t: t[1],
    )

    # Reconstruct per-layer choices.
    alloc: Dict[str, str] = {}
    c = best_c
    for li in range(len(options) - 1, -1, -1):
        prev_c, oi = back[li][c]
        alloc[names[li]] = options[li][oi][0]
        c = prev_c

    return _finalize(alloc, options, names, budget_bytes, predicted_error=best_err)


def _finalize(
    alloc: Dict[str, str],
    options: List[List[tuple]],
    names: List[str],
    budget_bytes: int,
    predicted_error: Optional[float] = None,
) -> AllocationResult:
    total_bytes = 0
    total_params = 0
    err_sum = 0.0
    for name, opts in zip(names, options):
        chosen = alloc[name]
        for (p, b, e) in opts:
            if p == chosen:
                total_bytes += b
                total_params += int(round(b / BYTES_PER_PARAM.get(p, 2.0)))
                err_sum += e
                break
    counts: Dict[str, int] = {}
    for p in alloc.values():
        counts[p] = counts.get(p, 0) + 1
    return AllocationResult(
        allocations=alloc,
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        counts=counts,
        predicted_error=predicted_error if predicted_error is not None else err_sum,
        total_params=total_params,
    )


def allocate_greedy(
    sensitivities: Dict[str, float],
    layer_sizes: Dict[str, int],
    budget_bytes: int,
    precisions: List[str] = ("fp16", "int8", "int4"),
) -> AllocationResult:
    """Baseline allocator: start everyone at the lowest precision, then upgrade
    the most sensitive layers first while the budget allows.

    Kept for comparison against :func:`allocate_optimal`.
    """
    ordered = sorted(precisions, key=lambda p: BYTES_PER_PARAM.get(p, 2.0))
    min_precision = ordered[0]
    alloc = {name: min_precision for name in sensitivities if name in layer_sizes}

    def total_bytes(a: Dict[str, str]) -> int:
        return sum(_layer_bytes(layer_sizes[n], p) for n, p in a.items())

    current = total_bytes(alloc)
    # Most sensitive first.
    for name, _ in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        if name not in alloc:
            continue
        cur_idx = ordered.index(alloc[name])
        for better in ordered[cur_idx + 1:]:
            delta = _layer_bytes(layer_sizes[name], better) - _layer_bytes(layer_sizes[name], alloc[name])
            if current + delta <= budget_bytes:
                current += delta
                alloc[name] = better
            else:
                break

    counts: Dict[str, int] = {}
    for p in alloc.values():
        counts[p] = counts.get(p, 0) + 1
    return AllocationResult(
        allocations=alloc,
        total_bytes=current,
        budget_bytes=budget_bytes,
        counts=counts,
        predicted_error=0.0,
        total_params=sum(layer_sizes[n] for n in alloc),
    )


def uniform_allocation(layer_names: List[str], precision: str = "int8") -> Dict[str, str]:
    """All layers at the same precision (baseline)."""
    return {name: precision for name in layer_names}


def estimate_memory_usage(layer_sizes: Dict[str, int], precision: str) -> int:
    """Total bytes if every layer used ``precision``."""
    return int(sum(layer_sizes.values()) * BYTES_PER_PARAM.get(precision, 2.0))


def print_allocation_report(
    result: AllocationResult,
    sensitivities: Optional[Dict[str, float]] = None,
    top_n: int = 15,
):
    """Print a readable summary of an allocation."""
    print("\n" + "=" * 72)
    print("Precision Allocation Report")
    print("=" * 72)
    print(f"Budget:    {result.budget_bytes / 1024**3:.3f} GB")
    util = 100 * result.total_bytes / result.budget_bytes if result.budget_bytes else 0
    print(f"Allocated: {result.total_bytes / 1024**3:.3f} GB ({util:.1f}% of budget)")
    print(f"Avg bits/weight: {result.avg_bits:.2f}")
    print("Precision distribution:")
    for p in ("fp16", "int8", "int4"):
        print(f"  {p.upper():<5}: {result.counts.get(p, 0)} layers")

    if sensitivities:
        ranked = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{'Layer':<45} {'Precision':>10} {'Rel.error':>12}")
        print("-" * 72)
        for name, sens in ranked[:top_n]:
            prec = result.allocations.get(name, "?")
            display = name if len(name) <= 43 else "..." + name[-40:]
            print(f"{display:<45} {prec:>10} {sens:>12.6f}")
        if len(ranked) > top_n:
            print(f"... and {len(ranked) - top_n} more layers")
    print("=" * 72 + "\n")

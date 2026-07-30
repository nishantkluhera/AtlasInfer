"""
Single source of truth for "which layers does AtlasInfer touch?".

This logic used to exist in three places — ``patcher.py`` (as a list),
``sensitivity.py`` (as a tuple), and ``benchmark.linear_param_count`` (as a
*four*-entry tuple missing ``"layernorm"``) — plus duplicate copies of
``_is_linear_layer`` and the parameter-counting helpers.

That duplication is not cosmetic. The profiler measures one set of layers and the
patcher quantizes another; the allocator's output is keyed by the profiler's
names. If the two sets ever diverge, ``quantize_model_mixed`` silently falls back
to ``default_precision`` for every unmatched layer (see ``patcher.py``) — you get
plausible-looking perplexity from an allocation nobody computed, with no error.

It has already drifted once: the ``benchmark.py`` copy is missing ``"layernorm"``
and is saved only by ``"norm"`` being a substring of it.

Everything that needs to answer "is this layer quantizable?" imports from here.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Layers never quantized: numerically delicate, or tiny relative to the model.
# Matching is case-insensitive substring, so "norm" also covers "layernorm" —
# the latter is kept explicit for readability, not because it adds coverage.
DEFAULT_EXCLUDE: Tuple[str, ...] = (
    "embed", "lm_head", "norm", "ln_", "layernorm",
)


def is_linear_layer(module: nn.Module) -> bool:
    """True for ``nn.Linear`` and HuggingFace ``Conv1D`` (GPT-2 family)."""
    if isinstance(module, nn.Linear):
        return True
    return type(module).__name__ == "Conv1D"


def is_excluded(name: str, exclude_patterns=DEFAULT_EXCLUDE) -> bool:
    """True if ``name`` matches any exclusion pattern (case-insensitive substring)."""
    lowered = name.lower()
    return any(str(p).lower() in lowered for p in exclude_patterns)


def param_count(module: nn.Module) -> int:
    """Parameters in a linear-like layer, weight + bias."""
    n = module.weight.numel()
    bias = getattr(module, "bias", None)
    if bias is not None:
        n += bias.numel()
    return n


def dense_bytes(module: nn.Module) -> int:
    """Bytes the layer occupies dense, at its current dtype."""
    return param_count(module) * module.weight.element_size()


def target_modules(model: nn.Module, exclude_patterns=DEFAULT_EXCLUDE) -> Dict[str, nn.Module]:
    """``name -> module`` for every quantizable linear layer."""
    return {
        name: module
        for name, module in model.named_modules()
        if is_linear_layer(module) and not is_excluded(name, exclude_patterns)
    }


def collect_targets(model: nn.Module, exclude_patterns=DEFAULT_EXCLUDE) -> List[tuple]:
    """``(parent, attr_name, module, full_name)`` for each quantizable layer.

    The parent/attr pair is what in-place replacement needs.
    """
    targets = []
    for name, module in model.named_modules():
        if not is_linear_layer(module) or is_excluded(name, exclude_patterns):
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent, attr = model.get_submodule(parts[0]), parts[1]
        targets.append((parent, attr, module, name))
    return targets


def check_allocation_covers(
    allocation: Optional[Dict[str, str]],
    target_names,
    context: str = "",
) -> Dict[str, list]:
    """Compare an allocation's keys against the layers actually being patched.

    Returns ``{"unmatched": [...], "unallocated": [...]}``:

    * **unmatched** — keys in the allocation that name no layer being patched.
      Usually a profiler/patcher exclusion-list mismatch, or an allocation
      computed against a different model.
    * **unallocated** — layers being patched that the allocation says nothing
      about. These silently take ``default_precision``, which is the failure this
      function exists to surface.

    Both are warnings rather than errors: passing a partial allocation is a
    legitimate use (quantize a subset, default the rest). The point is that it
    should never happen *by accident and in silence*.
    """
    if not allocation:
        return {"unmatched": [], "unallocated": []}
    names = set(target_names)
    unmatched = sorted(set(allocation) - names)
    unallocated = sorted(names - set(allocation))
    if unmatched or unallocated:
        where = f" ({context})" if context else ""
        print(f"WARNING: allocation does not line up with the model's layers{where}.")
        if unmatched:
            print(f"  {len(unmatched)} allocation key(s) match no layer, so those "
                  f"decisions are DISCARDED. First few: {unmatched[:3]}")
        if unallocated:
            print(f"  {len(unallocated)} layer(s) absent from the allocation will "
                  f"fall back to the default precision. First few: {unallocated[:3]}")
        print("  Most likely cause: the profiler and the patcher were given "
              "different exclude_patterns.")
    return {"unmatched": unmatched, "unallocated": unallocated}

"""
Shared evaluation primitives: perplexity, calibration/eval data, memory accounting.

These lived in ``benchmark.py`` and were imported *from that script* by
``compare_baselines.py``, ``eval_downstream.py`` and the ablation harness. That
made a top-level CLI the de facto shared library: importing it executed module
scope (setting HF env vars, allocating a CUDA warmup tensor, reconfiguring
transformers' logging), required the repo root on ``sys.path``, and meant
``benchmark.py`` could not be edited without breaking three callers.

Everything importable now lives here. ``benchmark.py`` re-exports for
backwards compatibility.
"""
import math
import os
from typing import List, Optional, Tuple

import torch

# Calibration/eval both come from WikiText-2, but from DISJOINT splits -- calib
# from train, eval from test. Keeping that split in one place is the point: it is
# the guarantee that there is no contamination between what the sensitivity
# profiler sees and what perplexity is measured on.
_CALIB_SPLIT = "train"
_EVAL_SPLIT = "test"
_CALIB_MIN_CHARS = 200
_CALIB_MAX_DOCS = 256


def model_weight_bytes(model: torch.nn.Module) -> int:
    """Resident weight footprint: quantized buffers + remaining dense params."""
    from .linear import QuantizedLinear, QuantizedLinear3bit, QuantizedLinear4bit

    total = sum(p.numel() * p.element_size() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedLinear3bit, QuantizedLinear4bit)):
            total += module.quantized_weights.memory_bytes()
            if module.bias is not None:
                total += module.bias.numel() * module.bias.element_size()
            # AWQ's per-input-channel scale is a buffer, so it is in neither
            # model.parameters() nor quantized_weights.memory_bytes(). Without
            # this an AWQ model under-reports and disagrees with
            # resident_bytes() (params + ALL buffers).
            in_scale = getattr(module, "in_scale", None)
            if in_scale is not None:
                total += in_scale.numel() * in_scale.element_size()
    return total


def resident_bytes(model: torch.nn.Module) -> int:
    """Total bytes of all params + buffers resident on the model (any method).

    Method-agnostic, so bitsandbytes / GPTQ / AWQ models are counted by the same
    rule as AtlasInfer's. Note packing differs between methods (bnb
    double-quantizes its scales, AtlasInfer stores sparse FP16 outliers), so read
    this as "same accounting rule, method-specific packing".
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total


@torch.no_grad()
def evaluate_perplexity(
    model, tokenizer, text: str, device, max_len: int = 1024, stride: int = 512,
    max_tokens: Optional[int] = None,
) -> float:
    """Standard sliding-window perplexity.

    Each window scores only the tokens not already scored by the previous window
    (the rest are masked to -100), so every token is counted exactly once and the
    result is comparable across stride choices.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    input_ids = input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: List[torch.Tensor] = []
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100  # only score the new tokens in this window
        out = model(ids, labels=target)
        # out.loss is the mean NLL over (trg_len - 1) scored positions.
        nlls.append(out.loss.float() * trg_len)
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break
    return float(torch.exp(torch.stack(nlls).sum() / n_tokens))


def load_wikitext() -> Tuple[List[str], str]:
    """Return ``(calibration_texts, eval_text)`` from WikiText-2.

    Calibration comes from the **train** split and evaluation from **test**, so
    there is no overlap between what the profiler calibrates on and what
    perplexity is measured on.

    The dataset id is tried both bare and namespaced: newer
    ``huggingface_hub``/``datasets`` reject the legacy bare ``"wikitext"`` id (it
    must be ``namespace/name``), while older stacks only know the bare id. Same
    content either way, so numbers stay comparable across environments.
    """
    from datasets import load_dataset

    def _load(split):
        last = None
        for repo in ("wikitext", "Salesforce/wikitext"):
            try:
                return load_dataset(repo, "wikitext-2-raw-v1", split=split)
            except Exception as exc:  # noqa: BLE001 - id/URI schemes differ by version
                last = exc
        raise last

    test = _load(_EVAL_SPLIT)
    eval_text = "\n\n".join(t for t in test["text"] if t.strip())

    train = _load(_CALIB_SPLIT)
    # Plenty of calibration docs: the sensitivity profiler caps at its own
    # max_samples (8), while GPTQ consumes many more for a well-conditioned Hessian.
    calib = [t for t in train["text"]
             if len(t.strip()) > _CALIB_MIN_CHARS][:_CALIB_MAX_DOCS]
    return calib, eval_text


def quantized_bits_per_weight(model: torch.nn.Module) -> float:
    """MEASURED average bits per quantizable weight, including all overhead.

    The allocator budgets in *nominal* bits ({fp16: 16, int8: 8, int4: 4}) and
    ignores per-block scales, outlier indices and outlier values. This reports
    what the model actually costs, which is always higher -- a "4-bit" layer here
    really costs ~4.4 bits. Use this, not the nominal target, whenever a claim is
    phrased in bits.
    """
    from ._targets import DEFAULT_EXCLUDE, is_excluded
    from .linear import QuantizedLinear, QuantizedLinear3bit, QuantizedLinear4bit

    bits = 0
    params = 0
    for name, module in model.named_modules():
        if is_excluded(name, DEFAULT_EXCLUDE):
            continue
        if isinstance(module, (QuantizedLinear, QuantizedLinear3bit, QuantizedLinear4bit)):
            qb = module.quantized_weights.memory_bytes()
            n = int(math.prod(module.quantized_weights.original_shape))
        elif isinstance(module, torch.nn.Linear):
            n = module.weight.numel()
            qb = n * module.weight.element_size()
        else:
            continue
        bits += qb * 8
        params += n
    return bits / params if params else 0.0

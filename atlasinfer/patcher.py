"""
AtlasInfer Model Patcher - replace dense linear layers with quantized ones.

Handles both ``nn.Linear`` and HuggingFace ``Conv1D`` (GPT-2 family) and supports
two modes:

* ``quantize_model`` - uniform precision for every layer (the simple baseline).
* ``quantize_model_mixed`` - per-layer precision from an allocation dict (the
  mixed-precision path that spends a memory budget where it matters most).
"""
import gc
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ._targets import (
    DEFAULT_EXCLUDE,
    check_allocation_covers,
    collect_targets as _collect_targets,
    dense_bytes as _dense_bytes,
    is_linear_layer as _is_linear_layer,
)
from .linear import QuantizedLinear, QuantizedLinear4bit, create_quantized_linear


def _quantized_bytes(layer: nn.Module, fallback: int) -> int:
    if hasattr(layer, "quantized_weights"):
        b = layer.quantized_weights.memory_bytes()
        if getattr(layer, "bias", None) is not None:
            b += layer.bias.numel() * layer.bias.element_size()
        return b
    if hasattr(layer, "memory_bytes"):  # W8A16Linear (kernel path)
        return layer.memory_bytes()
    return fallback  # fp16: kept dense


def quantize_model(
    model: nn.Module,
    precision: str = "int8",
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True,
    use_kernel: bool = False,
    quant_4bit: str = "nf4",
    double_quant: bool = False,
) -> nn.Module:
    """Quantize every eligible linear layer to a single uniform precision."""
    return quantize_model_mixed(
        model,
        allocation=None,
        default_precision=precision,
        exclude_patterns=exclude_patterns,
        verbose=verbose,
        use_kernel=use_kernel,
        quant_4bit=quant_4bit,
        double_quant=double_quant,
    )


def quantize_model_mixed(
    model: nn.Module,
    allocation: Optional[Dict[str, str]],
    default_precision: str = "int8",
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True,
    use_kernel: bool = False,
    quant_4bit: str = "nf4",
    double_quant: bool = False,
) -> nn.Module:
    """Replace linear layers in-place using a per-layer precision allocation.

    Args:
        model: Model to quantize (typically on CPU before this call).
        allocation: ``layer_name -> precision`` ("fp16"/"int8"/"int4"). Layers
            absent from the dict fall back to ``default_precision``. If
            ``allocation`` is None, every layer uses ``default_precision``.
        default_precision: Precision for layers not present in ``allocation``.
        exclude_patterns: Name substrings to skip (defaults to embeddings,
            norms, lm_head).
    """
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE
    allocation = allocation or {}

    targets = _collect_targets(model, exclude_patterns)
    # An allocation key that matches no layer is silently discarded below, and a
    # layer absent from the allocation silently takes `default_precision`. Both
    # produce a model that looks fine and is not the one the allocator designed,
    # so say so loudly. Warn-only: a partial allocation is a legitimate input.
    check_allocation_covers(allocation, [t[3] for t in targets],
                            context="quantize_model_mixed")
    stats = {"fp16": 0, "int8": 0, "int4": 0, "int3": 0}
    original_size = 0
    quantized_size = 0

    for parent, attr, module, full_name in targets:
        precision = allocation.get(full_name, default_precision).lower()
        precision = {"fp8": "int8", "fp4": "int4", "nf3": "int3"}.get(precision, precision)

        orig_bytes = _dense_bytes(module)
        original_size += orig_bytes
        mdev = next(module.parameters()).device  # keep each layer on its own device

        new_layer = create_quantized_linear(
            module, precision=precision, use_kernel=use_kernel, quant_4bit=quant_4bit,
            double_quant=double_quant,
        ).to(mdev)
        quantized_size += _quantized_bytes(new_layer, orig_bytes)

        setattr(parent, attr, new_layer)
        del module
        stats[precision] = stats.get(precision, 0) + 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        mode = "uniform" if not allocation else "mixed-precision"
        label_4bit = quant_4bit.upper()
        print(f"Quantization complete ({mode}):")
        print(f"  FP16: {stats['fp16']}  INT8: {stats['int8']}  "
              f"{label_4bit}: {stats['int4']}  NF3: {stats['int3']} layers")
        print(f"  {original_size / 1024**2:.1f} MB -> {quantized_size / 1024**2:.1f} MB "
              f"({ratio:.2f}x smaller)")

    return model


def get_model_info(model: nn.Module) -> dict:
    """Counts of dense vs quantized linear layers in the model."""
    from .linear import QuantizedLinear3bit
    from .triton_kernels import W8A16Linear, W4A16Linear

    info = {
        "total_modules": 0,
        "linear_count": 0,
        "quantized_linear_count": 0,
        "quantized_linear_4bit_count": 0,
        "quantized_linear_3bit_count": 0,
        "w8a16_kernel_count": 0,
        "w4a16_kernel_count": 0,
        "other_count": 0,
        "linear_names": [],
        "quantized_names": [],
    }
    for name, module in model.named_modules():
        info["total_modules"] += 1
        if isinstance(module, QuantizedLinear3bit):
            info["quantized_linear_3bit_count"] += 1
            info["quantized_names"].append(name)
        elif isinstance(module, QuantizedLinear4bit):
            info["quantized_linear_4bit_count"] += 1
            info["quantized_names"].append(name)
        elif isinstance(module, W8A16Linear):
            info["w8a16_kernel_count"] += 1
            info["quantized_names"].append(name)
        elif isinstance(module, W4A16Linear):
            info["w4a16_kernel_count"] += 1
            info["quantized_names"].append(name)
        elif isinstance(module, QuantizedLinear):
            info["quantized_linear_count"] += 1
            info["quantized_names"].append(name)
        elif isinstance(module, nn.Linear):
            info["linear_count"] += 1
            info["linear_names"].append(name)
        else:
            info["other_count"] += 1
    return info


def find_decoder_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Locate a model's decoder layer ModuleList across common architectures."""
    patterns = [
        "model.layers",          # Llama, Mistral, Gemma
        "model.decoder.layers",  # OPT
        "transformer.h",         # GPT-2, GPT-J
        "gpt_neox.layers",       # GPT-NeoX, Pythia
    ]
    for pattern in patterns:
        try:
            obj = model
            for attr in pattern.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (nn.ModuleList, list)):
                return obj
        except AttributeError:
            continue
    return None


def get_model_dtype(model: nn.Module) -> torch.dtype:
    for param in model.parameters():
        return param.dtype
    return torch.float32

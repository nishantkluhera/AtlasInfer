"""
AtlasInfer CPU Offload - move decoder blocks to the GPU only while they compute.

For models that don't fit in VRAM, each transformer block lives on the CPU and
is streamed to the GPU just-in-time for its forward pass, then evicted. Because
quantized weights are stored as registered buffers (see ``linear.py``), a plain
``module.to(device)`` moves everything - dense params, biases, and packed
quantized tensors alike - so the hook stays simple and correct.
"""
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple
from accelerate.hooks import ModelHook, add_hook_to_module

from .linear import QuantizedLinear, QuantizedLinear4bit


def _move(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors inside tuples/lists/dicts to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move(o, device) for o in obj)
    if isinstance(obj, list):
        return [_move(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj  # leave non-tensor objects (e.g. cache wrappers) untouched


class CPUOffloadHook(ModelHook):
    """Streams a module to the GPU for its forward pass, then back to the CPU.

    Crucially it also aligns the *activations*: inputs are moved to the GPU on
    entry and outputs back to the CPU on exit, so CPU-resident modules (the
    embeddings, final norm, etc.) always see CPU tensors while the streamed block
    runs on the GPU. Without this the block's GPU weights meet CPU inputs and the
    forward pass fails with a device-mismatch error.
    """

    def __init__(self, target_device: torch.device):
        self.target_device = target_device
        self.cpu_device = torch.device("cpu")

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Any, ...]:
        module.to(self.target_device)
        return _move(args, self.target_device), _move(kwargs, self.target_device)

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        module.to(self.cpu_device)
        return _move(output, self.cpu_device)


def setup_cpu_offload(
    model: nn.Module,
    target_device: torch.device,
    layer_patterns: Optional[list] = None,
) -> nn.Module:
    """Attach offload hooks to a model's decoder layers.

    Args:
        model: The model to set up offloading for.
        target_device: GPU device used for computation.
        layer_patterns: Attribute paths to the decoder ModuleList. If None,
            common architectures (Llama/Mistral, OPT, GPT-2, GPT-NeoX) are tried.

    Returns:
        The model with hooks attached.
    """
    if layer_patterns is None:
        layer_patterns = [
            "model.layers",          # Llama, Mistral, Gemma
            "model.decoder.layers",  # OPT
            "transformer.h",         # GPT-2, GPT-J
            "gpt_neox.layers",       # GPT-NeoX, Pythia
        ]

    layers = None
    for pattern in layer_patterns:
        try:
            obj = model
            for attr in pattern.split("."):
                obj = getattr(obj, attr)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError(
            f"Could not find decoder layers. Tried patterns: {layer_patterns}. "
            f"Please specify layer_patterns explicitly."
        )

    hook = CPUOffloadHook(target_device)
    for layer in layers:
        add_hook_to_module(layer, hook)

    if hasattr(model, "lm_head"):
        add_hook_to_module(model.lm_head, hook)

    return model


def estimate_model_memory(model: nn.Module) -> dict:
    """Estimate the resident weight footprint of (quantized) linear layers.

    Returns:
        Dict with ``total_bytes``, ``total_gb``, ``quantized_bytes`` and
        ``dense_bytes``.
    """
    from .triton_kernels import W8A16Linear, W4A16Linear

    quantized_bytes = 0
    dense_bytes = 0

    for _, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedLinear4bit)):
            quantized_bytes += module.quantized_weights.memory_bytes()
            if module.bias is not None:
                quantized_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, (W8A16Linear, W4A16Linear)):
            quantized_bytes += module.memory_bytes()
        elif isinstance(module, nn.Linear):
            dense_bytes += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                dense_bytes += module.bias.numel() * module.bias.element_size()

    total = quantized_bytes + dense_bytes
    return {
        "total_bytes": total,
        "total_gb": total / (1024 ** 3),
        "quantized_bytes": quantized_bytes,
        "dense_bytes": dense_bytes,
    }

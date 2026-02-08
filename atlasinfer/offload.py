"""
AtlasInfer CPU Offload - Efficient weight offloading hooks using accelerate
"""
import torch
import torch.nn as nn
from typing import Any, Tuple, Optional
from accelerate.hooks import ModelHook, add_hook_to_module

from .linear import QuantizedLinear


class CPUOffloadHook(ModelHook):
    """
    Hook that manages CPU ↔ GPU transfers for model layers during forward pass.
    
    This hook:
    - Moves layer weights to GPU before forward pass
    - Moves them back to CPU after forward pass
    - Works with both regular nn.Linear and QuantizedLinear layers
    """
    
    def __init__(self, target_device: torch.device):
        self.target_device = target_device
        self.cpu_device = torch.device('cpu')
    
    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Any, ...]:
        """Move module to GPU before forward."""
        self._move_module_to_device(module, self.target_device)
        return args, kwargs
    
    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Move module back to CPU after forward."""
        self._move_module_to_device(module, self.cpu_device)
        return output
    
    def _move_module_to_device(self, module: nn.Module, device: torch.device) -> None:
        """Move module's tensors to specified device."""
        if isinstance(module, QuantizedLinear):
            # Move quantized weights
            if hasattr(module, 'quantized_weights') and module.quantized_weights is not None:
                module.quantized_weights = module.quantized_weights.to(device)
            # Move bias
            if module.bias is not None:
                module.bias = module.bias.to(device)
        else:
            # Regular module - move all parameters and buffers
            for param in module.parameters(recurse=False):
                param.data = param.data.to(device)
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is not None:
                    setattr(module, buf_name, buf.to(device))


def setup_cpu_offload(
    model: nn.Module,
    target_device: torch.device,
    layer_patterns: Optional[list] = None
) -> nn.Module:
    """
    Set up CPU offloading for a model's decoder layers.
    
    Args:
        model: The model to set up offloading for
        target_device: GPU device to use for computation
        layer_patterns: List of attribute paths to decoder layers 
                       (e.g., ['model.layers', 'model.decoder.layers'])
                       If None, tries common patterns
                       
    Returns:
        The model with hooks attached
    """
    if layer_patterns is None:
        layer_patterns = [
            'model.layers',           # Llama, Mistral, Gemma
            'model.decoder.layers',   # OPT
            'transformer.h',          # GPT-2, GPT-J
            'gpt_neox.layers',        # GPT-NeoX
        ]
    
    # Find the decoder layers
    layers = None
    for pattern in layer_patterns:
        try:
            obj = model
            for attr in pattern.split('.'):
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
    
    # Add hooks to each decoder layer
    hook = CPUOffloadHook(target_device)
    for layer in layers:
        add_hook_to_module(layer, hook)
    
    # Also hook the lm_head if present
    if hasattr(model, 'lm_head'):
        add_hook_to_module(model.lm_head, hook)
    
    return model


def estimate_model_memory(model: nn.Module) -> dict:
    """
    Estimate memory requirements for a model.
    
    Returns:
        Dictionary with memory estimates in bytes
    """
    total_params = 0
    quantized_params = 0
    regular_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_params += module.quantized_weights.memory_bytes()
            if module.bias is not None:
                quantized_params += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, nn.Linear):
            size = module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                size += module.bias.numel() * module.bias.element_size()
            regular_params += size
    
    total_params = quantized_params + regular_params
    
    return {
        'total_bytes': total_params,
        'total_gb': total_params / (1024**3),
        'quantized_bytes': quantized_params,
        'regular_bytes': regular_params,
    }

# atlasinfer_core/utils/helpers.py
import torch
import torch.nn as nn # Need nn for Module type hint
import gc
from typing import TYPE_CHECKING

# Avoid circular import during type checking
if TYPE_CHECKING:
    from ..quant.adaptive_quantizer import QuantizedTensor
    # Import QuantizedLinear if needed for checks later
    # from ...integration.atlas_wrappers import QuantizedLinear

def estimate_quantized_model_vram(model: torch.nn.Module) -> int:
    """
    Estimates the VRAM required for model parameters if all QuantizedTensors
    and other parameters (embeddings, norms) were loaded onto the GPU.
    Does NOT include activations or KV cache.
    """
    total_bytes = 0
    seen_ids = set() # Avoid double counting shared parameters

    # Iterate through modules to find QuantizedLinear layers specifically
    # This is more reliable than checking parameters() directly for QuantizedTensor
    for module in model.modules():
        # Check if it's our custom layer holding the quantized weights
        # Need to import QuantizedLinear carefully or use getattr check
        if type(module).__name__ == 'QuantizedLinear': # Use string check to avoid import dep
             q_tensor = getattr(module, 'quantized_weights', None)
             if q_tensor is not None and hasattr(q_tensor, 'memory_footprint'):
                  if id(q_tensor) not in seen_ids:
                       total_bytes += q_tensor.memory_footprint(include_overhead=False)
                       seen_ids.add(id(q_tensor))
             # Add bias size if it exists and wasn't counted
             bias = getattr(module, 'bias', None)
             if bias is not None and id(bias) not in seen_ids:
                  total_bytes += bias.element_size() * bias.nelement()
                  seen_ids.add(id(bias))

        else:
             # For other layers (Embeddings, LayerNorms, etc.) add their parameters
             for param in module.parameters(recurse=False): # recurse=False avoids double counting
                  if id(param) not in seen_ids:
                       total_bytes += param.element_size() * param.nelement()
                       seen_ids.add(id(param))

    return total_bytes


def cleanup_memory():
    """Force garbage collection and attempt to clear CUDA cache."""
    # print("Running memory cleanup...") # Optional: Keep if useful
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional: print memory summary after cleanup
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print("Memory cleanup done.") # Optional
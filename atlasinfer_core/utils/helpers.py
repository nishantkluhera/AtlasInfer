# atlasinfer_core/utils/helpers.py
import torch
import torch.nn as nn # Need nn for Module type hint
import gc
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union # Add more types
from collections.abc import Mapping # Use abc for Mapping check

# --- Add map_structure function here ---
def map_structure(func: Callable, obj: Any):
    """
    Applies `func` recursively to tensors, lists, tuples, or dicts found nested in `obj`.
    Based on accelerate.utils.map_structure implementation. Handles Tensors,
    common collections, and NamedTuples.
    """
    if isinstance(obj, (dict, Dict, Mapping)): # Check against abc.Mapping
        # Recursively apply to dictionary values
        return {k: map_structure(func, v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Check if it's likely a NamedTuple (heuristic: has _fields attribute)
        if hasattr(obj, '_fields') and isinstance(getattr(obj, '_fields'), tuple):
             # Apply func to elements and recreate NamedTuple if possible
            original_type = type(obj)
            try: # Attempt NamedTuple recreation
                 mapped_elements = [map_structure(func, o) for o in obj]
                 return original_type(*mapped_elements)
            except TypeError: # Fallback if recreation fails
                 # print(f"Warning: Failed to recreate NamedTuple {original_type}, returning list.")
                 return [map_structure(func, o) for o in obj] # Fallback to list
        else: # Standard list or tuple
            # Apply func to elements and return same type of sequence
            return type(obj)(map_structure(func, o) for o in obj)
    elif isinstance(obj, torch.Tensor):
        # Apply the function directly to Tensors
        return func(obj)
    # --- Add specific check for QuantizedTensor ---
    # Need to import it carefully or check by name string
    elif type(obj).__name__ == 'QuantizedTensor':
        # Apply the function to the QuantizedTensor object itself
        # The function 'func' (which is _move_to_device in our hook)
        # needs to know how to handle QuantizedTensor.
        return func(obj)
    # --------------------------------------------
    else:
        # Keep non-mappable/non-tensor types as is
        return obj
# --------------------------------------------

# --- Existing functions ---
if TYPE_CHECKING:
    from ..quant.adaptive_quantizer import QuantizedTensor
    # Import QuantizedLinear if needed for type checks later
    # from ...integration.atlas_wrappers import QuantizedLinear

def estimate_quantized_model_vram(model: torch.nn.Module) -> int:
    """
    Estimates the VRAM required for model parameters if all QuantizedTensors
    and other parameters (embeddings, norms) were loaded onto the GPU.
    Does NOT include activations or KV cache.
    """
    total_bytes = 0
    seen_ids = set() # Avoid double counting shared parameters

    for module in model.modules():
        # Check for QuantizedLinear layers
        if type(module).__name__ == 'QuantizedLinear':
             q_tensor = getattr(module, 'quantized_weights', None)
             # Use QuantizedTensor's memory_footprint method
             if q_tensor is not None and hasattr(q_tensor, 'memory_footprint'):
                  if id(q_tensor) not in seen_ids:
                       total_bytes += q_tensor.memory_footprint(include_overhead=False)
                       seen_ids.add(id(q_tensor))
             bias = getattr(module, 'bias', None)
             if bias is not None and id(bias) not in seen_ids:
                  total_bytes += bias.element_size() * bias.nelement()
                  seen_ids.add(id(bias))
        else:
             # Add parameters from other standard layers
             for param in module.parameters(recurse=False):
                  if id(param) not in seen_ids:
                       total_bytes += param.element_size() * param.nelement()
                       seen_ids.add(id(param))
    return total_bytes


def cleanup_memory():
    """Force garbage collection and attempt to clear CUDA cache."""
    print("Running memory cleanup...") # Optional
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleanup done.") # Optional
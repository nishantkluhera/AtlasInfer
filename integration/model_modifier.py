# integration/model_modifier.py
import torch
import torch.nn as nn
from typing import Dict, Type, List, Optional, Tuple
import importlib
import gc
import traceback # For detailed error printing

# --- AtlasInfer Core Imports ---
# Use relative imports assuming standard structure
from .atlas_wrappers import QuantizedLinear, AtlasAttentionWrapper
from atlasinfer_core.quant.adaptive_quantizer import QuantizedTensor, quantize_tensor
from atlasinfer_core.memory.unified_memory_manager import UnifiedMemoryManager
from atlasinfer_core.utils.helpers import cleanup_memory
from atlasinfer_core.utils.weight_offloader import QuantizedTensorOffloadHook
# Import accelerate hook utility AFTER defining the hook
from accelerate.hooks import add_hook_to_module


# --- Model-Specific Layer Identification ---
# Store paths to layers relative to the main model object
# Format: { 'model_type': {'linear_patterns': [...], 'attention_patterns': [...], 'layer_block_module': Type} }
# Use lists for types/names to handle variations like SDPA
MODEL_LAYER_PATTERNS = {
    "opt": {
        "linear_types": [nn.Linear],
        "attention_type": ["OPTAttention", "OPTSdpaAttention"], # Handles standard and SDPA variant
        "decoder_layer_type": "OPTDecoderLayer",
        "decoder_block_path": "model.decoder.layers", # Path to nn.ModuleList
        "attention_subpath": "self_attn" # Name of attention module within decoder layer
    },
    "llama": {
        "linear_types": [nn.Linear],
        "attention_type": ["LlamaAttention", "LlamaSdpaAttention"],
        "decoder_layer_type": "LlamaDecoderLayer",
        "decoder_block_path": "model.layers",
        "attention_subpath": "self_attn"
    },
    "mistral": {
        "linear_types": [nn.Linear],
        "attention_type": ["MistralAttention", "MistralSdpaAttention"],
        "decoder_layer_type": "MistralDecoderLayer",
        "decoder_block_path": "model.layers",
        "attention_subpath": "self_attn"
    },
    "gemma": {
        "linear_types": [nn.Linear],
        "attention_type": ["GemmaAttention", "GemmaSdpaAttention"],
        "decoder_layer_type": "GemmaDecoderLayer",
        "decoder_block_path": "model.layers",
        "attention_subpath": "self_attn"
    },
    # Add other models here following the same pattern...
    # e.g., "falcon": { ... }
}

def _get_model_patterns(model_type: str) -> Optional[Dict]:
    """ Safely retrieves the patterns for a given model type. """
    patterns = MODEL_LAYER_PATTERNS.get(model_type)
    if patterns is None:
         print(f"Warning: No layer patterns defined for model type '{model_type}'. Modification might fail.")
    return patterns

def _replace_module(parent_module: nn.Module, child_name: str, new_module: nn.Module):
    """Safely replaces a submodule, deleting the old one."""
    old_module = getattr(parent_module, child_name, None)
    setattr(parent_module, child_name, new_module)
    # Try to delete the old module explicitly to help GC
    if old_module is not None:
         # If old module holds parameters, they might need explicit deletion too,
         # but Python's GC should handle it if no other references exist.
         # Consider explicitly deleting parameters if memory issues persist
         # for name, param in old_module.named_parameters(recurse=False):
         #      del param
         del old_module

def find_modules(
    model: nn.Module,
    target_types: List[Type], # Types like nn.Linear
    target_names: List[str] = [], # Class names like ['OPTAttention', 'OPTSdpaAttention']
    exclude_names: List[str] = [], # Names containing these substrings
) -> List[Tuple[str, nn.Module]]:
    """ Finds modules matching target types or names, respecting exclusions. """
    found_modules = []
    if not isinstance(target_names, list): target_names = [target_names] if target_names else [] # Ensure list
    if not isinstance(target_types, list): target_types = [target_types] if target_types else [] # Ensure list
    if not isinstance(exclude_names, list): exclude_names = [exclude_names] if exclude_names else [] # Ensure list

    for name, module in model.named_modules():
        # Check exclusion list first
        is_excluded = False
        for exclusion in exclude_names:
            # Check if exclusion is a direct match or substring based on need
            # Simple substring check for now:
            if exclusion and exclusion in name: # Ensure exclusion is not empty
                 is_excluded = True
                 break
        if is_excluded:
             continue

        # Check if type matches OR class name matches
        type_match = any(isinstance(module, t) for t in target_types if t is not None) # Check t is not None
        name_match = type(module).__name__ in target_names

        if type_match or name_match:
            found_modules.append((name, module))
    return found_modules


def apply_atlas_quantization_to_model(
    model: nn.Module,
    model_type: str,
    quantization_config: Dict,
    excluded_layers: List[str] = ["lm_head", "embed_tokens", "embeddings"], # Common exclusions updated
):
    """
    Applies Atlas adaptive quantization IN PLACE to linear layers of a model.

    Args:
        model: The PyTorch model (should be on CPU and preferably FP16 before calling).
        model_type: String identifying the model architecture (e.g., 'opt', 'llama').
        quantization_config: Dict with 'block_size', 'z_threshold'.
        excluded_layers: List of submodule names (or patterns) to exclude.
    """
    print(f"\n--- Applying Atlas Quantization ({model_type}) ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None:
        print("ERROR: Cannot apply quantization, no patterns found.")
        return # Exit if no patterns found

    # Ensure model is on CPU and correct dtype before proceeding
    try:
        model.to(dtype=torch.float16, device='cpu')
        cleanup_memory()
    except Exception as e:
         print(f"Warning: Failed to move model to CPU/FP16 before quantization: {e}")
         # Proceed cautiously, might fail later if dtype/device is wrong

    # Find target linear layers
    target_linear_types = patterns.get("linear_types", [])
    if not target_linear_types:
         print("Warning: No 'linear_types' defined in patterns. Cannot find layers to quantize.")
         return

    linear_modules_to_quantize = find_modules(
        model,
        target_types=target_linear_types,
        exclude_names=excluded_layers
    )

    if not linear_modules_to_quantize:
         print("Warning: No linear layers found to quantize (check patterns/exclusions).")
         return

    print(f"Found {len(linear_modules_to_quantize)} linear layers to quantize...")

    layers_replaced = 0
    layers_failed = 0
    # Using a list copy might be safer if replacement modifies named_modules iteration
    modules_list_copy = list(linear_modules_to_quantize)

    for name, module in modules_list_copy:
        # Double check it's not already quantized
        # Need to check the *current* module in the *actual model* in case it was replaced
        parent_name_check, child_name_check = name.rsplit('.', 1) if '.' in name else ('', name)
        parent_module_check = model.get_submodule(parent_name_check) if parent_name_check else model
        current_module_in_model = getattr(parent_module_check, child_name_check, None)
        if type(current_module_in_model).__name__ == 'QuantizedLinear':
             # print(f"Skipping already quantized layer: {name}")
             continue
        # Also check the original reference just in case
        if type(module).__name__ == 'QuantizedLinear':
             continue


        print(f"Quantizing layer: {name}...")
        try:
            # Ensure weights/bias are accessible and on CPU/FP16
            if not hasattr(module, 'weight') or module.weight is None:
                 print(f"Warning: Layer {name} (type: {type(module).__name__}) has no weight attribute. Skipping.")
                 layers_failed += 1
                 continue

            fp16_weight = module.weight.data.to(dtype=torch.float16, device='cpu')
            bias_data = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
            bias_tensor = bias_data.to(dtype=torch.float16, device='cpu') if bias_data is not None else None

            # Perform quantization
            quantized_weight_data = quantize_tensor(fp16_weight, **quantization_config)
            # Ensure new bias tensor is created if original existed
            new_bias_tensor = bias_tensor.clone() if bias_tensor is not None else None
            new_layer = QuantizedLinear(quantized_weight_data, new_bias_tensor)

            # Replace module - find parent to perform setattr
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module = model.get_submodule(parent_name) if parent_name else model
            if parent_module is None:
                print(f"ERROR: Could not find parent module for {name}. Skipping.")
                layers_failed += 1
                continue

            _replace_module(parent_module, child_name, new_layer)
            layers_replaced += 1

            # Explicitly delete temporary tensors to potentially save RAM
            del fp16_weight, bias_data, bias_tensor, quantized_weight_data, new_bias_tensor
            # Deleting old 'module' reference from the list doesn't affect the model structure

            # Optional: Clean up memory less frequently if it slows things down too much
            if (layers_replaced + layers_failed) % 20 == 0: # Clean every 20 layers processed
                 cleanup_memory()

        except Exception as e:
            print(f"!! FAILED to quantize layer {name}: {e}")
            traceback.print_exc() # Print stack trace for debugging
            layers_failed += 1
            # Decide whether to continue or raise? Continue for now.

    print(f"--- Quantization complete. Replaced: {layers_replaced}, Failed/Skipped: {layers_failed} ---")
    cleanup_memory() # Final cleanup


def apply_atlas_attention_wrapper(
    model: nn.Module,
    model_type: str,
    memory_manager: UnifiedMemoryManager,
):
    """
    Replaces attention modules with AtlasAttentionWrapper IN PLACE.
    Correctly handles finding the parent module for replacement.
    """
    print(f"\n--- Applying Atlas Attention Wrapper ({model_type}) ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None:
        print("ERROR: Cannot apply attention wrapper, no patterns found.")
        return

    attn_class_names = patterns.get("attention_type") # Expecting a list now
    decoder_block_path = patterns.get("decoder_block_path")
    attn_subpath_in_block = patterns.get("attention_subpath") # e.g., 'self_attn'

    if not isinstance(attn_class_names, list): # Ensure list
         attn_class_names = [attn_class_names] if attn_class_names else []

    if not all([attn_class_names, decoder_block_path, attn_subpath_in_block]):
        print("Warning: Missing required patterns (attention_type list, decoder_block_path, attention_subpath) "
              f"for model type '{model_type}'. Cannot wrap attention.")
        return

    # Find the nn.ModuleList containing the decoder layers
    try:
        decoder_layers_list = model.get_submodule(decoder_block_path)
        if not isinstance(decoder_layers_list, nn.ModuleList):
             print(f"Warning: Path '{decoder_block_path}' did not yield an nn.ModuleList. Cannot wrap attention.")
             return
    except AttributeError:
        print(f"Warning: Could not find decoder layers at path '{decoder_block_path}'. Cannot wrap attention.")
        return

    print(f"Found {len(decoder_layers_list)} decoder layers at '{decoder_block_path}'.")
    layers_wrapped = 0
    layers_failed = 0

    for layer_idx, decoder_layer in enumerate(decoder_layers_list):
        try:
            # === Correctly find the PARENT and NAME of the attention module ===
            parent_module = decoder_layer
            name_chain = attn_subpath_in_block.split('.')
            child_name = name_chain[-1]
            # Traverse the path if it's nested (e.g., 'block.attn')
            if len(name_chain) > 1:
                parent_path = ".".join(name_chain[:-1])
                # Need safe getattr with default None
                parent_module_candidate = decoder_layer.get_submodule(parent_path)
                if parent_module_candidate is None:
                     print(f"Warning: Could not find parent path '{parent_path}' in layer {layer_idx}. Skipping.")
                     layers_failed += 1
                     continue
                parent_module = parent_module_candidate

            # === Get the original module using getattr ===
            original_attn_module = getattr(parent_module, child_name, None)
            # =============================================

            if original_attn_module is None:
                 print(f"Warning: Could not find submodule '{child_name}' in parent '{type(parent_module).__name__}' "
                       f"(derived from path '{attn_subpath_in_block}') within decoder layer {layer_idx}. Skipping wrap.")
                 layers_failed += 1
                 continue # Skip to next layer

            current_attn_type_name = type(original_attn_module).__name__

            # Check if type name is in the list and not already wrapped
            if current_attn_type_name in attn_class_names and \
               not isinstance(original_attn_module, AtlasAttentionWrapper):

                print(f"Wrapping attention ({current_attn_type_name}) for layer {layer_idx} (replacing '{child_name}' on parent {type(parent_module).__name__})...")
                # Create wrapper with the *original* module instance and the PARENT decoder layer
                # Ensure parent_layer is the actual decoder layer block (e.g., OPTDecoderLayer)
                wrapped_attn = AtlasAttentionWrapper(
                    original_attn_module=original_attn_module,
                    parent_layer=decoder_layer, # Pass the whole decoder layer block
                    memory_manager=memory_manager,
                    layer_idx=layer_idx
                )

                # Replace the module on the PARENT
                _replace_module(parent_module, child_name, wrapped_attn)
                layers_wrapped += 1

            elif isinstance(original_attn_module, AtlasAttentionWrapper):
                 # This case should ideally not happen if logic is correct, but good to have
                 print(f"Skipping already wrapped attention for layer {layer_idx}.")
            # Check if expected type but skipped (e.g., not in list now, maybe log difference)
            elif current_attn_type_name not in attn_class_names:
                 # This was hit before, should be less common now with SDPA names included
                 print(f"Warning: Module '{child_name}' in layer {layer_idx} has unexpected type "
                       f"'{current_attn_type_name}', expected one of {attn_class_names}. Skipping wrap.")

        except AttributeError as ae: # Catch errors during submodule access more specifically
             print(f"Warning: AttributeError finding/accessing attention in layer {layer_idx} (Path: {attn_subpath_in_block}): {ae}. Skipping wrap.")
             layers_failed += 1
        except Exception as e:
             print(f"!! FAILED to process attention for layer {layer_idx}: {e}")
             traceback.print_exc()
             layers_failed += 1

    print(f"--- Attention wrapping complete. Wrapped: {layers_wrapped}, Failed/Skipped: {layers_failed} ---")
    cleanup_memory()


def setup_offloading_hooks(model: nn.Module, model_type: str, gpu_device: torch.device):
    """ Attaches the QuantizedTensorOffloadHook to relevant layers for CPU offloading. """
    print("\n--- Setting up CPU Weight Offloading Hooks ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None:
        print("ERROR: Cannot setup hooks, no patterns found.")
        return

    # Hook the decoder layer blocks - the hook will handle internal QuantizedLinear layers
    decoder_layer_type_name = patterns.get("decoder_layer_type")
    decoder_block_path = patterns.get("decoder_block_path") # Path to ModuleList

    if not decoder_layer_type_name or not decoder_block_path:
        print("Warning: Missing 'decoder_layer_type' or 'decoder_block_path' pattern. Cannot add hooks precisely.")
        return

    try:
        decoder_layers_list = model.get_submodule(decoder_block_path)
        if not isinstance(decoder_layers_list, nn.ModuleList):
             print(f"Warning: Path '{decoder_block_path}' did not yield an nn.ModuleList. Cannot add hooks.")
             return
    except AttributeError:
        print(f"Warning: Could not find decoder layers at path '{decoder_block_path}'. Cannot add hooks.")
        return

    hooks_added = 0
    print(f"Found {len(decoder_layers_list)} decoder layer blocks ({decoder_layer_type_name}) to hook.")
    for i, layer_module in enumerate(decoder_layers_list):
         # Attach the hook to the entire decoder layer block.
         # Use a dictionary for execution_device mapping specific to this module instance.
         try:
              # The hook's pre/post_forward will inspect submodules within this block.
              # The hook needs to know the target device for this specific layer module
              # Accelerate hook takes map: {module_instance: device}
              # Ensure the hook instance is unique per layer if needed, though state might be okay
              hook = QuantizedTensorOffloadHook(execution_device={layer_module: gpu_device})
              add_hook_to_module(layer_module, hook)
              hooks_added += 1
         except Exception as e:
              print(f"!! FAILED to add hook to decoder layer block {i}: {e}")
              traceback.print_exc()


    # Also hook the LM head if it was quantized
    lm_head = getattr(model, 'lm_head', None)
    # Check by type name to avoid direct import dependency on QuantizedLinear here
    if lm_head and type(lm_head).__name__ == 'QuantizedLinear':
         try:
              print("Adding hook to lm_head...")
              # Create a new hook instance for the lm_head specifically
              lm_head_hook = QuantizedTensorOffloadHook(execution_device={lm_head: gpu_device})
              add_hook_to_module(lm_head, lm_head_hook)
              hooks_added += 1
         except Exception as e:
              print(f"!! FAILED to add hook to lm_head: {e}")
              traceback.print_exc()

    print(f"--- Offloading hook setup complete. Hooks added to {hooks_added} modules/blocks. ---")
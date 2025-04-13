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
    """
    Safely replaces a submodule, attempting to aggressively clean up the old one
    to potentially prevent recursion errors during .to() or state_dict loading.
    """
    old_module = getattr(parent_module, child_name, None)
    # Set the new module first
    setattr(parent_module, child_name, new_module)

    # If there was an old module, try to clean it up
    if old_module is not None and old_module is not new_module: # Ensure not same object
         # === Attempt to clear children/parameters/buffers of old module ===
         # print(f"DEBUG: Cleaning up old module: {child_name} ({type(old_module).__name__})")
         try:
            # Recursively delete children to break potential cycles
            for name, child in list(old_module.named_children()):
                 # print(f"  Deleting child: {name}")
                 del child # Remove reference first
                 # setattr(old_module, name, None) # Set attribute to None? Maybe too aggressive

            # Clear internal dictionaries (use keys() + pop for safety during iteration)
            if hasattr(old_module, '_modules'):
                 for key in list(old_module._modules.keys()): old_module._modules.pop(key)
            if hasattr(old_module, '_parameters'):
                 for key in list(old_module._parameters.keys()): old_module._parameters.pop(key)
            if hasattr(old_module, '_buffers'):
                 for key in list(old_module._buffers.keys()): old_module._buffers.pop(key)
            if hasattr(old_module, '_backward_hooks'): old_module._backward_hooks.clear()
            if hasattr(old_module, '_forward_hooks'): old_module._forward_hooks.clear()
            if hasattr(old_module, '_forward_pre_hooks'): old_module._forward_pre_hooks.clear()

            # Finally delete the reference to the old module object itself
            del old_module
            gc.collect() # Encourage garbage collection immediately

         except Exception as e:
              print(f"Warning: Exception during old module cleanup for '{child_name}': {e}")
              # Continue even if cleanup fails partially


def find_modules(
    model: nn.Module,
    target_types: List[Type], # Types like nn.Linear
    target_names: List[str] = [], # Class names like ['OPTAttention', 'OPTSdpaAttention']
    exclude_names: List[str] = [], # Names containing these substrings
) -> List[Tuple[str, nn.Module]]:
    """ Finds modules matching target types or names, respecting exclusions. """
    found_modules = []
    # Ensure inputs are lists for consistent processing
    if not isinstance(target_names, list): target_names = [target_names] if target_names else []
    if not isinstance(target_types, list): target_types = [target_types] if target_types else []
    if not isinstance(exclude_names, list): exclude_names = [exclude_names] if exclude_names else []

    for name, module in model.named_modules():
        # Check exclusion list first
        is_excluded = False
        for exclusion in exclude_names:
            if exclusion and exclusion in name: # Ensure exclusion is not empty string
                 is_excluded = True
                 break
        if is_excluded:
             continue

        # Check if type matches OR class name matches
        type_match = any(isinstance(module, t) for t in target_types if t is not None)
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
        return

    # Ensure model is on CPU and correct dtype before proceeding
    try:
        model.to(dtype=torch.float16, device='cpu')
        cleanup_memory()
    except Exception as e:
         print(f"Warning: Failed to move model to CPU/FP16 before quantization: {e}")

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
    # Use a list copy for safety during iteration if needed, though usually fine
    modules_list_copy = list(linear_modules_to_quantize)

    for name, module in modules_list_copy:
        # Verify the module *currently* in the model at this name isn't already quantized
        try:
            parent_name_check, child_name_check = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module_check = model.get_submodule(parent_name_check) if parent_name_check else model
            current_module_in_model = getattr(parent_module_check, child_name_check, None)
            if type(current_module_in_model).__name__ == 'QuantizedLinear':
                 continue
        except AttributeError:
             print(f"Warning: Could not re-verify module {name}. Proceeding cautiously.")


        print(f"Quantizing layer: {name}...")
        try:
            # Ensure weights/bias are accessible
            if not hasattr(module, 'weight') or module.weight is None:
                 print(f"Warning: Layer {name} (type: {type(module).__name__}) has no weight attribute. Skipping.")
                 layers_failed += 1
                 continue

            # Ensure data is on CPU/FP16
            fp16_weight = module.weight.data.to(dtype=torch.float16, device='cpu')
            bias_data = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
            bias_tensor = bias_data.to(dtype=torch.float16, device='cpu') if bias_data is not None else None

            # Perform quantization
            quantized_weight_data = quantize_tensor(fp16_weight, **quantization_config)
            new_bias_tensor = bias_tensor.clone() if bias_tensor is not None else None # Clone bias
            new_layer = QuantizedLinear(quantized_weight_data, new_bias_tensor)

            # Replace module
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module = model.get_submodule(parent_name) if parent_name else model
            if parent_module is None:
                print(f"ERROR: Could not find parent module for {name}. Skipping.")
                layers_failed += 1
                continue

            _replace_module(parent_module, child_name, new_layer)
            layers_replaced += 1

            # Explicitly delete temporary tensors
            del fp16_weight, bias_data, bias_tensor, quantized_weight_data, new_bias_tensor

            # Optional cleanup
            if (layers_replaced + layers_failed) % 20 == 0:
                 cleanup_memory()

        except Exception as e:
            print(f"!! FAILED to quantize layer {name}: {e}")
            traceback.print_exc()
            layers_failed += 1

    print(f"--- Quantization complete. Replaced: {layers_replaced}, Failed/Skipped: {layers_failed} ---")
    cleanup_memory()


def apply_atlas_attention_wrapper(
    model: nn.Module,
    model_type: str,
    memory_manager: UnifiedMemoryManager,
):
    """
    Replaces attention modules with AtlasAttentionWrapper IN PLACE.
    Finds QKV/O projections WITHIN the identified attention module.
    Includes Debugging Prints (limited).
    """
    print(f"\n--- Applying Atlas Attention Wrapper ({model_type}) ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None:
        print("ERROR: Cannot apply attention wrapper, no patterns found.")
        return

    attn_class_names = patterns.get("attention_type", [])
    decoder_block_path = patterns.get("decoder_block_path")
    attn_subpath_in_block = patterns.get("attention_subpath")

    if not isinstance(attn_class_names, list): attn_class_names = [attn_class_names] if attn_class_names else []
    if not all([attn_class_names, decoder_block_path, attn_subpath_in_block]):
        print("Warning: Missing required patterns for attention wrapping. Cannot wrap.")
        return

    try:
        decoder_layers_list = model.get_submodule(decoder_block_path)
        if not isinstance(decoder_layers_list, nn.ModuleList):
             print(f"Warning: Path '{decoder_block_path}' did not yield an nn.ModuleList. Cannot wrap.")
             return
    except AttributeError:
        print(f"Warning: Path '{decoder_block_path}' not found. Cannot wrap.")
        return

    print(f"Found {len(decoder_layers_list)} decoder layers at '{decoder_block_path}'.")
    layers_wrapped = 0
    layers_failed = 0

    for layer_idx, decoder_layer in enumerate(decoder_layers_list):
        print_debug = layer_idx < 2 # Limit debug output
        try:
            # Find parent and name of attention module
            parent_module = decoder_layer
            name_chain = attn_subpath_in_block.split('.')
            child_attn_name = name_chain[-1]
            if len(name_chain) > 1:
                parent_path = ".".join(name_chain[:-1])
                parent_module_candidate = decoder_layer.get_submodule(parent_path)
                if parent_module_candidate is None:
                     print(f"Warning: Parent path '{parent_path}' not found layer {layer_idx}. Skip.")
                     layers_failed += 1; continue
                parent_module = parent_module_candidate

            original_attn_module = getattr(parent_module, child_attn_name, None)
            if original_attn_module is None:
                 print(f"Warning: Submodule '{child_attn_name}' not found layer {layer_idx}. Skip.")
                 layers_failed += 1; continue

            current_attn_type_name = type(original_attn_module).__name__

            # Check if wrapping needed
            if current_attn_type_name in attn_class_names and \
               not isinstance(original_attn_module, AtlasAttentionWrapper):

                if print_debug: # Conditional Debug Prints
                    print(f"\n[DEBUG] Layer {layer_idx}: Found Attention Module Path: "
                          f"{decoder_block_path}.{layer_idx}.{attn_subpath_in_block}")
                    print(f"[DEBUG] Layer {layer_idx}: Found Attention Module Type: {current_attn_type_name}")
                    # print(f"[DEBUG] Layer {layer_idx}: Instance: {original_attn_module}") # Might be too verbose
                    print(f"[DEBUG] Layer {layer_idx}: Checking attributes within original_attn_module ({current_attn_type_name})...")

                print(f"Attempting to wrap attention ({current_attn_type_name}) layer {layer_idx}...")

                # Find Q/K/V/O Projections WITHIN the original attention module
                q_proj = getattr(original_attn_module, 'q_proj', None)
                k_proj = getattr(original_attn_module, 'k_proj', None)
                v_proj = getattr(original_attn_module, 'v_proj', None)
                o_proj = getattr(original_attn_module, 'out_proj', None) # Corrected name 'out_proj'

                # Try alternate names only if needed
                if not all([q_proj, k_proj, v_proj, o_proj]):
                     if print_debug: print("[DEBUG] Trying alternate QKVO names (Wq, Wk, Wv, Wo)...")
                     if not q_proj: q_proj = getattr(original_attn_module, 'Wq', None)
                     if not k_proj: k_proj = getattr(original_attn_module, 'Wk', None)
                     if not v_proj: v_proj = getattr(original_attn_module, 'Wv', None)
                     if not o_proj: o_proj = getattr(original_attn_module, 'Wo', None)

                # Check if all projection layers were found
                if not all([q_proj, k_proj, v_proj, o_proj]):
                      print(f"ERROR: Could not find all Q/K/V/O projection layers *within* the attention module "
                            f"'{type(original_attn_module).__name__}' for layer {layer_idx}. Skipping wrap.")
                      if print_debug:
                          print(f"  Found: q={q_proj is not None}, k={k_proj is not None}, "
                                f"v={v_proj is not None}, o={o_proj is not None}")
                      layers_failed += 1
                      continue # Skip this layer

                # Create wrapper, passing projection layers explicitly
                wrapped_attn = AtlasAttentionWrapper(
                    # original_attn_module=original_attn_module, # Can remove if not needed by wrapper init
                    parent_layer=decoder_layer,
                    q_proj=q_proj,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    o_proj=o_proj,
                    memory_manager=memory_manager,
                    layer_idx=layer_idx
                )

                # Replace the original attention module on its parent
                _replace_module(parent_module, child_attn_name, wrapped_attn)
                print(f"Successfully wrapped attention for layer {layer_idx}.")
                layers_wrapped += 1

            elif isinstance(original_attn_module, AtlasAttentionWrapper):
                 print(f"Skipping already wrapped attention layer {layer_idx}.")
            elif current_attn_type_name not in attn_class_names:
                 print(f"Warning: Module '{child_attn_name}' layer {layer_idx} type '{current_attn_type_name}' not in target list {attn_class_names}. Skip.")

        except AttributeError as ae:
             print(f"Warning: AttributeError processing attention layer {layer_idx}: {ae}. Skipping.")
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

    decoder_layer_type_name = patterns.get("decoder_layer_type")
    decoder_block_path = patterns.get("decoder_block_path")

    if not decoder_layer_type_name or not decoder_block_path:
        print("Warning: Missing 'decoder_layer_type' or 'decoder_block_path' pattern. Cannot add hooks precisely.")
        return

    try:
        decoder_layers_list = model.get_submodule(decoder_block_path)
        if not isinstance(decoder_layers_list, nn.ModuleList):
             print(f"Warning: Path '{decoder_block_path}' not nn.ModuleList. Cannot add hooks.")
             return
    except AttributeError:
        print(f"Warning: Path '{decoder_block_path}' not found. Cannot add hooks.")
        return

    hooks_added = 0
    print(f"Found {len(decoder_layers_list)} decoder layer blocks ({decoder_layer_type_name}) to hook.")
    for i, layer_module in enumerate(decoder_layers_list):
         try:
              # Use a dictionary for execution_device mapping specific to this module instance.
              # Ensure unique hook instance per layer block if hook stores state per block (current impl doesn't heavily)
              hook = QuantizedTensorOffloadHook(execution_device={layer_module: gpu_device})
              add_hook_to_module(layer_module, hook)
              hooks_added += 1
         except Exception as e:
              print(f"!! FAILED to add hook to decoder layer block {i}: {e}")
              traceback.print_exc()


    # Also hook the LM head if it was quantized
    lm_head = getattr(model, 'lm_head', None)
    if lm_head and type(lm_head).__name__ == 'QuantizedLinear':
         try:
              print("Adding hook to lm_head...")
              lm_head_hook = QuantizedTensorOffloadHook(execution_device={lm_head: gpu_device})
              add_hook_to_module(lm_head, lm_head_hook)
              hooks_added += 1
         except Exception as e:
              print(f"!! FAILED to add hook to lm_head: {e}")
              traceback.print_exc()

    print(f"--- Offloading hook setup complete. Hooks added to {hooks_added} modules/blocks. ---")
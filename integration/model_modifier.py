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
MODEL_LAYER_PATTERNS = {
    "opt": {
        "linear_types": [nn.Linear],
        "attention_type": ["OPTAttention", "OPTSdpaAttention"],
        "decoder_layer_type": "OPTDecoderLayer",
        "decoder_block_path": "model.decoder.layers",
        "attention_subpath": "self_attn"
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
}

def _get_model_patterns(model_type: str) -> Optional[Dict]:
    """ Safely retrieves the patterns for a given model type. """
    patterns = MODEL_LAYER_PATTERNS.get(model_type)
    if patterns is None:
         print(f"Warning: No layer patterns defined for model type '{model_type}'.")
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
    if old_module is not None and old_module is not new_module:
         try:
            # Recursively delete children
            for _name_child, child in list(old_module.named_children()): del child
            # Clear internal dictionaries
            if hasattr(old_module, '_modules'): old_module._modules.clear()
            if hasattr(old_module, '_parameters'): old_module._parameters.clear()
            if hasattr(old_module, '_buffers'): old_module._buffers.clear()
            if hasattr(old_module, '_backward_hooks'): old_module._backward_hooks.clear()
            if hasattr(old_module, '_forward_hooks'): old_module._forward_hooks.clear()
            if hasattr(old_module, '_forward_pre_hooks'): old_module._forward_pre_hooks.clear()
            if hasattr(old_module, '_state_dict_hooks'): old_module._state_dict_hooks.clear()
            if hasattr(old_module, '_load_state_dict_pre_hooks'): old_module._load_state_dict_pre_hooks.clear()
            if hasattr(old_module, '_state_dict_pre_hooks'): old_module._state_dict_pre_hooks.clear()
            if hasattr(old_module, '_load_state_dict_post_hooks'): old_module._load_state_dict_post_hooks.clear()
            del old_module
            gc.collect()
         except Exception as e: print(f"Warning: Exception during module cleanup '{child_name}': {e}")


def find_modules(
    model: nn.Module, target_types: List[Type], target_names: List[str] = [], exclude_names: List[str] = []
) -> List[Tuple[str, nn.Module]]:
    """ Finds modules matching target types or names, respecting exclusions. """
    found_modules = []
    if not isinstance(target_names, list): target_names = [target_names] if target_names else []
    if not isinstance(target_types, list): target_types = [target_types] if target_types else []
    if not isinstance(exclude_names, list): exclude_names = [exclude_names] if exclude_names else []

    for name, module in model.named_modules():
        is_excluded = any(ex and ex in name for ex in exclude_names)
        if is_excluded: continue
        type_match = any(isinstance(module, t) for t in target_types if t is not None)
        name_match = type(module).__name__ in target_names
        if type_match or name_match: found_modules.append((name, module))
    return found_modules


def apply_atlas_quantization_to_model(
    model: nn.Module, model_type: str, quantization_config: Dict,
    excluded_layers: List[str] = ["lm_head", "embed_tokens", "embeddings"]
):
    """ Applies Atlas adaptive quantization IN PLACE to linear layers of a model. """
    print(f"\n--- Applying Atlas Quantization ({model_type}) ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None: print("ERROR: No patterns."); return

    try: model.to(dtype=torch.float16, device='cpu'); cleanup_memory()
    except Exception as e: print(f"Warning: Failed pre-quant move: {e}")

    target_linear_types = patterns.get("linear_types", [])
    if not target_linear_types: print("Warning: No linear types defined."); return

    linear_modules_to_quantize = find_modules(model, target_types=target_linear_types, exclude_names=excluded_layers)
    if not linear_modules_to_quantize: print("Warning: No linear layers found."); return

    print(f"Found {len(linear_modules_to_quantize)} linear layers to quantize...")
    layers_replaced, layers_failed = 0, 0
    modules_list_copy = list(linear_modules_to_quantize)

    for name, module in modules_list_copy:
        try: # Check if already replaced
            parent_name_check, child_name_check = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module_check = model.get_submodule(parent_name_check) if parent_name_check else model
            current_module_in_model = getattr(parent_module_check, child_name_check, None)
            if type(current_module_in_model).__name__ == 'QuantizedLinear': continue
        except AttributeError: print(f"Warning: Re-verify failed {name}.")

        print(f"Quantizing layer: {name}...")
        try:
            if not hasattr(module, 'weight') or module.weight is None:
                 print(f"Warning: No weight {name}. Skip."); layers_failed += 1; continue

            fp16_weight = module.weight.data.to(dtype=torch.float16, device='cpu')
            bias_data = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
            bias_tensor = bias_data.to(dtype=torch.float16, device='cpu') if bias_data is not None else None

            quantized_weight_data = quantize_tensor(fp16_weight, **quantization_config)
            new_bias_tensor = bias_tensor.clone() if bias_tensor is not None else None
            new_layer = QuantizedLinear(quantized_weight_data, new_bias_tensor)

            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module = model.get_submodule(parent_name) if parent_name else model
            if parent_module is None: print(f"ERROR: Parent not found {name}. Skip."); layers_failed += 1; continue

            _replace_module(parent_module, child_name, new_layer)
            layers_replaced += 1
            del fp16_weight, bias_data, bias_tensor, quantized_weight_data, new_bias_tensor
            if (layers_replaced + layers_failed) % 20 == 0: cleanup_memory()

        except Exception as e: print(f"!! FAILED quant layer {name}: {e}"); traceback.print_exc(); layers_failed += 1

    print(f"--- Quantization complete. Replaced: {layers_replaced}, Failed/Skipped: {layers_failed} ---")
    cleanup_memory()


def apply_atlas_attention_wrapper(
    model: nn.Module, model_type: str, memory_manager: UnifiedMemoryManager
):
    """ Replaces attention modules with AtlasAttentionWrapper IN PLACE. """
    print(f"\n--- Applying Atlas Attention Wrapper ({model_type}) ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None: print("ERROR: No patterns."); return

    attn_class_names = patterns.get("attention_type", [])
    decoder_block_path = patterns.get("decoder_block_path")
    attn_subpath_in_block = patterns.get("attention_subpath")

    if not isinstance(attn_class_names, list): attn_class_names = [attn_class_names] if attn_class_names else []
    if not all([attn_class_names, decoder_block_path, attn_subpath_in_block]): print("Warning: Missing patterns."); return

    try:
        decoder_layers_list = model.get_submodule(decoder_block_path)
        if not isinstance(decoder_layers_list, nn.ModuleList): print(f"Warning: Not ModuleList."); return
    except AttributeError: print(f"Warning: Path not found."); return

    print(f"Found {len(decoder_layers_list)} decoder layers at '{decoder_block_path}'.")
    layers_wrapped, layers_failed = 0, 0

    for layer_idx, decoder_layer in enumerate(decoder_layers_list):
        print_debug = layer_idx < 2
        try:
            parent_module = decoder_layer; name_chain = attn_subpath_in_block.split('.'); child_attn_name = name_chain[-1]
            if len(name_chain) > 1: parent_module = decoder_layer.get_submodule(".".join(name_chain[:-1]))

            original_attn_module = getattr(parent_module, child_attn_name, None)
            if original_attn_module is None: print(f"Warning: Submodule not found layer {layer_idx}. Skip."); layers_failed += 1; continue

            current_attn_type_name = type(original_attn_module).__name__

            if current_attn_type_name in attn_class_names and not isinstance(original_attn_module, AtlasAttentionWrapper):
                if print_debug:
                    print(f"\n[DEBUG] Layer {layer_idx}: Found Path: {decoder_block_path}.{layer_idx}.{attn_subpath_in_block}")
                    print(f"[DEBUG] Layer {layer_idx}: Found Type: {current_attn_type_name}")
                    print(f"[DEBUG] Layer {layer_idx}: Checking attributes within original module...")

                print(f"Attempting to wrap attention ({current_attn_type_name}) layer {layer_idx}...")
                q_proj = getattr(original_attn_module, 'q_proj', None)
                k_proj = getattr(original_attn_module, 'k_proj', None)
                v_proj = getattr(original_attn_module, 'v_proj', None)
                o_proj = getattr(original_attn_module, 'out_proj', None) # Corrected name 'out_proj'

                if not all([q_proj, k_proj, v_proj, o_proj]):
                    if print_debug: print("[DEBUG] Trying alternate QKVO names...")
                    if not q_proj: q_proj = getattr(original_attn_module, 'Wq', None)
                    if not k_proj: k_proj = getattr(original_attn_module, 'Wk', None)
                    if not v_proj: v_proj = getattr(original_attn_module, 'Wv', None)
                    if not o_proj: o_proj = getattr(original_attn_module, 'Wo', None)

                if not all([q_proj, k_proj, v_proj, o_proj]):
                      print(f"ERROR: Projs not found layer {layer_idx}. Skip."); layers_failed += 1; continue

                wrapped_attn = AtlasAttentionWrapper( q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
                                                     parent_layer=decoder_layer, memory_manager=memory_manager,
                                                     layer_idx=layer_idx, model_type=model_type)
                _replace_module(parent_module, child_attn_name, wrapped_attn)
                print(f"Successfully wrapped layer {layer_idx}.")
                layers_wrapped += 1

            elif isinstance(original_attn_module, AtlasAttentionWrapper): print(f"Skipping wrapped layer {layer_idx}.")
            elif current_attn_type_name not in attn_class_names: print(f"Warning: Type '{current_attn_type_name}' layer {layer_idx} not in target list. Skip.")

        except AttributeError as ae: print(f"Warning: AttrErr layer {layer_idx}: {ae}. Skip."); layers_failed += 1
        except Exception as e: print(f"!! FAILED layer {layer_idx}: {e}"); traceback.print_exc(); layers_failed += 1

    print(f"--- Attention wrapping complete. Wrapped: {layers_wrapped}, Failed/Skipped: {layers_failed} ---")
    cleanup_memory()


def setup_offloading_hooks(model: nn.Module, model_type: str, gpu_device: torch.device):
    """ Attaches the QuantizedTensorOffloadHook to relevant layers for CPU offloading. """
    print("\n--- Setting up CPU Weight Offloading Hooks ---")
    patterns = _get_model_patterns(model_type)
    if patterns is None: print("ERROR: No patterns."); return

    decoder_layer_type_name = patterns.get("decoder_layer_type")
    decoder_block_path = patterns.get("decoder_block_path")
    if not decoder_layer_type_name or not decoder_block_path: print("Warning: Missing patterns."); return

    try:
        decoder_layers_list = model.get_submodule(decoder_block_path)
        if not isinstance(decoder_layers_list, nn.ModuleList): print(f"Warning: Not ModuleList."); return
    except AttributeError: print(f"Warning: Path not found."); return

    hooks_added = 0
    print(f"Found {len(decoder_layers_list)} decoder layer blocks ({decoder_layer_type_name}) to hook.")
    for i, layer_module in enumerate(decoder_layers_list):
         try:
              hook = QuantizedTensorOffloadHook(execution_device={layer_module: gpu_device})
              add_hook_to_module(layer_module, hook)
              hooks_added += 1
         except Exception as e: print(f"!! FAILED hook block {i}: {e}"); traceback.print_exc()

    lm_head = getattr(model, 'lm_head', None)
    if lm_head and type(lm_head).__name__ == 'QuantizedLinear':
         try:
              print("Adding hook to lm_head...")
              lm_head_hook = QuantizedTensorOffloadHook(execution_device={lm_head: gpu_device})
              add_hook_to_module(lm_head, lm_head_hook)
              hooks_added += 1
         except Exception as e: print(f"!! FAILED hook lm_head: {e}"); traceback.print_exc()

    print(f"--- Offloading hook setup complete. Hooks added to {hooks_added} modules/blocks. ---")
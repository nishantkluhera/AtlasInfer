# atlasinfer_core/utils/weight_offloader.py
import torch
import torch.nn as nn # Import nn for type check
import os # For PID
# === Use local map_structure ===
from .helpers import map_structure # Import from local helpers.py
# ===============================
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from typing import Any, Dict, Optional, Union, Tuple, TYPE_CHECKING # Import necessary types
import threading # For cache lock
import gc
import weakref # To avoid strong references in cache potentially
import traceback # For debugging prints

# Use TYPE_CHECKING guard for QuantizedTensor hint if defined elsewhere
if TYPE_CHECKING:
    from ..quant.adaptive_quantizer import QuantizedTensor

# === Import ONLY QuantizedTensor from quantizer ===
from ..quant.adaptive_quantizer import QuantizedTensor
# ================================================


# === RESTORE Cache dict, lock, and clear functions LOCALLY ===
_dequantized_weights_cache: Dict[int, weakref.ReferenceType[torch.Tensor]] = {}
_cache_lock = threading.Lock()

def _clear_specific_dequant_cache(obj_id):
     """Clears a specific entry from the dequant cache."""
     global _dequantized_weights_cache
     with _cache_lock:
          if obj_id in _dequantized_weights_cache:
               # print(f"DEBUG: Clearing dequant cache for ID {obj_id}") # Verbose Debug
               del _dequantized_weights_cache[obj_id]

def clear_all_dequant_cache():
    """Clears the entire dequantization cache."""
    global _dequantized_weights_cache
    with _cache_lock:
        # print("DEBUG: Clearing ALL dequant cache.") # Verbose Debug
        _dequantized_weights_cache = {}
    if torch.cuda.is_available():
         gc.collect()
         torch.cuda.empty_cache()
# =========================================================

# --- Custom Hook ---
class QuantizedTensorOffloadHook(AlignDevicesHook):
    """
    Accelerate hook extending AlignDevicesHook. Overrides _move_to_device to handle
    QuantizedTensor/QuantizedLinear. Relies on module.to() in pre/post forward
    for moving hooked block and children. Manages dequantization cache clearing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooked_module_original_devices: dict[int, torch.device] = {}
        print(f"QuantizedTensorOffloadHook Initialized (Offload Mode: {self.offload}, ExecDev: {self.execution_device}).")

    def __deepcopy__(self, memo=None): return self

    def _move_to_device(self, obj: Any, device: torch.device):
        """ Override internal move method for custom types. """
        if obj is None: return None
        # Use LOCALLY defined clear function
        if isinstance(obj, QuantizedTensor):
            if obj.fp8_data.device != device:
                 _clear_specific_dequant_cache(id(obj))
                 return obj.to(device)
            else: return obj
        elif type(obj).__name__ == 'QuantizedLinear':
            q_tensor = getattr(obj, 'quantized_weights', None)
            bias = getattr(obj, 'bias', None)
            if q_tensor is not None and isinstance(q_tensor, QuantizedTensor):
                 if q_tensor.fp8_data.device != device:
                     _clear_specific_dequant_cache(id(q_tensor))
                     setattr(obj, 'quantized_weights', q_tensor.to(device))
            if bias is not None and bias.device != device:
                 setattr(obj, 'bias', bias.to(device))
            # Check if module needs move (needed if QuantizedLinear has other params/buffers)
            if hasattr(obj, 'to') and getattr(obj, 'device', device) != device:
                return obj.to(device)
            else:
                return obj # Return potentially modified module
        elif isinstance(obj, (torch.nn.Module, torch.Tensor)):
             if hasattr(obj, 'to') and getattr(obj, 'device', device) != device:
                 return obj.to(device)
             else: return obj
        else: return obj


    def pre_forward(self, module, *args, **kwargs):
        """ Moves hooked module and inputs (args/kwargs) to target device. """
        parent_module_id = id(module); pid_str=f"[PID:{os.getpid()}]"
        # print(f"\n{pid_str} >>> Hook Pre START {type(module).__name__} (ID: {parent_module_id})") # Debug

        # 1. Determine Target Device
        if isinstance(self.execution_device, dict): target_device = self.execution_device.get(module, self.execution_device.get(None))
        else: target_device = self.execution_device
        if target_device is None: return args, kwargs

        # 2. Store Original Device & Determine Current Device
        current_device = None
        if parent_module_id not in self.hooked_module_original_devices:
             try:
                  first_param = next(iter(module.parameters()), None); first_buffer = next(iter(module.buffers()), None)
                  if first_param is not None: param_device = first_param.device
                  elif first_buffer is not None: param_device = first_buffer.device
                  else: param_device = torch.device("cpu")
                  self.hooked_module_original_devices[parent_module_id] = param_device
                  current_device = param_device
             except Exception: self.hooked_module_original_devices[parent_module_id] = torch.device("cpu"); current_device = torch.device("cpu")
        else:
             try: current_device = next(iter(module.parameters()), torch.tensor(0)).device
             except StopIteration: current_device = torch.device("cpu")

        original_device = self.hooked_module_original_devices[parent_module_id]

        # 3. Move Parent Module if needed
        if current_device != target_device:
            # print(f"{pid_str}     Hook Pre: Moving parent module {type(module).__name__} ({current_device} -> {target_device})") # Debug
            try:
                # Calling module.to() SHOULD implicitly use our _move_to_device override
                # for children like QuantizedLinear / QuantizedTensor
                module.to(target_device)
            except NotImplementedError as nie:
                  print(f"!! Hook Pre: CAUGHT NotImplementedError moving {type(module).__name__}: {nie}")
                  raise nie
            except Exception as e:
                 print(f"{pid_str}   !! Hook Pre: Parent move ERROR: {e}"); traceback.print_exc(); raise e

        # 4. Move args/kwargs Tensors using map_structure
        # print(f"{pid_str}     Hook Pre: Moving args/kwargs START...") # Debug
        try:
             move_func = lambda t: self._move_to_device(t, target_device)
             moved_args = map_structure(move_func, args)
             moved_kwargs = map_structure(move_func, kwargs)
             # print(f"{pid_str}     Hook Pre: Moving args/kwargs END.") # Debug
        except Exception as e: print(f"ERROR map_structure pre_fwd: {e}"); traceback.print_exc(); return args, kwargs

        # print(f"{pid_str} <<< Hook Pre END {type(module).__name__}.") # Debug
        return moved_args, moved_kwargs # Return potentially modified args/kwargs


    def post_forward(self, module, output, *args, **kwargs):
        """ Moves hooked module back if offload=True and clears cache. """
        parent_module_id = id(module); pid_str=f"[PID:{os.getpid()}]"
        # print(f"\n{pid_str} >>> Hook Post START {type(module).__name__} (ID: {parent_module_id})") # Debug

        # 1. Move module back to original device if offload=True
        moved_back = False
        if self.offload and parent_module_id in self.hooked_module_original_devices:
            original_device = self.hooked_module_original_devices[parent_module_id]
            try: current_device = next(iter(module.parameters()), torch.tensor(0, device=original_device)).device
            except StopIteration: current_device = original_device

            if current_device != original_device:
                # print(f"{pid_str}     Hook Post: Moving module {type(module).__name__} back to {original_device}") # Debug
                try:
                     # module.to() should use our _move_to_device override for children
                     module.to(original_device)
                     moved_back = True
                except Exception as e: print(f"{pid_str}   !! Hook Post: Parent move back ERROR: {e}")

        # 2. Clear dequant cache AFTER potential move back
        caches_cleared = 0
        # print(f"{pid_str}     Hook Post: Clearing dequant caches...") # Debug
        for submodule_name, submodule in module.named_modules():
            if type(submodule).__name__ == 'QuantizedLinear':
                q_tensor = getattr(submodule, 'quantized_weights', None)
                if q_tensor is not None and isinstance(q_tensor, QuantizedTensor):
                    # === Use the LOCALLY defined clear function ===
                    _clear_specific_dequant_cache(id(q_tensor))
                    # ============================================
                    caches_cleared += 1

        # print(f"<<< Hook Post End {type(module).__name__}. Moved back: {moved_back}. Cleared {caches_cleared} caches.") # Debug
        return output # Return the original output
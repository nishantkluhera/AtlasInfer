# atlasinfer_core/utils/weight_offloader.py
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
# Use typing.TYPE_CHECKING or direct relative import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..quant.adaptive_quantizer import QuantizedTensor

# Import concrete classes needed at runtime
from ..quant.adaptive_quantizer import QuantizedTensor, dequantize_tensor, clear_dequantized_cache
import gc

# --- Custom Hook for Quantized Tensors ---
class QuantizedTensorOffloadHook(AlignDevicesHook):
    """
    An accelerate Hook to offload/reload QuantizedTensor components within modules.
    Specifically targets modules expected to contain a 'quantized_weights' attribute
    of type QuantizedTensor (like our QuantizedLinear). It iterates through submodules
    of the hooked module (e.g., a DecoderLayer) to move weights/bias before/after execution.
    """
    def __init__(self, execution_device=None):
        """
        Initializes the hook.

        Args:
            execution_device: The target device (e.g., 'cuda:0') or a dictionary mapping
                              modules to devices, as expected by Accelerate's AlignDevicesHook.
        """
        super().__init__(execution_device=execution_device)
        # Store original device of components for each hooked module instance's QuantizedLinear submodules
        # Key: id(QuantizedLinear submodule), Value: torch.device
        # Using submodule ID is more precise if multiple QLinear share a parent hook
        self.original_submodule_devices: dict[int, torch.device] = {}
        # Keep track of which submodules were moved in pre_forward to only move them back
        self.moved_in_pre_forward: set[int] = set()


    def pre_forward(self, module, *args, **kwargs):
        """
        Moves QuantizedTensor components and bias of relevant *submodules*
        to execution_device before the hooked module's forward executes.
        """
        parent_module_id = id(module)
        print(f"\n>>> Hook Pre-Fwd START Layer Block ID: {parent_module_id} ({type(module).__name__})") # DEBUG START
        self.moved_in_pre_forward.clear() # Reset for this forward pass

        # Determine target execution device for the parent module
        if isinstance(self.execution_device, dict):
            target_device = self.execution_device.get(module, self.execution_device.get(None))
        else:
            target_device = self.execution_device

        if target_device is None:
            print(f"Warning: Hook on {type(module).__name__}: No target device found. Skipping pre_forward moves.")
            return args, kwargs # Return original args/kwargs

        # --- Iterate through submodules of the hooked module ---
        moved_count = 0
        submodules_processed = 0
        for submodule_name, submodule in module.named_modules():
            # Check if the submodule is a QuantizedLinear layer
            # Using type name check for robustness against potential import issues
            if type(submodule).__name__ == 'QuantizedLinear':
                submodules_processed += 1
                submodule_id = id(submodule)
                q_tensor = getattr(submodule, 'quantized_weights', None)

                if q_tensor is not None and isinstance(q_tensor, QuantizedTensor):
                    # Store original device if not already stored
                    if submodule_id not in self.original_submodule_devices:
                        self.original_submodule_devices[submodule_id] = q_tensor.fp8_data.device
                        # print(f"    Hook Pre: Storing original device {self.original_submodule_devices[submodule_id]} for {submodule_name} ({submodule_id})") # Verbose Debug

                    # Move QuantizedTensor components if needed
                    if q_tensor.fp8_data.device != target_device:
                        # print(f"    Hook Pre: Moving weights {submodule_name} ({submodule_id}) -> {target_device}") # Debug Move
                        setattr(submodule, 'quantized_weights', q_tensor.to(target_device))
                        self.moved_in_pre_forward.add(submodule_id) # Track that we moved this one
                        moved_count += 1

                    # Move bias if needed
                    bias = getattr(submodule, 'bias', None)
                    if bias is not None and bias.device != target_device:
                        # print(f"    Hook Pre: Moving bias {submodule_name} ({submodule_id}) -> {target_device}") # Debug Move
                        setattr(submodule, 'bias', bias.to(target_device))
                        # If weights weren't moved but bias was, still track submodule? Yes.
                        self.moved_in_pre_forward.add(submodule_id)
                else:
                     print(f"    Warning: Hook Pre: {submodule_name} is QuantizedLinear but has no valid quantized_weights.")

        print(f"<<< Hook Pre-Fwd END Layer Block ID: {parent_module_id}. Processed {submodules_processed} QLinear. Moved {moved_count} weights.") # DEBUG END
        # === Corrected Return Statement ===
        # Return only args and kwargs, as expected by Accelerate's hook mechanism
        return args, kwargs
        # ================================


    def post_forward(self, module, output, *args, **kwargs):
        """
        Moves QuantizedTensor components and bias of relevant *submodules*
        back to their original device after the hooked module's forward executes.
        """
        parent_module_id = id(module)
        print(f"\n>>> Hook Post-Fwd START Layer Block ID: {parent_module_id} ({type(module).__name__})") # DEBUG START

        moved_back_count = 0
        submodules_processed = 0
        # --- Iterate through submodules to move them back ---
        # Need original_devices populated correctly in pre_forward
        for submodule_name, submodule in module.named_modules():
            if type(submodule).__name__ == 'QuantizedLinear':
                submodules_processed += 1
                submodule_id = id(submodule)

                # Only move back if it was moved by this hook instance in pre_forward
                # and if we know its original device
                if submodule_id in self.moved_in_pre_forward and \
                   submodule_id in self.original_submodule_devices:

                    original_device = self.original_submodule_devices[submodule_id]
                    q_tensor = getattr(submodule, 'quantized_weights', None)

                    # Determine current device (should be the target_device from pre_forward)
                    current_device = None
                    if q_tensor is not None and isinstance(q_tensor, QuantizedTensor):
                         current_device = q_tensor.fp8_data.device

                    # Only move if currently on execution device AND original is different
                    if current_device is not None and current_device != original_device:
                         # print(f"    Hook Post: Moving weights {submodule_name} ({submodule_id}) -> {original_device}") # Debug Move Back
                         setattr(submodule, 'quantized_weights', q_tensor.to(original_device))
                         moved_back_count += 1
                         # Clear the dequantization cache associated with this specific QTensor object ID
                         clear_dequantized_cache(id(q_tensor))

                         # Move bias back if needed
                         bias = getattr(submodule, 'bias', None)
                         if bias is not None and bias.device != original_device:
                              # print(f"    Hook Post: Moving bias {submodule_name} ({submodule_id}) -> {original_device}") # Debug Move Back
                              setattr(submodule, 'bias', bias.to(original_device))
                    # Handle case where maybe only bias was moved? Less likely but possible.
                    elif current_device == original_device:
                         bias = getattr(submodule, 'bias', None)
                         if bias is not None and bias.device != original_device:
                              # print(f"    Hook Post: Moving bias ONLY for {submodule_name} ({submodule_id}) -> {original_device}") # Debug
                              setattr(submodule, 'bias', bias.to(original_device))

                # else: # Debug if not moved back
                #     if submodule_id in self.moved_in_pre_forward:
                #         print(f"    Hook Post: {submodule_name} ({submodule_id}) was moved pre, but not moving back (orig dev not found or already there?).")
                #     else:
                #         print(f"    Hook Post: {submodule_name} ({submodule_id}) was not moved pre, skipping.")

        # Clear the tracking set for the next forward pass
        self.moved_in_pre_forward.clear()
        # Optional: Aggressive cleanup if memory is tight
        # gc.collect()
        # if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"<<< Hook Post-Fwd END Layer Block ID: {parent_module_id}. Processed {submodules_processed} QLinear. Moved back {moved_back_count} weights.") # DEBUG END
        # === Return the original output ===
        # Standard accelerate post_forward hook returns the output
        return output
        # ================================
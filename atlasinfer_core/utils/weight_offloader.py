# atlasinfer_core/utils/weight_offloader.py
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
# Import QuantizedTensor carefully to avoid circular deps if possible
# Use typing.TYPE_CHECKING or direct relative import
from ..quant.adaptive_quantizer import QuantizedTensor, dequantize_tensor, clear_dequantized_cache
import gc # For cleanup

# --- Custom Hook for Quantized Tensors ---
class QuantizedTensorOffloadHook(AlignDevicesHook):
    """
    An accelerate Hook to offload/reload QuantizedTensor components within modules.
    Specifically targets modules expected to contain a 'quantized_weights' attribute
    of type QuantizedTensor (like our QuantizedLinear).
    """
    def __init__(self, execution_device=None):
        super().__init__(execution_device=execution_device)
        # Store original device of components for each hooked module instance
        self.original_devices: dict[int, torch.device] = {}

    def pre_forward(self, module):
        """Moves QuantizedTensor components and bias to execution_device before forward."""
        module_id = id(module)
        q_tensor = getattr(module, 'quantized_weights', None)

        # Only proceed if the module has the quantized weights attribute
        if q_tensor is not None and isinstance(q_tensor, QuantizedTensor):
            # Store original device if not already stored (assume CPU initially)
            if module_id not in self.original_devices:
                # Assume all components start on the same device (CPU)
                self.original_devices[module_id] = q_tensor.fp8_data.device

            # Determine target execution device for this module instance
            # Accelerate >= 0.18 stores device per module in a dict
            if isinstance(self.execution_device, dict):
                target_device = self.execution_device.get(module, self.execution_device.get(None))
            else: # Older accelerate versions might just have one device
                target_device = self.execution_device

            if target_device is None:
                print(f"Warning: No execution device found for module {module}. Skipping pre_forward hook.")
                return module # Must return module

            # Move QuantizedTensor components if not already on target device
            if q_tensor.fp8_data.device != target_device:
                # print(f"Hook: Moving weights for {type(module).__name__} to {target_device}") # Debug
                setattr(module, 'quantized_weights', q_tensor.to(target_device))

            # Also move bias if it exists and is not on the target device
            bias = getattr(module, 'bias', None)
            if bias is not None and bias.device != target_device:
                setattr(module, 'bias', bias.to(target_device))

        # Must return module for Accelerate compatibility >= 0.18.0
        return module


    def post_forward(self, module, output):
        """Moves QuantizedTensor components and bias back to original device after forward."""
        module_id = id(module)
        q_tensor = getattr(module, 'quantized_weights', None)

        # Only proceed if we have state for this module and it has quantized weights
        if module_id in self.original_devices and \
           q_tensor is not None and isinstance(q_tensor, QuantizedTensor):

            original_device = self.original_devices[module_id]
            # Determine current execution device (where components *should* be)
            if isinstance(self.execution_device, dict):
                current_device = self.execution_device.get(module, self.execution_device.get(None))
            else:
                current_device = self.execution_device

            if current_device is None:
                 print(f"Warning: No execution device found for module {module}. Skipping post_forward hook.")
                 return output # Must return output

            # Move back only if it was moved (i.e., current != original)
            if q_tensor.fp8_data.device == current_device and current_device != original_device:
                # print(f"Hook: Moving weights for {type(module).__name__} back to {original_device}") # Debug
                setattr(module, 'quantized_weights', q_tensor.to(original_device))
                # Also move bias back
                bias = getattr(module, 'bias', None)
                if bias is not None and bias.device == current_device:
                     setattr(module, 'bias', bias.to(original_device))

                # Crucially, clear the dequantized weight cache for this tensor ID
                # This prevents holding onto a large dequantized tensor on the GPU
                clear_dequantized_cache(id(q_tensor))
                # Optional: More aggressive cleanup if VRAM pressure is extreme
                # gc.collect()
                # if torch.cuda.is_available(): torch.cuda.empty_cache()


        # Must return output for Accelerate compatibility >= 0.18.0
        return output

# Function to conveniently add the hook (no changes needed)
# def add_quantized_tensor_offload_hook(module: torch.nn.Module, execution_device: torch.device):
#     """Attaches the QuantizedTensor offload hook to relevant submodules."""
#     hook = QuantizedTensorOffloadHook(execution_device=execution_device)
#     add_hook_to_module(module, hook)
# This helper function might be better placed in model_modifier.py to avoid import issues
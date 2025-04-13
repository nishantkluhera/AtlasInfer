# atlasinfer_core/utils/weight_offloader.py
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from ..quant.adaptive_quantizer import QuantizedTensor, dequantize_tensor, clear_dequantized_cache
import gc

# --- Custom Hook for Quantized Tensors ---
class QuantizedTensorOffloadHook(AlignDevicesHook):
    """
    An accelerate Hook to offload/reload QuantizedTensor components within modules.
    Specifically targets modules expected to contain a 'quantized_weights' attribute
    of type QuantizedTensor (like our QuantizedLinear).
    """
    def __init__(self, execution_device=None):
        super().__init__(execution_device=execution_device)
        self.original_devices: dict[int, torch.device] = {}

    # === Modified Signature: Added *args, **kwargs ===
    def pre_forward(self, module, *args, **kwargs):
        """Moves QuantizedTensor components and bias to execution_device before forward."""
        module_id = id(module)
        # --- Find QuantizedLinear layers WITHIN the hooked module (e.g., OPTDecoderLayer) ---
        # We need to move weights for submodules just before the main module's forward uses them.
        for submodule_name, submodule in module.named_modules():
            # Check if the submodule itself is QuantizedLinear
            q_tensor = getattr(submodule, 'quantized_weights', None)
            if q_tensor is not None and type(submodule).__name__ == 'QuantizedLinear': # Check type name
                submodule_id = id(submodule) # Use submodule ID for device tracking? Or module_id? Let's stick to module_id for now.

                # Store original device if not already stored (using parent module's ID as key)
                if module_id not in self.original_devices:
                    self.original_devices[module_id] = q_tensor.fp8_data.device

                # Determine target execution device
                if isinstance(self.execution_device, dict):
                    target_device = self.execution_device.get(module, self.execution_device.get(None))
                else:
                    target_device = self.execution_device

                if target_device is None: continue # Skip if no device defined

                # Move QuantizedTensor components if needed
                if q_tensor.fp8_data.device != target_device:
                    setattr(submodule, 'quantized_weights', q_tensor.to(target_device))

                # Move bias if needed
                bias = getattr(submodule, 'bias', None)
                if bias is not None and bias.device != target_device:
                    setattr(submodule, 'bias', bias.to(target_device))

        # === Return the original args and kwargs ===
        # Accelerate >= 0.18.0 expects (module, args, kwargs)
        # Older versions might expect just (args, kwargs) - check accelerate docs if needed
        # Let's return all three for compatibility, assuming newer Accelerate
        return module, args, kwargs
        # ===========================================

    # === Modified Signature: Added *args, **kwargs ===
    def post_forward(self, module, output, *args, **kwargs):
        """Moves QuantizedTensor components and bias back to original device after forward."""
        module_id = id(module)

        if module_id in self.original_devices:
            original_device = self.original_devices[module_id]
            # Find target execution device again
            if isinstance(self.execution_device, dict):
                current_device = self.execution_device.get(module, self.execution_device.get(None))
            else:
                current_device = self.execution_device

            if current_device is None: return output # Skip if no device

            # --- Iterate through submodules again to move them back ---
            for submodule_name, submodule in module.named_modules():
                 q_tensor = getattr(submodule, 'quantized_weights', None)
                 if q_tensor is not None and type(submodule).__name__ == 'QuantizedLinear':
                     # Move back only if it was actually moved
                     if q_tensor.fp8_data.device == current_device and current_device != original_device:
                          setattr(submodule, 'quantized_weights', q_tensor.to(original_device))
                          # Move bias back
                          bias = getattr(submodule, 'bias', None)
                          if bias is not None and bias.device == current_device:
                               setattr(submodule, 'bias', bias.to(original_device))

                          # Clear dequant cache for the specific QuantizedTensor object
                          clear_dequantized_cache(id(q_tensor))

        # === Return the original output ===
        # Accelerate >= 0.18.0 expects just output
        # Older versions might expect (output, args, kwargs)? Check docs. Assume just output.
        return output
        # ================================
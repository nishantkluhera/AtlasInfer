# atlasinfer_core/quant/adaptive_quantizer.py
import torch
import numpy as np
from typing import NamedTuple, Tuple
import threading

# Define a structure to hold the quantized data
class QuantizedTensor(NamedTuple):
    """
    Represents a tensor quantized adaptively into FP8/FP16 hybrid format.
    Designed to be moved between devices.
    """
    fp8_data: torch.Tensor      # int8 tensor containing non-outlier data
    fp16_outliers: torch.Tensor # float16 tensor containing outlier values
    outlier_indices: torch.Tensor # Long tensor with indices of outliers in the original flattened tensor
    scale_factors: torch.Tensor # float16 tensor containing scale factor for each block
    original_shape: torch.Size  # Original shape of the tensor
    block_size: int             # Block size used for quantization

    def to(self, device):
        """Moves the underlying tensors to the specified device."""
        # Avoid moving if already on the target device
        if self.fp8_data.device == device:
             return self
        return QuantizedTensor(
            fp8_data=self.fp8_data.to(device),
            fp16_outliers=self.fp16_outliers.to(device),
            outlier_indices=self.outlier_indices.to(device),
            scale_factors=self.scale_factors.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size
        )

    def memory_footprint(self, include_overhead=True) -> int:
        """Calculates the memory footprint in bytes."""
        mem = self.fp8_data.element_size() * self.fp8_data.nelement()
        mem += self.fp16_outliers.element_size() * self.fp16_outliers.nelement()
        mem += self.outlier_indices.element_size() * self.outlier_indices.nelement()
        mem += self.scale_factors.element_size() * self.scale_factors.nelement()
        if include_overhead:
            mem += 128 # Approximate Python object overhead etc.
        return mem

def calculate_zscore_manual(data: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Calculates Z-scores manually, handling std dev near zero."""
    mean = torch.mean(data)
    std = torch.std(data)
    # Add epsilon to prevent division by zero
    return (data - mean) / (std + eps)

def quantize_tensor(
    fp16_tensor: torch.Tensor,
    block_size: int = 128,
    z_threshold: float = 3.0
) -> QuantizedTensor:
    """
    Quantizes an FP16 tensor into a hybrid FP8/FP16 format.
    Assumes input tensor is on CPU for offline quantization.
    """
    if not isinstance(fp16_tensor, torch.Tensor):
         raise TypeError(f"Input must be a PyTorch Tensor, got {type(fp16_tensor)}")
    if fp16_tensor.dtype != torch.float16:
        # Try converting, issue warning
        print(f"Warning: Input tensor was {fp16_tensor.dtype}, converting to FP16 for quantization.")
        fp16_tensor = fp16_tensor.to(torch.float16)

    input_device = fp16_tensor.device # Should ideally be CPU
    original_shape = fp16_tensor.shape
    flat_tensor = fp16_tensor.flatten()
    num_elements = flat_tensor.numel()

    if num_elements == 0:
        # Handle empty tensor case
        empty_int8 = torch.empty((0,), dtype=torch.int8, device=input_device)
        empty_fp16 = torch.empty((0,), dtype=torch.float16, device=input_device)
        empty_long = torch.empty((0,), dtype=torch.long, device=input_device)
        return QuantizedTensor(
            fp8_data=empty_int8, fp16_outliers=empty_fp16, outlier_indices=empty_long,
            scale_factors=empty_fp16, original_shape=original_shape, block_size=block_size
        )


    # Pad the tensor if its size is not a multiple of block_size
    padded_elements = (num_elements + block_size - 1) // block_size * block_size
    padding_size = padded_elements - num_elements
    if padding_size > 0:
        flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_size, dtype=torch.float16, device=input_device)])

    num_blocks = flat_tensor.numel() // block_size
    # Ensure num_blocks is not zero if flat_tensor was not empty
    if num_blocks == 0 and num_elements > 0: num_blocks = 1
    
    blocked_tensor = flat_tensor.view(num_blocks, block_size)

    all_fp8_data = []
    all_fp16_outliers = []
    all_outlier_indices = []
    all_scale_factors = []

    max_int8 = 127.0 # Max value for signed int8

    for i in range(num_blocks):
        block = blocked_tensor[i]

        # 1. Detect Outliers
        z_scores = calculate_zscore_manual(block) # Manual torch implementation
        is_outlier = torch.abs(z_scores) > z_threshold

        outliers = block[is_outlier].clone() # Clone to ensure data is separate
        non_outliers = block[~is_outlier].clone()

        # Store outlier indices relative to the start of the *original* flat tensor
        block_start_index = i * block_size
        outlier_indices_in_block = torch.where(is_outlier)[0]
        original_outlier_indices = block_start_index + outlier_indices_in_block
        all_fp16_outliers.append(outliers)
        all_outlier_indices.append(original_outlier_indices)

        # 2. Quantize Non-Outliers to FP8 (emulated with INT8)
        if non_outliers.numel() > 0:
            max_val = torch.max(torch.abs(non_outliers))
            # Ensure scale factor is float16 and at least a small value
            scale_factor = (max_val / max_int8).clamp_(min=1e-6).to(torch.float16)
            # Quantize: Scale -> Round -> Clamp -> Cast to int8
            quantized_block = torch.round(non_outliers / scale_factor)
            quantized_block = torch.clamp(quantized_block, min=-max_int8, max=max_int8).to(torch.int8)
        else:
            # Handle blocks with only outliers (or empty blocks due to padding)
            quantized_block = torch.empty((0,), dtype=torch.int8, device=input_device)
            scale_factor = torch.tensor(1.0, dtype=torch.float16, device=input_device) # Default scale

        all_fp8_data.append(quantized_block)
        all_scale_factors.append(scale_factor.unsqueeze(0)) # Store scale factor per block

    # Combine results from all blocks
    combined_fp8_data = torch.cat(all_fp8_data) if len(all_fp8_data) > 0 else torch.empty((0,), dtype=torch.int8, device=input_device)
    combined_fp16_outliers = torch.cat(all_fp16_outliers) if len(all_fp16_outliers) > 0 else torch.empty((0,), dtype=torch.float16, device=input_device)
    combined_outlier_indices = torch.cat(all_outlier_indices) if len(all_outlier_indices) > 0 else torch.empty((0,), dtype=torch.long, device=input_device)
    combined_scale_factors = torch.cat(all_scale_factors)

    return QuantizedTensor(
        fp8_data=combined_fp8_data,
        fp16_outliers=combined_fp16_outliers,
        outlier_indices=combined_outlier_indices,
        scale_factors=combined_scale_factors,
        original_shape=original_shape,
        block_size=block_size
    )

# --- Dequantization Cache ---
# Simple dict to cache dequantized weights on GPU to avoid recomputing if reused quickly
# Key: id(QuantizedTensor object), Value: torch.Tensor (on GPU)
_dequantized_weights_cache = {}
_cache_lock = threading.Lock() # Protect cache access if threading used later

def dequantize_tensor(quant_data: QuantizedTensor, target_device: torch.device) -> torch.Tensor:
    """
    Dequantizes a QuantizedTensor back into an FP16 tensor on the target device.
    Uses a cache to potentially reuse the dequantized tensor memory.
    """
    cache_key = id(quant_data)

    with _cache_lock:
        cached_tensor = _dequantized_weights_cache.get(cache_key)
        if cached_tensor is not None and cached_tensor.device == target_device:
            return cached_tensor
        elif cached_tensor is not None:
            # If cached but on wrong device, move it
            cached_tensor = cached_tensor.to(target_device)
            _dequantized_weights_cache[cache_key] = cached_tensor
            return cached_tensor

    # --- Cache miss or device mismatch, perform dequantization ---
    # Ensure quantized data components are on the target device for reconstruction
    quant_data_gpu = quant_data.to(target_device)
    block_size = quant_data_gpu.block_size

    original_numel = quant_data_gpu.original_shape.numel()
    padded_elements = (original_numel + block_size - 1) // block_size * block_size
    num_blocks = padded_elements // block_size
    
    # Handle case where original tensor was smaller than block size
    if original_numel > 0 and num_blocks == 0: num_blocks = 1 

    reconstructed_flat = torch.zeros(padded_elements, dtype=torch.float16, device=target_device)

    current_fp8_idx = 0
    for i in range(num_blocks):
        scale = quant_data_gpu.scale_factors[i]
        block_start_index = i * block_size
        block_end_index = block_start_index + block_size

        # Identify which outliers belong to this block
        outlier_mask_in_block = (quant_data_gpu.outlier_indices >= block_start_index) & \
                                (quant_data_gpu.outlier_indices < block_end_index)
        indices_of_block_outliers_in_combined = torch.where(outlier_mask_in_block)[0]

        # Indices relative to block start
        outlier_indices_relative = quant_data_gpu.outlier_indices[outlier_mask_in_block] - block_start_index
        # Outlier values for this block
        block_outlier_values = quant_data_gpu.fp16_outliers[indices_of_block_outliers_in_combined]

        num_outliers_in_block = outlier_indices_relative.numel()
        num_non_outliers_in_block = block_size - num_outliers_in_block
        
        # Safety check for num_non_outliers
        if num_non_outliers_in_block < 0 : num_non_outliers_in_block=0

        # Get the FP8 data for this block
        block_fp8_end_idx = current_fp8_idx + num_non_outliers_in_block
        block_fp8_data = quant_data_gpu.fp8_data[current_fp8_idx : block_fp8_end_idx]
        current_fp8_idx = block_fp8_end_idx

        # Dequantize FP8 data
        if num_non_outliers_in_block > 0:
            dequantized_non_outliers = block_fp8_data.to(torch.float16) * scale
        else:
             dequantized_non_outliers = torch.empty((0,), dtype=torch.float16, device=target_device)

        # Reconstruct the block
        block_reconstructed = torch.zeros(block_size, dtype=torch.float16, device=target_device)
        non_outlier_mask = torch.ones(block_size, dtype=torch.bool, device=target_device)

        if num_outliers_in_block > 0:
             # Ensure indices are valid and match value count
            valid_relative_indices = outlier_indices_relative[outlier_indices_relative < block_size]
            num_valid_outliers = valid_relative_indices.numel()
            if num_valid_outliers > 0:
                non_outlier_mask[valid_relative_indices] = False
                block_reconstructed[valid_relative_indices] = block_outlier_values[:num_valid_outliers]

        # Place dequantized non-outliers
        # Ensure we don't try to place more non-outliers than fit
        num_non_outlier_positions = non_outlier_mask.sum().item()
        block_reconstructed[non_outlier_mask] = dequantized_non_outliers[:num_non_outlier_positions]


        # Put the reconstructed block into the flat tensor
        reconstructed_flat[block_start_index:block_end_index] = block_reconstructed

    # Remove padding if any was added during quantization
    reconstructed_flat = reconstructed_flat[:original_numel]

    # Reshape back to original
    final_tensor = reconstructed_flat.view(quant_data_gpu.original_shape)

    # Store in cache
    with _cache_lock:
        _dequantized_weights_cache[cache_key] = final_tensor

    return final_tensor

def clear_dequantized_cache(quant_data_id=None):
    """Clears the cache for a specific item or the entire cache."""
    global _dequantized_weights_cache
    with _cache_lock:
        if quant_data_id is not None and quant_data_id in _dequantized_weights_cache:
            del _dequantized_weights_cache[quant_data_id]
        elif quant_data_id is None:
             _dequantized_weights_cache = {}
    # Consider torch.cuda.empty_cache() if memory pressure is high and GPU is available
    if torch.cuda.is_available():
         torch.cuda.empty_cache()
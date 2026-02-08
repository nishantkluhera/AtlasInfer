"""
AtlasInfer Quantizer - Adaptive FP8/FP16 Hybrid Quantization

Quantizes tensors to emulated FP8 format with outlier preservation in FP16,
achieving significant memory reduction with minimal accuracy loss.
"""
import torch
from typing import NamedTuple, Tuple, Optional
import math


class QuantizedTensor(NamedTuple):
    """
    Represents a tensor quantized to FP8 with outliers preserved in FP16.
    
    Memory layout:
    - fp8_data: INT8 storage (emulates FP8 E4M3 format)
    - scales: Per-block scale factors (FP32)
    - outlier_mask: Boolean mask indicating outlier positions
    - outlier_values: Original FP16 values for outliers
    - original_shape: Original tensor shape for reconstruction
    - block_size: Block size used during quantization
    """
    fp8_data: torch.Tensor        # INT8 storage for FP8-emulated values
    scales: torch.Tensor          # Per-block scales (FP32)
    outlier_mask: torch.Tensor    # Boolean mask for outliers
    outlier_values: torch.Tensor  # FP16 outlier values
    original_shape: torch.Size
    block_size: int
    
    def to(self, device: torch.device) -> 'QuantizedTensor':
        """Move all tensors to the specified device."""
        return QuantizedTensor(
            fp8_data=self.fp8_data.to(device),
            scales=self.scales.to(device),
            outlier_mask=self.outlier_mask.to(device),
            outlier_values=self.outlier_values.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size
        )
    
    def memory_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return (
            self.fp8_data.numel() * self.fp8_data.element_size() +
            self.scales.numel() * self.scales.element_size() +
            self.outlier_mask.numel() * self.outlier_mask.element_size() +
            self.outlier_values.numel() * self.outlier_values.element_size()
        )


# FP8 E4M3 range: [-448, 448], but we use conservative [-240, 240]
FP8_MAX = 240.0
FP8_MIN = -240.0


def quantize_tensor(
    tensor: torch.Tensor,
    block_size: int = 128,
    outlier_threshold: float = 3.0
) -> QuantizedTensor:
    """
    Quantize a FP16/FP32 tensor to FP8 with outlier preservation.
    
    Args:
        tensor: Input tensor (FP16 or FP32), expected on CPU for offline quantization
        block_size: Block size for per-block scaling
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        QuantizedTensor with FP8 data and outlier information
    """
    original_shape = tensor.shape
    original_device = tensor.device
    
    # Handle empty tensors
    if tensor.numel() == 0:
        return QuantizedTensor(
            fp8_data=torch.empty(0, dtype=torch.int8, device=original_device),
            scales=torch.empty(0, dtype=torch.float32, device=original_device),
            outlier_mask=torch.empty(0, dtype=torch.bool, device=original_device),
            outlier_values=torch.empty(0, dtype=torch.float16, device=original_device),
            original_shape=original_shape,
            block_size=block_size
        )
    
    # Flatten for block processing
    flat = tensor.float().flatten()
    num_elements = flat.numel()
    
    # Pad to multiple of block_size
    padded_size = math.ceil(num_elements / block_size) * block_size
    if padded_size > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded_size - num_elements), value=0.0)
    
    # Reshape into blocks
    num_blocks = padded_size // block_size
    blocks = flat.view(num_blocks, block_size)
    
    # Detect outliers using z-score method
    block_means = blocks.mean(dim=1, keepdim=True)
    block_stds = blocks.std(dim=1, keepdim=True)
    # Avoid division by zero
    block_stds = torch.clamp(block_stds, min=1e-6)
    z_scores = torch.abs((blocks - block_means) / block_stds)
    outlier_mask_blocks = z_scores > outlier_threshold
    
    # Create working copy with outliers zeroed for scale computation
    blocks_clean = blocks.clone()
    blocks_clean[outlier_mask_blocks] = 0.0
    
    # Compute per-block scales (max absolute value, avoiding outliers)
    block_max = blocks_clean.abs().max(dim=1, keepdim=True).values
    block_max = torch.clamp(block_max, min=1e-6)  # Avoid division by zero
    scales = block_max / FP8_MAX
    
    # Quantize: scale down and round to INT8 range
    scaled_blocks = blocks / scales
    # Clamp to FP8 range and round
    quantized_blocks = torch.clamp(scaled_blocks, FP8_MIN, FP8_MAX).round().to(torch.int8)
    
    # Flatten back
    quantized_flat = quantized_blocks.view(-1)[:num_elements]
    outlier_mask_flat = outlier_mask_blocks.view(-1)[:num_elements]
    scales_flat = scales.squeeze(1)
    
    # Extract outlier values
    outlier_values = tensor.flatten()[outlier_mask_flat].to(torch.float16)
    
    # Reshape mask to original shape
    outlier_mask = outlier_mask_flat.view(original_shape)
    
    return QuantizedTensor(
        fp8_data=quantized_flat.view(original_shape).to(original_device),
        scales=scales_flat.to(original_device),
        outlier_mask=outlier_mask.to(original_device),
        outlier_values=outlier_values.to(original_device),
        original_shape=original_shape,
        block_size=block_size
    )


def dequantize_tensor(
    qt: QuantizedTensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Dequantize a QuantizedTensor back to FP16.
    
    Args:
        qt: QuantizedTensor to dequantize
        device: Target device (defaults to device of fp8_data)
        
    Returns:
        Reconstructed FP16 tensor
    """
    if device is None:
        device = qt.fp8_data.device
    
    # Handle empty tensors
    if qt.fp8_data.numel() == 0:
        return torch.empty(qt.original_shape, dtype=torch.float16, device=device)
    
    original_shape = qt.original_shape
    block_size = qt.block_size
    
    # Move to target device
    fp8_data = qt.fp8_data.to(device)
    scales = qt.scales.to(device)
    outlier_mask = qt.outlier_mask.to(device)
    outlier_values = qt.outlier_values.to(device)
    
    # Flatten for block processing
    flat = fp8_data.flatten().float()
    num_elements = flat.numel()
    
    # Pad for block alignment
    padded_size = math.ceil(num_elements / block_size) * block_size
    if padded_size > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded_size - num_elements), value=0.0)
    
    # Reshape and dequantize per block
    num_blocks = padded_size // block_size
    blocks = flat.view(num_blocks, block_size)
    dequantized_blocks = blocks * scales.view(-1, 1)
    
    # Flatten and trim
    dequantized = dequantized_blocks.view(-1)[:num_elements]
    
    # Reshape to original
    output = dequantized.view(original_shape).to(torch.float16)
    
    # Restore outliers
    output_flat = output.view(-1)
    mask_flat = outlier_mask.view(-1)
    output_flat[mask_flat] = outlier_values
    
    return output.view(original_shape)


def compression_ratio(original: torch.Tensor, quantized: QuantizedTensor) -> float:
    """Calculate compression ratio achieved by quantization."""
    original_bytes = original.numel() * original.element_size()
    quantized_bytes = quantized.memory_bytes()
    return original_bytes / quantized_bytes if quantized_bytes > 0 else float('inf')


# ============================================================================
# FP4 Quantization (4-bit) - For maximum compression on less sensitive layers
# ============================================================================

class QuantizedTensor4bit(NamedTuple):
    """
    Represents a tensor quantized to 4-bit with outliers preserved in FP16.
    
    Two 4-bit values are packed into each INT8 byte for storage efficiency.
    """
    packed_data: torch.Tensor     # INT8 storage, 2 values per byte
    scales: torch.Tensor          # Per-block scales (FP32)
    outlier_mask: torch.Tensor    # Boolean mask for outliers
    outlier_values: torch.Tensor  # FP16 outlier values
    original_shape: torch.Size
    block_size: int
    num_elements: int             # Original element count (needed for unpacking)
    
    def to(self, device: torch.device) -> 'QuantizedTensor4bit':
        """Move all tensors to the specified device."""
        return QuantizedTensor4bit(
            packed_data=self.packed_data.to(device),
            scales=self.scales.to(device),
            outlier_mask=self.outlier_mask.to(device),
            outlier_values=self.outlier_values.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size,
            num_elements=self.num_elements
        )
    
    def memory_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return (
            self.packed_data.numel() * self.packed_data.element_size() +
            self.scales.numel() * self.scales.element_size() +
            self.outlier_mask.numel() * self.outlier_mask.element_size() +
            self.outlier_values.numel() * self.outlier_values.element_size()
        )


# FP4 uses range [-7, 7] (4-bit signed integer)
FP4_MAX = 7.0
FP4_MIN = -7.0


def quantize_tensor_fp4(
    tensor: torch.Tensor,
    block_size: int = 64,
    outlier_threshold: float = 2.5
) -> QuantizedTensor4bit:
    """
    Quantize a FP16/FP32 tensor to 4-bit with outlier preservation.
    
    Uses more aggressive outlier detection since 4-bit has less precision.
    Two 4-bit values are packed into each INT8 byte.
    
    Args:
        tensor: Input tensor (FP16 or FP32)
        block_size: Block size for per-block scaling
        outlier_threshold: Z-score threshold (lower = more outliers preserved)
        
    Returns:
        QuantizedTensor4bit with packed 4-bit data
    """
    original_shape = tensor.shape
    original_device = tensor.device
    num_elements = tensor.numel()
    
    # Handle empty tensors
    if num_elements == 0:
        return QuantizedTensor4bit(
            packed_data=torch.empty(0, dtype=torch.int8, device=original_device),
            scales=torch.empty(0, dtype=torch.float32, device=original_device),
            outlier_mask=torch.empty(0, dtype=torch.bool, device=original_device),
            outlier_values=torch.empty(0, dtype=torch.float16, device=original_device),
            original_shape=original_shape,
            block_size=block_size,
            num_elements=0
        )
    
    # Flatten for block processing
    flat = tensor.float().flatten()
    
    # Pad to multiple of block_size
    padded_size = math.ceil(num_elements / block_size) * block_size
    if padded_size > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded_size - num_elements), value=0.0)
    
    # Reshape into blocks
    num_blocks = padded_size // block_size
    blocks = flat.view(num_blocks, block_size)
    
    # Detect outliers (more aggressive for 4-bit)
    block_means = blocks.mean(dim=1, keepdim=True)
    block_stds = blocks.std(dim=1, keepdim=True)
    block_stds = torch.clamp(block_stds, min=1e-6)
    z_scores = torch.abs((blocks - block_means) / block_stds)
    outlier_mask_blocks = z_scores > outlier_threshold
    
    # Clean blocks for scale computation
    blocks_clean = blocks.clone()
    blocks_clean[outlier_mask_blocks] = 0.0
    
    # Compute per-block scales
    block_max = blocks_clean.abs().max(dim=1, keepdim=True).values
    block_max = torch.clamp(block_max, min=1e-6)
    scales = block_max / FP4_MAX
    
    # Quantize to 4-bit range
    scaled_blocks = blocks / scales
    quantized_blocks = torch.clamp(scaled_blocks, FP4_MIN, FP4_MAX).round().to(torch.int8)
    
    # Flatten
    quantized_flat = quantized_blocks.view(-1)[:num_elements]
    outlier_mask_flat = outlier_mask_blocks.view(-1)[:num_elements]
    
    # Pack two 4-bit values into each INT8
    # Pad to even number for packing
    pack_size = math.ceil(num_elements / 2) * 2
    if pack_size > num_elements:
        quantized_flat = torch.nn.functional.pad(quantized_flat, (0, pack_size - num_elements), value=0)
    
    # Map from [-7, 7] to [0, 15] for unsigned packing
    unsigned = (quantized_flat + 8).to(torch.uint8)
    
    # Pack: high nibble = even indices, low nibble = odd indices
    packed = (unsigned[0::2] << 4) | (unsigned[1::2] & 0x0F)
    packed = packed.to(torch.int8)
    
    # Extract outlier values
    outlier_values = tensor.flatten()[outlier_mask_flat].to(torch.float16)
    outlier_mask = outlier_mask_flat.view(original_shape)
    
    return QuantizedTensor4bit(
        packed_data=packed.to(original_device),
        scales=scales.squeeze(1).to(original_device),
        outlier_mask=outlier_mask.to(original_device),
        outlier_values=outlier_values.to(original_device),
        original_shape=original_shape,
        block_size=block_size,
        num_elements=num_elements
    )


def dequantize_tensor_fp4(
    qt: QuantizedTensor4bit,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Dequantize a QuantizedTensor4bit back to FP16.
    
    Args:
        qt: QuantizedTensor4bit to dequantize
        device: Target device (defaults to device of packed_data)
        
    Returns:
        Reconstructed FP16 tensor
    """
    if device is None:
        device = qt.packed_data.device
    
    if qt.num_elements == 0:
        return torch.empty(qt.original_shape, dtype=torch.float16, device=device)
    
    # Move to target device
    packed = qt.packed_data.to(device).to(torch.uint8)
    scales = qt.scales.to(device)
    outlier_mask = qt.outlier_mask.to(device)
    outlier_values = qt.outlier_values.to(device)
    
    # Unpack: extract high and low nibbles
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    
    # Interleave back
    unpacked = torch.zeros(packed.numel() * 2, dtype=torch.int8, device=device)
    unpacked[0::2] = high.to(torch.int8)
    unpacked[1::2] = low.to(torch.int8)
    
    # Map back from [0, 15] to [-7, 7]
    unpacked = unpacked - 8
    
    # Trim to original size
    unpacked = unpacked[:qt.num_elements].float()
    
    # Pad for block processing
    block_size = qt.block_size
    padded_size = math.ceil(qt.num_elements / block_size) * block_size
    if padded_size > qt.num_elements:
        unpacked = torch.nn.functional.pad(unpacked, (0, padded_size - qt.num_elements), value=0.0)
    
    # Dequantize per block
    num_blocks = padded_size // block_size
    blocks = unpacked.view(num_blocks, block_size)
    dequantized_blocks = blocks * scales.view(-1, 1)
    
    # Flatten and trim
    dequantized = dequantized_blocks.view(-1)[:qt.num_elements]
    
    # Reshape to original
    output = dequantized.view(qt.original_shape).to(torch.float16)
    
    # Restore outliers
    output_flat = output.view(-1)
    mask_flat = outlier_mask.view(-1)
    output_flat[mask_flat] = outlier_values
    
    return output.view(qt.original_shape)


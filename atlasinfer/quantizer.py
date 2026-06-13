"""
AtlasInfer Quantizer - block-wise integer quantization with sparse outliers.

Weights are quantized to symmetric 8-bit or 4-bit integers using per-block
scales. A small set of high-magnitude "outlier" weights (detected per block by
z-score) are kept in FP16 so the few values that dominate a layer's output are
not crushed by the low-bit grid.

Outliers are stored *sparsely* - as (index, value) pairs into the flattened
tensor - rather than as a dense boolean mask. Because typical outlier rates are
well under 1%, this keeps the overhead negligible; a dense mask would cost a full
byte per weight and erase the memory savings entirely.

Note on naming: the 8-bit path is symmetric INT8 (not IEEE FP8 E4M3). The field
``fp8_data`` is an int8 buffer; the name is retained only for API stability.
"""
import math
from typing import NamedTuple, Optional

import torch

# Symmetric integer ranges. INT8 fits [-127, 127]; INT4 uses a symmetric [-7, 7].
INT8_MAX = 127.0
INT4_MAX = 7.0

# Legacy aliases.
FP8_MAX, FP8_MIN = INT8_MAX, -INT8_MAX
FP4_MAX, FP4_MIN = INT4_MAX, -INT4_MAX


class QuantizedTensor(NamedTuple):
    """A tensor quantized to symmetric INT8 with sparse FP16 outliers.

    - fp8_data:        INT8 storage (symmetric 8-bit; name kept for API stability)
    - scales:          per-block scale factors (FP32)
    - outlier_indices: INT32 positions of outliers into the flattened tensor
    - outlier_values:  FP16 values at those positions
    - original_shape:  shape for reconstruction
    - block_size:      block size used for per-block scaling
    """
    fp8_data: torch.Tensor
    scales: torch.Tensor
    outlier_indices: torch.Tensor
    outlier_values: torch.Tensor
    original_shape: torch.Size
    block_size: int

    def to(self, device: torch.device) -> "QuantizedTensor":
        return QuantizedTensor(
            fp8_data=self.fp8_data.to(device),
            scales=self.scales.to(device),
            outlier_indices=self.outlier_indices.to(device),
            outlier_values=self.outlier_values.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size,
        )

    def memory_bytes(self) -> int:
        return (
            self.fp8_data.numel() * self.fp8_data.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.outlier_indices.numel() * self.outlier_indices.element_size()
            + self.outlier_values.numel() * self.outlier_values.element_size()
        )


def _find_outliers(blocks: torch.Tensor, threshold: float) -> torch.Tensor:
    """Boolean per-block mask of elements exceeding ``threshold`` std devs."""
    means = blocks.mean(dim=1, keepdim=True)
    stds = torch.clamp(blocks.std(dim=1, keepdim=True), min=1e-6)
    z = torch.abs((blocks - means) / stds)
    return z > threshold


def _quantize_core(tensor: torch.Tensor, block_size: int, int_max: float, threshold: float):
    """Shared quantization math; returns (q_int8, scales, outlier_idx, outlier_val)."""
    original_shape = tensor.shape
    num_elements = tensor.numel()

    flat = tensor.float().flatten()
    padded = math.ceil(num_elements / block_size) * block_size
    if padded > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded - num_elements))
    blocks = flat.view(-1, block_size)

    outlier_blocks = _find_outliers(blocks, threshold)

    # Compute scales from non-outlier magnitudes so a single spike can't blow
    # out the whole block's resolution.
    clean = blocks.clone()
    clean[outlier_blocks] = 0.0
    block_max = torch.clamp(clean.abs().amax(dim=1, keepdim=True), min=1e-6)
    scales = block_max / int_max

    q = torch.clamp((blocks / scales).round(), -int_max, int_max).to(torch.int8)
    q_flat = q.view(-1)[:num_elements]

    outlier_flat = outlier_blocks.view(-1)[:num_elements]
    outlier_idx = outlier_flat.nonzero(as_tuple=True)[0].to(torch.int32)
    outlier_val = tensor.flatten()[outlier_idx.long()].to(torch.float16)

    return q_flat, scales.squeeze(1), outlier_idx, outlier_val, original_shape


def quantize_tensor(
    tensor: torch.Tensor, block_size: int = 128, outlier_threshold: float = 3.0
) -> QuantizedTensor:
    """Quantize an FP16/FP32 tensor to symmetric INT8 with sparse outliers."""
    if tensor.numel() == 0:
        dev = tensor.device
        return QuantizedTensor(
            torch.empty(0, dtype=torch.int8, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
            torch.empty(0, dtype=torch.int32, device=dev),
            torch.empty(0, dtype=torch.float16, device=dev),
            tensor.shape, block_size,
        )
    dev = tensor.device
    q_flat, scales, oidx, oval, shape = _quantize_core(
        tensor, block_size, INT8_MAX, outlier_threshold
    )
    return QuantizedTensor(
        fp8_data=q_flat.view(shape).to(dev),
        scales=scales.to(dev),
        outlier_indices=oidx.to(dev),
        outlier_values=oval.to(dev),
        original_shape=shape,
        block_size=block_size,
    )


def dequantize_tensor(qt: QuantizedTensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct an FP16 tensor from a :class:`QuantizedTensor`."""
    if device is None:
        device = qt.fp8_data.device
    if qt.fp8_data.numel() == 0:
        return torch.empty(qt.original_shape, dtype=torch.float16, device=device)

    fp8 = qt.fp8_data.to(device)
    scales = qt.scales.to(device)
    block_size = qt.block_size
    num_elements = fp8.numel()

    flat = fp8.flatten().float()
    padded = math.ceil(num_elements / block_size) * block_size
    if padded > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded - num_elements))
    blocks = flat.view(-1, block_size) * scales.view(-1, 1)
    out = blocks.view(-1)[:num_elements].to(torch.float16)

    if qt.outlier_indices.numel() > 0:
        out[qt.outlier_indices.to(device).long()] = qt.outlier_values.to(device)
    return out.view(qt.original_shape)


def compression_ratio(original: torch.Tensor, quantized: "QuantizedTensor") -> float:
    """Bytes saved relative to the dense original."""
    original_bytes = original.numel() * original.element_size()
    quantized_bytes = quantized.memory_bytes()
    return original_bytes / quantized_bytes if quantized_bytes > 0 else float("inf")


# ============================================================================ #
# INT4 (4-bit) - packed two-per-byte, for the least sensitive layers
# ============================================================================ #
class QuantizedTensor4bit(NamedTuple):
    """A tensor quantized to packed 4-bit integers with sparse FP16 outliers.

    Two 4-bit values share each int8 byte.
    """
    packed_data: torch.Tensor      # int8, two 4-bit values per byte
    scales: torch.Tensor           # per-block scales (FP32)
    outlier_indices: torch.Tensor  # INT32 positions into the flattened tensor
    outlier_values: torch.Tensor   # FP16 values at those positions
    original_shape: torch.Size
    block_size: int
    num_elements: int

    def to(self, device: torch.device) -> "QuantizedTensor4bit":
        return QuantizedTensor4bit(
            packed_data=self.packed_data.to(device),
            scales=self.scales.to(device),
            outlier_indices=self.outlier_indices.to(device),
            outlier_values=self.outlier_values.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size,
            num_elements=self.num_elements,
        )

    def memory_bytes(self) -> int:
        return (
            self.packed_data.numel() * self.packed_data.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.outlier_indices.numel() * self.outlier_indices.element_size()
            + self.outlier_values.numel() * self.outlier_values.element_size()
        )


def quantize_tensor_fp4(
    tensor: torch.Tensor, block_size: int = 64, outlier_threshold: float = 2.5
) -> QuantizedTensor4bit:
    """Quantize an FP16/FP32 tensor to packed 4-bit with sparse outliers."""
    num_elements = tensor.numel()
    dev = tensor.device
    if num_elements == 0:
        return QuantizedTensor4bit(
            torch.empty(0, dtype=torch.int8, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
            torch.empty(0, dtype=torch.int32, device=dev),
            torch.empty(0, dtype=torch.float16, device=dev),
            tensor.shape, block_size, 0,
        )

    q_flat, scales, oidx, oval, shape = _quantize_core(
        tensor, block_size, INT4_MAX, outlier_threshold
    )

    # Pad to an even count and pack two signed 4-bit values into each byte.
    pack_size = math.ceil(num_elements / 2) * 2
    if pack_size > num_elements:
        q_flat = torch.nn.functional.pad(q_flat, (0, pack_size - num_elements))
    unsigned = (q_flat + 8).to(torch.uint8)  # map [-7,7] -> [1,15]
    packed = ((unsigned[0::2] << 4) | (unsigned[1::2] & 0x0F)).to(torch.int8)

    return QuantizedTensor4bit(
        packed_data=packed.to(dev),
        scales=scales.to(dev),
        outlier_indices=oidx.to(dev),
        outlier_values=oval.to(dev),
        original_shape=shape,
        block_size=block_size,
        num_elements=num_elements,
    )


def dequantize_tensor_fp4(qt: QuantizedTensor4bit, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct an FP16 tensor from a :class:`QuantizedTensor4bit`."""
    if device is None:
        device = qt.packed_data.device
    if qt.num_elements == 0:
        return torch.empty(qt.original_shape, dtype=torch.float16, device=device)

    packed = qt.packed_data.to(device).to(torch.uint8)
    scales = qt.scales.to(device)

    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.int8, device=device)
    unpacked[0::2] = high.to(torch.int8)
    unpacked[1::2] = low.to(torch.int8)
    unpacked = (unpacked - 8)[: qt.num_elements].float()  # map back to [-7,7]

    block_size = qt.block_size
    padded = math.ceil(qt.num_elements / block_size) * block_size
    if padded > qt.num_elements:
        unpacked = torch.nn.functional.pad(unpacked, (0, padded - qt.num_elements))
    blocks = unpacked.view(-1, block_size) * scales.view(-1, 1)
    out = blocks.view(-1)[: qt.num_elements].to(torch.float16)

    if qt.outlier_indices.numel() > 0:
        out[qt.outlier_indices.to(device).long()] = qt.outlier_values.to(device)
    return out.view(qt.original_shape)

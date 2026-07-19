"""
AtlasInfer Quantizer - block-wise integer quantization with sparse outliers.

Weights are quantized to symmetric 8-bit or 4-bit integers using per-block
scales. A small set of high-magnitude "outlier" weights (detected per block by a
robust median/MAD z-score) are kept in FP16 so the few values that dominate a
layer's output are not crushed by the low-bit grid. The detector is robust on
purpose: when a block holds several comparably large weights they inflate a plain
mean/std together and mask one another below the threshold, whereas median/MAD
tolerates up to ~50% such contamination and still flags them (see
``_find_outliers``).

Outliers are stored *sparsely* - as (index, value) pairs into the flattened
tensor - rather than as a dense boolean mask. Because typical outlier rates are
well under 1%, this keeps the overhead negligible; a dense mask would cost a full
byte per weight and erase the memory savings entirely.

Note on naming: the 8-bit path is symmetric INT8 (not IEEE FP8 E4M3) and the
4-bit path is integer/NF4 (not IEEE FP4). Fields and buffers are named ``int8_*``
accordingly.
"""
import math
from typing import NamedTuple, Optional

import torch

from .double_quant import DoubleQuantScales, double_quantize

# Symmetric integer ranges. INT8 fits [-127, 127]; INT4 uses a symmetric [-7, 7].
INT8_MAX = 127.0
INT4_MAX = 7.0

# Sanity cap on the per-block outlier rate. Outliers are meant to be rare (well
# under a few percent); a block that wants to flag more than this is either
# degenerate (>=50% identical / padded / pruned, so its robust spread collapsed)
# or just genuinely wide — either way, pulling that many values out into sparse
# FP16 would invert the compression, so we flag nothing there and quantize the
# block normally. See ``_find_outliers``.
_MAX_OUTLIER_FRACTION = 0.25

# NF4 (NormalFloat-4, from QLoRA): 16 levels placed at the quantiles of a unit
# normal distribution, normalized to [-1, 1] with an exact 0. Because LLM weights
# are roughly Gaussian, a grid matched to that distribution wastes far fewer codes
# than a uniform [-7,7] grid -> lower error at the same 4 bits.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


class QuantizedTensor(NamedTuple):
    """A tensor quantized to symmetric INT8 with sparse FP16 outliers.

    - int8_data:       INT8 storage (symmetric 8-bit)
    - scales:          per-block scale factors (FP32)
    - outlier_indices: INT32 positions of outliers into the flattened tensor
    - outlier_values:  FP16 values at those positions
    - original_shape:  shape for reconstruction
    - block_size:      block size used for per-block scaling
    """
    int8_data: torch.Tensor
    scales: torch.Tensor
    outlier_indices: torch.Tensor
    outlier_values: torch.Tensor
    original_shape: torch.Size
    block_size: int

    def to(self, device: torch.device) -> "QuantizedTensor":
        return QuantizedTensor(
            int8_data=self.int8_data.to(device),
            scales=self.scales.to(device),
            outlier_indices=self.outlier_indices.to(device),
            outlier_values=self.outlier_values.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size,
        )

    def memory_bytes(self) -> int:
        return (
            self.int8_data.numel() * self.int8_data.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.outlier_indices.numel() * self.outlier_indices.element_size()
            + self.outlier_values.numel() * self.outlier_values.element_size()
        )


def _find_outliers(
    blocks: torch.Tensor, threshold: float, num_valid: Optional[int] = None
) -> torch.Tensor:
    """Boolean per-block mask of elements ``threshold`` robust std-devs from center.

    Uses a **median / MAD** z-score rather than mean / std. Mean and standard
    deviation have a breakdown point of zero, so a *cluster* of comparably large
    weights inflates the block's std enough to pull each member's z-score back
    under the threshold — the classic *masking* effect — and the spikes this
    sparse-outlier path exists to rescue evade detection together. (A *single*
    spike is not the problem: its studentized deviation is bounded by
    ``(N-1)/sqrt(N)`` ~ 7.9-11.2 for the N=64-128 blocks we use, far above a
    3-sigma cut, so either estimator catches a lone one; it's several co-located
    spikes that mean/std misses.) The median and the median-absolute-deviation
    tolerate up to ~50% contamination, so the whole cluster is caught instead.

    That robustness has a failure mode of its own: when a block is >=50% identical
    (a pruned/sparse block of mostly zeros, or otherwise near-constant) the MAD
    collapses to 0 and *every* differing element scores as an outlier. More
    generally, any block where the mask covers a large fraction is one where
    pulling that fraction out into sparse FP16 would invert the compression the
    outlier path exists for — whether the block is degenerate or just genuinely
    wide, plain per-block quantization is the better call there. So a per-block
    outlier-fraction cap (:data:`_MAX_OUTLIER_FRACTION`) drops the mask for any
    block flagging more than that fraction; a real spike is a tiny fraction and
    stays under the cap, so lone/rare outliers are preserved. Note this also
    means a genuine spike sharing a *degenerate* block with a large secondary
    mass is deliberately left in-line (quantized normally) rather than extracted
    — the block is quantized as a whole, by design.

    ``num_valid``: when the flattened tensor was zero-padded up to a whole number
    of blocks, the count of real (non-padding) elements. The trailing padding
    lives entirely in the final block; left in, a mostly-padding tail block would
    have its median and MAD collapse to ~0 and flag its handful of *ordinary*
    weights as spurious outliers. So the final block's center/spread are computed
    from its real elements only, and padding positions are never flagged.
    """
    block_size = blocks.shape[1]
    n_pad = (blocks.numel() - num_valid) if num_valid is not None else 0

    median = blocks.median(dim=1, keepdim=True).values
    if n_pad > 0:  # recompute the padded tail block's center from real elements
        real = blocks[-1, : block_size - n_pad]
        median[-1, 0] = real.median()

    dev = torch.abs(blocks - median)
    mad = dev.median(dim=1, keepdim=True).values
    if n_pad > 0:  # ...and its spread, so trailing zeros don't shrink the MAD
        real = blocks[-1, : block_size - n_pad]
        mad[-1, 0] = (real - median[-1, 0]).abs().median()

    # sigma_hat = MAD / k. The asymptotic constant is k = Phi^{-1}(0.75) = 0.6745
    # (so `threshold` reads in sigma units), but sample MAD is downward-biased at
    # small block sizes, which would make sigma_hat too small and roughly double
    # the fraction flagged on clean weights (eroding compression). k = 0.60 is
    # calibrated so the flagged fraction on clean Gaussian blocks at the sizes we
    # use (64-128) matches the intended ~1% at threshold 2.5-3.0, while masked
    # clusters (the ones mean/std lets hide) are still caught. Clamp so a
    # (near-)constant block doesn't divide by ~0 and flag everything.
    robust_std = torch.clamp(mad / 0.60, min=1e-6)
    z = dev / robust_std
    mask = z > threshold
    if n_pad > 0:  # padding is an artifact, never an outlier
        mask[-1, block_size - n_pad :] = False
    # Drop the mask when it covers too much of a block (see docstring): flagging
    # more than _MAX_OUTLIER_FRACTION means the block is degenerate or genuinely
    # wide, and extracting that many values as sparse FP16 would invert the
    # compression, so quantize it normally instead.
    keep = mask.float().mean(dim=1, keepdim=True) <= _MAX_OUTLIER_FRACTION
    return mask & keep


def _quantize_core(tensor: torch.Tensor, block_size: int, int_max: float, threshold: float):
    """Shared quantization math; returns (q_int8, scales, outlier_idx, outlier_val)."""
    original_shape = tensor.shape
    num_elements = tensor.numel()

    flat = tensor.float().flatten()
    padded = math.ceil(num_elements / block_size) * block_size
    if padded > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded - num_elements))
    blocks = flat.view(-1, block_size)

    outlier_blocks = _find_outliers(blocks, threshold, num_valid=num_elements)

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
        int8_data=q_flat.view(shape).to(dev),
        scales=scales.to(dev),
        outlier_indices=oidx.to(dev),
        outlier_values=oval.to(dev),
        original_shape=shape,
        block_size=block_size,
    )


def dequantize_tensor(qt: QuantizedTensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct an FP16 tensor from a :class:`QuantizedTensor`."""
    if device is None:
        device = qt.int8_data.device
    if qt.int8_data.numel() == 0:
        return torch.empty(qt.original_shape, dtype=torch.float16, device=device)

    q8 = qt.int8_data.to(device)
    scales = qt.scales.to(device)
    block_size = qt.block_size
    num_elements = q8.numel()

    flat = q8.flatten().float()
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
    scales: torch.Tensor           # per-block scales (FP32); empty when scales_dq set
    outlier_indices: torch.Tensor  # INT32 positions into the flattened tensor
    outlier_values: torch.Tensor   # FP16 values at those positions
    original_shape: torch.Size
    block_size: int
    num_elements: int
    # Optional double-quantized block scales (QLoRA-style). When present the FP32
    # ``scales`` above is empty and the compressed form is what stays resident; the
    # dequant path reconstructs the FP32 scales transiently. See double_quant.py.
    scales_dq: Optional[DoubleQuantScales] = None

    def to(self, device: torch.device) -> "QuantizedTensor4bit":
        return QuantizedTensor4bit(
            packed_data=self.packed_data.to(device),
            scales=self.scales.to(device),
            outlier_indices=self.outlier_indices.to(device),
            outlier_values=self.outlier_values.to(device),
            original_shape=self.original_shape,
            block_size=self.block_size,
            num_elements=self.num_elements,
            scales_dq=self.scales_dq.to(device) if self.scales_dq is not None else None,
        )

    def block_scales(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """The per-block FP32 scales, reconstructed from the double-quantized form
        if that's how they're stored, else returned directly."""
        if self.scales_dq is not None:
            return self.scales_dq.reconstruct(device)
        return self.scales.to(device) if device is not None else self.scales

    def memory_bytes(self) -> int:
        scale_bytes = (
            self.scales_dq.memory_bytes() if self.scales_dq is not None
            else self.scales.numel() * self.scales.element_size()
        )
        return (
            self.packed_data.numel() * self.packed_data.element_size()
            + scale_bytes
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


# ============================================================================ #
# NF4 (NormalFloat-4) - a distribution-matched 4-bit codebook
# ============================================================================ #
def quantize_tensor_nf4(
    tensor: torch.Tensor, block_size: int = 64, outlier_threshold: float = 2.5,
    double_quant: bool = False,
) -> QuantizedTensor4bit:
    """Quantize to packed NF4 codes with per-block absmax scale and sparse outliers.

    Each weight is normalized by its block's absmax (into [-1, 1]) and mapped to
    the nearest NF4 level; the 4-bit *code* (0..15) is what gets packed. Reuses
    :class:`QuantizedTensor4bit` for storage - decode with
    :func:`dequantize_tensor_nf4` (the codes index the NF4 codebook, not [-7,7]).

    ``double_quant``: additionally compress the per-block FP32 scales to INT8 +
    per-group (scale, offset) (QLoRA-style), cutting the ~0.5 bit/weight scale
    overhead ~4x — the memory trick bitsandbytes uses at 4-bit. See double_quant.py.
    """
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

    flat = tensor.float().flatten()
    padded = math.ceil(num_elements / block_size) * block_size
    if padded > num_elements:
        flat = torch.nn.functional.pad(flat, (0, padded - num_elements))
    blocks = flat.view(-1, block_size)

    outlier_blocks = _find_outliers(blocks, outlier_threshold, num_valid=num_elements)
    clean = blocks.clone()
    clean[outlier_blocks] = 0.0
    absmax = clean.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)

    normalized = blocks / absmax
    boundaries = ((NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2).to(blocks.device)
    codes = torch.bucketize(normalized.reshape(-1), boundaries).clamp_(0, 15).to(torch.uint8)
    codes = codes[:num_elements]

    outlier_flat = outlier_blocks.view(-1)[:num_elements]
    oidx = outlier_flat.nonzero(as_tuple=True)[0].to(torch.int32)
    oval = tensor.flatten()[oidx.long()].to(torch.float16)

    pack_size = math.ceil(num_elements / 2) * 2
    if pack_size > num_elements:
        codes = torch.nn.functional.pad(codes, (0, pack_size - num_elements))
    packed = ((codes[0::2] << 4) | (codes[1::2] & 0x0F)).to(torch.int8)

    block_scales = absmax.squeeze(1).to(dev)
    scales_dq = None
    if double_quant:
        scales_dq = double_quantize(block_scales).to(dev)
        block_scales = torch.empty(0, dtype=torch.float32, device=dev)  # not resident

    return QuantizedTensor4bit(
        packed_data=packed.to(dev),
        scales=block_scales,
        outlier_indices=oidx.to(dev),
        outlier_values=oval.to(dev),
        original_shape=tensor.shape,
        block_size=block_size,
        num_elements=num_elements,
        scales_dq=scales_dq,
    )


def dequantize_tensor_nf4(qt: QuantizedTensor4bit, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct an FP16 tensor from NF4-coded :class:`QuantizedTensor4bit`."""
    if device is None:
        device = qt.packed_data.device
    if qt.num_elements == 0:
        return torch.empty(qt.original_shape, dtype=torch.float16, device=device)

    packed = qt.packed_data.to(device).to(torch.uint8)
    scales = qt.block_scales(device)  # reconstructs from double-quant if used

    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    codes = torch.empty(packed.numel() * 2, dtype=torch.long, device=device)
    codes[0::2] = high.long()
    codes[1::2] = low.long()
    codes = codes[: qt.num_elements]
    levels = NF4_LEVELS.to(device)[codes]  # codebook lookup

    block_size = qt.block_size
    padded = math.ceil(qt.num_elements / block_size) * block_size
    if padded > qt.num_elements:
        levels = torch.nn.functional.pad(levels, (0, padded - qt.num_elements))
    blocks = levels.view(-1, block_size) * scales.view(-1, 1)
    out = blocks.view(-1)[: qt.num_elements].to(torch.float16)

    if qt.outlier_indices.numel() > 0:
        out[qt.outlier_indices.to(device).long()] = qt.outlier_values.to(device)
    return out.view(qt.original_shape)

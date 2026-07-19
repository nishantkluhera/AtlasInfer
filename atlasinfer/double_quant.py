"""
Double quantization of block scales (the QLoRA / bitsandbytes memory trick).

Block-wise 4-bit quantization stores one FP32 scale per block. At block_size 64
that scale is 4 bytes per 64 weights = **0.5 bits/weight** of pure overhead — on
top of the 4-bit payload, so "4-bit NF4" actually costs ~4.5 bits before outliers.
bitsandbytes closes most of that by *quantizing the scales themselves*: the FP32
per-block absmax scales are grouped and stored as INT8 with a small second-level
FP32 scale (and offset) per group. This is exactly the ~10–13% memory gap between
AtlasInfer's plain NF4 and bnb's NF4 in the benchmarks.

This module implements it from scratch. The block scales are all positive
(per-block absmax), so per group of ``group_size`` scales we subtract the group
mean (an FP32 offset), quantize the centered residual to symmetric INT8 with one
FP32 second-level scale, and keep both. Storage per block-scale drops from 4 bytes
to ``1 + 8/group_size`` bytes (~1.03 at group_size 256) — a ~4x reduction of the
scale overhead, i.e. ~0.37 bits/weight saved at block_size 64.

The reconstruction is exact up to the INT8 rounding of the scales; because scales
vary smoothly and slowly relative to the weights, that second rounding adds
negligible perplexity (measured: within run-to-run noise) while recovering the
memory. Decode reconstructs the FP32 scales transiently, so the compressed form is
what actually stays resident — the memory saving is real, not just an accounting
change.
"""
from typing import NamedTuple, Optional

import torch

# Second-level group size: how many block-scales share one FP32 (scale, offset).
# 256 matches bitsandbytes; larger = smaller overhead but coarser second-level fit.
DEFAULT_DQ_GROUP = 256
_DQ_INT8_MAX = 127.0


class DoubleQuantScales(NamedTuple):
    """Compressed representation of a 1-D FP32 block-scale vector.

    - codes:        INT8 residual codes, one per original block-scale
    - second_scale: FP32 per-group second-level scale
    - offset:       FP32 per-group mean (subtracted before quantizing)
    - group_size:   block-scales per second-level group
    - num_scales:   original count (codes may be padded up to a whole group)
    """
    codes: torch.Tensor
    second_scale: torch.Tensor
    offset: torch.Tensor
    group_size: int
    num_scales: int

    def to(self, device) -> "DoubleQuantScales":
        return DoubleQuantScales(
            codes=self.codes.to(device),
            second_scale=self.second_scale.to(device),
            offset=self.offset.to(device),
            group_size=self.group_size,
            num_scales=self.num_scales,
        )

    def memory_bytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.second_scale.numel() * self.second_scale.element_size()
            + self.offset.numel() * self.offset.element_size()
        )

    def reconstruct(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Rebuild the FP32 block-scale vector (length ``num_scales``)."""
        dev = device if device is not None else self.codes.device
        codes = self.codes.to(dev).float()
        ss = self.second_scale.to(dev).repeat_interleave(self.group_size)
        off = self.offset.to(dev).repeat_interleave(self.group_size)
        scales = codes * ss + off
        return scales[: self.num_scales]


def double_quantize(scales: torch.Tensor, group_size: int = DEFAULT_DQ_GROUP) -> DoubleQuantScales:
    """Compress a 1-D FP32 block-scale vector to INT8 + per-group (scale, offset).

    Args:
        scales: 1-D tensor of per-block scales (FP32; typically all positive).
        group_size: block-scales per second-level group.
    """
    scales = scales.detach().reshape(-1).float()
    num_scales = scales.numel()
    dev = scales.device

    n_groups = (num_scales + group_size - 1) // group_size
    padded = n_groups * group_size
    if padded > num_scales:
        scales = torch.nn.functional.pad(scales, (0, padded - num_scales))
    groups = scales.view(n_groups, group_size)

    offset = groups.mean(dim=1, keepdim=True)                       # per-group mean
    centered = groups - offset
    second_scale = centered.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / _DQ_INT8_MAX
    codes = torch.clamp((centered / second_scale).round(), -127, 127).to(torch.int8)

    return DoubleQuantScales(
        codes=codes.reshape(-1).to(dev),
        second_scale=second_scale.reshape(-1).to(dev),
        offset=offset.reshape(-1).to(dev),
        group_size=group_size,
        num_scales=num_scales,
    )


def dq_relative_error(scales: torch.Tensor, group_size: int = DEFAULT_DQ_GROUP) -> float:
    """Relative L2 error of round-tripping ``scales`` through double quantization."""
    dq = double_quantize(scales, group_size)
    recon = dq.reconstruct()
    s = scales.detach().reshape(-1).float()
    return float((s - recon).norm() / (s.norm() + 1e-12))

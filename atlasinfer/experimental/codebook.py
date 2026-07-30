"""
EXPERIMENTAL — sub-4-bit **vector-quantized** codebook (research scaffold).

⚠️  Status: implemented and unit-tested for correctness, but **not validated at
scale**. The claim this reaches for — 2–3 bit weight quantization at usable
accuracy — is the current research frontier (AQLM, QuIP#, QTIP), and those papers
establish their results on Llama-2-7B/13B with the full WikiText-2/C4 protocol.
That does not fit on the 6 GB laptop this repo is developed on, so **no
SOTA-beating claim is made here** — this module exists to make the sub-4-bit
direction concrete and reproducible, not to assert it's competitive yet. Treat the
numbers it produces as a sanity check, not a benchmark.

Why vector quantization for sub-4-bit. Scalar 4-bit already spends one code per
weight; going to 2–3 bit scalar (a 4- or 8-level grid) is very lossy because each
weight is rounded independently. VQ instead groups ``d`` consecutive weights into
a vector and rounds the *whole vector* to the nearest of ``K`` learned codebook
entries — so correlations between neighbouring weights are exploited and the
effective grid is far richer than ``K^(1/d)`` per-axis. With ``K=256`` (one byte
per code) the rate is ``8/d`` bits/weight: ``d=4 -> 2 bit``, ``d=3 -> 2.67 bit``,
``d=2 -> 4 bit``. The codebook is learned per-tensor with a few Lloyd (k-means)
iterations on absmax-normalized blocks.

This is a *single*-codebook VQ. The AQLM family stacks several additive codebooks
(residual VQ) and learns them end-to-end; QuIP#/QTIP use incoherence
pre-processing + lattice/trellis codes. Those are the natural extensions and where
the real accuracy at 2 bit comes from — noted as future work, deliberately not
implemented-and-overclaimed here.
"""
from typing import NamedTuple, Optional

import torch

# One byte per code keeps storage byte-aligned (no bit-packing needed) and honest:
# resident code memory is exactly 1 byte per d-vector. Rate = 8/d bits/weight.
CODEBOOK_SIZE = 256


class VectorQuantizedTensor(NamedTuple):
    """A tensor vector-quantized against a learned per-tensor codebook.

    - codes:     uint8 index per d-vector into ``codebook`` (1 byte each)
    - codebook:  (K, vector_dim) FP16 centroids (absmax-normalized space)
    - scales:    per-block absmax used to normalize before VQ (FP32)
    - vector_dim, block_size, original_shape, num_elements: reconstruction metadata
    """
    codes: torch.Tensor
    codebook: torch.Tensor
    scales: torch.Tensor
    vector_dim: int
    block_size: int
    original_shape: torch.Size
    num_elements: int

    def to(self, device) -> "VectorQuantizedTensor":
        return VectorQuantizedTensor(
            codes=self.codes.to(device), codebook=self.codebook.to(device),
            scales=self.scales.to(device), vector_dim=self.vector_dim,
            block_size=self.block_size, original_shape=self.original_shape,
            num_elements=self.num_elements,
        )

    def memory_bytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.codebook.numel() * self.codebook.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def bits_per_weight(self) -> float:
        """Effective bits/weight including the amortized codebook + scale overhead."""
        return 8.0 * self.memory_bytes() / max(self.num_elements, 1)


def _kmeans(vectors: torch.Tensor, k: int, iters: int, seed: int = 0):
    """Lloyd's algorithm. vectors: (N, d) -> (codebook (k, d), codes (N,) uint8-range)."""
    n = vectors.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(n, generator=gen)[:k]
    centroids = vectors[init.to(vectors.device)].clone()
    if centroids.shape[0] < k:  # tiny tensor: pad codebook by repeating
        pad = k - centroids.shape[0]
        centroids = torch.cat([centroids, centroids[:1].expand(pad, -1)], dim=0)

    codes = torch.zeros(n, dtype=torch.long, device=vectors.device)
    for _ in range(iters):
        # Assign: nearest centroid by squared distance (chunked to bound memory).
        for start in range(0, n, 8192):
            chunk = vectors[start:start + 8192]
            d2 = torch.cdist(chunk, centroids)
            codes[start:start + 8192] = d2.argmin(dim=1)
        # Update: mean of assigned points; keep old centroid for empty clusters.
        new = centroids.clone()
        counts = torch.bincount(codes, minlength=k).clamp(min=1).unsqueeze(1).float()
        summed = torch.zeros_like(centroids).index_add_(0, codes, vectors)
        assigned = torch.bincount(codes, minlength=k) > 0
        new[assigned] = (summed / counts)[assigned]
        centroids = new
    return centroids, codes


def quantize_tensor_vq(
    tensor: torch.Tensor, vector_dim: int = 4, block_size: int = 256,
    kmeans_iters: int = 15, seed: int = 0,
) -> VectorQuantizedTensor:
    """Vector-quantize a weight tensor to ``8/vector_dim`` bits/weight (K=256).

    Weights are absmax-normalized per block, split into ``vector_dim``-wide
    vectors, and each vector is mapped to the nearest of 256 learned centroids.
    ``vector_dim=4`` gives the 2-bit target; ``3`` gives ~2.67-bit.
    """
    original_shape = tensor.shape
    num_elements = tensor.numel()
    dev = tensor.device
    flat = tensor.float().flatten()

    # Per-block absmax normalization (same idea as the scalar paths).
    import math
    padded = math.ceil(num_elements / block_size) * block_size
    fb = torch.nn.functional.pad(flat, (0, padded - num_elements)) if padded > num_elements else flat
    blocks = fb.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normed = (blocks / scales).reshape(-1)[:num_elements]

    # Split into d-vectors (pad the tail to a whole vector).
    vpad = math.ceil(num_elements / vector_dim) * vector_dim
    nv = torch.nn.functional.pad(normed, (0, vpad - num_elements)) if vpad > num_elements else normed
    vectors = nv.view(-1, vector_dim)

    codebook, codes = _kmeans(vectors, CODEBOOK_SIZE, kmeans_iters, seed)

    return VectorQuantizedTensor(
        codes=codes.to(torch.uint8).to(dev),
        codebook=codebook.to(torch.float16).to(dev),
        scales=scales.squeeze(1).to(dev),
        vector_dim=vector_dim, block_size=block_size,
        original_shape=original_shape, num_elements=num_elements,
    )


def dequantize_tensor_vq(qt: VectorQuantizedTensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct an FP16 tensor from a :class:`VectorQuantizedTensor`."""
    import math
    dev = device if device is not None else qt.codes.device
    codebook = qt.codebook.to(dev).float()
    codes = qt.codes.to(dev).long()

    vectors = codebook[codes]                       # (num_vectors, d)
    normed = vectors.reshape(-1)[: qt.num_elements]  # drop vector padding

    padded = math.ceil(qt.num_elements / qt.block_size) * qt.block_size
    if padded > qt.num_elements:
        normed = torch.nn.functional.pad(normed, (0, padded - qt.num_elements))
    blocks = normed.view(-1, qt.block_size) * qt.scales.to(dev).view(-1, 1)
    return blocks.reshape(-1)[: qt.num_elements].to(torch.float16).view(qt.original_shape)

"""
Tests for AtlasInfer Quantizer
"""
import torch
import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.quantizer import (
    quantize_tensor,
    dequantize_tensor,
    QuantizedTensor,
    compression_ratio,
    quantize_tensor_nf4,
    dequantize_tensor_nf4,
    quantize_tensor_fp4,
    dequantize_tensor_fp4,
    NF4_LEVELS,
)


class TestNF4:
    """NormalFloat-4 codebook quantization."""

    def test_roundtrip(self):
        torch.manual_seed(0)
        original = torch.randn(256, 256, dtype=torch.float16)
        qt = quantize_tensor_nf4(original)
        recon = dequantize_tensor_nf4(qt)
        assert recon.shape == original.shape and recon.dtype == torch.float16
        rel = (original.float() - recon.float()).norm() / original.float().norm()
        assert rel < 0.15

    def test_beats_symmetric_int4_on_gaussian(self):
        # The whole point of NF4: lower error than uniform int4 on normal weights.
        torch.manual_seed(0)
        w = torch.randn(512, 512, dtype=torch.float16)
        nf4_err = (w.float() - dequantize_tensor_nf4(quantize_tensor_nf4(w)).float()).norm()
        int4_err = (w.float() - dequantize_tensor_fp4(quantize_tensor_fp4(w)).float()).norm()
        assert nf4_err < int4_err

    def test_same_memory_as_int4(self):
        w = torch.randn(512, 512, dtype=torch.float16)
        assert quantize_tensor_nf4(w).memory_bytes() == quantize_tensor_fp4(w).memory_bytes()

    def test_levels_sorted_with_zero(self):
        assert torch.all(NF4_LEVELS[1:] > NF4_LEVELS[:-1])  # strictly ascending
        assert (NF4_LEVELS == 0.0).any()                     # has an exact zero
        assert len(NF4_LEVELS) == 16

    def test_empty(self):
        qt = quantize_tensor_nf4(torch.empty((0, 8), dtype=torch.float16))
        assert dequantize_tensor_nf4(qt).numel() == 0


class TestQuantizer:
    """Test suite for the adaptive quantizer."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Verify quantization/dequantization preserves accuracy."""
        # Create a test tensor with some variation
        original = torch.randn(256, 512, dtype=torch.float16)
        
        # Quantize
        quantized = quantize_tensor(original, block_size=128, outlier_threshold=3.0)
        
        # Verify it's a QuantizedTensor
        assert isinstance(quantized, QuantizedTensor)
        assert quantized.original_shape == original.shape
        
        # Dequantize
        reconstructed = dequantize_tensor(quantized)
        
        # Check shape matches
        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == torch.float16
        
        # Check accuracy (allow for INT8 quantization error)
        mse = torch.mean((original.float() - reconstructed.float()) ** 2)
        assert mse < 2.0, f"MSE too high: {mse.item()}"
    
    def test_outlier_preservation(self):
        """Verify that outliers are detected and preserved."""
        # Create tensor with clear outliers
        original = torch.randn(128, 128, dtype=torch.float16)
        # Add some extreme values
        original[0, 0] = 100.0
        original[64, 64] = -100.0
        original[127, 127] = 50.0
        
        # Quantize with low threshold to catch these as outliers
        quantized = quantize_tensor(original, block_size=64, outlier_threshold=2.5)

        # Should have some outliers (stored sparsely as flat indices)
        num_outliers = quantized.outlier_indices.numel()
        assert num_outliers > 0, "No outliers detected"

        # Dequantize
        reconstructed = dequantize_tensor(quantized)

        # Outlier values should be exactly preserved
        recon_flat = reconstructed.reshape(-1)
        orig_flat = original.reshape(-1)
        for idx in quantized.outlier_indices[:min(10, num_outliers)].long():
            orig_val = orig_flat[idx].item()
            recon_val = recon_flat[idx].item()
            assert abs(orig_val - recon_val) < 1e-3, f"Outlier not preserved at {idx}"
    
    def test_masked_outlier_cluster_is_caught(self):
        """Regression: a cluster of co-located large weights must be caught.

        A single spike is caught by either estimator (its z-score is bounded by
        (N-1)/sqrt(N) ~ 7.9-11.2 for N=64-128, far above a 3-sigma cut), so a
        lone-spike test would pass under the old mean/std detector too and guard
        nothing. The case mean/std actually misses is *masking*: several
        comparably large weights in one block inflate the mean/std together and
        pull each other's z-scores under the threshold. The robust median/MAD
        detector must still flag the whole cluster.
        """
        from atlasinfer.quantizer import _find_outliers

        block = torch.full((1, 64), 0.05)
        idx = list(range(0, 8))          # 8/64 co-located spikes -> mean/std masks
        for i in idx:
            block[0, i] = 50.0

        # First assert this configuration really does mask under mean/std (z < 3.0
        # at every spike), so the test exercises the failure mode it claims and
        # would fail against the old detector.
        mean = block.mean(dim=1, keepdim=True)
        std = block.std(dim=1, keepdim=True).clamp(min=1e-6)
        z_meanstd = (block - mean).abs() / std
        assert not bool((z_meanstd[0, idx] > 3.0).any()), \
            "test no longer exercises masking: mean/std already catches these"

        # The robust detector catches the whole masked cluster (8/64 < the 25% cap).
        mask = _find_outliers(block, threshold=3.0)
        assert bool(mask[0, idx].all()), "robust detector missed a masked cluster"

        recon = dequantize_tensor(
            quantize_tensor(block, block_size=64, outlier_threshold=3.0)
        ).reshape(-1)
        for i in idx:
            assert abs(recon[i].item() - 50.0) < 1e-2, "clustered outlier not preserved"

    def test_degenerate_block_does_not_mass_flag(self):
        """A >=50%-constant/sparse block must not dump its whole minority into FP16.

        MAD collapses to 0 on a majority-constant block, so without the per-block
        outlier-fraction cap every differing element gets flagged (compression
        inverted). A lone spike, being a tiny fraction, must still be caught.
        """
        from atlasinfer.quantizer import _find_outliers

        sparse = torch.zeros(1, 64)
        sparse[0, :20] = torch.randn(20)  # 44/64 exact zeros -> MAD == 0
        assert int(_find_outliers(sparse, 3.0).sum()) == 0, "degenerate block mass-flagged"

        spike = torch.full((1, 64), 0.05)
        spike[0, 32] = 100.0
        m = _find_outliers(spike, 3.0)
        assert bool(m[0, 32]) and int(m.sum()) == 1, "lone spike must still be caught, alone"

    def test_clean_weights_stay_compressed(self):
        """The robust detector must not over-flag clean Gaussian weights.

        Guards the calibration constant in ``_find_outliers``: the asymptotic
        MAD->sigma factor over-flags at small block sizes and would erode INT4
        compression below the INT8 line. On clean weights outliers stay ~1%.
        """
        from atlasinfer.quantizer import _find_outliers

        torch.manual_seed(0)
        blocks = torch.randn(4000, 64)
        frac = _find_outliers(blocks, threshold=2.5).float().mean().item()
        assert frac < 0.02, f"robust detector over-flags clean weights: {frac:.3%}"

    def test_padded_tail_block_not_mass_flagged(self):
        """A zero-padded tail block must not turn its real weights into outliers.

        When numel isn't block-aligned the final block is mostly padding zeros;
        left in the stats its median/MAD collapse to ~0 and every ordinary weight
        in it scores as an outlier (compression wasted on sparse FP16). Passing
        num_valid excludes the padding, so a clean tensor with no true outliers
        flags essentially none regardless of alignment.
        """
        from atlasinfer.quantizer import _find_outliers, quantize_tensor, dequantize_tensor

        torch.manual_seed(0)
        t = torch.randn(128 * 3 + 10) * 0.02  # tail block: 10 real + 118 padding
        qt = quantize_tensor(t, block_size=128, outlier_threshold=3.0)
        # Only a couple genuine tail-of-Gaussian flags in the full blocks; nowhere
        # near the whole 10-weight tail (which the padding-blind detector flagged).
        assert qt.outlier_indices.numel() <= 3, \
            f"padded tail over-flagged: {qt.outlier_indices.numel()} outliers"
        assert (dequantize_tensor(qt).float() - t).abs().max().item() < 1e-2

        # Directly: no flags fall inside the padded tail region.
        blocks = torch.nn.functional.pad(t.float(), (0, 128 - (t.numel() % 128))).view(-1, 128)
        mask = _find_outliers(blocks, 3.0, num_valid=t.numel())
        assert int(mask[-1, 10:].sum()) == 0, "padding flagged as outliers"

    def test_compression_ratio(self):
        """INT8 with sparse outliers must actually shrink the tensor.

        Regression guard: a dense boolean outlier mask used to cost a full byte
        per weight, making the 'quantized' tensor as big as (or bigger than) the
        FP16 original. With sparse outliers INT8 should be ~1.7x+ smaller.
        """
        original = torch.randn(1024, 1024, dtype=torch.float16)
        quantized = quantize_tensor(original)

        ratio = compression_ratio(original, quantized)
        assert ratio > 1.5, f"INT8 compression ratio too low: {ratio}"

    def test_int4_compression_ratio(self):
        """INT4 should compress more aggressively than INT8."""
        from atlasinfer.quantizer import quantize_tensor_fp4, QuantizedTensor4bit

        original = torch.randn(1024, 1024, dtype=torch.float16)
        q4 = quantize_tensor_fp4(original)
        assert isinstance(q4, QuantizedTensor4bit)

        orig_bytes = original.numel() * original.element_size()
        ratio = orig_bytes / q4.memory_bytes()
        assert ratio > 3.0, f"INT4 compression ratio too low: {ratio}"
    
    def test_small_tensor(self):
        """Test with tensor smaller than block size."""
        original = torch.randn(10, dtype=torch.float16)
        quantized = quantize_tensor(original, block_size=128)
        reconstructed = dequantize_tensor(quantized)
        
        assert reconstructed.shape == original.shape
        mse = torch.mean((original.float() - reconstructed.float()) ** 2)
        assert mse < 1.0  # Small tensors may have higher relative error
    
    def test_empty_tensor(self):
        """Test with empty tensor."""
        original = torch.empty((0, 10), dtype=torch.float16)
        quantized = quantize_tensor(original)
        reconstructed = dequantize_tensor(quantized)
        
        assert reconstructed.shape == original.shape
        assert reconstructed.numel() == 0
    
    def test_device_transfer(self):
        """Test moving quantized tensor between devices."""
        original = torch.randn(64, 64, dtype=torch.float16)
        quantized = quantize_tensor(original)
        
        # Move to same device (should work even without GPU)
        moved = quantized.to(torch.device('cpu'))
        
        assert moved.fp8_data.device == torch.device('cpu')
        assert moved.scales.device == torch.device('cpu')
        
        # Verify can still dequantize
        reconstructed = dequantize_tensor(moved)
        assert reconstructed.shape == original.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_dequantize(self):
        """Test dequantizing directly to CUDA."""
        original = torch.randn(128, 128, dtype=torch.float16)
        quantized = quantize_tensor(original)
        
        # Dequantize to GPU
        cuda_device = torch.device('cuda:0')
        reconstructed = dequantize_tensor(quantized, device=cuda_device)
        
        assert reconstructed.device == cuda_device
        assert reconstructed.shape == original.shape


class TestQuantizedTensor:
    """Test QuantizedTensor helper methods."""
    
    def test_memory_bytes(self):
        """Test memory footprint calculation."""
        original = torch.randn(256, 256, dtype=torch.float16)
        quantized = quantize_tensor(original)
        
        mem = quantized.memory_bytes()
        
        # Should be positive
        assert mem > 0
        
        # Note: with outliers + scales, quantized may not always be smaller
        # Just check the calculation works
        original_mem = original.numel() * original.element_size()
        assert mem > 0 and original_mem > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

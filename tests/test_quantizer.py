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
    compression_ratio
)


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
        
        # Check accuracy (should be close, allowing for quantization error)
        mse = torch.mean((original.float() - reconstructed.float()) ** 2)
        assert mse < 0.1, f"MSE too high: {mse.item()}"
    
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
        
        # Should have some outliers
        num_outliers = quantized.outlier_mask.sum().item()
        assert num_outliers > 0, "No outliers detected"
        
        # Dequantize
        reconstructed = dequantize_tensor(quantized)
        
        # Outlier values should be exactly preserved
        outlier_positions = quantized.outlier_mask.nonzero()
        for pos in outlier_positions[:min(10, len(outlier_positions))]:  # Check first 10
            orig_val = original[tuple(pos)].item()
            recon_val = reconstructed[tuple(pos)].item()
            assert abs(orig_val - recon_val) < 1e-3, f"Outlier not preserved at {pos}"
    
    def test_compression_ratio(self):
        """Verify we achieve meaningful compression."""
        original = torch.randn(1024, 1024, dtype=torch.float16)
        quantized = quantize_tensor(original)
        
        ratio = compression_ratio(original, quantized)
        
        # Should achieve at least 1.5x compression (conservative estimate)
        assert ratio > 1.5, f"Compression ratio too low: {ratio}"
    
    def test_small_tensor(self):
        """Test with tensor smaller than block size."""
        original = torch.randn(10, dtype=torch.float16)
        quantized = quantize_tensor(original, block_size=128)
        reconstructed = dequantize_tensor(quantized)
        
        assert reconstructed.shape == original.shape
        mse = torch.mean((original.float() - reconstructed.float()) ** 2)
        assert mse < 0.1
    
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
        
        # Should be less than original
        original_mem = original.numel() * original.element_size()
        assert mem < original_mem, "Quantized should use less memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

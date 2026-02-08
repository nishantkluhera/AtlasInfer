"""
Tests for AtlasInfer Linear layer and Model Patcher
"""
import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlasinfer.quantizer import quantize_tensor
from atlasinfer.linear import QuantizedLinear
from atlasinfer.patcher import quantize_model, get_model_info


class TestQuantizedLinear:
    """Test suite for QuantizedLinear layer."""
    
    def test_forward_pass(self):
        """Test basic forward pass through quantized linear."""
        # Create a regular linear layer
        linear = nn.Linear(64, 32)
        linear.weight.data = torch.randn_like(linear.weight.data)
        
        # Create quantized version
        qlinear = QuantizedLinear.from_linear(linear)
        
        # Test forward pass
        x = torch.randn(4, 64)
        output = qlinear(x)
        
        assert output.shape == (4, 32)
    
    def test_matches_original_approximately(self):
        """Verify quantized layer output is close to original."""
        linear = nn.Linear(128, 64)
        qlinear = QuantizedLinear.from_linear(linear)
        
        x = torch.randn(2, 128)
        
        with torch.no_grad():
            original_out = linear(x)
            quantized_out = qlinear(x.float())  # Quantized uses FP16 internally
        
        # Outputs should be close (within quantization error)
        diff = (original_out - quantized_out).abs().mean()
        assert diff < 0.5, f"Output difference too large: {diff}"
    
    def test_with_bias(self):
        """Test layer with bias."""
        linear = nn.Linear(32, 16, bias=True)
        qlinear = QuantizedLinear.from_linear(linear)
        
        assert qlinear.bias is not None
        
        x = torch.randn(1, 32)
        output = qlinear(x)
        assert output.shape == (1, 16)
    
    def test_without_bias(self):
        """Test layer without bias."""
        linear = nn.Linear(32, 16, bias=False)
        qlinear = QuantizedLinear.from_linear(linear)
        
        assert qlinear.bias is None
        
        x = torch.randn(1, 32)
        output = qlinear(x)
        assert output.shape == (1, 16)


class TestModelPatcher:
    """Test suite for model patching utilities."""
    
    def test_quantize_simple_model(self):
        """Test quantizing a simple model."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        
        # Quantize it
        quantize_model(model, verbose=False)
        
        # Check that linear layers were replaced
        info = get_model_info(model)
        assert info['quantized_linear_count'] == 2
        assert info['linear_count'] == 0
    
    def test_exclude_patterns(self):
        """Test layer exclusion patterns."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(64, 32)  # Should be excluded
                self.hidden = nn.Linear(32, 32)  # Should be quantized
                self.lm_head = nn.Linear(32, 10)  # Should be excluded
            
            def forward(self, x):
                x = self.embed(x)
                x = self.hidden(x)
                return self.lm_head(x)
        
        model = SimpleModel()
        quantize_model(model, exclude_patterns=['embed', 'lm_head'], verbose=False)
        
        # Only hidden should be quantized
        assert isinstance(model.embed, nn.Linear)
        assert isinstance(model.hidden, QuantizedLinear)
        assert isinstance(model.lm_head, nn.Linear)
    
    def test_forward_after_quantize(self):
        """Test that model works after quantization."""
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        
        x = torch.randn(2, 32)
        
        # Get original output
        with torch.no_grad():
            original_out = model(x).clone()
        
        # Quantize
        quantize_model(model, verbose=False)
        
        # Get quantized output
        with torch.no_grad():
            quantized_out = model(x)
        
        # Should still produce valid output
        assert quantized_out.shape == original_out.shape
        assert not torch.isnan(quantized_out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

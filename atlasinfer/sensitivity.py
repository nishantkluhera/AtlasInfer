"""
AtlasInfer Sensitivity Profiler - Measures layer sensitivity to quantization

This module implements the core novelty of AtlasInfer: determining which layers
are most sensitive to quantization error, enabling intelligent precision allocation.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
import hashlib

from .quantizer import quantize_tensor, dequantize_tensor


class LayerSensitivityProfiler:
    """
    Measures how sensitive each linear layer is to quantization error.
    
    Sensitivity is measured as the relative change in layer output when
    weights are quantized vs kept at full precision. Higher sensitivity
    means the layer should be kept at higher precision.
    
    This is the CORE NOVELTY of AtlasInfer:
    - AWQ uses activations to optimize quantization scales
    - LADQ uses sensitivity to allocate precision budgets
    """
    
    def __init__(
        self,
        block_size: int = 128,
        outlier_threshold: float = 3.0,
        device: Optional[torch.device] = None
    ):
        self.block_size = block_size
        self.outlier_threshold = outlier_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def profile_layer(
        self,
        layer: nn.Linear,
        sample_inputs: List[torch.Tensor]
    ) -> float:
        """
        Measure sensitivity of a single layer.
        
        Args:
            layer: The linear layer to profile
            sample_inputs: List of sample input tensors
            
        Returns:
            Sensitivity score (higher = more sensitive to quantization)
        """
        layer.eval()
        total_sensitivity = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inp in sample_inputs:
                inp = inp.to(self.device)
                if inp.dim() == 1:
                    inp = inp.unsqueeze(0)
                
                # Ensure input has correct feature dimension
                if inp.shape[-1] != layer.in_features:
                    # Reshape or skip incompatible inputs
                    if inp.numel() >= layer.in_features:
                        inp = inp.view(-1, layer.in_features)[:8]  # Take up to 8 samples
                    else:
                        continue
                
                # Original output (FP16/FP32)
                original_weight = layer.weight.data
                original_output = layer(inp.to(original_weight.dtype))
                
                # Quantized output
                quantized_weight = quantize_tensor(
                    original_weight.float(),
                    block_size=self.block_size,
                    outlier_threshold=self.outlier_threshold
                )
                dequantized_weight = dequantize_tensor(quantized_weight, device=self.device)
                
                # Temporarily replace weight
                layer.weight.data = dequantized_weight.to(original_weight.dtype)
                quantized_output = layer(inp.to(original_weight.dtype))
                
                # Restore original weight
                layer.weight.data = original_weight
                
                # Compute relative error (sensitivity)
                diff = (original_output - quantized_output).float()
                orig_norm = original_output.float().norm() + 1e-8
                relative_error = diff.norm() / orig_norm
                
                total_sensitivity += relative_error.item()
                num_samples += 1
        
        return total_sensitivity / max(num_samples, 1)
    
    def profile_model(
        self,
        model: nn.Module,
        calibration_texts: Optional[List[str]] = None,
        tokenizer = None,
        num_random_samples: int = 32,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Profile all linear layers in a model.
        
        Args:
            model: The model to profile
            calibration_texts: Optional list of text strings for calibration
            tokenizer: Tokenizer (required if calibration_texts provided)
            num_random_samples: Number of random samples if no calibration texts
            exclude_patterns: Layer name patterns to exclude
            
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        if exclude_patterns is None:
            exclude_patterns = ['embed', 'lm_head', 'norm', 'ln_', 'layernorm']
        
        sensitivities = {}
        
        # Generate sample inputs
        sample_inputs = self._generate_sample_inputs(
            model, calibration_texts, tokenizer, num_random_samples
        )
        
        # Profile each linear layer
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            # Check exclusions
            name_lower = name.lower()
            if any(pat.lower() in name_lower for pat in exclude_patterns):
                continue
            
            try:
                # Generate appropriately sized inputs for this layer
                layer_inputs = [
                    torch.randn(4, module.in_features, dtype=torch.float16, device=self.device)
                    for _ in range(min(8, num_random_samples))
                ]
                
                sensitivity = self.profile_layer(module, layer_inputs)
                sensitivities[name] = sensitivity
            except Exception as e:
                print(f"Warning: Could not profile layer {name}: {e}")
                sensitivities[name] = 0.5  # Default mid-range sensitivity
        
        return sensitivities
    
    def _generate_sample_inputs(
        self,
        model: nn.Module,
        calibration_texts: Optional[List[str]],
        tokenizer,
        num_random_samples: int
    ) -> List[torch.Tensor]:
        """Generate sample inputs for calibration."""
        inputs = []
        
        if calibration_texts and tokenizer:
            for text in calibration_texts[:16]:  # Limit samples
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs.append(tokens.input_ids)
        
        # Always add some random samples
        # Get hidden size from model config or first linear layer
        hidden_size = 768  # Default
        for module in model.modules():
            if isinstance(module, nn.Linear):
                hidden_size = module.in_features
                break
        
        for _ in range(num_random_samples):
            inputs.append(torch.randn(1, hidden_size, dtype=torch.float16))
        
        return inputs


def compute_sensitivity_scores(
    model: nn.Module,
    calibration_texts: Optional[List[str]] = None,
    tokenizer = None,
    block_size: int = 128,
    outlier_threshold: float = 3.0,
    cache_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute sensitivity scores for all linear layers in a model.
    
    This is the main entry point for sensitivity profiling.
    
    Args:
        model: Model to profile
        calibration_texts: Optional calibration texts
        tokenizer: Tokenizer for calibration texts
        block_size: Quantization block size
        outlier_threshold: Outlier detection threshold
        cache_path: Path to cache results (optional)
        
    Returns:
        Dictionary of layer_name -> sensitivity_score
    """
    # Check cache first
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
            print(f"Loaded cached sensitivity scores from {cache_path}")
            return cached
    
    # Profile
    profiler = LayerSensitivityProfiler(
        block_size=block_size,
        outlier_threshold=outlier_threshold
    )
    
    print("Profiling layer sensitivities...")
    sensitivities = profiler.profile_model(
        model,
        calibration_texts=calibration_texts,
        tokenizer=tokenizer
    )
    
    # Cache if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(sensitivities, f, indent=2)
        print(f"Cached sensitivity scores to {cache_path}")
    
    return sensitivities


def get_cache_path(model_name: str) -> str:
    """Generate a cache path for sensitivity scores."""
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    return f".atlasinfer_cache/{safe_name}_sensitivity.json"


def print_sensitivity_report(sensitivities: Dict[str, float], top_n: int = 10):
    """Print a human-readable sensitivity report."""
    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 60)
    print("Layer Sensitivity Report")
    print("=" * 60)
    print(f"{'Layer':<40} {'Sensitivity':>15}")
    print("-" * 60)
    
    for name, score in sorted_layers[:top_n]:
        # Truncate long names
        display_name = name if len(name) <= 38 else "..." + name[-35:]
        print(f"{display_name:<40} {score:>15.6f}")
    
    if len(sorted_layers) > top_n:
        print(f"... and {len(sorted_layers) - top_n} more layers")
    
    print("-" * 60)
    print(f"Total layers profiled: {len(sensitivities)}")
    print(f"Most sensitive: {sorted_layers[0][0]} ({sorted_layers[0][1]:.6f})")
    print(f"Least sensitive: {sorted_layers[-1][0]} ({sorted_layers[-1][1]:.6f})")
    print("=" * 60 + "\n")

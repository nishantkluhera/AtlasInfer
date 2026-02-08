"""
AtlasInfer Model Patcher - Non-invasive model modification utilities
"""
import torch
import torch.nn as nn
from typing import List, Optional, Set, Dict
import gc

from .quantizer import quantize_tensor
from .linear import QuantizedLinear, QuantizedLinear4bit, create_quantized_linear


def quantize_model(
    model: nn.Module,
    exclude_patterns: Optional[List[str]] = None,
    block_size: int = 128,
    outlier_threshold: float = 3.0,
    verbose: bool = True
) -> nn.Module:
    """
    Quantize all Linear layers in a model in-place (uniform FP8).
    
    This replaces nn.Linear layers with QuantizedLinear, reducing memory
    footprint while preserving accuracy through outlier detection.
    
    Args:
        model: Model to quantize (should be on CPU)
        exclude_patterns: Layer name patterns to exclude (default: embeddings, lm_head, norms)
        block_size: Block size for quantization
        outlier_threshold: Z-score threshold for outlier detection
        verbose: Print progress information
        
    Returns:
        The model with quantized layers (modified in-place)
    """
    if exclude_patterns is None:
        exclude_patterns = [
            'embed',      # Embedding layers (embed_tokens, embeddings, etc.)
            'lm_head',    # Output head
            'norm',       # Layer norms
            'ln_',        # Layer norms (GPT-2 style)
            'layernorm',  # Layer norms
        ]
    
    layers_quantized = 0
    layers_skipped = 0
    original_size = 0
    quantized_size = 0
    
    # Collect layers to replace (can't modify during iteration)
    replacements = []
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
            
        # Check exclusion patterns
        name_lower = name.lower()
        should_exclude = any(pattern.lower() in name_lower for pattern in exclude_patterns)
        
        if should_exclude:
            layers_skipped += 1
            continue
        
        # Find parent module and attribute name
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            attr_name = parts[0]
        else:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        
        replacements.append((parent, attr_name, module, name))
    
    # Perform replacements
    for parent, attr_name, linear_module, full_name in replacements:
        # Calculate original size
        orig_bytes = linear_module.weight.numel() * linear_module.weight.element_size()
        if linear_module.bias is not None:
            orig_bytes += linear_module.bias.numel() * linear_module.bias.element_size()
        original_size += orig_bytes
        
        # Create quantized replacement
        quantized_linear = QuantizedLinear.from_linear(
            linear_module,
            block_size=block_size,
            outlier_threshold=outlier_threshold
        )
        
        # Calculate quantized size
        quant_bytes = quantized_linear.quantized_weights.memory_bytes()
        if quantized_linear.bias is not None:
            quant_bytes += quantized_linear.bias.numel() * quantized_linear.bias.element_size()
        quantized_size += quant_bytes
        
        # Replace in model
        setattr(parent, attr_name, quantized_linear)
        
        # Clean up old module
        del linear_module
        layers_quantized += 1
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose:
        ratio = original_size / quantized_size if quantized_size > 0 else float('inf')
        print(f"Quantization complete:")
        print(f"  Layers quantized: {layers_quantized}")
        print(f"  Layers skipped: {layers_skipped}")
        print(f"  Original size: {original_size / (1024**2):.1f} MB")
        print(f"  Quantized size: {quantized_size / (1024**2):.1f} MB")
        print(f"  Compression ratio: {ratio:.2f}x")
    
    return model


def quantize_model_ladq(
    model: nn.Module,
    precision_allocation: Dict[str, str],
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Apply LADQ mixed-precision quantization based on allocation.
    
    This is the NOVEL CONTRIBUTION of AtlasInfer: each layer gets
    precision (FP16/FP8/FP4) based on its sensitivity score.
    
    Args:
        model: Model to quantize (should be on CPU)
        precision_allocation: Dict mapping layer names to precision strings
        exclude_patterns: Additional patterns to exclude
        verbose: Print progress
        
    Returns:
        Model with mixed-precision quantized layers
    """
    if exclude_patterns is None:
        exclude_patterns = ['embed', 'lm_head', 'norm', 'ln_', 'layernorm']
    
    stats = {'fp16': 0, 'fp8': 0, 'fp4': 0, 'skipped': 0}
    original_size = 0
    quantized_size = 0
    
    replacements = []
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        name_lower = name.lower()
        if any(pat.lower() in name_lower for pat in exclude_patterns):
            stats['skipped'] += 1
            continue
        
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            attr_name = parts[0]
        else:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        
        # Get precision from allocation (default to FP8)
        precision = precision_allocation.get(name, 'fp8').lower()
        
        replacements.append((parent, attr_name, module, name, precision))
    
    for parent, attr_name, linear_module, full_name, precision in replacements:
        orig_bytes = linear_module.weight.numel() * linear_module.weight.element_size()
        if linear_module.bias is not None:
            orig_bytes += linear_module.bias.numel() * linear_module.bias.element_size()
        original_size += orig_bytes
        
        # Create appropriate quantized layer
        new_layer = create_quantized_linear(linear_module, precision=precision)
        
        # Calculate new size
        if precision == 'fp16':
            quant_bytes = orig_bytes
        elif hasattr(new_layer, 'quantized_weights'):
            quant_bytes = new_layer.quantized_weights.memory_bytes()
            if new_layer.bias is not None:
                quant_bytes += new_layer.bias.numel() * new_layer.bias.element_size()
        else:
            quant_bytes = orig_bytes
        
        quantized_size += quant_bytes
        
        setattr(parent, attr_name, new_layer)
        del linear_module
        stats[precision] += 1
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose:
        ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        print(f"LADQ Mixed-Precision Quantization:")
        print(f"  FP16 (high precision): {stats['fp16']} layers")
        print(f"  FP8 (medium precision): {stats['fp8']} layers")
        print(f"  FP4 (low precision): {stats['fp4']} layers")
        print(f"  Skipped: {stats['skipped']} layers")
        print(f"  Original: {original_size / (1024**2):.1f} MB → Quantized: {quantized_size / (1024**2):.1f} MB")
        print(f"  Compression: {ratio:.2f}x")
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about model layers and their types.
    
    Returns:
        Dictionary with counts and lists of layer types
    """
    info = {
        'total_modules': 0,
        'linear_count': 0,
        'quantized_linear_count': 0,
        'quantized_linear_4bit_count': 0,
        'other_count': 0,
        'linear_names': [],
        'quantized_names': [],
    }
    
    for name, module in model.named_modules():
        info['total_modules'] += 1
        
        if isinstance(module, QuantizedLinear4bit):
            info['quantized_linear_4bit_count'] += 1
            info['quantized_names'].append(name)
        elif isinstance(module, QuantizedLinear):
            info['quantized_linear_count'] += 1
            info['quantized_names'].append(name)
        elif isinstance(module, nn.Linear):
            info['linear_count'] += 1
            info['linear_names'].append(name)
        else:
            info['other_count'] += 1
    
    return info


def find_decoder_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """
    Find the decoder layer list in a model.
    
    Tries common patterns for popular model architectures.
    
    Returns:
        ModuleList of decoder layers, or None if not found
    """
    patterns = [
        'model.layers',           # Llama, Mistral, Gemma
        'model.decoder.layers',   # OPT
        'transformer.h',          # GPT-2, GPT-J
        'gpt_neox.layers',        # GPT-NeoX, Pythia
    ]
    
    for pattern in patterns:
        try:
            obj = model
            for attr in pattern.split('.'):
                obj = getattr(obj, attr)
            if isinstance(obj, (nn.ModuleList, list)):
                return obj
        except AttributeError:
            continue
    
    return None


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Get the primary dtype of model parameters."""
    for param in model.parameters():
        return param.dtype
    return torch.float32


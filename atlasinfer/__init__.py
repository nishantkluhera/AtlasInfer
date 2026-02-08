"""
AtlasInfer - Efficient LLM Inference with Layer-Adaptive Dynamic Quantization (LADQ)

Novel features:
- Sensitivity-based precision allocation (FP16/FP8/FP4)
- Memory-budget-aware quantization
- CPU offloading for large models
"""
__version__ = "2.0.0"

from .quantizer import (
    quantize_tensor, dequantize_tensor, QuantizedTensor,
    quantize_tensor_fp4, dequantize_tensor_fp4, QuantizedTensor4bit
)
from .linear import QuantizedLinear, QuantizedLinear4bit, create_quantized_linear
from .patcher import quantize_model, quantize_model_ladq, get_model_info
from .offload import setup_cpu_offload, CPUOffloadHook
from .sensitivity import LayerSensitivityProfiler, compute_sensitivity_scores
from .allocator import PrecisionAllocator, PrecisionLevel, get_layer_sizes
from .inference import AtlasInference

__all__ = [
    # Core quantization
    "quantize_tensor", "dequantize_tensor", "QuantizedTensor",
    "quantize_tensor_fp4", "dequantize_tensor_fp4", "QuantizedTensor4bit",
    # Layers
    "QuantizedLinear", "QuantizedLinear4bit", "create_quantized_linear",
    # Model patching
    "quantize_model", "quantize_model_ladq", "get_model_info",
    # Offloading
    "setup_cpu_offload", "CPUOffloadHook",
    # LADQ (Novel)
    "LayerSensitivityProfiler", "compute_sensitivity_scores",
    "PrecisionAllocator", "PrecisionLevel", "get_layer_sizes",
    # Inference
    "AtlasInference",
]

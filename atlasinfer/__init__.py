"""
AtlasInfer - mixed-precision LLM inference for consumer GPUs.

Quantizes each linear layer to FP16 / INT8 / INT4 based on its *measured*
sensitivity to quantization, spending a memory budget where it buys the most
accuracy. The precision assignment is solved exactly as a budget-constrained
knapsack over a per-layer error-vs-memory frontier.

Pipeline:
    1. quantizer  - block-wise INT8/INT4 with FP16 outlier preservation
    2. sensitivity - per-layer, per-bit-width error on real calibration activations
    3. allocator  - exact (DP) precision assignment within a memory budget
    4. patcher    - swap dense layers for quantized ones in-place
"""
__version__ = "2.1.0"

from .quantizer import (
    QuantizedTensor3bit, quantize_tensor_nf3, dequantize_tensor_nf3,
    quantize_tensor, dequantize_tensor, QuantizedTensor,
    quantize_tensor_fp4, dequantize_tensor_fp4, QuantizedTensor4bit,
    quantize_tensor_nf4, dequantize_tensor_nf4, NF4_LEVELS,
    compression_ratio,
)
from .linear import (
    QuantizedLinear, QuantizedLinear3bit, QuantizedLinear4bit,
    create_quantized_linear,
)
from .patcher import quantize_model, quantize_model_mixed, get_model_info
from .offload import setup_cpu_offload, CPUOffloadHook, estimate_model_memory
from .sensitivity import (
    SensitivityProfiler, LayerProfile, print_sensitivity_report,
)
from .allocator import (
    PrecisionLevel, AllocationResult, get_layer_sizes,
    allocate_optimal, allocate_greedy, uniform_allocation,
    estimate_memory_usage, print_allocation_report,
)
from .triton_kernels import (
    HAS_TRITON, kernel_available, W8A16Linear, W4A16Linear,
    quantize_w8a16, quantize_w4a16,
)
from .gptq import gptq_quantize_nf4, quantize_model_gptq
from .double_quant import DoubleQuantScales, double_quantize, dq_relative_error
from .awq import quantize_model_awq, search_awq_scale
from .experimental.codebook import (
    VectorQuantizedTensor, quantize_tensor_vq, dequantize_tensor_vq,
)
from .reproducibility import seed_everything
from .inference import AtlasInference

__all__ = [
    # Quantization primitives
    "quantize_tensor", "dequantize_tensor", "QuantizedTensor",
    "quantize_tensor_fp4", "dequantize_tensor_fp4", "QuantizedTensor4bit",
    "quantize_tensor_nf4", "dequantize_tensor_nf4", "NF4_LEVELS",
    "compression_ratio",
    # Layers
    "QuantizedLinear", "QuantizedLinear3bit", "QuantizedLinear4bit",
    "create_quantized_linear",
    # Patching
    "quantize_model", "quantize_model_mixed", "get_model_info",
    # Offloading
    "setup_cpu_offload", "CPUOffloadHook", "estimate_model_memory",
    # Sensitivity profiling
    "SensitivityProfiler", "LayerProfile", "print_sensitivity_report",
    # Allocation
    "PrecisionLevel", "AllocationResult", "get_layer_sizes",
    "allocate_optimal", "allocate_greedy", "uniform_allocation",
    "QuantizedTensor3bit", "quantize_tensor_nf3", "dequantize_tensor_nf3",
    "estimate_memory_usage", "print_allocation_report",
    # Fused kernel
    "HAS_TRITON", "kernel_available", "W8A16Linear", "W4A16Linear",
    "quantize_w8a16", "quantize_w4a16",
    # GPTQ
    "gptq_quantize_nf4", "quantize_model_gptq",
    # Double-quantized scales
    "DoubleQuantScales", "double_quantize", "dq_relative_error",
    # AWQ
    "quantize_model_awq", "search_awq_scale",
    # Experimental sub-4-bit vector-quantized codebook (not validated at scale)
    "VectorQuantizedTensor", "quantize_tensor_vq", "dequantize_tensor_vq",
    # Reproducibility
    "seed_everything",
    # Inference
    "AtlasInference",
]

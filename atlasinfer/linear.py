"""
AtlasInfer QuantizedLinear - Drop-in replacement for nn.Linear with quantized weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .quantizer import (
    QuantizedTensor, dequantize_tensor,
    QuantizedTensor4bit, dequantize_tensor_fp4
)


class QuantizedLinear(nn.Module):
    """
    A Linear layer that uses FP8-quantized weights, dequantizing on-the-fly.
    
    This is the standard precision tier for most layers.
    """
    
    def __init__(
        self,
        quantized_weights: QuantizedTensor,
        bias: Optional[torch.Tensor] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None
    ):
        super().__init__()
        self.quantized_weights = quantized_weights
        self.precision = "fp8"
        
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        if in_features is None or out_features is None:
            shape = quantized_weights.original_shape
            self.out_features = shape[0]
            self.in_features = shape[1] if len(shape) > 1 else shape[0]
        else:
            self.in_features = in_features
            self.out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_tensor(self.quantized_weights, device=x.device)
        return F.linear(x, weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, precision={self.precision}'
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 128,
        outlier_threshold: float = 3.0
    ) -> 'QuantizedLinear':
        from .quantizer import quantize_tensor
        
        weight_cpu = linear.weight.data.cpu()
        quantized_weights = quantize_tensor(
            weight_cpu,
            block_size=block_size,
            outlier_threshold=outlier_threshold
        )
        bias = linear.bias.data.clone() if linear.bias is not None else None
        
        return cls(
            quantized_weights=quantized_weights,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features
        )


class QuantizedLinear4bit(nn.Module):
    """
    A Linear layer that uses FP4-quantized weights for maximum compression.
    
    Used for less sensitive layers in LADQ allocation.
    """
    
    def __init__(
        self,
        quantized_weights: QuantizedTensor4bit,
        bias: Optional[torch.Tensor] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None
    ):
        super().__init__()
        self.quantized_weights = quantized_weights
        self.precision = "fp4"
        
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        if in_features is None or out_features is None:
            shape = quantized_weights.original_shape
            self.out_features = shape[0]
            self.in_features = shape[1] if len(shape) > 1 else shape[0]
        else:
            self.in_features = in_features
            self.out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_tensor_fp4(self.quantized_weights, device=x.device)
        return F.linear(x, weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, precision={self.precision}'
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 64,
        outlier_threshold: float = 2.5
    ) -> 'QuantizedLinear4bit':
        from .quantizer import quantize_tensor_fp4
        
        weight_cpu = linear.weight.data.cpu()
        quantized_weights = quantize_tensor_fp4(
            weight_cpu,
            block_size=block_size,
            outlier_threshold=outlier_threshold
        )
        bias = linear.bias.data.clone() if linear.bias is not None else None
        
        return cls(
            quantized_weights=quantized_weights,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features
        )


def create_quantized_linear(
    linear: nn.Linear,
    precision: str = "fp8",
    block_size_fp8: int = 128,
    block_size_fp4: int = 64,
    outlier_threshold_fp8: float = 3.0,
    outlier_threshold_fp4: float = 2.5
) -> Union[QuantizedLinear, QuantizedLinear4bit, nn.Linear]:
    """
    Factory function to create the appropriate quantized linear layer.
    
    Args:
        linear: Source nn.Linear layer
        precision: Target precision ("fp16", "fp8", "fp4")
        block_size_fp8: Block size for FP8 quantization
        block_size_fp4: Block size for FP4 quantization
        outlier_threshold_fp8: Outlier threshold for FP8
        outlier_threshold_fp4: Outlier threshold for FP4
        
    Returns:
        Appropriate quantized layer (or original for fp16)
    """
    precision = precision.lower()
    
    if precision == "fp16":
        # Keep original layer
        return linear
    elif precision == "fp8":
        return QuantizedLinear.from_linear(
            linear,
            block_size=block_size_fp8,
            outlier_threshold=outlier_threshold_fp8
        )
    elif precision == "fp4":
        return QuantizedLinear4bit.from_linear(
            linear,
            block_size=block_size_fp4,
            outlier_threshold=outlier_threshold_fp4
        )
    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'fp16', 'fp8', or 'fp4'.")


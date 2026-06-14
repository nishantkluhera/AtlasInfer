"""
AtlasInfer QuantizedLinear - Drop-in replacements for nn.Linear with quantized weights.

The quantized tensor components are stored as registered buffers so that the
standard ``module.to(device)`` / ``model.cuda()`` machinery moves them with the
rest of the model. This avoids re-copying the packed weights from CPU to GPU on
every forward pass (the most expensive mistake an offline-quantized layer can
make).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .quantizer import (
    QuantizedTensor, dequantize_tensor,
    QuantizedTensor4bit, dequantize_tensor_fp4,
)


class QuantizedLinear(nn.Module):
    """Linear layer backed by 8-bit integer weights, dequantized on the fly.

    This is the medium-precision tier used by the allocator.
    """

    def __init__(
        self,
        quantized_weights: QuantizedTensor,
        bias: Optional[torch.Tensor] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.precision = "int8"

        # Store packed components as buffers so .to(device) moves them and they
        # live persistently on the compute device (no per-forward host copy).
        self.register_buffer("q_data", quantized_weights.fp8_data)
        self.register_buffer("q_scales", quantized_weights.scales)
        self.register_buffer("q_outlier_indices", quantized_weights.outlier_indices)
        self.register_buffer("q_outlier_values", quantized_weights.outlier_values)
        # Shape / block-size are plain Python metadata, not tensors.
        self._original_shape = quantized_weights.original_shape
        self._block_size = quantized_weights.block_size

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        if in_features is None or out_features is None:
            shape = quantized_weights.original_shape
            self.out_features = shape[0]
            self.in_features = shape[1] if len(shape) > 1 else shape[0]
        else:
            self.in_features = in_features
            self.out_features = out_features

    @property
    def quantized_weights(self) -> QuantizedTensor:
        """Reconstruct the QuantizedTensor view over the registered buffers."""
        return QuantizedTensor(
            fp8_data=self.q_data,
            scales=self.q_scales,
            outlier_indices=self.q_outlier_indices,
            outlier_values=self.q_outlier_values,
            original_shape=self._original_shape,
            block_size=self._block_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dtype-transparent: dequantize/compute in the weight dtype, but return
        # output in the input's dtype so quantized layers compose with dense
        # (e.g. fp32) layers in the surrounding model.
        in_dtype = x.dtype
        weight = dequantize_tensor(self.quantized_weights, device=x.device)
        bias = self.bias.to(weight.dtype) if self.bias is not None else None
        out = F.linear(x.to(weight.dtype), weight, bias)
        return out.to(in_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, precision={self.precision}"
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 128,
        outlier_threshold: float = 3.0,
    ) -> "QuantizedLinear":
        from .quantizer import quantize_tensor

        weight_cpu = linear.weight.data.cpu()
        quantized_weights = quantize_tensor(
            weight_cpu,
            block_size=block_size,
            outlier_threshold=outlier_threshold,
        )
        bias = linear.bias.data.clone() if linear.bias is not None else None

        return cls(
            quantized_weights=quantized_weights,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )


class QuantizedLinear4bit(nn.Module):
    """Linear layer backed by packed 4-bit integer weights for maximum compression.

    Used for the least sensitive layers in a mixed-precision allocation.
    """

    def __init__(
        self,
        quantized_weights: QuantizedTensor4bit,
        bias: Optional[torch.Tensor] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        scheme: str = "int4",
    ):
        super().__init__()
        # scheme: "int4" (symmetric [-7,7]) or "nf4" (NormalFloat codebook).
        self.scheme = scheme
        self.precision = scheme

        self.register_buffer("q_packed", quantized_weights.packed_data)
        self.register_buffer("q_scales", quantized_weights.scales)
        self.register_buffer("q_outlier_indices", quantized_weights.outlier_indices)
        self.register_buffer("q_outlier_values", quantized_weights.outlier_values)
        self._original_shape = quantized_weights.original_shape
        self._block_size = quantized_weights.block_size
        self._num_elements = quantized_weights.num_elements

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        if in_features is None or out_features is None:
            shape = quantized_weights.original_shape
            self.out_features = shape[0]
            self.in_features = shape[1] if len(shape) > 1 else shape[0]
        else:
            self.in_features = in_features
            self.out_features = out_features

    @property
    def quantized_weights(self) -> QuantizedTensor4bit:
        return QuantizedTensor4bit(
            packed_data=self.q_packed,
            scales=self.q_scales,
            outlier_indices=self.q_outlier_indices,
            outlier_values=self.q_outlier_values,
            original_shape=self._original_shape,
            block_size=self._block_size,
            num_elements=self._num_elements,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .quantizer import dequantize_tensor_nf4
        in_dtype = x.dtype
        if self.scheme == "nf4":
            weight = dequantize_tensor_nf4(self.quantized_weights, device=x.device)
        else:
            weight = dequantize_tensor_fp4(self.quantized_weights, device=x.device)
        bias = self.bias.to(weight.dtype) if self.bias is not None else None
        out = F.linear(x.to(weight.dtype), weight, bias)
        return out.to(in_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, precision={self.precision}"
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 64,
        outlier_threshold: float = 2.5,
        scheme: str = "int4",
    ) -> "QuantizedLinear4bit":
        from .quantizer import quantize_tensor_fp4, quantize_tensor_nf4

        weight_cpu = linear.weight.data.cpu()
        quantizer = quantize_tensor_nf4 if scheme == "nf4" else quantize_tensor_fp4
        quantized_weights = quantizer(
            weight_cpu,
            block_size=block_size,
            outlier_threshold=outlier_threshold,
        )
        bias = linear.bias.data.clone() if linear.bias is not None else None

        return cls(
            quantized_weights=quantized_weights,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
            scheme=scheme,
        )


def _conv1d_to_linear(conv1d: nn.Module) -> nn.Linear:
    """Convert a HuggingFace Conv1D (GPT-2 style) into an equivalent nn.Linear."""
    # Conv1D stores weight as (in_features, out_features); nn.Linear wants (out, in).
    weight = conv1d.weight.data.t().contiguous()
    bias = conv1d.bias.data if conv1d.bias is not None else None

    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    linear.weight.data = weight
    if bias is not None:
        linear.bias.data = bias
    return linear


def create_quantized_linear(
    linear: nn.Linear,
    precision: str = "int8",
    block_size_int8: int = 128,
    block_size_int4: int = 64,
    outlier_threshold_int8: float = 3.0,
    outlier_threshold_int4: float = 2.5,
    use_kernel: bool = False,
    quant_4bit: str = "nf4",
) -> Union[QuantizedLinear, QuantizedLinear4bit, nn.Linear]:
    """Create the appropriate quantized layer for the requested precision.

    Args:
        linear: Source ``nn.Linear`` or HuggingFace ``Conv1D`` layer.
        precision: ``"fp16"`` (keep dense), ``"int8"``, or ``"int4"``.
            Legacy aliases ``"fp8"`` -> ``int8`` and ``"fp4"`` -> ``int4`` are accepted.
        use_kernel: if True, INT8/INT4 layers use the fused-kernel modules
            (``W8A16Linear`` / ``W4A16Linear``; per-channel, Triton-accelerated on
            CUDA) instead of the block-wise + outlier eager layers.
        quant_4bit: 4-bit scheme for the eager path - ``"nf4"`` (NormalFloat,
            default, better for Gaussian weights) or ``"int4"`` (symmetric).

    Returns:
        A quantized layer, or the (possibly converted) dense layer for fp16.
    """
    precision = precision.lower()
    # Backwards-compatible aliases from the old (misleading) FP8/FP4 naming.
    precision = {"fp8": "int8", "fp4": "int4"}.get(precision, precision)

    # Normalise Conv1D into nn.Linear up front.
    if type(linear).__name__ == "Conv1D":
        linear = _conv1d_to_linear(linear)

    if precision == "fp16":
        return linear
    elif precision == "int8":
        if use_kernel:
            from .triton_kernels import W8A16Linear
            return W8A16Linear.from_linear(linear)
        return QuantizedLinear.from_linear(
            linear,
            block_size=block_size_int8,
            outlier_threshold=outlier_threshold_int8,
        )
    elif precision == "int4":
        if use_kernel:
            from .triton_kernels import W4A16Linear
            return W4A16Linear.from_linear(linear)
        return QuantizedLinear4bit.from_linear(
            linear,
            block_size=block_size_int4,
            outlier_threshold=outlier_threshold_int4,
            scheme=quant_4bit,
        )
    else:
        raise ValueError(
            f"Unknown precision: {precision!r}. Use 'fp16', 'int8', or 'int4'."
        )

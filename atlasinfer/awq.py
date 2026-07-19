"""
AWQ — Activation-aware Weight Quantization (Lin et al., 2023) on the NF4 path.

Plain block-wise NF4 treats every weight column the same. AWQ's observation: a
small set of input channels carry most of the activation magnitude, and the
weights multiplying them dominate the layer's output — so quantization error on
*those* columns matters far more. Instead of keeping them in FP16 (what
AtlasInfer's mixed-precision path does), AWQ **scales the salient columns up**
before quantization and scales the corresponding activations down by the same
factor, so those weights occupy more of the low-bit grid and round more finely.
The two scalings cancel exactly in exact arithmetic — ``(W·diag(s))·(x/s) = W·x``
— so the only effect is where the quantizer spends its resolution.

Per input channel ``j`` the scale is ``s_j = (mean_t |x_{t,j}|)^alpha``,
normalized to unit mean, with ``alpha in [0,1]`` grid-searched per layer to
minimize the layer's quantized output MSE on calibration activations (alpha=0 →
no scaling = plain NF4; alpha=1 → full activation-proportional scaling). The
scaled weight ``W·diag(s)`` is stored in the ordinary block-NF4 format and the
layer keeps ``s`` as an ``in_scale`` buffer that divides the input at run time
(see ``QuantizedLinear4bit``). This is the same accuracy tier as GPTQ; the two are
complementary (AWQ reshapes what's quantized, GPTQ compensates the residual).
"""
import gc
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .quantizer import quantize_tensor_nf4, dequantize_tensor_nf4
from .linear import QuantizedLinear4bit, _conv1d_to_linear
from .patcher import _collect_targets, DEFAULT_EXCLUDE
from .sensitivity import SensitivityProfiler

# Grid of scaling exponents searched per layer. 0.0 is plain NF4 (no scaling), so
# AWQ can never do worse than NF4 on the calibration proxy — it's a safe floor.
DEFAULT_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


@torch.no_grad()
def _capture_awq_stats(model, targets, calib_batches, max_rows: int = 128):
    """Per-target: (mean |activation| per input channel, a sample of input rows)."""
    absmean: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {name: 0 for name in targets}
    rows: Dict[str, list] = {name: [] for name in targets}
    row_counts: Dict[str, int] = {name: 0 for name in targets}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0]
            if x is None:
                return
            x = x.reshape(-1, x.shape[-1]).float()
            a = x.abs().sum(dim=0)
            absmean[name] = a if name not in absmean else absmean[name] + a
            counts[name] += x.shape[0]
            if row_counts[name] < max_rows:
                take = x[: max_rows - row_counts[name]].cpu()
                rows[name].append(take)
                row_counts[name] += take.shape[0]
        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    dev = next(model.parameters()).device
    try:
        for ids in calib_batches:
            model(ids.to(dev))
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name in targets:
        if name in absmean and counts[name] > 0 and rows[name]:
            out[name] = (absmean[name] / counts[name], torch.cat(rows[name], dim=0))
    return out


@torch.no_grad()
def search_awq_scale(
    W: torch.Tensor, act_absmean: torch.Tensor, x_rows: torch.Tensor,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> torch.Tensor:
    """Grid-search the per-input-channel AWQ scale that minimizes NF4 output MSE.

    Args:
        W: (out_features, in_features) float weight, on the compute device.
        act_absmean: (in_features,) mean |activation| per input channel.
        x_rows: (R, in_features) sample of real input rows (proxy for the MSE).
    Returns:
        s: (in_features,) positive per-channel scale, unit mean.
    """
    dev = W.device
    act = act_absmean.to(dev).clamp(min=1e-8)
    salience = act / act.mean().clamp(min=1e-8)          # unitless per-channel salience
    x = x_rows.to(dev, dtype=W.dtype)
    ref = x @ W.t()                                       # exact output on the sample
    ref_norm = ref.pow(2).mean().clamp(min=1e-12)

    best_s = torch.ones_like(act)
    best_err = None
    for alpha in alphas:
        s = salience.pow(alpha)
        s = (s / s.mean().clamp(min=1e-8)).clamp(min=1e-2, max=1e2)
        Ws = (W * s.unsqueeze(0))                         # amplify salient columns
        qWs = dequantize_tensor_nf4(quantize_tensor_nf4(Ws.cpu())).to(dev).to(W.dtype)
        out = (x / s) @ qWs.t()
        err = (out - ref).pow(2).mean() / ref_norm
        if best_err is None or err < best_err:
            best_err, best_s = err, s
    return best_s


@torch.no_grad()
def quantize_model_awq(
    model: nn.Module,
    tokenizer=None,
    calibration_texts: Optional[List[str]] = None,
    nsamples: int = 128,
    seqlen: int = 512,
    block_size: int = 64,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    double_quant: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Quantize all eligible linear layers to NF4 with AWQ activation-aware scaling.

    One calibration pass captures each layer's input statistics; then each layer's
    scale exponent is searched and the scaled weight is quantized to block-NF4 with
    a runtime ``in_scale``. ``double_quant`` additionally compresses the block
    scales (see double_quant.py), so AWQ composes with the memory trick too.
    """
    exclude_patterns = list(exclude_patterns or DEFAULT_EXCLUDE)
    targets = _collect_targets(model, exclude_patterns)
    target_mods = {name: mod for (_, _, mod, name) in targets}

    batches = SensitivityProfiler(max_samples=nsamples, seq_len=seqlen)._build_calibration_batches(
        model, tokenizer, calibration_texts
    )
    if verbose:
        print(f"AWQ: capturing activation stats for {len(targets)} layers "
              f"({len(batches)} calibration batches)...")
    stats = _capture_awq_stats(model, target_mods, batches)

    n_awq = n_fallback = 0
    for parent, attr, module, name in targets:
        linear = _conv1d_to_linear(module) if type(module).__name__ == "Conv1D" else module
        ldev = linear.weight.device
        W = linear.weight.data.float()
        bias = linear.bias.data.clone() if linear.bias is not None else None

        if name in stats and W.shape[1] % block_size == 0:
            act_absmean, x_rows = stats[name]
            s = search_awq_scale(W.to(ldev), act_absmean, x_rows, alphas)
            Ws = (W.to(ldev) * s.unsqueeze(0))
            qt = quantize_tensor_nf4(Ws.cpu(), block_size=block_size, double_quant=double_quant)
            layer = QuantizedLinear4bit(
                qt.to(ldev), bias=bias, in_features=linear.in_features,
                out_features=linear.out_features, scheme="nf4",
                in_scale=s.to(ldev).to(torch.float16),
            )
            n_awq += 1
        else:  # no stats (layer never fired) or unaligned in_features -> plain NF4
            qt = quantize_tensor_nf4(W.cpu(), block_size=block_size, double_quant=double_quant)
            layer = QuantizedLinear4bit(
                qt.to(ldev), bias=bias, in_features=linear.in_features,
                out_features=linear.out_features, scheme="nf4",
            )
            n_fallback += 1
        setattr(parent, attr, layer)
        del module

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(f"AWQ NF4 complete: {n_awq} layers activation-scaled, "
              f"{n_fallback} fell back to plain NF4.")
    return model

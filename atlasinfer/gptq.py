"""
GPTQ-style error-compensated quantization on top of AtlasInfer's NF4 codebook.

Plain NF4 rounds each weight to the nearest codebook level independently. GPTQ
(Frantar et al., 2022) does better at the same bit-width: it quantizes the weight
matrix one input-column at a time and, after each column, updates the *remaining*
columns to cancel the error that column just introduced — weighting the
correction by the inverse Hessian of the layer's input activations
(H = sum_x x x^T) so it fixes the dimensions that actually matter for the output.

The result is stored in AtlasInfer's existing block-NF4 format (a
``QuantizedTensor4bit`` whose per-block scales are exactly the per-(output-channel,
input-group) scales GPTQ uses), so the unchanged ``QuantizedLinear4bit(scheme=
"nf4")`` decodes it. Layers whose in_features isn't a multiple of ``group_size``
fall back to plain NF4 (rare for transformers).
"""
import gc

import torch
import torch.nn as nn

from .quantizer import NF4_LEVELS, QuantizedTensor4bit, quantize_tensor_nf4, _find_outliers
from .linear import QuantizedLinear4bit, _conv1d_to_linear
from .patcher import _collect_targets, DEFAULT_EXCLUDE
from .sensitivity import SensitivityProfiler
from .double_quant import double_quantize


def _in_features(module: nn.Module) -> int:
    if hasattr(module, "in_features"):
        return module.in_features
    return module.weight.shape[0]  # Conv1D stores (in_features, out_features)


def _nf4_codes(norm: torch.Tensor) -> torch.Tensor:
    """Nearest NF4 code (0..15) for normalized values in ~[-1, 1]."""
    boundaries = ((NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2).to(norm.device)
    return torch.bucketize(norm, boundaries).clamp_(0, 15)


@torch.no_grad()
def gptq_quantize_nf4(W: torch.Tensor, H: torch.Tensor, group_size: int = 64,
                      percdamp: float = 0.01, outlier_threshold: float = 2.5,
                      double_quant: bool = False) -> QuantizedTensor4bit:
    """GPTQ-compensated NF4 quantization with sparse-outlier preservation.

    Combines three things: (1) per-(output-channel, input-group) NF4 scales
    computed *excluding* outliers, (2) the few high-magnitude outlier weights kept
    exact in FP16 (sparse), and (3) GPTQ error compensation on the rest — each
    column's quantization error is propagated to the not-yet-quantized columns via
    the inverse-Hessian operator. Output is AtlasInfer's block-NF4 format.

    Args:
        W: (out_features, in_features) float weight.
        H: (in_features, in_features) Hessian = sum over calibration tokens of x x^T.
        group_size: input-group size for per-group scales (must divide in_features).
        percdamp: Hessian diagonal damping as a fraction of mean(diag(H)).
        outlier_threshold: per-group z-score above which a weight is kept in FP16.
    """
    dev = W.device
    W = W.clone().float()
    out, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"in_features {cols} not divisible by group_size {group_size}")
    num_groups = cols // group_size

    # Outlier mask on the original weights, per (row, group). Uses the same
    # robust median/MAD detector as the block quantizer so a cluster of large
    # weights in a group can't inflate its spread and mask one another (see
    # quantizer._find_outliers).
    omask = _find_outliers(W.reshape(-1, group_size), outlier_threshold).reshape(out, cols)

    H = H.clone().float().to(dev)
    diag = torch.arange(cols, device=dev)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    # Upper-triangular Cholesky factor of H^{-1}; escalate damping if ill-conditioned.
    mean_diag = torch.mean(torch.diag(H)).clamp(min=1e-8)
    Hinv = None
    for damp in (percdamp, percdamp * 5, percdamp * 25, 0.1):
        try:
            Hd = H.clone()
            Hd[diag, diag] += damp * mean_diag
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except torch.linalg.LinAlgError:
            continue
    if Hinv is None:
        raise torch.linalg.LinAlgError("Hessian not factorable even with heavy damping")

    nf4 = NF4_LEVELS.to(dev)
    codes = torch.zeros(out, cols, dtype=torch.uint8, device=dev)
    scales = torch.zeros(out, num_groups, device=dev)
    o_rows, o_cols, o_vals = [], [], []

    # Block-wise GPTQ: process one group (= block) at a time. Within the block the
    # error is applied column-by-column (cheap, only group_size wide); the error's
    # effect on all *later* columns is applied once per block as a single matmul.
    # This is mathematically identical to the per-column form but ~group_size x
    # fewer large updates -> practical at 7B+ instead of hours.
    for g in range(num_groups):
        i1 = g * group_size
        i2 = i1 + group_size
        grp = W[:, i1:i2].clone()
        grp[omask[:, i1:i2]] = 0.0                         # scale ignores outliers
        cur_scale = grp.abs().amax(dim=1).clamp(min=1e-8)
        scales[:, g] = cur_scale

        W1 = W[:, i1:i2].clone()                           # local working copy
        Hinv1 = Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for j in range(group_size):
            col = i1 + j
            w = W1[:, j]
            d = Hinv1[j, j]
            c = _nf4_codes(w / cur_scale)
            codes[:, col] = c.to(torch.uint8)
            q = nf4[c] * cur_scale
            om = omask[:, col]
            if bool(om.any()):                             # keep outliers exact
                q = q.clone()
                q[om] = w[om]
                rows = om.nonzero(as_tuple=True)[0]
                o_rows.append(rows)
                o_cols.append(torch.full_like(rows, col))
                o_vals.append(w[om])
            err = (w - q) / d                              # zero at outlier rows
            W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err
        if i2 < cols:                                      # propagate to later cols
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    codes_flat = codes.reshape(-1)
    if codes_flat.numel() % 2:
        codes_flat = torch.nn.functional.pad(codes_flat, (0, 1))
    packed = ((codes_flat[0::2] << 4) | (codes_flat[1::2] & 0x0F)).to(torch.int8)

    if o_rows:
        r = torch.cat(o_rows); c2 = torch.cat(o_cols)
        oidx = (r * cols + c2).to(torch.int32)
        oval = torch.cat(o_vals).to(torch.float16)
    else:
        oidx = torch.empty(0, dtype=torch.int32, device=dev)
        oval = torch.empty(0, dtype=torch.float16, device=dev)

    block_scales = scales.reshape(-1).to(torch.float32).to(dev)
    scales_dq = None
    if double_quant:
        scales_dq = double_quantize(block_scales).to(dev)
        block_scales = torch.empty(0, dtype=torch.float32, device=dev)  # not resident

    return QuantizedTensor4bit(
        packed_data=packed.to(dev),
        scales=block_scales,
        outlier_indices=oidx.to(dev),
        outlier_values=oval.to(dev),
        original_shape=torch.Size([out, cols]),
        block_size=group_size,
        num_elements=out * cols,
        scales_dq=scales_dq,
    )


@torch.no_grad()
def _compute_hessians(model, targets, calib_batches):
    """Accumulate H = sum_x x x^T (on CPU) for each target layer's input."""
    Hs = {}
    handles = []

    def make_hook(name):
        def hook(_mod, inp, _out):
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).float()
            xtx = (x.t() @ x).cpu()
            Hs[name] = xtx if name not in Hs else Hs[name] + xtx
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_hook(make_hook(name)))
    model.eval()
    dev = next(model.parameters()).device
    try:
        for ids in calib_batches:
            model(ids.to(dev))
    finally:
        for h in handles:
            h.remove()
    return Hs


def quantize_model_gptq(
    model: nn.Module,
    tokenizer=None,
    calibration_texts=None,
    group_size: int = 64,
    nsamples: int = 128,
    seqlen: int = 512,
    hessian_budget_gb: float = 4.0,
    exclude_patterns=None,
    double_quant: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Quantize all eligible linear layers to NF4 with GPTQ error compensation.

    Layers are processed in memory-bounded *chunks*: the Hessians for a chunk are
    accumulated, those layers quantized and replaced, then the Hessians freed
    before the next chunk. This keeps peak memory ~``hessian_budget_gb`` instead
    of holding every layer's Hessian at once (which OOMs on larger models), and
    has a bonus: later chunks' Hessians are measured with earlier chunks already
    quantized -> true sequential GPTQ, which compensates for accumulated error.

    GPTQ is data-hungry: a rank-deficient Hessian gives unreliable compensation,
    so use many, longer calibration sequences (``nsamples`` x ``seqlen``).
    """
    exclude_patterns = list(exclude_patterns or DEFAULT_EXCLUDE)
    all_targets = _collect_targets(model, exclude_patterns)

    batches = SensitivityProfiler(max_samples=nsamples, seq_len=seqlen)._build_calibration_batches(
        model, tokenizer, calibration_texts
    )

    # Greedily pack layers into chunks under the Hessian memory budget
    # (H is in_features^2 * 4 bytes).
    budget = hessian_budget_gb * 1e9
    chunks, cur, cur_bytes = [], [], 0.0
    for t in all_targets:
        hb = _in_features(t[2]) ** 2 * 4
        if cur and cur_bytes + hb > budget:
            chunks.append(cur)
            cur, cur_bytes = [], 0.0
        cur.append(t)
        cur_bytes += hb
    if cur:
        chunks.append(cur)

    if verbose:
        print(f"GPTQ: {len(all_targets)} layers in {len(chunks)} chunk(s) "
              f"(<= {hessian_budget_gb:g} GB Hessians each), "
              f"{len(batches)} calibration batches...")

    n_gptq = n_fallback = 0
    for ci, chunk in enumerate(chunks):
        chunk_mods = {name: mod for (_, _, mod, name) in chunk}
        Hs = _compute_hessians(model, chunk_mods, batches)
        for parent, attr, module, name in chunk:
            linear = _conv1d_to_linear(module) if type(module).__name__ == "Conv1D" else module
            ldev = linear.weight.device  # keep each layer on its own device (multi-GPU safe)
            W = linear.weight.data.float()
            bias = linear.bias.data.clone() if linear.bias is not None else None
            qt = None
            if W.shape[1] % group_size == 0 and name in Hs:
                try:
                    qt = gptq_quantize_nf4(W, Hs[name].to(ldev), group_size=group_size,
                                           double_quant=double_quant)
                    n_gptq += 1
                except torch.linalg.LinAlgError:
                    qt = None
            if qt is None:  # unaligned in_features or unfactorable Hessian -> plain NF4
                qt = quantize_tensor_nf4(W.cpu(), block_size=group_size,
                                         double_quant=double_quant)
                n_fallback += 1
            setattr(parent, attr, QuantizedLinear4bit(
                qt.to(ldev), bias=bias,
                in_features=linear.in_features, out_features=linear.out_features,
                scheme="nf4",
            ))
            Hs.pop(name, None)
        del Hs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"  chunk {ci + 1}/{len(chunks)} done ({len(chunk)} layers)")

    if verbose:
        print(f"GPTQ NF4 quantization complete: {n_gptq} layers compensated, "
              f"{n_fallback} fell back to plain NF4.")
    return model

"""
AtlasInfer Sensitivity Profiler.

Estimates how much each linear layer's output is perturbed when its weights are
quantized to a given bit-width. Unlike a naive analysis that feeds random noise
through each layer in isolation, this profiler captures the *real* activations
each layer sees on a small calibration set (via forward hooks) and measures the
relative output error those activations actually produce under quantization:

    error(layer, bits) = || W·x - dequant(quant(W, bits))·x || / || W·x ||

averaged over the captured activations. These per-layer, per-bit-width errors are
the input to the precision allocator (see ``allocator.py``), which spends the
memory budget where it buys the most accuracy.
"""
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .linear import create_quantized_linear

# Default precisions the profiler measures (besides the implicit lossless fp16).
DEFAULT_PRECISIONS = ("int8", "int4")

# Linear layers we never quantize (numerically delicate / tiny relative to model).
DEFAULT_EXCLUDE = ("embed", "lm_head", "norm", "ln_", "layernorm")

# Generic English calibration text used when the caller supplies none.
_FALLBACK_CALIBRATION = [
    "The history of science is the study of how human understanding of the "
    "natural world has changed over time.",
    "In a distributed system, consistency, availability, and partition "
    "tolerance cannot all be fully guaranteed at once.",
    "She walked along the shoreline at dawn, watching the tide pull the last "
    "of the night back out to sea.",
    "Compound interest is the addition of interest to the principal sum of a "
    "loan or deposit, so that interest also earns interest.",
    "def quicksort(items): return items if len(items) <= 1 else "
    "quicksort([x for x in items[1:] if x < items[0]])",
    "The mitochondria is often described as the powerhouse of the cell because "
    "it produces most of the cell's supply of ATP.",
    "Markets tend to overreact to short-term news while underreacting to slow, "
    "structural shifts in the underlying economy.",
    "Whether or not the treaty would hold depended less on its wording than on "
    "the willingness of each party to honour it.",
]


def _is_linear_layer(module: nn.Module) -> bool:
    """True for nn.Linear and HuggingFace Conv1D (GPT-2 style) layers."""
    if isinstance(module, nn.Linear):
        return True
    return type(module).__name__ == "Conv1D"


def _linear_in_features(module: nn.Module) -> Optional[int]:
    """Best-effort input feature count for a linear-like layer."""
    if hasattr(module, "in_features"):
        return module.in_features
    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() == 2:
        # Conv1D weight is (in_features, out_features).
        return weight.shape[0]
    return None


def _param_count(module: nn.Module) -> int:
    n = module.weight.numel()
    if getattr(module, "bias", None) is not None:
        n += module.bias.numel()
    return n


@dataclass
class LayerProfile:
    """Per-bit-width quantization error for one layer."""
    name: str
    param_count: int
    # precision label -> mean relative output error on calibration activations.
    errors: Dict[str, float] = field(default_factory=dict)

    def sensitivity(self, reference: str = "int4") -> float:
        """A single scalar sensitivity score (default: error under the most
        aggressive precision)."""
        if reference in self.errors:
            return self.errors[reference]
        return max(self.errors.values()) if self.errors else 0.0


class SensitivityProfiler:
    """Measures real-activation quantization error for every linear layer."""

    def __init__(
        self,
        precisions: tuple = DEFAULT_PRECISIONS,
        exclude_patterns: tuple = DEFAULT_EXCLUDE,
        max_rows: int = 64,
        max_samples: int = 8,
        seq_len: int = 128,
    ):
        """
        Args:
            precisions: Precision labels to measure (e.g. ``("int8", "int4")``).
            exclude_patterns: Substrings of layer names to skip.
            max_rows: Cap on captured activation rows per layer (controls cost).
            max_samples: Number of calibration sequences to run.
            seq_len: Max tokens per calibration sequence.
        """
        self.precisions = tuple(precisions)
        self.exclude_patterns = tuple(p.lower() for p in exclude_patterns)
        self.max_rows = max_rows
        self.max_samples = max_samples
        self.seq_len = seq_len

    def _target_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        targets = {}
        for name, module in model.named_modules():
            if not _is_linear_layer(module):
                continue
            if any(pat in name.lower() for pat in self.exclude_patterns):
                continue
            targets[name] = module
        return targets

    def _capture_activations(
        self,
        model: nn.Module,
        targets: Dict[str, nn.Module],
        input_batches: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run calibration data through the model, recording the input each
        target layer actually receives (capped at ``max_rows`` rows)."""
        captured: Dict[str, List[torch.Tensor]] = {name: [] for name in targets}
        counts: Dict[str, int] = {name: 0 for name in targets}
        handles = []

        def make_hook(name):
            def hook(_module, inputs, _output):
                if counts[name] >= self.max_rows:
                    return
                x = inputs[0]
                if x is None:
                    return
                rows = x.reshape(-1, x.shape[-1]).detach()
                remaining = self.max_rows - counts[name]
                rows = rows[:remaining]
                captured[name].append(rows.to(torch.float16).cpu())
                counts[name] += rows.shape[0]
            return hook

        for name, module in targets.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        model_device = next(model.parameters()).device
        try:
            model.eval()
            with torch.no_grad():
                for input_ids in input_batches:
                    model(input_ids.to(model_device))
                    if all(c >= self.max_rows for c in counts.values()):
                        break
        finally:
            for h in handles:
                h.remove()

        return {
            name: torch.cat(chunks, dim=0)
            for name, chunks in captured.items()
            if chunks
        }

    def _measure_layer(
        self,
        module: nn.Module,
        activations: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """Relative output error of ``module`` at each target precision."""
        errors: Dict[str, float] = {}
        act = activations.to(device)

        with torch.no_grad():
            ref_module = module.to(device)
            ref_out = ref_module(act.to(next(ref_module.parameters()).dtype)).float()
            ref_norm = ref_out.norm() + 1e-8

            for precision in self.precisions:
                qlayer = create_quantized_linear(deepcopy(module), precision=precision)
                qlayer = qlayer.to(device)
                q_out = qlayer(act).float()
                rel = ((ref_out - q_out).norm() / ref_norm).item()
                errors[precision] = rel
                del qlayer

        return errors

    def profile(
        self,
        model: nn.Module,
        tokenizer=None,
        calibration_texts: Optional[List[str]] = None,
    ) -> Dict[str, LayerProfile]:
        """Profile every quantizable linear layer in ``model``.

        Returns a mapping ``layer_name -> LayerProfile``.
        """
        targets = self._target_layers(model)
        input_batches = self._build_calibration_batches(
            model, tokenizer, calibration_texts
        )

        activations = self._capture_activations(model, targets, input_batches)

        device = next(model.parameters()).device
        profiles: Dict[str, LayerProfile] = {}
        for name, module in targets.items():
            if name not in activations:
                continue
            try:
                errors = self._measure_layer(module, activations[name], device)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Warning: could not profile layer {name}: {exc}")
                errors = {p: 0.5 for p in self.precisions}
            profiles[name] = LayerProfile(
                name=name, param_count=_param_count(module), errors=errors
            )
        return profiles

    @staticmethod
    def _parent_and_attr(model: nn.Module, name: str):
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            return model, parts[0]
        return model.get_submodule(parts[0]), parts[1]

    @torch.no_grad()
    def _calibration_loss(self, model, batches, device) -> float:
        """Mean next-token cross-entropy (NLL) over the calibration batches."""
        total, n = 0.0, 0
        for input_ids in batches:
            ids = input_ids.to(device)
            loss = model(ids, labels=ids).loss
            total += float(loss)
            n += 1
        return total / max(n, 1)

    def profile_end_to_end(
        self,
        model: nn.Module,
        tokenizer=None,
        calibration_texts: Optional[List[str]] = None,
    ) -> Dict[str, LayerProfile]:
        """Profile each layer by its *actual effect on the model's output loss*.

        For every layer and candidate precision, the layer is temporarily swapped
        for its quantized version, the calibration cross-entropy is re-measured,
        and the error is recorded as the increase over the FP16 baseline:

            error(layer, bits) = NLL(model with only this layer quantized) - NLL_fp16

        This end-to-end signal tracks perplexity far better than a layer-local
        output-error proxy, at the cost of one forward pass per (layer, precision).
        """
        targets = self._target_layers(model)
        batches = self._build_calibration_batches(model, tokenizer, calibration_texts)
        device = next(model.parameters()).device
        model.eval()

        base_nll = self._calibration_loss(model, batches, device)

        profiles: Dict[str, LayerProfile] = {}
        for name, module in targets.items():
            parent, attr = self._parent_and_attr(model, name)
            errors: Dict[str, float] = {}
            try:
                for precision in self.precisions:
                    qlayer = create_quantized_linear(deepcopy(module), precision=precision).to(device)
                    setattr(parent, attr, qlayer)
                    nll = self._calibration_loss(model, batches, device)
                    errors[precision] = max(0.0, nll - base_nll)
                    setattr(parent, attr, module)  # restore original
                    del qlayer
            except Exception as exc:  # pragma: no cover - defensive
                setattr(parent, attr, module)
                print(f"Warning: could not profile layer {name}: {exc}")
                errors = {p: 0.5 for p in self.precisions}
            profiles[name] = LayerProfile(
                name=name, param_count=_param_count(module), errors=errors
            )
        return profiles

    def _build_calibration_batches(
        self,
        model: nn.Module,
        tokenizer,
        calibration_texts: Optional[List[str]],
    ) -> List[torch.Tensor]:
        """Tokenize calibration text into a list of input_id tensors."""
        texts = calibration_texts or _FALLBACK_CALIBRATION
        texts = texts[: self.max_samples]

        if tokenizer is not None:
            batches = []
            for text in texts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.seq_len,
                )
                batches.append(enc["input_ids"])
            return batches

        # No tokenizer: fall back to random token ids within the embedding range.
        vocab = getattr(getattr(model, "config", None), "vocab_size", 1000)
        gen = torch.Generator().manual_seed(0)
        return [
            torch.randint(0, vocab, (1, self.seq_len), generator=gen)
            for _ in range(self.max_samples)
        ]


def compute_sensitivity_scores(
    model: nn.Module,
    calibration_texts: Optional[List[str]] = None,
    tokenizer=None,
    precisions: tuple = DEFAULT_PRECISIONS,
    reference: str = "int4",
    cache_path: Optional[str] = None,
) -> Dict[str, float]:
    """Convenience wrapper returning a single scalar sensitivity per layer.

    The scalar is the relative output error under ``reference`` precision and is
    consumed by the greedy allocator. Results are cached to ``cache_path`` as
    full per-precision error tables when provided.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        print(f"Loaded cached sensitivity scores from {cache_path}")
        return {name: errs.get(reference, max(errs.values()))
                for name, errs in cached.items()}

    profiler = SensitivityProfiler(precisions=precisions)
    print("Profiling layer sensitivities on calibration data...")
    profiles = profiler.profile(model, tokenizer, calibration_texts)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({n: p.errors for n, p in profiles.items()}, f, indent=2)
        print(f"Cached sensitivity scores to {cache_path}")

    return {n: p.sensitivity(reference) for n, p in profiles.items()}


def get_cache_path(model_name: str) -> str:
    """Cache path for a model's sensitivity profile."""
    safe = model_name.replace("/", "_").replace("\\", "_")
    return f".atlasinfer_cache/{safe}_sensitivity.json"


def print_sensitivity_report(sensitivities: Dict[str, float], top_n: int = 10):
    """Print a human-readable ranking of the most sensitive layers."""
    ranked = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("Layer Sensitivity Report")
    print("=" * 60)
    print(f"{'Layer':<40} {'Rel. error':>15}")
    print("-" * 60)
    for name, score in ranked[:top_n]:
        display = name if len(name) <= 38 else "..." + name[-35:]
        print(f"{display:<40} {score:>15.6f}")
    if len(ranked) > top_n:
        print(f"... and {len(ranked) - top_n} more layers")
    print("-" * 60)
    print(f"Total layers profiled: {len(sensitivities)}")
    if ranked:
        print(f"Most sensitive:  {ranked[0][0]} ({ranked[0][1]:.6f})")
        print(f"Least sensitive: {ranked[-1][0]} ({ranked[-1][1]:.6f})")
    print("=" * 60 + "\n")


# Backwards-compatible alias for the previous public class name.
LayerSensitivityProfiler = SensitivityProfiler

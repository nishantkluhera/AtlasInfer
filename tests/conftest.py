"""Pytest bootstrap.

Force a clean CUDA context init with a tiny warmup op *before* any test module
imports ``atlasinfer`` (and, transitively, ``transformers``). On some Windows
torch builds, importing transformers before a CUDA context exists races the lazy
init and hard-crashes with an access violation (0xC0000005) — the same gotcha the
benchmark harnesses guard against. No-op on CPU / CI (Linux, CPU-only torch).
"""
import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

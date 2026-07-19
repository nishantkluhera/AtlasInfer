"""Pytest bootstrap.

Force a clean CUDA context init with a tiny warmup op *before* any test module
imports ``atlasinfer`` (and, transitively, ``transformers``). On some Windows
torch builds, importing transformers before a CUDA context exists races the lazy
init and hard-crashes with an access violation (0xC0000005) — the same gotcha the
benchmark harnesses guard against. No-op on CPU / CI (Linux, CPU-only torch).
"""
import os
import sys

import torch

# Repo root on sys.path so tests can import the top-level harnesses (benchmark,
# eval_downstream, ...) without each file re-deriving it — and without depending
# on some other test module having inserted it first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if torch.cuda.is_available():
    try:
        torch.zeros(1, device="cuda")
    except Exception as exc:  # noqa: BLE001
        # A present-but-unusable GPU (OOM, claimed by another process, driver
        # mismatch) must not abort collection: most of the suite is CPU-only and
        # should still run. The CUDA-only tests skip themselves on failure.
        print(f"conftest: CUDA warmup skipped ({type(exc).__name__}: {exc})")

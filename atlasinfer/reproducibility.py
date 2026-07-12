"""Determinism helpers shared by every benchmark/eval harness.

Quantization error, calibration sampling, and perplexity all depend on RNG state
(calibration document selection, GPTQ's Hessian sampling, dropout-free but
CUDA-nondeterministic matmuls). Seeding once at the top of each harness makes a
reported number reproducible run-to-run on the same hardware, which is the whole
point of a benchmark table someone else is meant to trust.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = True) -> int:
    """Seed Python/NumPy/Torch RNGs (and, if present, CUDA) for reproducibility.

    Args:
        seed: the seed to apply everywhere.
        deterministic: also pin cuDNN to deterministic kernels and disable the
            autotuner. Costs a little speed; makes CUDA matmuls repeatable. Left
            on by default for benchmarks (correctness of the number > throughput);
            pass ``False`` for latency benchmarks where you want the fast kernels.

    Returns:
        The seed, so callers can log exactly what was used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS GEMMs on CUDA >= 10.2; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed

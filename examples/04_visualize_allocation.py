"""
Visualize how AtlasInfer allocates precision across a model's layers.

Profiles each layer's end-to-end sensitivity, solves the budget allocation, then
plots every linear layer as a point (depth vs. sensitivity) coloured by the
precision it was assigned. The picture makes the method's logic concrete: the
most loss-sensitive layers are kept at higher precision, the robust ones dropped
to INT4.

    python examples/04_visualize_allocation.py --model EleutherAI/pythia-410m --bits 5

Requires the benchmark extra (matplotlib): pip install -e ".[benchmark]"
"""
import argparse
import os
import re

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from atlasinfer import SensitivityProfiler, allocate_optimal

hf_logging.set_verbosity_error()

PRECISION_COLOR = {"fp16": "#2ca02c", "int8": "#1f77b4", "int4": "#d62728"}


def _depth_key(name: str) -> int:
    """Sort layers by the block index embedded in their name (else stable 0)."""
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="EleutherAI/pythia-410m")
    ap.add_argument("--bits", type=float, default=5.0, help="target avg bits/weight")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()

    print("Profiling end-to-end sensitivity (this runs many forward passes)...")
    profiles = SensitivityProfiler().profile_end_to_end(model, tokenizer=tok)

    n_params = sum(p.param_count for p in profiles.values())
    alloc = allocate_optimal(profiles, budget_bytes=int(n_params * args.bits / 8))
    print(f"  {alloc.summary()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(profiles, key=lambda n: (_depth_key(n), n))
    xs = list(range(len(names)))
    ys = [profiles[n].sensitivity("int4") for n in names]
    colors = [PRECISION_COLOR.get(alloc.allocations[n], "#999") for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys, c=colors, s=42, edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Linear layer (ordered by depth)")
    ax.set_ylabel("End-to-end sensitivity (loss increase if INT4)")
    ax.set_title(
        f"Per-layer precision allocation — {args.model} "
        f"(avg {alloc.avg_bits:.1f} bits)"
    )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c,
                   markeredgecolor="black", markersize=8,
                   label=f"{p.upper()} ({alloc.counts.get(p, 0)})")
        for p, c in PRECISION_COLOR.items()
    ]
    ax.legend(handles=handles, title="assigned precision")
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()

    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    path = os.path.join(args.out, f"allocation_{safe}.png")
    fig.savefig(path, dpi=130)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()

"""
Phase-2 gate experiment: does the knapsack allocation actually do anything?

Answers two pre-registered kill criteria at once (see PAPER/01_go_nogo.md):

  K5 (attribution) — at a MATCHED memory budget, compare
        knapsack DP  vs  greedy sensitivity-ranking  vs  random-at-matched-memory
     If the DP doesn't beat the others by more than seed noise, the multiple-choice
     knapsack formulation is not the contribution.

  K3 (iso-memory) — a mixed allocation over {fp16, int8, int4} can never be
     cheaper than uniform int4, so "mixed vs uniform-NF4 at equal memory" is
     unmeasurable by construction. The honest baseline at an INTERMEDIATE budget
     is what a practitioner would otherwise do: quantize a RANDOM subset of layers
     to INT8 and the rest to NF4. That is exactly the `random` arm here. So the
     real question is whether the mixed curve sits below the random-allocation
     curve, not below uniform-NF4.

Everything is written to structured JSON — no number here is meant to be
transcribed by hand.

    python PAPER/exp/allocator_ablation.py --model Qwen/Qwen2.5-0.5B
"""
import argparse
import gc
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

if torch.cuda.is_available():  # see benchmark.py: Windows CUDA init order
    torch.zeros(1, device="cuda")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from atlasinfer import seed_everything  # noqa: E402
from atlasinfer.allocator import BYTES_PER_PARAM, allocate_greedy, allocate_optimal  # noqa: E402
from atlasinfer.patcher import quantize_model, quantize_model_mixed  # noqa: E402
from atlasinfer.sensitivity import SensitivityProfiler  # noqa: E402
from benchmark import evaluate_perplexity, load_wikitext, model_weight_bytes  # noqa: E402

hf_logging.set_verbosity_error()


def random_allocation(profiles, budget_bytes, rng):
    """Control arm: spend the budget on a RANDOM subset of layers.

    Start every layer at int4 (the cheapest tier), then walk layers in random
    order upgrading int4->int8->fp16 while the budget allows. Structurally
    identical to `allocate_greedy` except the ordering carries no information,
    so the difference between them is exactly what the sensitivity ranking buys,
    and the difference from `allocate_optimal` is what the DP buys on top.
    """
    order = ["int4", "int8", "fp16"]
    alloc = {n: "int4" for n in profiles}
    total = sum(int(p.param_count * BYTES_PER_PARAM["int4"]) for p in profiles.values())

    names = list(profiles)
    rng.shuffle(names)
    for name in names:
        pc = profiles[name].param_count
        while True:
            cur = order.index(alloc[name])
            if cur + 1 >= len(order):
                break
            nxt = order[cur + 1]
            delta = int(pc * BYTES_PER_PARAM[nxt]) - int(pc * BYTES_PER_PARAM[alloc[name]])
            if total + delta > budget_bytes:
                break
            total += delta
            alloc[name] = nxt
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--eval-tokens", type=int, default=30000)
    ap.add_argument("--bits", type=float, nargs="+", default=[4.25, 4.5, 5.0, 6.0])
    ap.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="PAPER/exp/results")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model}  device={device}  seed={args.seed}  "
          f"eval_tokens={args.eval_tokens}")

    tok = AutoTokenizer.from_pretrained(args.model)
    calib, eval_text = load_wikitext()
    cfg = AutoConfig.from_pretrained(args.model)
    max_len = min(getattr(cfg, "max_position_embeddings", 1024) or 1024, 1024)

    def load():
        return AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, low_cpu_mem_usage=True)

    rows = []

    def measure(arm, budget_bits, model, extra=None):
        model = model.to(device).eval()
        t0 = time.time()
        ppl = evaluate_perplexity(model, tok, eval_text, device, max_len=max_len,
                                  stride=max_len // 2, max_tokens=args.eval_tokens)
        mb = model_weight_bytes(model) / 1024 ** 2
        dt = time.time() - t0
        row = {"arm": arm, "target_bits": budget_bits, "ppl": ppl, "mb": mb,
               "seconds": round(dt, 1)}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {arm:<22} bits={budget_bits!s:<5} ppl={ppl:8.4f}  {mb:8.1f} MB  ({dt:.0f}s)")
        model.to("cpu"); del model; gc.collect(); torch.cuda.empty_cache()
        return row

    # ---- anchors -----------------------------------------------------------
    print("\n[anchors]")
    measure("fp16", 16.0, load())
    measure("uniform-int8", 8.0, quantize_model(load(), precision="int8", verbose=False))
    measure("uniform-nf4", 4.0,
            quantize_model(load(), precision="int4", quant_4bit="nf4", verbose=False))

    # ---- profile once ------------------------------------------------------
    print("\n[profiling end-to-end sensitivities once]")
    t0 = time.time()
    base = load().to(device).eval()
    profiles = SensitivityProfiler().profile_end_to_end(
        base, tokenizer=tok, calibration_texts=calib)
    base.to("cpu"); del base; gc.collect(); torch.cuda.empty_cache()
    print(f"  profiled {len(profiles)} layers in {time.time()-t0:.0f}s")

    n_params = sum(p.param_count for p in profiles.values())
    sens = {n: p.sensitivity("int4") for n, p in profiles.items()}
    sizes = {n: p.param_count for n, p in profiles.items()}

    # ---- the ablation ------------------------------------------------------
    for bits in args.bits:
        budget = int(n_params * bits / 8)
        print(f"\n[budget {bits} bits = {budget/1024**2:.1f} MB over profiled layers]")

        a = allocate_optimal(profiles, budget_bytes=budget)
        measure("knapsack-dp", bits,
                quantize_model_mixed(load(), allocation=a.allocations, verbose=False),
                {"counts": a.counts, "alloc_bytes": a.total_bytes,
                 "predicted_error": a.predicted_error})

        g = allocate_greedy(sens, sizes, budget)
        measure("greedy-sensitivity", bits,
                quantize_model_mixed(load(), allocation=g.allocations, verbose=False),
                {"counts": g.counts, "alloc_bytes": g.total_bytes})

        for rs in args.random_seeds:
            rng = random.Random(rs)
            ra = random_allocation(profiles, budget, rng)
            counts = {}
            for p in ra.values():
                counts[p] = counts.get(p, 0) + 1
            measure(f"random-s{rs}", bits,
                    quantize_model_mixed(load(), allocation=ra, verbose=False),
                    {"counts": counts, "random_seed": rs})

    # ---- write -------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    payload = {
        "model": args.model,
        "eval_tokens": args.eval_tokens,
        "seed": args.seed,
        "max_len": max_len,
        "n_profiled_layers": len(profiles),
        "n_profiled_params": n_params,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "rows": rows,
    }
    path = os.path.join(args.out, f"ablation_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

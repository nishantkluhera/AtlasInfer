"""
The iso-memory experiment: does mixed precision beat uniform 4-bit at EQUAL
measured memory?

This was previously unmeasurable. With tiers {fp16, int8, int4} the allocator's
cheapest option IS uniform int4, so at uniform-NF4's budget the DP simply returns
uniform NF4 and the comparison is vacuous (PAPER/01_go_nogo.md 2b). Adding the
3-bit NF3 tier removes that floor: robust layers drop to 3 bits, the freed budget
buys INT8 on the fragile ones, and the whole allocation can land at or below
uniform NF4's footprint.

Method
------
Memory is compared on MEASURED megabytes, not nominal bits -- the nominal budget
excludes per-block scales and sparse outliers, so a "4-bit" model really costs
~4.4 bits. The script sweeps nominal budgets, measures the real footprint and
perplexity of each, and reports the mixed point that lands at or under uniform
NF4's measured MB. Any point at LOWER memory and LOWER perplexity is a strict
Pareto win and needs no interpolation to defend.

    python PAPER/exp/iso_memory.py --model Qwen/Qwen2.5-0.5B
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from atlasinfer import seed_everything  # noqa: E402
from atlasinfer.allocator import allocate_greedy, allocate_optimal  # noqa: E402
from atlasinfer.evaluation import (  # noqa: E402
    evaluate_perplexity, load_wikitext, model_weight_bytes, quantized_bits_per_weight,
)
from atlasinfer.patcher import quantize_model, quantize_model_mixed  # noqa: E402
from atlasinfer.sensitivity import SensitivityProfiler  # noqa: E402

hf_logging.set_verbosity_error()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--eval-tokens", type=int, default=30000)
    ap.add_argument("--bits", type=float, nargs="+",
                    default=[3.4, 3.6, 3.8, 4.0, 4.25],
                    help="nominal budgets to sweep; the interesting ones land at "
                         "or below uniform-NF4's MEASURED footprint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="PAPER/exp/results")
    args = ap.parse_args()

    seed_everything(args.seed)
    assert torch.cuda.is_available(), "needs CUDA"
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    calib, eval_text = load_wikitext()
    cfg = AutoConfig.from_pretrained(args.model)
    max_len = min(getattr(cfg, "max_position_embeddings", 1024) or 1024, 1024)
    print(f"model={args.model}  eval_tokens={args.eval_tokens}  seed={args.seed}\n")

    def load():
        return AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, low_cpu_mem_usage=True)

    rows = []

    def measure(arm, model, nominal_bits, extra=None):
        model = model.to(dev).eval()
        t0 = time.time()
        ppl = evaluate_perplexity(model, tok, eval_text, dev, max_len=max_len,
                                  stride=max_len // 2, max_tokens=args.eval_tokens)
        mb = model_weight_bytes(model) / 1024 ** 2
        real_bits = quantized_bits_per_weight(model)
        row = {"arm": arm, "nominal_bits": nominal_bits, "ppl": ppl, "mb": mb,
               "measured_bits_per_weight": real_bits, "seconds": round(time.time() - t0, 1)}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {arm:<26} ppl={ppl:8.4f}  {mb:8.1f} MB  "
              f"({real_bits:.2f} real bits/wt)")
        model.to("cpu"); del model; gc.collect(); torch.cuda.empty_cache()
        return row

    print("[anchors]")
    fp16_row = measure("fp16", load(), 16.0)
    nf4_row = measure("uniform-nf4", quantize_model(
        load(), precision="int4", quant_4bit="nf4", verbose=False), 4.0)
    measure("uniform-nf3", quantize_model(load(), precision="int3", verbose=False), 3.0)

    print("\n[profiling end-to-end sensitivities once (int8/int4/int3)]")
    t0 = time.time()
    base = load().to(dev).eval()
    # This is the experiment that studies the sub-4-bit tier, so it opts into int3
    # explicitly (it is NOT a library default — see allocator.allocate_optimal).
    profiles = SensitivityProfiler(precisions=("int8", "int4", "int3")).profile_end_to_end(
        base, tokenizer=tok, calibration_texts=calib)
    base.to("cpu"); del base; gc.collect(); torch.cuda.empty_cache()
    n_params = sum(p.param_count for p in profiles.values())
    print(f"  profiled {len(profiles)} layers in {time.time()-t0:.0f}s")

    sens = {n: p.sensitivity("int4") for n, p in profiles.items()}
    sizes = {n: p.param_count for n, p in profiles.items()}

    for bits in args.bits:
        budget = int(n_params * bits / 8)
        print(f"\n[nominal {bits} bits]")
        a = allocate_optimal(profiles, budget_bytes=budget,
                             precisions=("fp16", "int8", "int4", "int3"))
        measure(f"knapsack-{bits}", quantize_model_mixed(
            load(), allocation=a.allocations, verbose=False), bits, {"counts": a.counts})
        g = allocate_greedy(sens, sizes, budget, profiles=profiles,
                            precisions=("fp16", "int8", "int4", "int3"))
        measure(f"greedy-{bits}", quantize_model_mixed(
            load(), allocation=g.allocations, verbose=False), bits, {"counts": g.counts})

    # ---- the verdict --------------------------------------------------------
    base_ppl, target_mb = fp16_row["ppl"], nf4_row["mb"]
    nf4_delta = nf4_row["ppl"] - base_ppl
    print(f"\n{'='*72}")
    print(f"ISO-MEMORY vs uniform NF4 ({target_mb:.1f} MB, delta {nf4_delta:+.4f})")
    print(f"{'='*72}")
    verdict = []
    for r in rows:
        if not r["arm"].startswith(("knapsack", "greedy")):
            continue
        if r["mb"] > target_mb:            # only strict wins count
            continue
        d = r["ppl"] - base_ppl
        cut = (nf4_delta - d) / nf4_delta * 100 if nf4_delta > 0 else float("nan")
        verdict.append({**r, "delta_vs_fp16": d, "pct_cut_vs_uniform_nf4": cut,
                        "mb_vs_uniform_nf4": r["mb"] - target_mb})
        print(f"  {r['arm']:<26} {r['mb']:8.1f} MB ({r['mb']-target_mb:+6.1f}) "
              f"delta {d:+.4f}  -> cuts the 4-bit penalty {cut:5.1f}%")
    if not verdict:
        print("  (no allocation landed at or below uniform NF4's footprint)")

    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    path = os.path.join(args.out, f"iso_memory_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "eval_tokens": args.eval_tokens, "seed": args.seed,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "fp16_ppl": base_ppl,
            "uniform_nf4": {"mb": target_mb, "ppl": nf4_row["ppl"], "delta": nf4_delta},
            "iso_memory_or_better": verdict,
            "rows": rows,
        }, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

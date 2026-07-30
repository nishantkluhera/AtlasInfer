# results/

**Committed evidence.** Every number quoted in the top-level `README.md` comes
from a file in this directory, and `tests/test_readme_consistency.py` fails CI if
any of them drift apart. Nothing here should be edited by hand.

## What belongs here

| Pattern | Produced by | Contents |
| --- | --- | --- |
| `<model>.{json,md,png}` | `benchmark.py` | perplexity/memory sweep, uniform + mixed |
| `comparison_<model>.{json,md}` | `compare_baselines.py` | head-to-head vs bnb/GPTQ/AWQ |
| `downstream_<model>.{json,md}` | `eval_downstream.py` | zero-shot task accuracy |
| `latency_<model>.{json,md}` | `bench_latency.py` | decode tok/s + peak GPU memory |
| `triton_kernel_<gpu>.json` | `bench_triton_kernel.py` | fused-kernel microbenchmark |
| `allocation_<model>.png` | `examples/04_visualize_allocation.py` | per-layer precision figure |

**The `.json` is the source of truth.** The `.md` is a rendered view for humans
and the `.png` a figure; both are regenerated from the same run. If a `.md` and
its `.json` disagree, the `.json` is right and the `.md` is stale.

## What does NOT belong here

- **`results/_smoke/`** — short-eval preflight runs (`run_lightning.sh setup`).
  Deliberately separate so a 2k-token smoke can never overwrite a committed
  full-length result for the same model. Not tracked.
- **`results/_logs/`** — raw stdout from runs. Useful locally, not tracked.
- **`PAPER/exp/results/`** — one-off experiments supporting the analysis in
  `PAPER/`. Kept apart because they use non-standard settings (shorter evals,
  ablation-only arms) and are *not* the headline benchmark numbers.

## Regenerating

```bash
python reproduce.py --readme
```

Runs exactly the models the README tables were generated from. Any other model
list will regenerate only some files, and the consistency test will then fail on
the ones that moved — which is the intended behaviour, not a bug.

Note the numbers depend on the GPU and library versions; each `.json` records the
GPU, torch version, seed and eval-token count that produced it. Compare like with
like.

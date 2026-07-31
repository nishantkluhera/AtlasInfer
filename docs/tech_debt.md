# Technical Debt Audit — AtlasInfer

Audit date: 2026-07-30, at commit `e85bc2e`. Scope: first-party code only
(`atlasinfer/` 3,437 lines + harnesses 1,250 lines). Excludes the claim and
reproducibility issues already fixed today (see `PAPER/00_audit.md`).

**Priority = (Impact + Risk) × (6 − Effort)**, each scored 1–5.

---

## Summary

The codebase is in better shape than most projects this size — the quantization
math is well-commented, the non-obvious decisions carry their rationale inline,
and a CI test guards the README against drift. The debt is concentrated in three
places: **duplicated layer-selection logic that can silently mis-allocate**,
**zero test coverage on the public API**, and **an architectural split between the
fast path and the accurate path** that no amount of tidying will fix cheaply.

| # | Item | Type | I | R | E | **Pri** |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Silent fallback when an allocation key doesn't match a layer | Code | 3 | 5 | 1 | **40** |
| 2 | Layer-selection logic triplicated across 3 modules | Code | 3 | 4 | 1 | **35** |
| 3 | `inference.py` (421 lines, the public API) has zero tests | Test | 3 | 4 | 2 | **28** |
| 4 | Harnesses import `benchmark.py` as a library | Arch | 3 | 3 | 2 | **24** |
| 5 | `main()` functions 84–275 lines, untestable | Arch | 4 | 3 | 3 | **21** |
| 6 | **Fast path ≠ accurate path (two quantizers)** | Arch | 5 | 5 | 4 | **20** |
| 7 | `test_readme_consistency` parses markdown, not JSON | Test | 2 | 2 | 1 | **20** |
| 8 | `autoawq` is deprecated and version-pinned to old transformers | Dep | 2 | 3 | 2 | **20** |
| 9 | `codebook.py` ships unvalidated in the public package | Doc | 2 | 2 | 1 | **20** |
| 10 | `offload.py` has no dedicated tests | Test | 2 | 4 | 3 | **18** |
| 11 | Tuned magic constants are neither configurable nor swept | Code | 2 | 2 | 2 | **16** |
| 12 | README is ~600 lines doing four jobs | Doc | 3 | 2 | 3 | **15** |
| 13 | GPU kernel tests run in no automated environment | Test | 3 | 4 | 4 | **14** |
| 14 | `results/` mixes committed evidence with build output | Infra | 2 | 1 | 2 | **12** |
| 15 | `inference.py` style drift (33 trailing-whitespace lines) | Code | 1 | 1 | 1 | **10** |
| 16 | Orphaned `.pyc` from deleted KV-cache modules; dup `.gitignore` lines | Infra | 1 | 1 | 1 | **10** |

**Note on the formula:** it ranks #6 at 20 because the fix is expensive. On
Impact+Risk alone it is the top item in the repo (10/10). Don't let the score
hide that — see the discussion below.

---

## P1 — Correctness hazards (fix before the paid 7B run)

### 1. Silent fallback when an allocation key doesn't match a layer — Pri 40

`patcher.py:119`:

```python
precision = allocation.get(full_name, default_precision).lower()
```

If a layer name in the allocation dict doesn't match a name the patcher walks,
that layer is **silently quantized to `default_precision` (`"int8"`)** with no
warning. The allocator's careful per-layer decision is discarded and replaced
with a guess, and nothing anywhere reports it.

This is latent today because the profiler and the patcher happen to select the
same layers — but see #2 for why that is not guaranteed, and note the failure is
invisible: you get plausible-looking perplexity from an allocation you did not
compute. On a 7B cloud run, that is hours spent measuring the wrong thing.

**Fix (15 min):** count unmatched keys and unallocated targets; warn loudly, and
report both counts in the returned stats.

**Justification:** cheapest possible insurance against silently invalid results
on paid hardware.

### 2. Layer-selection logic triplicated — Pri 35

Three independent copies of "which layers are quantizable," which must agree or
#1 fires:

| Location | Form |
| --- | --- |
| `patcher.py:19` | `DEFAULT_EXCLUDE = ["embed", "lm_head", "norm", "ln_", "layernorm"]` (list) |
| `sensitivity.py:51` | `DEFAULT_EXCLUDE = ("embed", ...)` (**tuple**) |
| `benchmark.py:75` | `exclude=("embed", "lm_head", "norm", "ln_")` (**4 entries, not 5**) |

Plus `_is_linear_layer` duplicated verbatim in `patcher.py:22` and
`sensitivity.py:74`, and `_param_count`/`_dense_bytes` as near-duplicates.

The `benchmark.py` copy is already missing `"layernorm"`. It happens to be
harmless because `"norm"` is a substring of `"layernorm"` — i.e. this has already
drifted once and was saved by luck.

**Fix (30 min):** one `atlasinfer/_targets.py` exporting `DEFAULT_EXCLUDE`,
`is_linear_layer()`, `param_count()`, `collect_targets()`. Import everywhere.

**Justification:** removes the drift that makes #1 dangerous. Both together are
under an hour and eliminate an entire class of silent-wrong-result bug.

---

## P2 — Test and structural debt (next 2 weeks)

### 3. `inference.py` has zero test coverage — Pri 28

421 lines, the package's public API (`AtlasInference`) and its console entry
point (`atlasinfer = "atlasinfer.inference:main"`). It is mentioned in **zero**
test files. Every other core module has one. It is also the most likely thing a
reader evaluating this repo will actually run.

**Fix (half a day):** `tests/test_inference.py` — construction with
`quantize=False`, `kernel="off"`, memory-budget path invokes the allocator,
`generate()` returns a string, CLI arg parsing. Use a tiny local `nn.Module`
where possible; mark anything needing a real HF download as `@pytest.mark.slow`.

### 4. Harnesses import `benchmark.py` as a library — Pri 24

```python
# compare_baselines.py:56, eval_downstream.py:55, PAPER/exp/allocator_ablation.py:48
from benchmark import evaluate_perplexity, load_wikitext, model_weight_bytes
```

A top-level script is the de facto shared library. Importing it executes module
scope: setting `HF_HUB_DISABLE_PROGRESS_BARS`, running a CUDA warmup allocation,
and calling `hf_logging.set_verbosity_error()` — side effects every consumer
inherits whether or not they want them. It also means `benchmark.py` can never be
refactored without breaking three callers, and requires the repo root on
`sys.path` (which `conftest.py` has to patch in).

**Fix (2 hours):** move `evaluate_perplexity`, `load_wikitext`,
`model_weight_bytes` into `atlasinfer/evaluation.py`. Have `benchmark.py` import
*from* it. Keeps the CLI working, makes the shared code importable and testable.

### 5. `main()` functions of 84–275 lines — Pri 21

| Function | Lines |
| --- | ---: |
| `compare_baselines.main()` | **275** |
| `eval_downstream.main()` | 121 |
| `gptq.quantize_model_gptq()` | 133 |
| `gptq.gptq_quantize_nf4()` | 121 |
| `inference.parse_args()` | 104 |

`compare_baselines.main()` now holds a dozen measurement arms, JSON flushing,
table rendering, and CLI parsing in one scope. No unit test can exercise a single
arm; the only way to test the composition path is to run everything.

**Fix (half a day):** extract the arms into a `METHODS` registry of
`(key, label, builder, bits)`; `main()` becomes parse → loop → render. This also
makes `--only <key>` trivial, which is directly useful for cheap cloud re-runs of
one method.

*Note:* the two `gptq.py` functions are long because the algorithm is genuinely
sequential and they are heavily commented. Lower priority — length there is not
the same problem.

---

## P3 — The one that matters most and costs most

### 6. The fast path is not the accurate path — Pri 20 (Impact+Risk 10/10)

Two independent quantizers ship in this package:

| | Eager (`linear.py` + `quantizer.py`) | Kernel (`triton_kernels.py`) |
| --- | --- | --- |
| Granularity | per-block (64/128) | per-output-channel |
| Outliers | sparse FP16, median/MAD | none |
| 4-bit grid | NF4 codebook | symmetric int4 |
| Used for | **every accuracy number** | **every speed number** |

`sensitivity.py:135-141` documents that INT8's error is ~10× higher under the
kernel scheme, which is why `SensitivityProfiler` needs a `use_kernel` flag at
all — the allocator must be told which quantizer it is optimizing for. No
experiment in the repo measures accuracy and speed on the same configuration, and
a reader asking "what is the perplexity of the thing you benchmarked for speed?"
gets no answer.

**This is the deepest problem in the codebase**, and the priority formula
under-ranks it purely because the fix is a Triton kernel that handles block-wise
scales plus a sparse outlier pass — days of specialist work, on a platform
(WSL2) where you can't even run the tests from the dev box.

**Options, cheapest first:**

1. **Document and scope it (1 hour).** State plainly in the README that the
   kernels implement a different, coarser format, and that kernel-path accuracy
   is unmeasured. *Partially done today.*
2. **Measure the kernel path's accuracy (2 hours + GPU).** Run
   `benchmark.py`-equivalent with `use_kernel=True`. Cheap, and turns an unknown
   into a known — probably the right call.
3. **Make the kernel support block-wise scales (2–4 days).** Real fix. Only worth
   it if the kernels are going to be a headline claim, and per
   `PAPER/01_go_nogo.md` §2c they are not competitive with Marlin anyway.

**Recommendation: do (2), not (3).** Given the kernels won't be a headline claim,
knowing the number is worth far more than closing the gap.

### RESULT (option 2, run 2026-07-30) — worse than expected

`PAPER/exp/kernel_format_accuracy.py` on Qwen2.5-0.5B, WikiText-2, 30k tokens.
(Runs via the eager fallback, which the kernel test asserts is numerically
identical to the fused path to 3e-4 — so this is measurable without Triton.)

| Config | format | ~bits | MB | Perplexity | Δ vs FP16 |
| --- | --- | ---: | ---: | ---: | ---: |
| fp16 | dense | 16 | 942.3 | 11.9066 | +0.0000 |
| eager int8 (block+outlier) | eager | 8 | 620.9 | 11.9113 | +0.0047 |
| kernel W8A16 (per-channel) | kernel | 8 | 601.6 | 11.9240 | **+0.0174** |
| eager nf4 (block+outlier) | eager | 4 | 484.1 | 12.6795 | +0.7729 |
| kernel W4A16 (per-channel) | kernel | 4 | 431.0 | 26.2992 | **+14.3926** |

**W8A16 is fine** (+0.017, essentially lossless) — the INT8 kernel is a genuine,
usable win: 98% of its ideal bandwidth speedup at no real accuracy cost.

**W4A16 more than doubles perplexity.** Per-channel symmetric int4 with no block
scales, no outlier handling and no codebook is not a deployable format. Its
1.78–2.74× batch-1 speedup is therefore a speedup of something nobody would run.

**Cross-validated on both execution paths.** The table above was first measured on
Windows via the eager dequant fallback (no Triton). Re-running it under WSL2 with
the fused Triton kernels actually active gives W8A16 +0.0174 and W4A16 +14.3926 —
within 0.004 ppl of the fallback numbers. So the result is a property of the
quantization *format*, not an artifact of how it was executed, and the script's
claim that the fallback is numerically equivalent to the kernel is now verified
rather than assumed.

This *raises* the severity of #6 from "unknown gap" to "known, and one half of it
is unusable", but does not change the recommendation: the fix is still days of
kernel work for something that still would not reach Marlin. The correct action
is what has now been done — measure it, state it plainly in the README, and stop
presenting the W4A16 speedup without the accuracy caveat attached.

---

## P4 — Lower priority, batch when convenient

- **#7 (Pri 20)** `test_readme_consistency._comparison_deltas` re-parses markdown
  tables. `comparison_*.json` now exists — read that instead. ~30 min, removes a
  brittle regex.
- **#8 (Pri 20)** `autoawq` is deprecated by its own maintainers and, per the
  comment at `compare_baselines.py:94`, last tested on transformers 4.51. It is
  already guarded to skip. Decide: drop the baseline, or pin a compatible
  transformers in a separate extra. Don't leave it ambiguous.
- **#9 (Pri 20)** `codebook.py` is honestly labelled experimental in its own
  docstring but still ships in the installed package. Either move it to
  `atlasinfer/experimental/` or drop it. It adds public surface area you don't
  want questions about.
- **#10 (Pri 18)** `offload.py` implements the "run a model that doesn't fit"
  feature and has no dedicated test — and no benchmark showing it works at scale.
- **#11 (Pri 16)** `_MAX_OUTLIER_FRACTION = 0.25`, the MAD constant `k = 0.60`,
  `num_buckets = 4096`, `_FAILED_PROFILE_PENALTY = 1e6`. All are *well justified
  in comments* — genuinely above average — but none is configurable or swept, so
  "how sensitive is this to the threshold?" is unanswerable without an edit.
- **#12 (Pri 15)** The README is ~600 lines serving as paper, tutorial, API
  reference, and lab notebook. Split: README (what/why/quickstart) →
  `docs/benchmarks.md` → `docs/design.md`.
- **#13 (Pri 14)** Kernel correctness tests run in no automated environment. CI
  is CPU-only and now warns about it. A self-hosted runner is the real fix; short
  of that, a documented pre-release manual checklist.
- **#14 (Pri 12)** `results/` holds 30 tracked files including PNGs — generated
  artifacts in version control. They *are* the evidence base for README claims,
  so keep them; but the boundary between "committed evidence" and "scratch" needs
  stating. Partly addressed by `PAPER/exp/results/`.
- **#15/#16 (Pri 10)** `inference.py` has 33 trailing-whitespace lines and older
  docstring style — it is visibly the oldest file. `atlasinfer/__pycache__/` holds
  orphaned `.pyc` for deleted `kv_cache`/`kv_quant` modules (and `tests/` for
  three deleted KV test files); `.gitignore` lists `.venv`/`venv/` twice
  (lines 2–3 and 137–142).

---

## Phased plan

Designed to run alongside the Lightning work, not block it.

### Phase 0 — before spending credits (~1 hour) — **DONE (`e85bc2e`+1)**
- ~~**#1** warn on unmatched allocation keys~~ → `_targets.check_allocation_covers`,
  called from `quantize_model_mixed`. Warns on both unmatched keys and
  unallocated layers; silent on a correct or deliberately-empty allocation.
- ~~**#2** consolidate layer selection into `atlasinfer/_targets.py`~~ →
  `patcher.py`, `sensitivity.py` and `benchmark.linear_param_count` now all
  import from it. `tests/test_targets.py` (14 tests) asserts the profiler and the
  patcher select the identical layer set, including under custom exclusions.
- Also swept #16: orphaned `kv_cache`/`kv_quant`/`test_kv_*` `.pyc` removed.

Both were correctness hazards on paid hardware. Nothing else on this list should
have preceded them.

### Phase 1 — **DONE**
- ~~**#4** extract `atlasinfer/evaluation.py`~~ → `evaluate_perplexity`,
  `load_wikitext`, `model_weight_bytes`, `resident_bytes` and a new
  `quantized_bits_per_weight` (the *measured* bits, answering #4's sibling
  complaint that `avg_bits` is nominal). `benchmark.py` re-exports for
  compatibility; the three harnesses import the library.
- ~~**#7** point the consistency test at JSON~~ → prefers
  `comparison_*.json`, falls back to markdown for pre-JSON result sets.
- ~~**#9** move or drop `codebook.py`~~ → `atlasinfer/experimental/codebook.py`
  with a subpackage docstring saying nothing in it backs a claim.
- ~~**#15/#16**~~ → 33 trailing-whitespace lines stripped, orphan `.pyc` removed,
  3 duplicate `.gitignore` entries dropped.

### Phase 2 — **DONE**
- ~~**#3** `tests/test_inference.py`~~ → 16 tests. Required one small library
  change: `parse_args()` read `sys.argv` directly and now takes an optional
  `argv`, which is what made the CLI testable at all.
- ~~**#5** `METHODS` registry~~ → `build_methods(ctx)` is module-level and
  inspectable without a GPU; `main()` 275 → 178 lines. Adds `--only`, so a
  failed arm can be re-run for the cost of that arm instead of the whole
  comparison. Profiling is memoized and skipped entirely when no arm needs it.
- ~~**#6 option 2** measure kernel-path accuracy~~ → done, and the answer was
  worse than expected (see the RESULT block above).

### Phase 3 — **DONE**
- ~~**#10** offload tests~~ → 18 tests. Found that unknown architectures raise
  rather than no-op, which is the better contract; test now pins it, and pins
  the `layer_patterns` escape hatch the error message advertises.
- ~~**#12** README split~~ → 702 → 339 lines; benchmarks moved to
  `docs/benchmarks.md` with a docs index in the README. The consistency test now
  searches both files, so a table stays guarded wherever it lives.
- ~~**#13** GPU CI runner~~ → no self-hosted runner, but
  `docs/gpu_test_checklist.md` documents exactly what CI cannot verify and how to
  verify it by hand, and CI emits a warning naming the skipped tests.
- ~~**#11** tuned constants~~ → `MAD_SCALE` promoted to a named module constant;
  both it and `_MAX_OUTLIER_FRACTION` are now overridable per call, so their
  sensitivity is sweepable without editing the library.
- ~~**#8** `autoawq`~~ → removed from the `baselines` extra (deprecated, pinned to
  transformers 4.51, predictably fails to install). AtlasInfer's own `awq-nf4`
  arm covers the mechanism; the external reference is now an explicit opt-in.
- ~~**#14** `results/` boundary~~ → `results/README.md` states what is committed
  evidence vs scratch; `results/_smoke/` and `results/_logs/` are gitignored, and
  `compare_baselines.py --out` keeps a smoke run from overwriting a real result.

### Explicitly not recommended
- **#6 option 3** (block-wise Triton kernel). Days of specialist work to make a
  kernel competitive that, per `PAPER/01_go_nogo.md` §2c, still would not reach
  Marlin. Measuring it (option 2) was the right call and is now done: the W4A16
  format costs +14.39 ppl, so the honest action is to caveat the number, not to
  chase the kernel.

---

## Status after this pass

All 16 items are addressed. Test count went 84 → 152. The two highest-scoring
items were silent-wrong-result hazards and are fixed; the highest Impact+Risk item
(#6) is now measured and documented rather than closed, which was the deliberate
call.

What remains genuinely open, and is a judgement call rather than debt:

- **#6 is documented, not fixed.** The fast path still isn't the accurate path.
  That is now stated in the README with numbers attached.
- **`eval_downstream.main()` (121 lines) and the two `gptq.py` functions** were
  left long. The GPTQ ones are long because the algorithm is sequential and
  heavily commented; splitting them would hurt readability, not help it.

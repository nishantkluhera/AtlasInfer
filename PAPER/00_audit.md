# Phase 1 — Repository Audit

Audit date: 2026-07-30. Branch: `quant-methods-and-7b-validation` @ `7018451`.
Scope: what the code actually does, what it actually measures, and which of the
three headline claims survive contact with the repository.

---

## 1. Architecture map

~4,700 lines of first-party Python (`atlasinfer/` = 3,332; harnesses = 1,149).

```
                calibration text (WikiText-2 train)
                             │
   ┌─────────────────────────▼──────────────────────────┐
   │ sensitivity.py — SensitivityProfiler               │
   │   .profile()            activation-local proxy     │   ──┐
   │   .profile_end_to_end()  ΔNLL per (layer, bits)    │  DEFAULT
   └─────────────────────────┬──────────────────────────┘     │
                             │ {layer: LayerProfile(errors)}  │
   ┌─────────────────────────▼──────────────────────────┐     │
   │ allocator.py                                       │     │
   │   allocate_optimal()  MCKP via DP, 4096 buckets    │◄────┘
   │   allocate_greedy()   sensitivity-ranked baseline  │  (tests only)
   └─────────────────────────┬──────────────────────────┘
                             │ {layer: "fp16"|"int8"|"int4"}
   ┌─────────────────────────▼──────────────────────────┐
   │ patcher.py — quantize_model_mixed()                │
   │   walks named_modules, swaps in place              │
   └─────────────────────────┬──────────────────────────┘
                             │
            ┌────────────────┴─────────────────┐
            │                                  │
   ┌────────▼──────────┐            ┌──────────▼─────────┐
   │ EAGER PATH        │            │ KERNEL PATH        │
   │ linear.py         │            │ triton_kernels.py  │
   │ QuantizedLinear   │            │ W8A16Linear        │
   │ QuantizedLinear4b │            │ W4A16Linear        │
   │                   │            │                    │
   │ quantizer.py:     │            │ per-out-channel    │
   │  block-wise 64/128│            │  symmetric int8/4  │
   │  + sparse FP16    │            │  NO blocks         │
   │    outliers       │            │  NO outliers       │
   │  + NF4 codebook   │            │  NO NF4            │
   └───────────────────┘            └────────────────────┘
     ▲ all accuracy claims           ▲ all speed claims
```

**The single most important structural fact in this repo: the two paths quantize
differently, and each headline claim rests on only one of them.**
`SensitivityProfiler(use_kernel=...)` exists precisely because the per-layer error
profiles differ between them — the docstring at `sensitivity.py:135-141` states
INT8's error is "~10x higher under the per-channel kernel than under block-wise +
outliers". Every perplexity number in `results/` is eager-path. Every speed number
is kernel-path. **No experiment in the repository measures accuracy and speed on
the same quantized model.**

### Module inventory

| File | LoC | Role | Status |
| --- | ---: | --- | --- |
| `allocator.py` | 315 | MCKP DP + greedy baseline + reporting | core, exercised |
| `sensitivity.py` | 398 | forward-hook activation capture; local + ΔNLL profilers | core, exercised |
| `quantizer.py` | 480 | block-wise INT8/INT4, NF4 codebook, median/MAD sparse outliers | core, exercised |
| `linear.py` | 326 | `QuantizedLinear`, `QuantizedLinear4bit`, Conv1D adapter | core, exercised |
| `patcher.py` | 213 | in-place module swap, uniform + mixed | core, exercised |
| `triton_kernels.py` | 393 | W8A16 / W4A16 fused dequant-GEMM, autotuned, eager fallback | core, GPU-only |
| `gptq.py` | 280 | GPTQ error compensation onto the NF4 format | working, benchmarked |
| `awq.py` | 195 | AWQ per-input-channel scaling onto NF4 | working, 1 model |
| `double_quant.py` | 126 | QLoRA-style scale compression | working, 1 model |
| `codebook.py` | 154 | VQ sub-4-bit scaffold | **explicitly unvalidated** (own docstring) |
| `offload.py` | 135 | just-in-time block streaming | working, unbenchmarked |
| `inference.py` | 421 | `AtlasInference` high-level API | working, smoke-tested |
| `reproducibility.py` | 40 | `seed_everything` | used by all harnesses |

### Key algorithmic details worth knowing before writing a method section

- **DP formulation** (`allocator.py:98-199`): standard MCKP. Memory axis
  discretized into 4096 buckets scaled to the budget; per-layer option cost
  `max(1, round(bytes/scale))`. Complexity `O(L · B · P)` = 32 layers × 4096 ×
  3 ≈ 4×10⁵ — microseconds. **The solver is not the expensive part; profiling is.**
- **Tie-break** (`allocator.py:181-189`): on equal predicted error, prefer the
  *largest* feasible capacity. Deliberate and documented.
- **Budget is soft**: `total_bytes` is the true footprint and can exceed
  `budget_bytes` slightly. Tests assert `<= budget * 1.01`.
- **Budget accounting ignores overhead**: `BYTES_PER_PARAM` is exactly
  {2.0, 1.0, 0.5}. Per-block scales, outlier indices, and outlier values are
  **not** counted in the DP's budget but **are** counted in the reported MB. This
  is why `mixed-4.5bit` reports 590.6 MB on Qwen3 while uniform-NF4 (nominally
  4.0 bits) reports 568.7 MB — a 4-bit weight actually costs ~4.4 real bits here.
  **Consequence: "avg_bits" in the tables is a nominal target, not the measured
  rate, and the mixed rows carry unbudgeted overhead the uniform rows also carry.
  This does not invalidate the MB column, which is measured — but any claim
  phrased in *bits* rather than *MB* is approximate.**
- **Failed-profile penalty** (`sensitivity.py:32-48`): unmeasurable layers get
  error `1e6/bytes_per_param`, forcing FP16. Sound, but means a silent profiling
  failure inflates memory rather than accuracy loss — worth a reviewer-facing note.
- **Profiler cost**: `profile_end_to_end` is `O(L × P)` full forward passes over
  the calibration set. For 8 samples × 128 tokens × 2 precisions × ~168 layers
  (7B) this is the dominant cost of the whole pipeline.
- **Outlier detector** (`quantizer.py:90-159`): median/MAD z-score with a 25%
  per-block cap and explicit padding handling. This is a genuine, well-reasoned
  piece of engineering and is unit-tested.

---

## 2. Everything reproducible from this repo, with the exact command

| # | Result | Command | Output artifact | Hardware |
| --- | --- | --- | --- | --- |
| R1 | WikiText-2 ppl + weight MB for {fp16, uniform-int8, uniform-nf4, mixed@{4.5,5,6,7}} | `python benchmark.py --model <name> --eval-tokens 40000 --bits 4.5 5 6 7` | `results/<safe>.{json,md,png}` | any CUDA |
| R2 | Head-to-head vs bnb int8/nf4 + AtlasInfer gptq-nf4/awq-nf4/mixed (+optional auto-gptq, autoawq) | `python compare_baselines.py --model <name>` | `results/comparison_<safe>.md` **(md only — no JSON)** | Linux/CUDA |
| R3 | Downstream zero-shot on 5 lm-eval tasks | `python eval_downstream.py --model <name> --limit 2000` | `results/downstream_<safe>.{json,md}` | Linux/CUDA |
| R4 | Peak GPU MB + decode tok/s, eager path | `python bench_latency.py --model <name>` | **stdout only** | any CUDA |
| R5 | Fused-kernel GEMM microbenchmark, 5 synthetic shapes | `python bench_triton_kernel.py` | **stdout only** | Linux/WSL2 + Triton |
| R6 | Per-layer allocation figure | `python examples/04_visualize_allocation.py` | `results/allocation_<safe>.png` | any |
| R7 | Full suite over N models | `python reproduce.py --readme` | all of the above | Linux/CUDA |
| R8 | Unit tests (11 files) | `pytest tests/ -v` | pass/fail | CPU ok |
| R9 | Double-quant memory study | `compare_baselines.py --double-quant` | `results/double_quant_*.md` | Linux/CUDA |

Positives worth keeping: `seed_everything(0)` is called at the top of every
harness; `--seed` is threaded through; `reproduce.py` runs each stage in its own
subprocess so GPU memory is released; `tests/test_readme_consistency.py` fails CI
if a README number drifts from its source file. That last one is unusually good
practice and should survive into the paper workflow.

---

## 3. Claims that are NOT reproducible from a script in this repo

This is the blunt section.

### Claim 1 — "~35% reduction in 4-bit perplexity loss at equal memory"

**Status: the number is roughly right; "at equal memory" is false.**

Computed directly from `results/*.json` (uniform-NF4 vs the nearest mixed point):

| Model | uniform-NF4 | mixed-4.5bit | Δppl reduction | **memory change** |
| --- | --- | --- | ---: | ---: |
| Qwen3-0.6B | +1.173 @ 568.7 MB | +0.787 @ 590.6 MB | 32.9% | **+3.9%** |
| Qwen2.5-0.5B | +0.788 @ 484.1 MB | +0.533 @ 502.8 MB | 32.4% | **+3.9%** |
| GPT-2 | +1.390 @ 127.6 MB | +1.027 @ 131.7 MB | 26.1% | **+3.2%** |
| Pythia-1.4B | +0.848 @ 1118.3 MB | +0.644 @ 1174.4 MB | 24.1% | **+5.0%** |
| Pythia-410M | +5.466 @ 378.9 MB | +1.257 @ 392.4 MB | 77.0% | **+3.6%** |

The honest statement is: *"a 3–5% memory increase over uniform NF4 buys a 24–77%
(median ~33%) reduction in the perplexity penalty."* That is still a real result —
but it is a **memory-for-accuracy trade at a favourable ratio**, not a free lunch
at matched memory. The sweep's cheapest mixed point is 4.5 bits; **the repository
has never measured mixed precision at or below uniform-NF4's actual footprint**,
so the iso-memory comparison the claim asserts does not exist as data.

Fixing this is cheap and is experiment #1 of any paper plan: extend the sweep
to 4.0/4.2 bits and, better, plot Δppl against measured MB rather than nominal bits.

### Claim 2 — "matches bnb INT8 and beats its NF4 perplexity"

**Status: perplexity is true; it is not iso-memory, and the framing omits the
result that matters most.**

From `results/comparison_*.md` — AtlasInfer NF4 beats bnb NF4 on ppl on all four
models, but costs 10–13% more memory every time (484.1 vs 430.4 MB on Qwen2.5;
568.7 vs 506.9 on Qwen3). The extra bytes are the sparse FP16 outliers. bnb also
double-quantizes its scales. So "beats bnb NF4" = "spends 12% more memory and gets
better perplexity" — a point on a different part of the curve, not a dominating one.
`--double-quant` narrows the gap to ~9%, measured on exactly one model.

**The buried lede:** the repo's own GPTQ-NF4 Pareto-dominates the knapsack mixed
precision on 3 of 4 models — better perplexity at strictly less memory:

| Model | gptq-nf4 | mixed-5bit | verdict |
| --- | --- | --- | --- |
| Qwen3-0.6B | **+0.508 @ 568.7 MB** | +0.651 @ 611.7 MB | gptq dominates |
| Pythia-1.4B | **+0.267 @ 1120.4 MB** | +0.424 @ 1238.0 MB | gptq dominates |
| Qwen2.5-0.5B | +0.480 @ 484.1 MB | +0.460 @ 518.8 MB | gptq: −7% memory for +0.02 ppl (effectively dominates) |
| Pythia-410M | +3.883 @ 378.9 MB | **+0.919 @ 409.0 MB** | mixed dominates decisively |

The README states this openly ("better than even 5-bit mixed precision"), to its
credit. But it means **the headline method of the proposed paper is beaten by a
2022 baseline that this same repository implements**, on the majority of tested
models. Phase 2 has to confront this directly.

Note also what has *never been run*: **GPTQ-NF4 + knapsack allocation combined.**
The two are orthogonal (one compensates residual error, one chooses bit-widths)
and `gptq.py` already emits the same `QuantizedTensor4bit` format the allocator's
INT4 tier consumes. This is the single highest-value unrun experiment in the repo.

### Claim 3 — "1.4–1.95× faster batch-1 decode than FP16"

**Status: materially overstated. It is a synthetic single-GEMM microbenchmark,
not decode.**

What `bench_triton_kernel.py` actually measures: `torch.randn` matrices at five
hardcoded shapes, one isolated `w8a16_linear` call vs one `F.linear` call. No
model, no KV cache, no attention, no sampling, no real weights.

What the repository's *actual* end-to-end decode numbers say (README lines
183–189, from `bench_latency.py`, eager path, RTX 3060):

| Model | INT8 speed vs FP16 | NF4 | mixed-5bit |
| --- | ---: | ---: | ---: |
| gpt2 | 0.52× | 0.32× | 0.37× |
| Qwen2.5-0.5B | 0.61× | 0.31× | 0.35× |
| Qwen3-0.6B | 0.64× | 0.36× | 0.43× |
| Pythia-410M | 0.52× | 0.21× | 0.25× |
| Pythia-1.4B | 0.24× | 0.09× | 0.10× |

**End-to-end, AtlasInfer decode is 1.6×–11× *slower* than FP16.** The only
kernel-path end-to-end artifact in the repo is
`results/_logs/wsl_engine_e2e.log`, which is a **correctness smoke test** — it
prints a generated sentence and "E2E OK". It reports no tokens/sec.

So the accurate claim today is: *"a fused W8A16 GEMM is 1.5–1.95× faster than an
FP16 GEMM at M=1 on synthetic shapes; end-to-end model decode with the fused
kernels has not been measured."* The gap between that and "1.4–1.95× faster
batch-1 decode" is exactly the kind of thing a reviewer finds and does not forgive.

### Other gaps

- **No comparison against any quantized kernel.** Not Marlin, not ExLlamaV2, not
  bnb's kernels, not AWQ's GEMM, not `torch.compile`. The only comparator is
  `F.linear`. Beating FP16 at batch-1 with 4-bit weights is arithmetic, not a
  contribution.
- **No ablation isolating the knapsack.** `allocate_greedy` exists but is called
  **only from `tests/test_allocator.py`** — never on a real model. There is no
  random-allocation control, no sensitivity-threshold control. Claim 1's causal
  attribution is untested at model scale.
- **No calibration-sensitivity study.** `max_samples=8`, `seq_len=128`,
  `max_rows=64` are hardcoded defaults; nothing sweeps them.
- **No variance anywhere.** Every number is a single seed-0 run. No error bars,
  no repeats, on either perplexity or latency.
- **No C4 perplexity.** WikiText-2 only.
- **7B is aspirational.** `docs/lightning_7b.md` and `run_lightning.sh` are a
  well-prepared plan for a cloud run; `results/` contains nothing above 1.4B
  locally. The 3B table in the README is from a Kaggle T4 session and has no
  committed JSON.

**Contamination check — this one is clean.** `benchmark.load_wikitext()` draws
calibration from the **train** split and evaluation from the **test** split
(`benchmark.py:141-148`). No overlap. Every harness routes through this function.
Good.

---

## 4. Hardcoded, cached, or hand-transcribed results

| Artifact | Provenance | Risk |
| --- | --- | --- |
| `results/triton_kernel_rtx3060.md` | **hand-transcribed** from `_logs/wsl_triton_kernel.log`; script prints to stdout only | typo-prone; not regenerable by a script |
| README kernel table (lines 207–211) | hand-copied from the above | second-order transcription |
| README runtime table (lines 183–189) | hand-copied from `bench_latency.py` stdout; **no JSON, no log committed** | **unverifiable — the source data does not exist in the repo** |
| README Kaggle T4 3B tables (lines 251–279) | hand-copied from an uncommitted Kaggle session | **unverifiable** |
| README T4 "~7× slower / 215 vs 17 GB/s" narrative | prose recollection of a debugging session | no artifact |
| `results/comparison_*.md` | generated, but **markdown only** | not machine-readable; `test_readme_consistency` re-parses the markdown |
| `results/*.json` | **generated properly** ✓ | fine |
| `results/downstream_*.json` | **generated properly** ✓ | fine |
| `results/_logs/` | **not git-tracked** | local-only evidence |

**Rule for the paper: no number may enter LaTeX except from a committed JSON.**
Today three of the most quotable tables (runtime, kernel, 3B) violate that, and
two of them have no committed source at all. `bench_latency.py` and
`bench_triton_kernel.py` need JSON output before anything else happens.

---

## 5. Test coverage and kernel correctness

11 test files, CPU-only in CI (`ubuntu-latest`, no GPU) — so **the Triton kernel
tests never run in CI**. They are gated on `HAS_TRITON and torch.cuda.is_available()`
and skip silently.

### Kernel correctness verification — the answer to "do we verify against FP16?"

Yes, at two different levels, and the distinction matters:

| Test | Reference | Tolerance | What it isolates |
| --- | --- | --- | --- |
| `TestFusedKernel::test_matches_fp16_linear` | `F.linear` on the **original FP16 weight** | rel < **0.05** | kernel + int8 quantization error together |
| `TestFusedKernel::test_with_bias` | same, with bias | rel < 0.05 | bias path |
| `TestFusedKernel::test_w4a16_matches_reference` | `F.linear` on the **dequantized packed weight** | rel < **0.02** | **kernel math alone** ✓ |
| `TestW8A16LinearEager::test_from_linear_matches` | `nn.Linear` FP16 | rel < 0.03 | CPU fallback |
| `TestW4A16LinearEager::test_from_linear_matches` | `nn.Linear` FP16 | rel < **0.12** | CPU fallback |

The W4A16 test is methodologically the right one — comparing against the
dequantized reference separates kernel bugs from quantization error, and it
achieves 0.0003 measured against a 0.02 bound. The W8A16 test conflates the two,
though its 0.0085 measured error against a 0.05 bound leaves ample headroom.

**Weaknesses a reviewer would flag:** tolerances are relative Frobenius norm on a
single seed with `torch.randn * 0.05` weights — no adversarial shapes, no
non-power-of-2 K, no per-element max-error bound, no accumulation-order check at
large K. Shapes tested top out at K=512, N=384; the benchmark runs K=4096–11008.
And nothing tests the two paths *against each other*: no test asserts that
`QuantizedLinear` (eager, block-wise+outliers) and `W8A16Linear` (kernel,
per-channel) produce comparable outputs — which is correct, because they don't,
which is exactly the structural problem in §1.

### Coverage by module

Well covered: `allocator` (5 targeted tests incl. the greedy-vs-optimal
counterexample), `quantizer`, `linear`, `codebook`, `double_quant`, `gptq`, `awq`,
`triton_kernels` (CPU parts).
**Not covered:** `sensitivity.py` has **no dedicated test file** — the profiler
that produces every input to the headline method is exercised only indirectly via
`test_pipeline.py`. `offload.py`, `inference.py`: no dedicated tests.

Dependencies are unpinned (`torch>=2.0.0`, `transformers>=4.30.0`, …) — fine for a
library, unacceptable for artifact evaluation.

---

## 6. Summary of what Phase 2 must resolve

1. **Novelty.** Is MCKP-for-bit-allocation new? (HAWQ-V3 used ILP; SqueezeLLM,
   SpQR, OWQ are all sensitivity-driven mixed precision.) → §2a
2. **Is the accuracy win real and general, and does it survive an iso-memory
   comparison it has never actually been given?** → §2b
3. **The kernels have never been compared to a quantized kernel, and end-to-end
   decode is 1.6–11× slower than FP16.** → §2c
4. **Attribution: is the win from the DP, or from NF4 + sparse outliers + the ΔNLL
   profiler?** `allocate_greedy` has never been run on a model. → §2d
5. **The repo's own GPTQ-NF4 beats the proposed method on 3 of 4 models.** This is
   the finding most likely to sink the submission, and it is already in the
   committed results.

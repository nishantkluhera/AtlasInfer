# Benchmarks

Every number here is generated, not transcribed: each table's source is a
`.json` under [`results/`](../results/), and `tests/test_readme_consistency.py`
fails CI if a table drifts from its source file. Two tables are explicitly
marked as lacking committed source data — they say so inline.

Paths in this file are relative to the repository root.

See also: [tech debt](tech_debt.md) ·
[GPU test checklist](gpu_test_checklist.md) · [7B on Lightning](lightning_7b.md)

---


WikiText-2 perplexity (lower is better) and resident weight memory, measured on
an RTX 3060. `delta vs FP16` is the perplexity increase over the dense baseline.
Reproduce any row with `python benchmark.py --model <name>`.

<!-- RESULTS:Qwen3-0.6B-Base -->
### Qwen/Qwen3-0.6B-Base  (2025)

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 1136.9 | 11.804 | +0.000 |
| uniform-int8 | 8.0 | 739.0 | 11.812 | +0.008 |
| uniform-nf4 | 4.0 | 568.7 | 12.977 | +1.173 |
| mixed-4.5bit | 4.5 | 590.6 | 12.591 | +0.787 |
| mixed-5bit | 5.0 | 611.7 | 12.455 | +0.651 |
| mixed-6bit | 5.9 | 652.3 | 12.227 | +0.424 |
| mixed-7bit | 7.0 | 701.7 | 12.110 | +0.306 |
<!-- /RESULTS:Qwen3-0.6B-Base -->

<!-- RESULTS:Qwen2.5-0.5B -->
### Qwen/Qwen2.5-0.5B  (2024)

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 942.3 | 12.279 | +0.000 |
| uniform-int8 | 8.0 | 620.9 | 12.279 | +0.001 |
| uniform-nf4 | 4.0 | 484.1 | 13.067 | +0.788 |
| mixed-4.5bit | 4.5 | 502.8 | 12.812 | +0.533 |
| mixed-5bit | 5.0 | 518.8 | 12.739 | +0.460 |
| mixed-6bit | 6.0 | 553.6 | 12.638 | +0.359 |
| mixed-7bit | 7.0 | 589.1 | 12.591 | +0.313 |

At ~4.5 bits mixed precision cuts the perplexity penalty by about a third
(here +0.79 → +0.53) by spending the extra half-bit only on the layers that hurt
most — **at a 3.9% memory increase** (484.1 → 502.8 MB), not at equal memory.
Across the five models the reduction is 24–77% (median ~33%) for +3–5% memory.

Two things this table does **not** say, both worth knowing up front:

- **Mixed precision cannot be cheaper than uniform 4-bit.** INT4 is the cheapest
  tier the allocator can pick, so an "equal-memory" comparison against uniform
  NF4 is not measurable — at that budget the allocator *returns* uniform NF4. The
  meaningful baseline at an in-between footprint is what you'd otherwise do:
  upgrade some subset of layers to INT8. Measured against a *random* such subset
  at matched memory, the allocator wins by 16–43%
  ([ablation](../PAPER/exp/results/ablation_Qwen_Qwen2.5-0.5B.json)).
- **The exact knapsack solve is not what's doing the work.** In that same
  ablation, a plain benefit-per-byte greedy matches the DP to within 0.006 ppl on
  average — **7.5× inside the run-to-run spread** — and beats it at 2 of 4
  budgets. What buys the 16–43% is the *measured sensitivity signal*, not solving
  the multiple-choice knapsack optimally. Reported because it's the kind of thing
  that's easy to leave unmeasured and quietly overclaim.
- **GPTQ-NF4 does better still, at the *same* 4-bit footprint** (+0.48 vs +0.53)
  and at zero extra memory — see [vs bitsandbytes](#vs-bitsandbytes). On 3 of 4
  models GPTQ-NF4 beats mixed precision outright. The two mechanisms are
  orthogonal and `compare_baselines.py` now measures them composed
  (`gptq-mixed`), which is the open question.
<!-- /RESULTS:Qwen2.5-0.5B -->

Also validated on older architectures — GPT-2 (124M) and Pythia-410M / 1.4B —
under [`results/`](../results/); the same mixed-precision win holds there too (it's
largest on models where uniform 4-bit is most lossy, e.g. +5.5→+1.3 on Pythia-410M).

**Reading the tables:** INT8 is effectively lossless. Uniform 4-bit (NF4) is much
smaller but costs real perplexity. The mixed-precision rows land *between* those
uniform points in memory while staying *closer to FP16* in perplexity than the
uniform trade-off line — that gap is what per-layer allocation buys you.

### Runtime: memory vs latency (the honest tradeoff)

The tables above measure *stored* size. The win also shows up at **runtime** —
because only one layer is dequantized to FP16 at a time, peak GPU memory during
generation drops too. But that on-the-fly dequant happens on every matmul with no
fused low-bit kernel, so it costs latency rather than saving it. Peak GPU memory
and decode tok/s vs FP16, all models, 64-token decode on an RTX 3060
(`python bench_latency.py --model <name>`), reported as **fraction of FP16**:

| Model | FP16 (MB, tok/s) | INT8 (mem, spd) | NF4 (mem, spd) | mixed-5bit (mem, spd) |
| --- | ---: | ---: | ---: | ---: |
| gpt2 | 267, 67.9 | 0.77×, 0.52× | 0.73×, 0.32× | 0.76×, 0.37× |
| Qwen2.5-0.5B | 961, 16.9 | 0.70×, 0.61× | 0.60×, 0.31× | 0.63×, 0.35× |
| Qwen3-0.6B | 1154, 18.5 | 0.69×, 0.64× | 0.56×, 0.36× | 0.60×, 0.43× |
| Pythia-410M | 789, 46.9 | 0.71×, 0.52× | 0.60×, 0.21× | 0.64×, 0.25× |
| Pythia-1.4B | 2720, 42.7 | 0.66×, 0.24× | 0.53×, 0.09× | 0.58×, 0.10× |

> ⚠️ **Provenance:** these five rows were transcribed by hand from an early
> `bench_latency.py` stdout run whose output was never committed, so unlike every
> other table here they cannot be re-checked against a source file. The script now
> writes `results/latency_<model>.{md,json}` (and reports the spread over repeated
> runs); **these numbers will be replaced by generated ones on the next run.**
> Treat the exact digits as indicative and the direction — quantized eager decode
> is slower than FP16, markedly so at 1.4B — as the finding.

The pattern is consistent: **~30–45% less peak GPU memory for slower decode** in
the eager path (the codebook dequant is the most expensive) — the right call when
the goal is *running a model that otherwise wouldn't fit*. The fix for the latency
cost is a fused kernel, below.

### Fused INT8 kernel (Triton): turning the memory win into a speed win

[`atlasinfer/triton_kernels.py`](../atlasinfer/triton_kernels.py) implements **fused
dequant-GEMM** Triton kernels for both INT8 (`W8A16`) and packed INT4 (`W4A16`):
they read the low-bit weights directly (half / a quarter of FP16's bytes),
dequantize them in-register, and do the matmul in a single pass — no full-weight
materialization. Measured on an RTX 3060 Laptop under WSL2 (torch 2.5.1+cu124 /
Triton 3.1.0). **Generated**, not transcribed — full table, per-window samples
and correctness notes in
[`results/triton_kernel_NVIDIA_GeForce_RTX_3060_Laptop_GPU.md`](../results/triton_kernel_NVIDIA_GeForce_RTX_3060_Laptop_GPU.md)
and its `.json`.

Because batch-1 is bandwidth-bound, reading half (int8) or a quarter (int4) of
the weight bytes makes ~2× and ~4× *available for free*. So the honest column is
not the speedup — it's how much of that headroom the kernel actually captures:

| matmul shape (M, K, N) | fp16 | fused W8A16 | fused W4A16 | W8 vs fp16 (of 2.0× ideal) | W4 vs fp16 (of 4.0× ideal) |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1, 4096, 4096)  | 0.113 ms | 0.076 ms | 0.078 ms | 1.49× (74%) | 1.45× (36%) |
| (1, 4096, 11008) | 0.288 ms | 0.146 ms | 0.110 ms | **1.97×** (98%) | 2.62× (66%) |
| (1, 5120, 5120)  | 0.172 ms | 0.098 ms | 0.095 ms | 1.76× (88%) | 1.82× (45%) |
| (4, 4096, 4096)  | 0.108 ms | 0.073 ms | 0.078 ms | 1.48× (74%) | 1.39× (35%) |
| (16, 4096, 4096) | 0.114 ms | 0.079 ms | 0.108 ms | 1.45× (72%) | 1.06× (27%) |

**Being straight about what this shows.** The FP16 baseline is *not* a straw man —
`F.linear` reaches 313 GB/s, 93% of the card's 336 GB/s peak (192-bit at 14 Gbps;
`nvidia-smi` reports a 7001 MHz memory clock). Against it, **W8A16 is genuinely
good**: 1.97× of an available 2.0× at the largest shape. **W4A16 is not**: it
captures only 27–66% of its headroom, and by M=16 it has collapsed to 1.06× —
where [Marlin](https://arxiv.org/abs/2408.11743) sustains close to the full 4×.
So these kernels beat FP16, but the 4-bit one does **not** compete with a
state-of-the-art quantized kernel, and no such comparison has been run here.

> **On quoting these numbers.** Within a single run the FP16 baseline is stable to
> 1–6%, but *across* invocations the small shapes swing far more — `(1, 4096, 4096)`
> has been observed anywhere from 1.01× to 1.66×, because Triton autotunes once
> per shape and the tile it lands on depends on the clock state at that moment.
> Only `(1, 4096, 11008)` is consistently ~2.0× for W8A16 across runs. Quote that
> one, or quote a range — not a small-shape figure to two decimals.

### What the kernel format costs in accuracy

The kernels use **per-output-channel symmetric** int8/int4 — *not* the block-wise
NF4 + sparse-outlier format every other perplexity number on this page was
measured with. So the speed table above and the accuracy tables above describe
**different quantizers**. Measuring both on one model
(`PAPER/exp/kernel_format_accuracy.py`, Qwen2.5-0.5B, WikiText-2, 30k tokens):

| Config | format | ~bits | MB | Perplexity | Δ vs FP16 |
| --- | --- | ---: | ---: | ---: | ---: |
| fp16 | dense | 16 | 942.3 | 11.906 | +0.000 |
| eager int8 (block+outlier) | eager | 8 | 620.9 | 11.911 | +0.005 |
| **kernel W8A16 (per-channel)** | kernel | 8 | 601.6 | 11.925 | **+0.019** |
| eager nf4 (block+outlier) | eager | 4 | 484.1 | 12.679 | +0.773 |
| **kernel W4A16 (per-channel)** | kernel | 4 | 431.0 | 26.296 | **+14.389** |

**W8A16 is fine** — +0.019 is essentially lossless, so the INT8 kernel is a real,
usable speedup: 1.95× on a batch-1 GEMM at no meaningful accuracy cost.

**W4A16 is not usable.** Per-channel symmetric int4 with no blocks, no outlier
handling and no codebook more than doubles perplexity. You would never deploy it,
which means its 1.78–2.74× batch-1 speedup is not a speedup of anything you'd
actually run. Reported rather than quietly dropped, because the number was
previously unmeasured and the speed table on its own reads as though it were free.

Fixing this means a Triton kernel that handles per-block scales plus a sparse
outlier pass — days of work for a kernel that still would not reach
[Marlin](https://arxiv.org/abs/2408.11743). Not planned; see
[docs/tech_debt.md](tech_debt.md) #6.

Triton is Linux/GPU-only, so this needs **WSL2** on Windows; the module
import-guards on `HAS_TRITON` so the rest of the library is unaffected. See
[docs/wsl_triton.md](wsl_triton.md) for the 3-command setup.

**It's wired into the engine.** `AtlasInference(..., kernel="auto")` (the default)
routes INT8 layers through `W8A16Linear` and INT4 layers through `W4A16Linear`
whenever Triton + CUDA are present, and falls back to the eager block-wise path
everywhere else — so the same code is fast on Linux/WSL2 and still correct on
Windows/CPU:

```python
engine = AtlasInference("gpt2", kernel="auto")   # "on" to force, "off" to disable
```

Both kernel layers carry an eager dequant fallback, so a kernel-quantized model
still runs (just unaccelerated) without Triton.

> Note on the memory floor: token embeddings and the LM head are left in FP16
> (quantizing them hurts accuracy disproportionately), so for a small model like
> GPT-2 they dominate the footprint and mute the headline compression. The effect
> is much larger on models where the transformer blocks, not the embeddings, hold
> most of the parameters.

### Kaggle (Tesla T4): a bigger model, and kernel portability

> ⚠️ **Provenance:** every number in this section comes from an interactive
> Kaggle session whose result files were never committed to this repo. Unlike the
> tables above, they are **not backed by anything under `results/`** and are not
> covered by the README-consistency test. They are reported because the 3B
> datapoint is informative, but they should be re-run and committed before being
> relied on — or cited as anecdote, not measurement.

All the runtime/kernel tables above are on an Ampere RTX 3060. Kaggle's free
**Tesla T4** (Turing) is the most accessible GPU for reproducing this, so these T4
measurements are reported **separately** rather than mixed into the 3060 numbers.

**Accuracy holds up at 3B (and still beats bitsandbytes).** WikiText-2 perplexity
on Qwen2.5-3B (`python benchmark.py` + `python compare_baselines.py`, run on the
T4):

| Method | ~bits | Weights (MB) | Perplexity | Δ vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 5886.0 | 7.527 | +0.000 |
| AtlasInfer int8 | 8 | 3396.3 | 7.529 | **+0.002** |
| bnb int8 (LLM.int8) | 8 | 3240.0 | 7.607 | +0.080 |
| AtlasInfer mixed-7bit | 7.0 | 3141.0 | 7.637 | +0.110 |
| AtlasInfer mixed-6bit | 6.0 | 2867.8 | 7.669 | +0.142 |
| AtlasInfer mixed-5bit | 5.0 | 2572.8 | 7.725 | **+0.198** |
| AtlasInfer mixed-4.5bit | 4.4 | 2420.7 | 7.757 | +0.230 |
| AtlasInfer nf4 | 4 | 2304.1 | 7.859 | **+0.332** |
| bnb nf4 | 4 | 1917.0 | 7.965 | +0.438 |
| AtlasInfer int4 (symmetric) | 4 | 2304.1 | 8.086 | +0.559 |

Same pattern as the smaller models, at 6× the size: INT8 is essentially lossless
and ahead of bnb (+0.002 vs +0.080); NF4 beats bnb's NF4 (+0.332 vs +0.438); and
the mixed-precision sweep traces the curve between. (GPTQ-NF4, the best 4-bit
method on the smaller models, wasn't run here — it would slot in below NF4. bnb
stays ~17% smaller at 4-bit via scale double-quantization.)

**Eager decode reaches 3B — and the latency cost grows with model size.**
Qwen2.5-3B, 64-token decode, eager block-wise path
(`python bench_latency.py --model Qwen/Qwen2.5-3B`):

| Config | Peak GPU (MB) | vs FP16 | tok/s | rel. speed |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 6008.3 | 1.00x | 19.1 | 1.00x |
| uniform-int8 | 3684.6 | 0.61x | 3.6 | 0.19x |
| uniform-nf4 | 2761.3 | 0.46x | 1.0 | 0.05x |
| mixed-5bit | 3045.1 | 0.51x | 1.2 | 0.06x |

Same shape as the smaller models — **~40–55% less peak memory at a decode-speed
cost** — but more pronounced at 3B on a T4, where the eager codebook dequant
dominates (NF4 at 1.0 tok/s). This is squarely the "fits but slow" regime the
fused kernel exists to remove.

**Kernel portability (Turing vs Ampere).** The fused kernel's batch-1 speedup was
tuned on Ampere. A single fixed tile that streams at ~215 GB/s on the 3060 stalled
at ~17 GB/s on the T4 — roughly **7× *slower* than FP16** — because Turing lacks
the `cp.async` software pipelining Ampere relies on. Two fixes
([`triton_kernels.py`](../atlasinfer/triton_kernels.py)): (1) the kernel **autotunes
tile size / warps / pipeline depth per `(M, N, K)`**, and — the one that actually
mattered — (2) the int8/int4 weights are stored **transposed `(K, N)`** so the
kernel's fast tile axis is contiguous and the weight loads **coalesce** (with the
natural `(N, K)` layout each load was strided by K, which Ampere hides via
`cp.async` but Turing can't). Autotuning alone left the T4 ~7× slow — confirming
the bottleneck was the access pattern, not the schedule. The coalesced-layout fix
is in `quantize_w8a16`/`quantize_w4a16`; end-to-end T4 kernel throughput is
reproducible from the [Kaggle notebook](../notebooks/kaggle_benchmark.ipynb) with
`bench_triton_kernel.py` (Triton is Linux/GPU-only, so it can't run on the Windows
dev box these perplexity numbers came from).

### vs bitsandbytes (the accessible-quant baseline)

Head-to-head, same WikiText-2 eval and same memory accounting
(`python compare_baselines.py`), across **four models** — current (Qwen3/Qwen2.5),
a larger one (Pythia-1.4B), and a hard small one (Pythia-410M). **Δ perplexity vs
FP16** per method (lower is better; INT8 columns are 8-bit, the rest 4-bit):

| Model | Atlas int8 | bnb int8 | bnb nf4 | Atlas nf4 | **Atlas gptq-nf4** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B (2025) | +0.008 | +0.057 | +1.834 | +1.173 | **+0.508** |
| Qwen2.5-0.5B (2024) | +0.001 | +0.070 | +1.323 | +0.788 | **+0.480** |
| Pythia-1.4B | +0.003 | +0.036 | +0.760 | +0.713 | **+0.267** |
| Pythia-410M | +0.024 | +0.122 | +5.939 | +5.466 | **+3.883** |

On **every** model: AtlasInfer's **INT8 matches/beats** bitsandbytes' LLM.int8(),
and at **4-bit, GPTQ-NF4 is the best method** — beating bnb's NF4, plain NF4, and
(on Qwen3-0.6B and Pythia-1.4B) even AtlasInfer's own 5-bit mixed precision, at
the same 4-bit memory. The win is largest on the modern models (~3× closer to
FP16 than bnb); even the hard Pythia-410M case improves +5.9→+3.9.

**The important caveat, since the table above is perplexity-only:** AtlasInfer's
4-bit rows are **10–13% larger** than bnb's, because bnb double-quantizes its
scales and omits sparse FP16 outliers entirely — and those outliers are exactly
what buys the accuracy lead. So "beats bnb NF4" means *better perplexity at more
memory*, a different point on the curve, not a dominating one. The
[`--double-quant`](#pushing-further-double-quant-awq-and-2-bit) flag narrows the
gap to ~9% at unchanged perplexity.

**And note what beats what:** GPTQ-NF4 costs no extra memory, while the
mixed-precision allocation costs 7–10%. On 3 of these 4 models GPTQ-NF4 therefore
Pareto-dominates the allocator — better perplexity at strictly less memory. The
allocator wins decisively only on Pythia-410M, where uniform 4-bit collapses. That
is why `compare_baselines.py` now also measures the two **composed**
(`gptq-mixed`): the mechanisms are orthogonal, and whether they stack is the open
question. See [PAPER/01_go_nogo.md](../PAPER/01_go_nogo.md) for the full accounting.
Per-model detail incl. mixed-precision and symmetric-int4 rows:
[Qwen3-0.6B](../results/comparison_Qwen_Qwen3-0.6B-Base.md) ·
[Qwen2.5-0.5B](../results/comparison_Qwen_Qwen2.5-0.5B.md) ·
[Pythia-1.4B](../results/comparison_EleutherAI_pythia-1.4b.md) ·
[Pythia-410M](../results/comparison_EleutherAI_pythia-410m.md).

### Downstream accuracy (not just perplexity)

Perplexity is a proxy; the accuracy that matters is on real tasks. Same quantized
models, run through `lm-evaluation-harness` on five zero-shot multiple-choice
tasks (ARC-easy/challenge, HellaSwag, PIQA, WinoGrande), 2000 examples each
(`python eval_downstream.py --model Qwen/Qwen2.5-0.5B --limit 2000`). Mean accuracy
across the five tasks, on **Qwen2.5-0.5B**:

| Method | ~bits | Weights (MB) | Avg acc (5 tasks) | Δ vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 942 | 0.5325 | +0.0000 |
| AtlasInfer int8 | 8 | 621 | 0.5328 | +0.0003 |
| AtlasInfer mixed-5bit | 5.0 | 519 | 0.5320 | −0.0005 |
| **AtlasInfer gptq-nf4** | 4 | 484 | 0.5314 | **−0.0010** |
| AtlasInfer nf4 | 4 | 484 | 0.5234 | −0.0090 |
| bnb nf4 | 4 | 430 | 0.5209 | −0.0115 |

At ~0.5-point standard error on the 5-task mean (2000 examples/task), INT8,
mixed-5bit, and GPTQ-NF4 are all **within noise of FP16** — the accuracy is
preserved, not just the perplexity. The two *plain* 4-bit methods lose a real
~1 point, and AtlasInfer's GPTQ-NF4 recovers essentially all of that gap while
still beating bnb's NF4 by ~1 point at 4-bit. Per-task numbers and the Qwen3-0.6B
run: [Qwen2.5-0.5B](../results/downstream_Qwen_Qwen2.5-0.5B.md) ·
[Qwen3-0.6B](../results/downstream_Qwen_Qwen3-0.6B-Base.md).

### Pushing further: double-quant, AWQ, and 2-bit

Three additions on top of the 4-bit story above, each implemented from scratch.
The framing is deliberately narrow: they push AtlasInfer to the accessible-quant
frontier (matching the memory *and* accuracy levers bitsandbytes/AWQ/GPTQ have),
not past the research frontier — the honest boundary is spelled out per item.

**1. Double-quantized scales** ([`double_quant.py`](../atlasinfer/double_quant.py),
`--double-quant`). Block-wise NF4 stores one FP32 scale per 64 weights — ~0.5
bit/weight of overhead, and exactly the ~10–13% memory gap to bnb. Quantizing the
scales themselves to INT8 + per-group (scale, offset), the QLoRA trick, recovers
most of it. Measured on Qwen2.5-0.5B (WikiText-2, 40k tokens, seed 0):

| Method | Weights (MB) | Perplexity | Δ vs FP16 | vs bnb NF4 mem |
| --- | ---: | ---: | ---: | ---: |
| nf4 | 484.1 | 13.067 | +0.788 | 1.12× |
| **nf4+dq** | 468.2 | 13.066 | **+0.788** | 1.09× |
| gptq-nf4 | 484.0 | 12.758 | +0.479 | 1.12× |
| **gptq-nf4+dq** | 468.2 | 12.757 | **+0.478** | 1.09× |

Double-quant removes the scale overhead at **zero perplexity cost** (+0.788 →
+0.788), narrowing the 4-bit memory gap to bnb from +12.5% to +8.8%. It doesn't
fully close it — the residual is AtlasInfer's sparse FP16 outliers, which bnb omits
and which are what keep its accuracy ahead. So **gptq-nf4+dq is the best 4-bit
config here**: it beats bnb NF4 on accuracy by a wide margin (**+0.48 vs +1.32**)
at within ~9% of bnb's memory. Reproduce:
`python compare_baselines.py --model Qwen/Qwen2.5-0.5B --double-quant`
(and [results/double_quant_Qwen_Qwen2.5-0.5B.md](../results/double_quant_Qwen_Qwen2.5-0.5B.md)).

**2. AWQ activation-aware scaling** ([`awq.py`](../atlasinfer/awq.py)). A second,
independent route to the 4-bit accuracy tier. Rather than keep the salient
(high-activation) input channels in FP16 like the mixed-precision path, AWQ scales
those weight columns *up* before NF4 quantization and divides the activation back
out at run time — so the salient weights round more finely, with the scaling
cancelling exactly (`(W·diag(s))·(x/s) = W·x`). A per-layer exponent `α∈[0,1]` is
grid-searched to minimize each layer's quantized output error. On Qwen2.5-0.5B:

| Method | Weights (MB) | Perplexity | Δ vs FP16 |
| --- | ---: | ---: | ---: |
| nf4 | 484.1 | 13.067 | +0.788 |
| **awq-nf4** | 513.5 | 12.762 | **+0.483** |
| gptq-nf4 | 484.0 | 12.758 | +0.479 |

AWQ (**+0.483**) lands right on GPTQ (+0.479) from a completely different,
Hessian-free mechanism — roughly halving the plain-NF4 penalty. The honest catch:
it uses *more* memory than plain NF4 here (513 vs 484 MB), because the per-channel
up-scaling widens per-block variance and pushes more weights over the sparse-outlier
threshold — so GPTQ is the more memory-efficient route to the same accuracy in this
implementation, and the two are complementary rather than redundant. Full table:
[results/awq_Qwen_Qwen2.5-0.5B.md](../results/awq_Qwen_Qwen2.5-0.5B.md).

**3. Sub-4-bit vector quantization** —
*experimental* ([`codebook.py`](../atlasinfer/experimental/codebook.py)). A single-codebook vector
quantizer (group `d` weights, round each vector to the nearest of 256 k-means
centroids → `8/d` bits/weight) that reaches the 2–3 bit range scalar quantization
can't. It's implemented and unit-tested for correctness, but the accuracy it would
need to be *useful* at 2 bit is the current research frontier (AQLM, QuIP#, QTIP),
established on 7B models the 6 GB dev GPU here can't run — so this ships as a
**clearly-labeled scaffold with no SOTA claim**, and the natural next steps
(residual/additive codebooks, incoherence pre-processing) are called out in the
module. It's the honest edge of what this repo demonstrates versus what it merely
sets up.

---

# AtlasInfer

**Mixed-precision quantization for running LLMs on consumer GPUs — from scratch, and benchmarked.**

[![CI](https://github.com/nishantkluhera/AtlasInfer/actions/workflows/ci.yml/badge.svg)](https://github.com/nishantkluhera/AtlasInfer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

AtlasInfer quantizes each linear layer of a transformer to **FP16, INT8, or 4-bit
(NF4) independently**, choosing the precision per layer to **minimize accuracy
loss within a memory budget**. Instead of forcing every layer to the same bit-width,
it measures how much each layer actually suffers under quantization and spends
the budget where it buys the most accuracy.

Everything here — the block-wise integer quantizer, the calibration-based
sensitivity profiler, and the budget allocator — is implemented from scratch on
top of PyTorch (no `bitsandbytes`, no `auto-gptq`), so the whole pipeline is
inspectable in a few hundred lines. There are also optional **fused W8A16/W4A16
Triton kernels** (Linux/WSL2) that beat an FP16 `F.linear` by **1.5–1.95×** on an
isolated batch-1 GEMM.

> **Read that kernel number carefully — it is a single-matmul microbenchmark, not
> decode.** End-to-end generation through the default *eager* path is
> **slower** than FP16 (0.09–0.64×, [table below](#runtime-memory-vs-latency));
> AtlasInfer buys **memory**, and the fused kernels claw back matmul time in
> isolation. End-to-end decode with the fused kernels has not been measured.
> A fuller accounting of what these kernels do and don't achieve — including the
> fact that W4A16 captures only 26–68% of the bandwidth headroom a 4-bit weight
> makes available — is in [PAPER/01_go_nogo.md](PAPER/01_go_nogo.md#2c).

---

## The result

Three measured claims, on **current models** (Qwen3, 2025; Qwen2.5, 2024):

1. **It beats bitsandbytes**, the standard accessible-quant library: AtlasInfer's
   INT8 is lossless and edges LLM.int8(), and at 4-bit its **GPTQ-NF4** path is
   ~3x closer to FP16 than bitsandbytes' NF4 (see [vs bitsandbytes](#vs-bitsandbytes)).
2. **GPTQ error compensation closes the 4-bit gap**: on Qwen3-0.6B the 4-bit
   penalty drops from +1.17 (plain NF4) to **+0.51** (GPTQ-NF4) at the *same*
   memory — better than even 5-bit mixed precision (+0.65).
3. **Per-layer mixed precision beats any uniform bit-width** at a given footprint
   — a knob bitsandbytes doesn't have.

On **Qwen3-0.6B**, uniform 4-bit costs **+1.17 perplexity**; GPTQ-NF4 cuts that to
**+0.51** at the same 4 bits, and INT8 is effectively free (**+0.01**). The blue
mixed-precision curve sits strictly below the uniform line:

![Accuracy vs memory — Qwen3-0.6B](results/Qwen_Qwen3-0.6B-Base.png)

*WikiText-2 perplexity vs. weight memory on Qwen3-0.6B. Grey = uniform
quantization (NF4 / INT8 / FP16); blue = AtlasInfer's mixed-precision allocation
at several budgets. Lower-left is better — spending a few extra bits on the most
loss-sensitive layers recovers most of the 4-bit→FP16 accuracy gap.*

Full numbers for every model (Qwen3-0.6B, Qwen2.5-0.5B, plus Pythia/GPT-2) are in
[Benchmarks](#benchmarks) and under [`results/`](results/), reproducible with
`python benchmark.py --model <name>`.

---

## How it works

A model is quantized in four stages, one module per file:

```
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
weights │  quantizer   │   │ sensitivity  │   │  allocator   │   │   patcher    │ quantized
───────▶│ block-wise   │──▶│ per-layer,   │──▶│ exact budget │──▶│ swap layers  │──▶ model
        │ INT8 / NF4   │   │ per-bit error│   │ assignment   │   │ in-place     │
        └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

**1. Block-wise quantization with sparse outliers** (`quantizer.py`)
Weights are quantized with a separate scale per 64–128 element block: symmetric
INT8 for the 8-bit tier, and **NF4** (NormalFloat — a 16-level codebook matched
to the Gaussian distribution of weights, from QLoRA) for the 4-bit tier, with
symmetric INT4 also available. The handful of high-magnitude "outlier" weights
that dominate a layer's output (detected per block by a **robust median/MAD
z-score** — when a block holds several comparably large weights they inflate a
plain mean/std together and mask one another under the threshold, whereas
median/MAD tolerates that contamination and still flags them) are kept in FP16
and stored **sparsely** as `(index, value)` pairs — a dense boolean mask would
cost a full byte per weight and wipe out the savings.

**2. End-to-end sensitivity profiling** (`sensitivity.py`)
The key question is *which layers can tolerate aggressive quantization?* The
default profiler answers it directly: for each layer and candidate bit-width it
temporarily swaps in the quantized layer, re-evaluates the model's cross-entropy
on a small calibration set, and records the **increase in loss** that one layer
caused:

```
error(layer, bits) = NLL(model with only this layer quantized) − NLL_fp16
```

This measures a layer's true contribution to output degradation, so the
allocator optimizes the thing we actually care about (perplexity) rather than a
proxy. Switching from a layer-local error proxy to this end-to-end signal cut
the mixed-precision perplexity gap on Pythia-410M by ~4× at the same memory. A
faster activation-local profiler (`SensitivityProfiler.profile`) is also
included for quick experiments.

> Either way, sensitivity is measured on **real calibration activations** — not
> random noise fed through a layer in isolation, a common shortcut that produces
> sensitivity numbers unrelated to how the layer behaves on real text.

**3. Budget-constrained allocation** (`allocator.py`)
Given each layer's `(bytes, error)` options, choosing one precision per layer to
minimize total error under a byte budget is a **multiple-choice knapsack
problem**. AtlasInfer solves it with dynamic programming over a discretized
memory axis — optimal on that grid, whose resolution is far finer than the
per-block scale/outlier overhead the budget already approximates. A greedy
sensitivity-ranked baseline is kept for comparison; the DP beats it whenever the
most *sensitive* layer isn't the most *byte-efficient* to upgrade (covered by a
unit test).

![Per-layer precision allocation — Qwen3-0.6B](results/allocation_Qwen_Qwen3-0.6B-Base.png)

*The allocator's decisions on Qwen3-0.6B at a 5-bit budget: each dot is a linear
layer (y = how much quantizing it to 4-bit hurts the model's loss). The most
sensitive layers are kept at INT8 (blue) while the robust majority drop to 4-bit
(red). It's not a simple sensitivity threshold — the DP also weighs each layer's
byte cost, which is why a few mid-sensitivity but cheap layers stay at INT8.
Generate it with `python examples/04_visualize_allocation.py`.*

**4. In-place patching** (`patcher.py`)
Dense `nn.Linear` (and GPT-2 `Conv1D`) layers are swapped for quantized
equivalents in place. Quantized weights are stored as registered **buffers**, so
`model.to("cuda")` moves them with the rest of the model and they live
persistently on the GPU.

---

## Benchmarks

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
  ([ablation](PAPER/exp/results/ablation_Qwen_Qwen2.5-0.5B.json)).
- **GPTQ-NF4 does better still, at the *same* 4-bit footprint** (+0.48 vs +0.53)
  and at zero extra memory — see [vs bitsandbytes](#vs-bitsandbytes). On 3 of 4
  models GPTQ-NF4 beats mixed precision outright. The two mechanisms are
  orthogonal and `compare_baselines.py` now measures them composed
  (`gptq-mixed`), which is the open question.
<!-- /RESULTS:Qwen2.5-0.5B -->

Also validated on older architectures — GPT-2 (124M) and Pythia-410M / 1.4B —
under [`results/`](results/); the same mixed-precision win holds there too (it's
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

[`atlasinfer/triton_kernels.py`](atlasinfer/triton_kernels.py) implements **fused
dequant-GEMM** Triton kernels for both INT8 (`W8A16`) and packed INT4 (`W4A16`):
they read the low-bit weights directly (half / a quarter of FP16's bytes),
dequantize them in-register, and do the matmul in a single pass — no full-weight
materialization. At batch-1 decode (memory-bandwidth bound) both beat FP16.
Measured on an RTX 3060 under WSL2 (`bench_triton_kernel.py`, best-of-3, torch
2.5.1+cu124 / Triton 3.1.0; full table + correctness notes in
[results/triton_kernel_rtx3060.md](results/triton_kernel_rtx3060.md)):

Because batch-1 is bandwidth-bound, reading half (int8) or a quarter (int4) of
the weight bytes makes ~2× and ~4× *available for free*. So the honest column is
not the speedup — it's how much of that headroom the kernel actually captures:

| matmul shape (M, K, N) | fp16 | fused W8A16 | fused W4A16 | W8 vs fp16 (of 2.0× ideal) | W4 vs fp16 (of 4.0× ideal) |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1, 4096, 4096)  | 0.156 ms | 0.099 ms | 0.088 ms | **1.58×** (79%) | 1.78× (45%) |
| (1, 4096, 11008) | 0.337 ms | 0.173 ms | 0.123 ms | **1.95×** (98%) | 2.74× (69%) |
| (1, 5120, 5120)  | 0.198 ms | 0.112 ms | 0.109 ms | **1.78×** (89%) | 1.82× (46%) |
| (16, 4096, 4096) | 0.134 ms | 0.088 ms | 0.129 ms | 1.52× (76%) | 1.04× (26%) |

**Being straight about what this shows.** The FP16 baseline is *not* a straw man —
`F.linear` reaches 267 GB/s, about 93% of the card's ~288 GB/s peak. Against it,
**W8A16 is genuinely good**: 1.95× of an available 2.0×. **W4A16 is not**: it
captures only 26–68% of its headroom, and by M=16 it has collapsed to 1.04× —
where [Marlin](https://arxiv.org/abs/2408.11743) sustains close to the full 4×.
So these kernels beat FP16, but the 4-bit one does **not** compete with a
state-of-the-art quantized kernel, and no such comparison has been run here.

One further caveat that matters for reading the accuracy tables alongside these:
the kernels use **per-output-channel symmetric** int8/int4, which is *not* the
block-wise NF4 + sparse-outlier format every perplexity number on this page was
measured with. The fast path and the accurate path are different quantizers, and
no experiment here measures both on the same model.

Triton is Linux/GPU-only, so this needs **WSL2** on Windows; the module
import-guards on `HAS_TRITON` so the rest of the library is unaffected. See
[docs/wsl_triton.md](docs/wsl_triton.md) for the 3-command setup.

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
([`triton_kernels.py`](atlasinfer/triton_kernels.py)): (1) the kernel **autotunes
tile size / warps / pipeline depth per `(M, N, K)`**, and — the one that actually
mattered — (2) the int8/int4 weights are stored **transposed `(K, N)`** so the
kernel's fast tile axis is contiguous and the weight loads **coalesce** (with the
natural `(N, K)` layout each load was strided by K, which Ampere hides via
`cp.async` but Turing can't). Autotuning alone left the T4 ~7× slow — confirming
the bottleneck was the access pattern, not the schedule. The coalesced-layout fix
is in `quantize_w8a16`/`quantize_w4a16`; end-to-end T4 kernel throughput is
reproducible from the [Kaggle notebook](notebooks/kaggle_benchmark.ipynb) with
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
question. See [PAPER/01_go_nogo.md](PAPER/01_go_nogo.md) for the full accounting.
Per-model detail incl. mixed-precision and symmetric-int4 rows:
[Qwen3-0.6B](results/comparison_Qwen_Qwen3-0.6B-Base.md) ·
[Qwen2.5-0.5B](results/comparison_Qwen_Qwen2.5-0.5B.md) ·
[Pythia-1.4B](results/comparison_EleutherAI_pythia-1.4b.md) ·
[Pythia-410M](results/comparison_EleutherAI_pythia-410m.md).

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
run: [Qwen2.5-0.5B](results/downstream_Qwen_Qwen2.5-0.5B.md) ·
[Qwen3-0.6B](results/downstream_Qwen_Qwen3-0.6B-Base.md).

### Pushing further: double-quant, AWQ, and 2-bit

Three additions on top of the 4-bit story above, each implemented from scratch.
The framing is deliberately narrow: they push AtlasInfer to the accessible-quant
frontier (matching the memory *and* accuracy levers bitsandbytes/AWQ/GPTQ have),
not past the research frontier — the honest boundary is spelled out per item.

**1. Double-quantized scales** ([`double_quant.py`](atlasinfer/double_quant.py),
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
(and [results/double_quant_Qwen_Qwen2.5-0.5B.md](results/double_quant_Qwen_Qwen2.5-0.5B.md)).

**2. AWQ activation-aware scaling** ([`awq.py`](atlasinfer/awq.py)). A second,
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
[results/awq_Qwen_Qwen2.5-0.5B.md](results/awq_Qwen_Qwen2.5-0.5B.md).

**3. Sub-4-bit vector quantization** —
*experimental* ([`codebook.py`](atlasinfer/codebook.py)). A single-codebook vector
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

## Install

```bash
git clone https://github.com/nishantkluhera/AtlasInfer.git
cd AtlasInfer

# Install PyTorch for your CUDA version (see pytorch.org), e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install AtlasInfer (editable) + benchmark extras
pip install -e ".[benchmark]"
```

## Quickstart

```python
from atlasinfer import AtlasInference

# Uniform INT8 — near-lossless, ~1.5x smaller.
engine = AtlasInference("Qwen/Qwen2.5-0.5B")
print(engine.generate("The future of on-device AI is", max_tokens=40))

# Mixed precision within a memory budget (per-layer FP16/INT8/NF4).
engine = AtlasInference("Qwen/Qwen3-0.6B-Base", memory_budget_gb=0.6)
print(engine.generate("The future of on-device AI is", max_tokens=40))

# Best 4-bit accuracy: GPTQ error-compensated NF4 (needs a tokenizer + calib text).
from transformers import AutoModelForCausalLM, AutoTokenizer
from atlasinfer import quantize_model_gptq, quantize_model_awq
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base").cuda()
quantize_model_gptq(model, tokenizer=tok)                    # ~3x closer to FP16 than bnb NF4
quantize_model_gptq(model, tokenizer=tok, double_quant=True) # + QLoRA scale compression, same accuracy
quantize_model_awq(model, tokenizer=tok)                     # AWQ: same 4-bit tier, Hessian-free route
```

CLI:

```bash
python -m atlasinfer.inference --model Qwen/Qwen2.5-0.5B --prompt "Hello world"
python -m atlasinfer.inference --model Qwen/Qwen3-0.6B-Base --prompt "Hello" --memory-budget 0.6
```

The [`examples/`](examples/) folder has runnable scripts for basic inference,
the full manual mixed-precision pipeline, layer-sensitivity inspection, and the
per-layer allocation visualization shown above.

## Run the benchmarks

Every harness is seeded (`--seed 0`, deterministic kernels) and writes its table
+ JSON under `results/`, so a number is reproducible run-to-run on the same
hardware.

```bash
# 1. Perplexity / memory sweep (uniform + mixed).
python benchmark.py --model Qwen/Qwen3-0.6B-Base --eval-tokens 40000 --bits 4.5 5 6 7

# 2. Head-to-head vs the 4-bit stack: bitsandbytes + real GPTQ (auto-gptq) + AWQ.
pip install -e ".[baselines]"
python compare_baselines.py --model Qwen/Qwen2.5-0.5B

# 3. Downstream zero-shot accuracy (ARC / HellaSwag / PIQA / WinoGrande), not just perplexity.
pip install -e ".[eval]"
python eval_downstream.py --model Qwen/Qwen2.5-0.5B --limit 1000

# ...or all three, for several models, in one command:
python reproduce.py --models gpt2 Qwen/Qwen2.5-0.5B Qwen/Qwen3-0.6B-Base
```

The external baselines (`auto-gptq`/`gptqmodel`, `autoawq`) and `lm-eval` are
Linux/CUDA-only and version-fragile; each harness **skips any that fail to
import** with a clear note, so a partial install still produces a full table of
whatever's present.

### Bigger models (7–13B) on a cloud GPU

The benchmarks here cap at ~1.4B (6 GB laptop). To go bigger — and to run the
head-to-head against **real bitsandbytes / GPTQ / AWQ** (Linux/CUDA-only, so they
don't install on the Windows dev box) — use a cloud GPU:

- **Lightning AI** (single A100/L4): a priority-ordered runner + hour-budget plan
  in [`docs/lightning_7b.md`](docs/lightning_7b.md) — `bash run_lightning.sh compare`
  gives the AtlasInfer-vs-bitsandbytes head-to-head in ~30 min, across a **spread of
  families** (SmolLM3 / Phi-4-mini / Mistral / OLMo-2 / Qwen3, + gated Llama/Gemma),
  not just one vendor.
- **Kaggle** (free T4×2 / P100): the ready-made
  [`notebooks/kaggle_benchmark.ipynb`](notebooks/kaggle_benchmark.ipynb) clones,
  installs, and runs the full suite. A single 16 GB GPU fits ~≤4B for the FP16
  baseline; **`--device-map`** shards across both T4s for 7–13B:

```bash
python benchmark.py        --model Qwen/Qwen3.5-9B-Base --device-map
python compare_baselines.py --model Qwen/Qwen3.5-9B-Base --device-map
```

Two changes make this practical: GPTQ now uses a **block-batched** column update
(≈ group-size× fewer matmuls — minutes instead of hours at 7B) with
**memory-bounded chunked Hessians**, and quantization is **device-preserving** so
each layer stays on its shard. *(Multi-GPU is validated on single-GPU here; Kaggle
T4×2 is the intended test bed — see the notebook.)*

---

## Project layout

```
atlasinfer/
├── quantizer.py     # block-wise INT8 / NF4 / INT4 + sparse FP16 outliers
├── linear.py        # QuantizedLinear / QuantizedLinear4bit (buffer-backed)
├── sensitivity.py   # calibration-based per-layer error profiler
├── allocator.py     # exact (DP) + greedy budget allocators
├── gptq.py          # GPTQ error-compensated NF4 (Hessian-based) + outliers
├── awq.py           # AWQ activation-aware per-channel scaling on the NF4 path
├── double_quant.py  # QLoRA-style double-quantized block scales (--double-quant)
├── codebook.py      # EXPERIMENTAL sub-4-bit vector-quantized codebook (unvalidated)
├── patcher.py       # in-place layer replacement (nn.Linear & Conv1D)
├── offload.py       # optional CPU<->GPU layer streaming
├── triton_kernels.py# fused W8A16/W4A16 dequant-GEMM kernels (Linux/WSL2, guarded)
└── inference.py     # high-level API + CLI
benchmark.py         # WikiText-2 perplexity / stored-size harness
bench_latency.py     # runtime peak-GPU-memory + decode tok/s harness
bench_triton_kernel.py # fused-kernel microbenchmark (Linux/WSL2)
compare_baselines.py # head-to-head vs bitsandbytes (Linux/WSL2)
examples/            # runnable usage scripts
notebooks/kaggle_benchmark.ipynb  # run the suite on bigger models (Kaggle GPUs)
tests/               # pytest suite
docs/wsl_triton.md   # WSL2 setup for the Triton kernel
```

## What this does and doesn't do

- **Two paths, by precision.** The default eager path dequantizes weights to
  FP16 per `matmul` — a memory win at a latency cost (runtime table above). The
  **fused Triton W8A16 kernel** removes that cost and is *faster* than FP16 at
  batch-1 decode (kernel table above), but it's Linux/GPU-only (WSL2 on Windows).
  A note on what *doesn't* work: the obvious shortcut — INT8×INT8 GEMM via
  `torch._int_mm` (W8A8) — measured *slower* than FP16 on this consumer Ampere
  card once activation quant/dequant overhead is counted, and can't do batch-1
  decode, so it isn't used. The fused dequant kernel is the right approach.
- **Where it sits in the literature.** The pieces are established ideas —
  sensitivity-driven mixed precision (HAWQ, SqueezeLLM), NF4 + double-quant (QLoRA),
  Hessian-based error compensation (GPTQ), activation-aware scaling (AWQ). AtlasInfer
  is a clean, self-contained, from-scratch implementation that *combines* them (GPTQ
  and AWQ on top of NF4 + sparse outliers + double-quantized scales, under an exact
  budget allocator) with reproducible benchmarks — not a new algorithm, but it lands
  at the GPTQ/AWQ accuracy tier at 4-bit, ahead of the bitsandbytes baseline. For
  **2–3 bit** there's now an *experimental* single-codebook vector-quantization path
  ([`codebook.py`](atlasinfer/codebook.py)) — implemented and unit-tested, but
  explicitly **not validated at scale** against the AQLM/QuIP#/QTIP frontier (which
  is established on 7B models this repo's 6 GB dev GPU can't run), so no
  SOTA-beating claim is made there.
- **GPTQ is calibration-hungry, memory-heavy, and compute-heavy** — three gotchas
  I hit and fixed. (1) Too few tokens -> rank-deficient Hessian -> compensation
  *hurts* (~1k tokens made 4-bit worse than plain NF4); ~65k tokens (128×512)
  gives the gains above. (2) Holding every layer's Hessian at once OOMs above
  ~0.6B, so the driver uses memory-bounded **chunked** Hessians (also making it
  true *sequential* GPTQ). (3) The naive per-column update is hours at 7B, so it's
  **block-batched** (one matmul per group instead of per column). Together these
  let GPTQ scale from 0.6B to 7–13B (Kaggle T4×2).
- **Perplexity** is measured on a capped slice of WikiText-2 test for speed; the
  exact token count is a benchmark flag, so absolute numbers shift slightly with
  it while the *relative* ordering (the point of the comparison) is stable.

## Roadmap

- **Done:** NF4 codebook for the 4-bit tier (beats bitsandbytes' NF4).
- **Done:** GPTQ Hessian-based error compensation on the NF4 path — takes 4-bit
  to the GPTQ/AWQ tier (Qwen3-0.6B +0.51, Qwen2.5-0.5B +0.48), see above.
- **Done:** AWQ activation-aware scaling on the NF4 path
  ([`awq.py`](atlasinfer/awq.py)) — a second, complementary route to the GPTQ/AWQ
  4-bit tier, from scratch.
- **Done:** double-quantized block scales
  ([`double_quant.py`](atlasinfer/double_quant.py), `--double-quant`) — removes the
  FP32 scale overhead at ~unchanged perplexity, narrowing the 4-bit memory gap to
  bnb.
- **Done:** fused W8A16 **and** W4A16 Triton kernels (1.6–2.7× over FP16 at
  batch-1 decode, WSL2/RTX 3060), wired into `AtlasInference` (`kernel="auto"`).
- **Experimental:** 2–3 bit via a single-codebook vector quantizer
  ([`codebook.py`](atlasinfer/codebook.py)) — the direction toward the
  AQLM/QuIP#/QTIP frontier; correctness-tested, **not yet validated at scale**.
  Next: residual/additive codebooks and incoherence pre-processing.
- Activation/KV-cache quantization, not just weights.

## License

MIT — see [LICENSE](LICENSE).

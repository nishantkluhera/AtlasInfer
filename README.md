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
inspectable in a few hundred lines. There's also an optional **fused W8A16 Triton
kernel** that makes batch-1 decode **1.4–1.6× faster than FP16** (Linux/WSL2).

---

## The result

Three measured claims, on **current models** (Qwen3, 2025; Qwen2.5, 2024):

1. **It beats bitsandbytes**, the standard accessible-quant library: AtlasInfer's
   INT8 is lossless and edges LLM.int8(), and at 4-bit its **GPTQ-NF4** path is
   ~3–4x closer to FP16 than bitsandbytes' NF4 (see [vs bitsandbytes](#vs-bitsandbytes)).
2. **GPTQ error compensation closes the 4-bit gap**: on Qwen3-0.6B the 4-bit
   penalty drops from +1.34 (plain NF4) to **+0.45** (GPTQ-NF4) at the *same*
   memory — better than even 5-bit mixed precision.
3. **Per-layer mixed precision beats any uniform bit-width** at a given footprint
   — a knob bitsandbytes doesn't have.

On **Qwen3-0.6B**, uniform 4-bit costs **+1.34 perplexity**; GPTQ-NF4 cuts that to
**+0.45** at the same 4 bits, and INT8 is effectively free (**+0.01**). The blue
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
that dominate a layer's output (detected per block by z-score) are kept in FP16
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
| fp16 | 16.0 | 1136.9 | 11.803 | +0.000 |
| uniform-int8 | 8.0 | 738.8 | 11.813 | +0.010 |
| uniform-nf4 | 4.0 | 564.7 | 13.143 | +1.340 |
| mixed-4.5bit | 4.5 | 587.0 | 12.682 | +0.879 |
| mixed-5bit | 5.0 | 608.2 | 12.535 | +0.732 |
| mixed-6bit | 5.9 | 650.0 | 12.300 | +0.497 |
| mixed-7bit | 7.0 | 699.8 | 12.120 | +0.317 |
<!-- /RESULTS:Qwen3-0.6B-Base -->

<!-- RESULTS:Qwen2.5-0.5B -->
### Qwen/Qwen2.5-0.5B  (2024)

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 942.3 | 12.279 | +0.000 |
| uniform-int8 | 8.0 | 620.5 | 12.286 | +0.007 |
| uniform-nf4 | 4.0 | 479.6 | 13.222 | +0.943 |
| mixed-4.5bit | 4.5 | 498.8 | 12.878 | +0.599 |
| mixed-5bit | 5.0 | 515.2 | 12.778 | +0.499 |
| mixed-6bit | 6.0 | 550.8 | 12.660 | +0.381 |
| mixed-7bit | 6.9 | 586.1 | 12.561 | +0.282 |

At ~4.5 bits — essentially the same footprint as uniform 4-bit NF4 — mixed
precision roughly **halves** the perplexity penalty (here +0.94 → +0.60) by
spending the extra half-bit only on the layers that hurt most.
<!-- /RESULTS:Qwen2.5-0.5B -->

Also validated on older architectures — GPT-2 (124M) and Pythia-410M / 1.4B —
under [`results/`](results/); the same mixed-precision win holds there too (it's
largest on models where uniform 4-bit is most lossy, e.g. +5.7→+1.4 on Pythia-410M).

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
Measured on an RTX 3060 (`bench_triton_kernel.py`, best-of-3):

| matmul shape (M, K, N) | fp16 | fused W8A16 | fused W4A16 | W8 vs fp16 | W4 vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1, 4096, 4096)  | 0.120 ms | 0.078 ms | 0.080 ms | **1.55x** | **1.51x** |
| (1, 4096, 11008) | 0.297 ms | 0.208 ms | 0.152 ms | **1.43x** | **1.95x** |
| (1, 5120, 5120)  | 0.177 ms | 0.105 ms | 0.101 ms | **1.68x** | **1.76x** |

The edge narrows as batch grows and the matmul becomes compute- rather than
bandwidth-bound (≈1.0–1.4x at M=16), exactly as expected. The W8A16 path is
near-lossless (<1% error); the W4A16 kernel uses per-channel int4 (coarser than
the default block-wise + outlier INT4), so it trades a little accuracy for the
4-bit bandwidth — use it when speed matters most. Triton is Linux/GPU-only, so
this needs **WSL2** on Windows; the module import-guards on `HAS_TRITON` so the
rest of the library is unaffected. See [docs/wsl_triton.md](docs/wsl_triton.md)
for the 3-command setup.

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

All the runtime/kernel tables above are on an Ampere RTX 3060. Kaggle's free
**Tesla T4** (Turing) is the most accessible GPU for reproducing this, so these T4
measurements are reported **separately** rather than mixed into the 3060 numbers.

**Eager decode reaches 3B — and the latency cost grows with model size.**
Qwen2.5-3B, 64-token decode, eager block-wise path
(`python bench_latency.py --model Qwen/Qwen2.5-3B`):

| Config | Peak GPU (MB) | vs FP16 | tok/s | rel. speed |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 6008.3 | 1.00x | 19.8 | 1.00x |
| uniform-int8 | 3684.6 | 0.61x | 3.6 | 0.18x |
| uniform-nf4 | 2761.3 | 0.46x | 1.0 | 0.05x |
| mixed-5bit | 3045.1 | 0.51x | 1.2 | 0.06x |

Same shape as the smaller models — **~40–55% less peak memory at a decode-speed
cost** — but more pronounced at 3B on a T4, where the eager codebook dequant
dominates (NF4 at 1.0 tok/s). This is squarely the "fits but slow" regime the
fused kernel exists to remove.

**Kernel portability (Turing vs Ampere).** The fused kernel's batch-1 speedup was
tuned on Ampere. A single fixed tile that streams at ~215 GB/s on the 3060 stalled
at ~17 GB/s on the T4 — roughly **7× *slower* than FP16** — because Turing lacks
the `cp.async` software pipelining Ampere relies on, so that schedule can't hide
HBM latency. The fix isn't T4-specific code: the kernel now **autotunes its tile
size, warp count, and pipeline depth per `(M, N, K)`**
([`triton_kernels.py`](atlasinfer/triton_kernels.py)), so Triton picks an Ampere
schedule on the 3060 and a higher-occupancy, shallower-pipeline schedule on the
T4. *(Post-autotune T4 kernel throughput: re-measure with
`bench_triton_kernel.py` — pending.)*

### vs bitsandbytes (the accessible-quant baseline)

Head-to-head, same WikiText-2 eval and same memory accounting
(`python compare_baselines.py`), across **four models** — current (Qwen3/Qwen2.5),
a larger one (Pythia-1.4B), and a hard small one (Pythia-410M). **Δ perplexity vs
FP16** per method (lower is better; INT8 columns are 8-bit, the rest 4-bit):

| Model | Atlas int8 | bnb int8 | bnb nf4 | Atlas nf4 | **Atlas gptq-nf4** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B (2025) | +0.009 | +0.057 | +1.834 | +1.339 | **+0.453** |
| Qwen2.5-0.5B (2024) | +0.007 | +0.070 | +1.323 | +0.943 | **+0.558** |
| Pythia-1.4B | −0.005 | +0.056 | +0.922 | +0.754 | **+0.362** |
| Pythia-410M | +0.024 | +0.121 | +5.938 | +5.681 | **+4.019** |

On **every** model: AtlasInfer's **INT8 matches/beats** bitsandbytes' LLM.int8(),
and at **4-bit, GPTQ-NF4 is the best method** — beating bnb's NF4, plain NF4, and
(on the Qwen/Pythia-1.4B models) even AtlasInfer's own 5-bit mixed precision, at
the same 4-bit memory. The win is largest on the modern models (~3–4× closer to
FP16 than bnb); even the hard Pythia-410M case improves +5.9→+4.0. (bnb is ~10–13%
smaller at 4-bit — it double-quantizes its scales, a roadmap item.) Per-model
detail incl. mixed-precision and symmetric-int4 rows:
[Qwen3-0.6B](results/comparison_Qwen3-0.6B-Base.md) ·
[Qwen2.5-0.5B](results/comparison_Qwen2.5-0.5B.md) ·
[Pythia-1.4B](results/comparison_pythia-1.4b.md) ·
[Pythia-410M](results/comparison_pythia-410m.md).

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
from atlasinfer import quantize_model_gptq
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base").cuda()
quantize_model_gptq(model, tokenizer=tok)   # ~3-4x closer to FP16 than bnb NF4
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

```bash
python benchmark.py --model Qwen/Qwen3-0.6B-Base --eval-tokens 40000 --bits 4.5 5 6 7
python compare_baselines.py --model Qwen/Qwen2.5-0.5B   # head-to-head vs bitsandbytes
```

Outputs a markdown table, a JSON dump, and a perplexity-vs-memory figure under
`results/`.

### Bigger models (7–13B) on a free Kaggle GPU

The benchmarks here cap at ~1.4B (6 GB laptop). To go bigger, run the ready-made
[`notebooks/kaggle_benchmark.ipynb`](notebooks/kaggle_benchmark.ipynb) on Kaggle's
free **T4×2 (32 GB)** or **P100 (16 GB)**: it clones, installs, and runs the full
suite on a model you pick. A single 16 GB GPU fits ~≤4B for the FP16 baseline;
**`--device-map`** shards across both T4s for 7–13B:

```bash
python benchmark.py        --model Qwen/Qwen2.5-7B --device-map
python compare_baselines.py --model Qwen/Qwen2.5-7B --device-map
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
  sensitivity-driven mixed precision (HAWQ, SqueezeLLM), NF4 (QLoRA), Hessian-based
  error compensation (GPTQ). AtlasInfer is a clean, self-contained, from-scratch
  implementation that *combines* them (GPTQ on top of NF4 + sparse outliers, under
  an exact budget allocator) with reproducible benchmarks — not a new algorithm,
  but it lands at the GPTQ/AWQ accuracy tier at 4-bit, ahead of the bitsandbytes
  baseline. (It does *not* do 2–3 bit; that needs vector/trellis codebooks like
  AQLM/QuIP#/QTIP.)
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
  to the GPTQ/AWQ tier (Qwen3-0.6B +0.45, Qwen2.5-0.5B +0.56), see above.
- **Done:** fused W8A16 **and** W4A16 Triton kernels (1.4–1.95x over FP16 at
  decode), wired into `AtlasInference` (`kernel="auto"`).
- Double-quantize the block scales (as bitsandbytes does) to close the remaining
  ~10% memory gap at 4-bit.
- 2–3 bit via a vector/trellis codebook (AQLM/QuIP#/QTIP family) — the current
  frontier we don't yet reach.
- Activation/KV-cache quantization, not just weights.

## License

MIT — see [LICENSE](LICENSE).

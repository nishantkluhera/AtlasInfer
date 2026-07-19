# Validating AtlasInfer at 7B on Lightning AI

The laptop benchmarks cap at ~1.4B (6 GB). This runs the **same harness at 7B on a
cloud GPU**, which does two things the laptop can't:

1. **The real head-to-head.** `bitsandbytes`, `gptqmodel` (auto-GPTQ), and `autoawq`
   are Linux/CUDA-only and don't install on the Windows dev box — on Lightning they
   do, so you finally compare AtlasInfer's INT8 / NF4 / GPTQ-NF4 against **actual
   bitsandbytes, real GPTQ, and real AWQ** on one eval, at the scale the field uses.
2. **Credibility.** "Validated at 7B vs the standard 4-bit stack" is a different
   claim than "reimplemented and tested at ≤1.4B."

No novelty is asserted here — this is a **validation/credibility** run.

## 1. Studio + GPU

Pick by what you want and how many credits you'll spend (interruptible prices):

| GPU | Mem | ~$/hr | Use it for |
| --- | ---: | ---: | --- |
| **L40S** | 48 GB | 2.49 | **recommended** — one clean full run incl. the external GPTQ/AWQ baselines; a full `compare+sweep+downstream` is ~3–5 h (~$8–12) |
| **L4** | 24 GB | 0.68 | cheapest; fits 7B for the AtlasInfer-vs-**bitsandbytes** headline. Skip the external GPTQ/AWQ baselines here (24 GB is too tight for auto-gptq on 7B) |
| A100 / H100 | 40 / 80 GB | 3.32 / 3.82 | if you want it faster; H100 is ~6× the FLOPs |
| RTX Pro 6000 / B200 | 96 / — | 1.97 / 9.86 | **avoid for this** — Blackwell (sm_120); `bitsandbytes`/`autoawq` wheels may not load, and those baselines are the point |
| T4 | 16 GB | 0.36 | ≤3B only — 7B FP16 won't fit |

- **13B+** needs two GPUs: `export DEVICE_MAP=--device-map` and attach ≥2.
- **Interruptible** GPUs can be preempted, and each stage writes results only when
  it finishes — so run stages **separately** (not `all`), so a preemption costs one
  stage, not the session.
- In the Studio terminal:

```bash
git clone https://github.com/nishantkluhera/AtlasInfer.git && cd AtlasInfer
```

## 2. Models — test a spread of FAMILIES, latest open (July 2026)

A quant result on one vendor (Qwen) is weak evidence; **cross-family** results —
different weight distributions, outlier structure, and norms — are what convince a
reviewer you didn't tune for one model. Prefer **dense** transformers (what
AtlasInfer is validated on) and **ungated** ones (no login). Verify tags on HF —
they rev fast.

> **What "diverse" means here:** architectural/distributional diversity, not brand
> count. Eight Llama-shaped dense transformers ≈ one datapoint repeated. Span
> different design choices (norm placement, activation, GQA vs MHA, dense vs MoE).

**Top open families with a ≥7B model.** Confirm every tag on HF before running —
these rev fast and some are gated.

| Family | Model (≥7B) | Gating | Arch note |
| --- | --- | --- | --- |
| Mistral | `mistralai/Mistral-7B-v0.3` | open (Apache) | dense; sliding-window + GQA |
| Alibaba Qwen | `Qwen/Qwen3-8B-Base` | open (Apache) | dense |
| Google Gemma | `google/gemma-2-9b` (or `gemma-3-12b-pt`) | click-through | dense; **distinct** norm/GeGLU/logit-capping |
| Microsoft Phi | `microsoft/Phi-4` (14B) / `Phi-4-mini` (3.8B) | open (MIT) | dense; distinct data recipe |
| AI2 OLMo | `allenai/OLMo-2-…-7B` | open (Apache) | dense; fully open data |
| Meta Llama | `meta-llama/Llama-3.1-8B` | gated | dense; the reference arch |
| 01.AI Yi | `01-ai/Yi-1.5-9B` | open | dense (Llama-like) |
| InternLM | `internlm/internlm2_5-7b` | open | dense |
| Falcon | `tiiuae/Falcon3-7B-Base` | open | dense |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | open | ⚠️ **Qwen arch underneath** — DeepSeek *data*, not DeepSeek *architecture*. True DeepSeek arch (V2/V3 MoE) is 236B–671B, out of budget. Their own dense `deepseek-llm-7b-base` is 2023. |
| OpenAI gpt-oss | `openai/gpt-oss-20b` | open | **MoE** — genuinely different arch class |
| Alibaba Qwen (latest) | `Qwen/Qwen3.5-9B-Base` | open | **hybrid** Gated-Delta + MoE (see caveat) |
| field-standard | `meta-llama/Llama-2-7b-hf` | gated | what QuIP#/AQLM/GPTQ report on |

**Recommended subset for ~15 credits** (maximum architectural spread, mostly
ungated — ~4 × `compare` ≈ 2 h ≈ $5 on L40S):

`mistralai/Mistral-7B-v0.3` · `Qwen/Qwen3-8B-Base` · `google/gemma-2-9b` ·
`microsoft/Phi-4-mini-instruct` — four genuinely different design lineages. Add
`openai/gpt-oss-20b` if you want an MoE datapoint, or `Llama-3.1-8B` for the
reference arch.

Run the whole cross-family suite (benchmark + baselines + downstream per model —
several hours, do this only with credits to spare):

```bash
python reproduce.py --device-map --models \
  mistralai/Mistral-7B-v0.3 Qwen/Qwen3-8B-Base \
  google/gemma-2-9b microsoft/Phi-4-mini-instruct
```

On a **budget**, the credible-spread-per-credit move is just the headline `compare`
across the four lineages (~30 min each):

```bash
for m in mistralai/Mistral-7B-v0.3 Qwen/Qwen3-8B-Base \
         google/gemma-2-9b microsoft/Phi-4-mini-instruct; do
  bash run_lightning.sh compare "$m"
done
```

> ⚠️ **Arch caveat (Qwen3.5 & any MoE model):** AtlasInfer quantizes
> `nn.Linear`/`Conv1D` and is validated on **dense** transformers. Qwen3.5 (Gated
> Delta Networks + sparse MoE) and MoE models get their linear/expert layers
> quantized, but the per-layer sensitivity profiler assumes every layer fires per
> token (weaker under routing). Run the `setup` smoke first; if it misbehaves, stick
> to the dense families above.

**Auth:** ungated models need no login. For gated Llama/Gemma:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
```

## 3a. Unattended / overnight — one command, then walk away

```bash
mkdir -p results/_logs
nohup bash run_lightning.sh overnight > results/_logs/overnight.log 2>&1 &
```

Runs `setup`, then `compare` across **3 ungated families** (nothing can stall on a
Hugging Face login while you're asleep), then `sweep` + `downstream` + `latency` on
the main model. Each stage writes its own log, **a failing stage never kills the
run**, and a summary table prints at the end. ~5 h ≈ 12 credits on an L40S.

`nohup … &` keeps it alive if your browser disconnects. In the morning:

```bash
tail -40 results/_logs/overnight.log        # summary table is at the bottom
cat results/_logs/overnight_summary.txt     # per-stage OK/FAIL + minutes
```

Options: `FAMILIES="a b c"` to change the family list (add gated Gemma/Llama only
if you ran `huggingface-cli login` first), `AUTOSTOP=1` to try powering the machine
off when done so idle credits aren't burned — verify it works for your Studio, and
set the Lightning UI idle-timeout as the reliable backstop.

> Prefer a **non-interruptible** GPU for an unattended run: an interruptible one can
> be preempted mid-stage, and results are only written when a stage completes.

## 3. Run — priority-ordered so you can stop when hours run out

```bash
bash run_lightning.sh setup                 # install + 0.5B smoke (~5-10 min)
bash run_lightning.sh compare               # <-- the headline result
bash run_lightning.sh sweep                 # mixed-precision Pareto
bash run_lightning.sh downstream            # task accuracy, not just ppl
bash run_lightning.sh latency               # decode tok/s + peak mem
# or: bash run_lightning.sh all
# pick a model:  bash run_lightning.sh compare meta-llama/Llama-2-7b-hf
```

### Hour budget (rough, on one A100-40GB)

| Stage | ~time | What you get | Priority |
| --- | ---: | --- | --- |
| `setup` | 5–10 min | env + 0.5B smoke (fail fast before spending 7B hours) | required |
| `compare` | 30–120 min | AtlasInfer int8/nf4/gptq-nf4(+dq)/awq **vs bnb (+ real GPTQ/AWQ)** at 7B | **do this first** |
| `sweep` | 40–70 min | uniform + mixed-precision perplexity/memory curve at 7B | high |
| `downstream` | 45–90 min | ARC/HellaSwag/PIQA/WinoGrande accuracy at 7B | medium-high |
| `latency` | 10–15 min | decode tok/s + peak GPU memory | nice-to-have |

`compare`'s spread is because the external GPTQ/AWQ baselines dominate it (each
quantizes the 7B on load). For a lean, high-value first pass, install without them
(`pip install -e '.[benchmark,eval]' bitsandbytes`) — you still get the AtlasInfer
vs **bitsandbytes** head-to-head, which is 80% of the credibility for ~30 min.

**Budget guide:** ~2 h → `setup`+`compare` (bnb only). ~4 h → add `sweep` + the
external GPTQ/AWQ baselines. ~6 h+ → add `downstream`, and/or repeat `compare` on
`meta-llama/Llama-2-7b-hf` (the exact model QuIP#/AQLM/GPTQ papers report on).

## 4. Collect results

Everything lands in `results/` (`comparison_*.md/.json`, `<model>.md/.json/.png`,
`downstream_*.md/.json`). Zip and download:

```bash
zip -r atlasinfer_7b_results.zip results && echo "download atlasinfer_7b_results.zip"
```

Then commit them and update the README's benchmark tables (the CI consistency test
in `tests/test_readme_consistency.py` will fail if the numbers you paste don't match
the JSON — that's intended: it keeps the README honest).

## Gotchas

- **The Studio's shared conda env is often broken — install into a venv.** Lightning's
  `cloudspace` env can ship half-installed distributions (`Ignoring invalid
  distribution ~umpy`) and missing `dist-info` dirs that make `pip` abort with
  `OSError`, and it commonly has `numpy 2.x` alongside `scipy`/`sklearn` compiled
  for NumPy 1.x. That last one breaks **every** harness, because `transformers`
  imports `sklearn` → `scipy` → `ImportError: numpy.core.multiarray failed to
  import`. `setup` therefore builds an isolated `.venv` by default (adds ~2-4 min
  for a fresh torch wheel, and removes the whole class of failure). Override with
  `USE_VENV=0` to use the host env — then pin `numpy<2` yourself.

- **External baselines are version-fragile.** If `gptqmodel`/`autoawq` fail to
  install or import against the current `transformers`, the harness prints a SKIP
  and continues — you still get bnb + all AtlasInfer methods. Don't fight it.
- **Memory.** 7B FP16 baseline ~14 GB; GPTQ peak ~18–22 GB. Fits A100-40GB easily,
  L4-24GB tightly. 13B+ → `export DEVICE_MAP=--device-map` and attach ≥2 GPUs.
- **`latency` is single-GPU only** — don't pass `--device-map` to it.
- **Reproducibility.** Every harness is seeded (`--seed 0`, deterministic kernels),
  so a number is stable run-to-run on the same GPU.

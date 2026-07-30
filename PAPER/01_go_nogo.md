# Phase 2 — GO / NO-GO

Written 2026-07-30. **§1 (kill criteria) was committed before any Phase 2
experiment was run** — see git history for `PAPER/01_go_nogo.md`. Results were
appended afterwards in §2 onward.

---

## 1. Kill criteria — PRE-REGISTERED

Each criterion states, in advance, the observation that would make me abandon or
downgrade the submission. Verdicts get filled in after testing; **the criteria
themselves do not get edited.** If I later want to revise one, that fact gets
written down as a finding rather than applied silently.

| # | Criterion | Kill condition | Testable now? |
| --- | --- | --- | --- |
| **K1** | **Novelty of the formulation** | ≥1 prior work exists that (a) targets LLMs, (b) assigns bit-width per layer/block, (c) under an explicit memory budget, (d) by knapsack / DP / ILP over *measured* per-layer sensitivity. → The "novel method" framing is dead. | Yes — literature |
| **K2** | **Cross-family generality** | Mixed precision fails to beat the best uniform baseline on ≥3 of the 4 tested families (Qwen, Pythia/NeoX, GPT-2). → It's a Qwen artifact. | Partly — have 4 models, need matched memory |
| **K3** | **Iso-memory survival** | At *matched measured MB* (not nominal bits), mixed precision reduces Δppl vs the best uniform baseline by **<15%** on the majority of models. → The headline is an artifact of an unmatched comparison. | **Needs running** |
| **K4** | **Not dominated by a known baseline** | A published baseline the repo already implements (GPTQ-NF4 / AWQ-NF4) Pareto-dominates mixed precision — better ppl at ≤ memory — on the majority of models, **and** mixed precision fails to improve on it when composed. → The method is not worth publishing as the headline. | Partly — dominance known; composition **needs running** |
| **K5** | **Attribution to the knapsack** | At matched budget, `allocate_optimal` (DP) fails to beat `allocate_greedy` (sensitivity-ranked) by more than run-to-run noise. → The MCKP formulation is not the contribution; the profiler is. | **Needs running** |
| **K6** | **Kernel contribution** | The fused kernels lose to an existing quantized kernel (Marlin / ExLlamaV2 / bnb / `torch.compile`) at matched accuracy, **and** end-to-end decode remains slower than FP16. → No systems contribution. | Partly — literature; measurement blocked on Linux/Triton |
| **K7** | **Scale credibility** | No honest ≥7B evaluation is obtained before the deadline. → MLSys reviewers dismiss on scale alone. | Known: not yet obtained |

**Decision rule fixed in advance:**
- K1 fails **and** (K3 or K5) fails → **Option 3** (no paper; release + blog).
- K1 fails, K3 and K5 hold, K4 rescued by composition → **Option 2** (workshop),
  upgradeable to Option 1 only if K6 and K7 both clear.
- K1 holds → reassess for Option 1.

---

## 2a. Novelty — VERDICT: **K1 FAILS. Decisively.**

I searched arXiv, MLSys/PPoPP/ICML/ICCV proceedings, and 2024–2026 quantization
surveys. The formulation is not new. It is not even recently not-new.

### Direct hits on the exact framing

**1. MPQCO — "Towards Mixed-Precision Quantization of Neural Networks via
Constrained Optimization", ICCV 2021 (arXiv:2110.06554).**
Explicitly formulates bit-width assignment as a **Multiple-Choice Knapsack
Problem**, derived from a Hessian-based per-layer error proxy. This is
AtlasInfer's `allocate_optimal` formulation, five years earlier. CNNs, not LLMs —
which is the only daylight, and it is not much.

**2. Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh — "Pushing the Limits
of Large Language Model Quantization via the Linearity Theorem", arXiv:2411.17525
(Nov 2024).** This one is the problem. It:
- states that "optimal non-uniform compression can be reduced to **knapsack-style
  dynamic programming** over the set of quantization choices at each layer" —
  AtlasInfer's exact sentence;
- measures per-layer sensitivity by **perturbing one layer at a time and reading
  the perplexity change**, which is functionally `SensitivityProfiler.profile_end_to_end`;
- and then does the thing AtlasInfer does not: **proves** the linearity theorem
  that justifies summing independent per-layer errors — the additivity assumption
  AtlasInfer's DP silently relies on and never defends;
- evaluates on Llama-3.1-8B/70B, Llama-3.2-1B/3B, Qwen2.5-7B.

From Alistarh's group — the GPTQ / SpQR / AQLM / Marlin lineage. A reviewer in
this area will know this paper.

**3. HAWQ-V3 (Yao et al., ICML 2021).** Bit-precision by **integer linear
programming** under memory/BOPS/latency constraints, Hessian sensitivity. The
canonical citation for "bit allocation as constrained optimization."

**4. LLM-MQ (2024).** Integer programming for per-layer LLM bit-widths under a
budget, first-order sensitivity.

**5. SqueezeLLM (ICLR 2024), SpQR (2023), OWQ (2023).** All sensitivity-driven
mixed-precision LLM weight quantization with **sparse FP16 outlier extraction** —
i.e. also prior art for AtlasInfer's *second* mechanism, the median/MAD sparse
outlier path. SpQR and SqueezeLLM both store outliers as a sparse FP16 matrix.

### Where AtlasInfer actually sits

| Component | Prior art | AtlasInfer's delta |
| --- | --- | --- |
| MCKP for bit allocation | MPQCO 2021 | none (CNN→LLM port) |
| Knapsack-DP for **LLM** layers | Linearity Theorem 2024 | none |
| Sensitivity = ΔPPL from single-layer perturbation | Linearity Theorem 2024 | none |
| ILP under memory budget | HAWQ-V3 2021, LLM-MQ 2024 | DP instead of ILP (an implementation choice, not a contribution) |
| Sparse FP16 outliers | SpQR, SqueezeLLM, OWQ 2023 | median/MAD detector instead of Hessian-based — **genuinely different, and defensible, but small** |
| NF4 codebook | QLoRA 2023 | none (reimplementation) |
| GPTQ on NF4 | GPTQ 2022 + QLoRA 2023 | the combination is uncommon; the components are not |
| Fused W4A16 dequant-GEMM | Marlin (PPoPP 2025), ExLlamaV2, AWQ-GEMM, bnb | none — and see §2c |

**The one thing I could not find pre-empted:** the *median/MAD robust outlier
detector* as an alternative to Hessian/magnitude criteria. That is a paragraph in
someone else's paper, not a paper.

**This also confirms a conclusion reached in an earlier adversarial search on this
project** (recorded in project memory): the PTQ mixed-precision/outlier design
space is saturated; multiple previously-considered "novel" directions were killed
against specific 2024–2026 papers. This search independently reproduces that
result and now names the single most damaging citation (Linearity Theorem, 2024).

**K1: FAILED.** There is no novel-method paper here. Anything written must be
framed as reproduction, systems engineering, or empirical study — never as a new
allocation method.

---

### Even the rescue path is pre-empted

If the pivot were "GPTQ error compensation *composed with* per-layer mixed
precision" (the obvious remaining combination — see §2d), that is also taken:
**APTQ** uses Hessian traces to guide selective mixed-precision on top of GPTQ,
and **oQ / oQ+** is an existing system described as measuring "each layer's actual
quantization sensitivity through calibration and allocat[ing] bits where the data
says they matter most," with a GPTQ variant. And the fallback framing — a
controlled empirical study of *what actually buys accuracy* — collides with
"A Comprehensive Evaluation on Quantization Techniques for LLMs" (Jan 2026) and
"An empirical study of LLaMA3 quantization" (2024), both done at scale.

There is no direction out of this that is not already occupied.

---

## 2b. Is the accuracy result real, or an artifact? — **K3: MARGINAL PASS,
## with a much weaker honest claim than the one being made**

### The comparison as stated is not measurable

A mixed allocation over {fp16, int8, int4} **can never be cheaper than uniform
int4** — int4 is the cheapest option, so the all-int4 assignment is the DP's
memory floor. "Mixed precision at equal memory to uniform NF4" is therefore
*unmeasurable by construction*: at that budget the DP returns uniform NF4.

So the honest question is not "mixed vs uniform-NF4 at equal memory." It is:
**at an intermediate budget, does the allocation beat what a practitioner would
trivially do instead — quantize some layers to INT8 and the rest to NF4?**
The natural reference is the line between the two uniform operating points.

### Against that reference, the win shrinks — and inverts at high budgets

Gain = (interpolated ppl at the same MB) − (measured mixed ppl), expressed as a
percentage of the perplexity gap the interpolation line still has to FP16:

| Model | @4.5 bit | @5 bit | @6 bit | @7 bit |
| --- | ---: | ---: | ---: | ---: |
| Pythia-410M | **+74.1%** | **+77.5%** | **+73.7%** | +43.9% |
| Qwen3-0.6B | +23.1% | +25.8% | +29.5% | **−16.3%** |
| Qwen2.5-0.5B | +21.6% | +21.8% | +7.3% | **−70.4%** |
| GPT-2 | +15.7% | +29.5% | +29.1% | +0.2% |
| Pythia-1.4B | +13.9% | +14.2% | +24.3% | **−8.1%** |

Three findings, none of them in the README:

1. **The advantage is confined to a band, roughly 4.5–6 bits.** At 7 bits the
   allocation is *worse than trivial interpolation* on three of five models. The
   claim "mixed precision dominates uniform at any footprint" is false as stated.
2. **The magnitude is 14–30% on four of five models,** not 35%. The ~35% figure
   comes from comparing against uniform-NF4 at *lower* memory (§Phase 1), which
   credits the allocation with the memory increase.
3. **Pythia-410M is a 3–5× outlier** and is doing a lot of work in the average.
   It is the model where uniform NF4 is catastrophic (+5.47 ppl); the allocator's
   job there is mostly "rescue the handful of layers that fall apart," which is a
   real but different result from "spends budget optimally."

*Caveat, stated because it cuts against me too:* linear interpolation in
(MB, ppl) space is an approximation of the true random-subset baseline. Perplexity
need not be linear in the fraction of layers upgraded. The §2d experiment measures
the real random baseline and is the arbiter; if random lands *above* the line, these
gains are understated.

**Why it inverts at 7 bits** — worth understanding, because it is diagnostic. At
high budgets nearly all layers can afford INT8, whose measured ΔNLL is ~0. The DP
is then choosing among options whose error estimates are indistinguishable from
profiler noise, and its documented tie-break (`allocator.py:181-189`, prefer the
largest capacity) resolves ties arbitrarily rather than informatively. Compounding
this, the DP's objective **assumes per-layer errors are additive** — the exact
assumption the Linearity Theorem paper had to prove, and which AtlasInfer neither
proves nor tests.

### Contamination: clean ✓

Calibration is drawn from WikiText-2 **train**, evaluation from **test**
(`benchmark.py:141-148`), through a single shared loader used by every harness.
No overlap. This is one thing that needs no defending.

### Is the uniform baseline a straw man? Partly, in AtlasInfer's favour

The uniform 4-bit baseline is AtlasInfer's *own* NF4 + sparse-outlier
implementation, which is a strong baseline (it beats bnb's NF4 on perplexity on
all four models). That is fair. **But the comparison against GPTQ/AWQ at matched
memory is the one that matters, and it goes the other way** — see K4 in §2d.

### Downstream survival: the effect largely vanishes

From `results/downstream_*.json`, mean accuracy over five zero-shot tasks:

| Model | FP16 | NF4 | mixed-5bit | GPTQ-NF4 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 0.5325 | 0.5234 | 0.5320 (−0.0005) | 0.5314 (−0.0010) |
| Qwen3-0.6B | 0.5474 | 0.5542 | 0.5548 (+0.0075) | 0.5466 (−0.0008) |

On Qwen3-0.6B **uniform NF4 scores *above* FP16** (+0.0068). That is noise, and it
tells you the measurement cannot resolve the effect: the perplexity differences
being optimized (+0.65 ppl) are below the resolution of a 2000-example zero-shot
suite with no error bars. A reviewer will say — correctly — that the paper has not
shown its perplexity win matters downstream. Fixing this needs full task sets,
multiple seeds, and reported confidence intervals.

**K3 verdict: marginal pass at 4.5–5 bits (4 of 5 models ≥15%), fail at 7 bits,
and the honest claim is "14–30% in a narrow band, plus one outlier model at 75%",
not "35% at equal memory."**

---

## 2c. Are the kernel speedups meaningful? — **K6: FAILS**

### The claim measures a GEMM, not decode

`bench_triton_kernel.py` times one `w8a16_linear` call against one `F.linear` call
on `torch.randn` matrices at five hardcoded shapes. No model, no KV cache, no
attention. The repo's only end-to-end kernel artifact
(`results/_logs/wsl_engine_e2e.log`) prints a generated sentence and "E2E OK" — a
**correctness smoke test with no timing**.

Meanwhile the repo's measured end-to-end decode (eager path, RTX 3060) is
**0.09×–0.64× of FP16** — i.e. 1.6× to 11× *slower*. So "1.4–1.95× faster batch-1
decode" is not a measurement that exists in this repository.

### Is the FP16 baseline competent? Yes — and that is the bad news

Recomputing achieved weight-streaming bandwidth from the committed log
(RTX 3060 Laptop, 192-bit @ 12 Gbps ⇒ ~288 GB/s peak):

| shape (M,K,N) | fp16 GB/s | W8 GB/s | W4 GB/s | W8 speedup / ideal | W4 speedup / ideal | W4 bandwidth eff. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (1, 4096, 4096) | 214.5 | 169.8 | 95.4 | 1.58× / 2.0× | 1.78× / 4.0× | 44.5% |
| (1, 4096, 11008) | **267.5** | 260.9 | 183.3 | 1.95× / 2.0× | 2.74× / 4.0× | 68.5% |
| (1, 5120, 5120) | 264.3 | 235.1 | 120.1 | 1.78× / 2.0× | 1.82× / 4.0× | 45.5% |
| (4, 4096, 4096) | 255.8 | 204.6 | 95.1 | 1.60× / 2.0× | 1.49× / 4.0× | 37.2% |
| (16, 4096, 4096) | 250.4 | 190.4 | 64.9 | 1.52× / 2.0× | 1.04× / 4.0× | 25.9% |

**The FP16 baseline reaches 267.5 GB/s = 93% of theoretical peak.** It is not a
straw man — `F.linear` is doing its job. Good for honesty; it removes the
"we beat an unoptimized baseline" escape hatch in both directions.

**W8A16 is genuinely good:** 1.95× against an ideal 2.0× on the large shape (98%
of what the bandwidth argument allows). This is a competent kernel and the one
defensible piece of the systems work.

**W4A16 is not:** it delivers 1.78–2.74× where **4.0× is available**, sustaining
only 26–68% of the bandwidth FP16 itself achieves. Over half the 4-bit win is
left on the floor.

### Against the real competition

Marlin (Frantar et al., **PPoPP 2025** — same group as the Linearity Theorem
paper) states that "batchsizes up to 16-32 can be supported with close to maximum
(4×) quantization speedup." AtlasInfer's W4A16 at M=16 is **1.04×**. That is
roughly **3.8× off Marlin at M=16** and ~2.2× off at M=1. ExLlamaV2's EXL2 kernels
are reported ahead of Marlin on newer architectures.

**No comparison against any quantized kernel has ever been run in this repo** —
not Marlin, not ExLlamaV2, not bnb's kernels, not AWQ-GEMM, not `torch.compile`.
The only comparator is FP16, and beating FP16 at batch-1 by reading fewer weight
bytes is arithmetic, not a contribution.

### The structural problem underneath

The kernels quantize **per-output-channel symmetric int8/int4 with no blocks, no
outliers, and no NF4**. The accuracy results use **block-wise NF4 with sparse FP16
outliers**. These are different quantizers, and `sensitivity.py:135-141` documents
that INT8's error is ~10× higher under the kernel scheme. **So the fast path is
not the accurate path, and no experiment in the repo measures both on one model.**
A reviewer will ask "what is the perplexity of the configuration you benchmarked
for speed?" and the answer does not exist.

**K6 verdict: FAILS.** W4A16 is far off the state of the art, no quantized-kernel
comparison exists, end-to-end decode is slower than FP16, and the benchmarked
configuration is not the evaluated one.

---

## 2d. Attribution — **K5: PARTIAL PASS. The allocation signal is real; the
## *exact DP* is not shown to be necessary.**

Experiment: `PAPER/exp/allocator_ablation.py`, Qwen2.5-0.5B, WikiText-2, 30k eval
tokens, seed 0, 168 profiled layers, ~25 min on the 6 GB 3060. Raw JSON:
`PAPER/exp/results/ablation_Qwen_Qwen2.5-0.5B.json`. Anchors: FP16 11.9064;
uniform-NF4 12.6793 @ 484.1 MB; uniform-INT8 11.9111 @ 620.9 MB.

| target bits | knapsack DP | greedy-sensitivity | random (mean ± sd, n=3) | DP vs greedy | **DP vs random** |
| ---: | --- | --- | --- | ---: | ---: |
| 4.25 | **12.5392** @ 492.6 MB | 12.5987 @ 493.3 | 12.6586 ± 0.0135 @ 493.2 | 8.6% | **15.9%** |
| 4.5 | **12.4283** @ 502.9 MB | 12.5930 @ 503.1 | 12.6421 ± 0.0063 @ 502.8 | 24.0% | **29.1%** |
| 5.0 | **12.3660** @ 518.6 MB | 12.5023 @ 522.0 | 12.6141 ± 0.0174 @ 521.9 | 22.9% | **35.1%** |
| 6.0 | **12.2668** @ 553.8 MB | 12.4458 @ 560.4 | 12.5349 ± 0.0349 @ 560.1 | 33.2% | **42.7%** |

(Percentages are reduction in Δppl-vs-FP16. At 5.0 and 6.0 bits the DP also uses
*less* memory than both baselines, so it strictly dominates them.)

### What this establishes

**The sensitivity signal is doing real work.** The DP beats random-at-matched-
memory by 16–43%, and the gaps (0.10–0.27 ppl) are 3–20× the random-seed spread
(0.006–0.035 ppl). This is a clean, well-separated effect and it is the strongest
result the project has produced. K5's kill condition — "fails to beat the
alternatives by more than run-to-run noise" — is not met.

### What this does *not* establish, and why it matters more

**The "greedy" baseline is broken, and it is the repo's own.** Look at the
precision histograms at a 4.5-bit budget:

| arm | fp16 | int8 | int4 |
| --- | ---: | ---: | ---: |
| knapsack DP | 1 | **49** | 118 |
| greedy-sensitivity | 6 | **2** | 160 |
| random-s0 | 6 | **2** | 160 |

Greedy and random produce **near-identical allocations**. The cause is in
`allocator.py:253-263`: greedy walks layers in sensitivity order and upgrades each
one *as far as the budget allows* (`for better in ordered[cur_idx+1:]`) before
moving on — so the few most-sensitive layers get pushed all the way to FP16,
exhausting the budget, and almost nothing reaches INT8. That is "max out each
layer in sensitivity order," not a sensible greedy. A reviewer who sees two
baselines with identical precision histograms will conclude the baseline is
broken, and they will be right.

**So the DP's 8.6–33.2% margin over "greedy" is a margin over a straw man.** The
honest comparison — a *benefit-per-byte* greedy that upgrades in single steps by
Δerror/Δbytes — has never been run. And theory says it would be close: the LP-
relaxation greedy for the multiple-choice knapsack problem is provably within one
item of optimal. **The exact DP is therefore very unlikely to be necessary, and
the paper's central algorithmic claim — that solving the MCKP exactly is the
contribution — is both untested and theoretically improbable.**

### Correction to §2b, cutting in AtlasInfer's favour

The measured random baseline is **worse** than the (MB, ppl) interpolation line at
every budget (e.g. at 5.0 bits: random 12.6141 vs interpolation 12.4670).
Perplexity is convex in the fraction of randomly-upgraded layers, so the
interpolation line is not achievable by any simple method — it is a fiction, and a
*harsher* reference than anything a practitioner would actually reach.

Consequences, stated because they weaken my own §2b argument:
- The §2b gains (14–30%) were computed against that fictional line and are
  therefore **understated**; against the achievable random baseline the same
  comparison roughly doubles (18.0% → 35.1% at 5 bits on this model).
- The §2b finding that mixed precision "inverts at 7 bits" is an inversion
  **against the fictional line only** (reproduced here: −4.1% at 6 bits). Against
  achievable baselines the advantage persists. That finding should be stated as
  "fails to reach an unachievable lower bound at high budgets," not "is worse than
  uniform."
- **K3 is upgraded from marginal pass to clear pass** on this model. It remains
  measured on **one model** — the ablation must be repeated on ≥1 more family
  before it is load-bearing.

### Mechanism decomposition, from committed results

Where does the accuracy actually come from? Δppl-vs-FP16 steps at 4-bit:

| Model | sym-INT4 → NF4 (codebook) | NF4 → GPTQ (free) | NF4 → mixed-5bit (costs memory) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | −0.502 | **−0.665** | −0.522 (+7.6% mem) |
| Qwen2.5-0.5B | **−0.649** | −0.308 | −0.328 (+7.2% mem) |
| Pythia-1.4B | −0.192 | **−0.446** | −0.289 (+10.5% mem) |
| Pythia-410M | −2.390 | −1.583 | **−4.547** (+7.9% mem) |

**On 3 of 4 models, GPTQ — which costs zero extra memory — buys as much or more
than the knapsack allocation, which costs 7–10% extra memory.** The NF4 codebook,
a QLoRA reimplementation, is comparable to both. The allocator is one of three
roughly co-equal mechanisms, and the cheapest of the three in memory terms it is
not.

**K4 remains unresolved.** GPTQ-NF4 Pareto-dominates mixed precision on 3 of 4
models (Phase 1 §3). The rescue is to *compose* them — and `quantize_model_gptq`
has **no `allocation` parameter** (`gptq.py:193-204`), so this has never been run
and needs code. It is also pre-empted (APTQ, oQ+).

---

## 2e. VERDICT

### Scorecard against the pre-registered criteria

| # | Criterion | Verdict |
| --- | --- | --- |
| K1 | Novelty of formulation | **FAIL** — decisive, multiple prior works |
| K2 | Cross-family generality | PASS — holds on Qwen, Pythia/NeoX, GPT-2 |
| K3 | Iso-memory survival | **PASS** — 16–43% over achievable baselines (1 model measured) |
| K4 | Not dominated by known baseline | **UNRESOLVED** — GPTQ dominates 3/4; composition unimplemented *and* pre-empted |
| K5 | Attribution to the knapsack | **PARTIAL** — signal real; *exact DP* untested vs. competent greedy and theoretically unnecessary |
| K6 | Kernel contribution | **FAIL** — W4A16 at 26–68% of available bandwidth; 1.04× at M=16 vs Marlin's ~4×; no quantized-kernel comparison; e2e decode slower than FP16 |
| K7 | Scale credibility | **FAIL** — no ≥7B; 6 GB local cannot even load 7B FP16 |

**My decision rule did not cover the observed state.** I wrote "K1 fails **and**
(K3 or K5) fails → Option 3", but K3 passed and K5 landed partial. I am flagging
that rather than quietly reinterpreting it. What follows is judgment, argued
explicitly, not a rule being applied.

### Recommendation: **Option 3 — do not write the MLSys submission.**

Release the repository well, write a strong technical report with the benchmark
tables, and spend the three months elsewhere.

**The argument:**

1. **Both halves of the paper are pre-empted by the same lab.** The method is
   pre-empted by the Linearity Theorem paper (knapsack-DP + ΔPPL sensitivity, *with
   the additivity proof this repo lacks*, at 8B–70B). The kernels are pre-empted by
   Marlin. Both are Alistarh's group. An MLSys reviewer drawn from this area will
   recognise both instantly. There is no framing that survives that.

2. **The systems contribution is not there.** W4A16 leaves over half the available
   bandwidth unused and is ~3.8× off Marlin at M=16. End-to-end decode is slower
   than FP16. And the benchmarked configuration (per-channel symmetric) is not the
   evaluated one (block-wise NF4 + outliers) — a reviewer asking "what is the
   perplexity of the thing you timed?" gets no answer.

3. **The surviving empirical claim is thinner than it looks.** "Sensitivity-driven
   allocation beats random allocation" is true, cleanly measured, and not news.
   The sharper claim — "the exact MCKP solution is what matters" — is contradicted
   by the broken greedy baseline and by MCKP theory, and would need a proper
   benefit-per-byte greedy to test. That test will probably lose.

4. **The scale floor is a hard cost.** Credibility starts at 7B. On 6 GB that is
   cloud-only: the L40S run in `docs/lightning_7b.md` is ~$8–12 per full pass, and
   several passes across families plus ablations plus reruns is realistically
   $150–400 and a lot of wall-clock — spent to reproduce at 7B what the prior work
   established at 70B.

5. **It is the wrong instrument for the actual goal.** This is a portfolio project
   that must survive expert scrutiny. A rejected MLSys submission does not do that;
   a repository that a senior engineer can read in an afternoon and find *correct,
   tested, and honestly benchmarked* does. This repo is already unusually close to
   that: README-consistency tests wired into CI, a real limitations section, a
   documented negative result on INT8×INT8 GEMM, cross-family evaluation, clean
   train/test calibration separation. Very few portfolio repos have any of that.
   The gap to "excellent artifact" is weeks; the gap to "accepted MLSys paper" is
   not bridgeable from here.

**What I would do with the three months instead** (in priority order):

- **(1 week) Run the K4 composition experiment.** Add an `allocation` parameter to
  `quantize_model_gptq` and measure GPTQ-NF4 ∘ knapsack allocation. This is the
  single best number the project can still produce, it is cheap, and it resolves
  the one genuinely open question. If it beats GPTQ alone, it is the headline of
  the technical report.
- **(1 week) Fix the reproducibility holes.** JSON output from `bench_latency.py`
  and `bench_triton_kernel.py`; pin dependencies; add a `sensitivity.py` test file;
  delete or clearly quarantine the hand-transcribed tables that have no committed
  source (runtime, kernel, 3B).
- **(1 week) Correct the README's claims** to what §Phase 1 and §2b–2d actually
  support. Specifically: the "1.6–1.9× faster decode" line in the opening paragraph
  is a GEMM microbenchmark, not decode, and currently contradicts the repo's own
  runtime table 170 lines below it.
- **(2–3 days) Fix `allocate_greedy`** to a benefit-per-byte greedy and re-run the
  ablation. If the DP still wins, that is a much stronger claim than the current
  one. If it does not — which I expect — that is worth knowing and worth saying.
- **(optional, ~$50) One 7B cross-family run** on Lightning for the report's
  credibility, not for a paper.
- **(optional) A workshop submission** (ENLSP / efficient-ML) of the allocation
  ablation *only*, if it replicates on a second family with the fixed greedy. Low
  cost, citable, honest about being a reproduction-plus-ablation. This is the one
  path I would not argue against — but I would not spend three months on it.

**What I am not saying:** the engineering is not bad. The outlier detector, the
chunked GPTQ Hessian, the coalesced-layout kernel fix, and the CI consistency
guard are all genuinely good work. The problem is not quality. It is that the
ideas were published first by people with 70B-scale compute, and the parts that
were not published first are not, on measurement, competitive.



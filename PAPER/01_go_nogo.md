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

## 2b, 2c, 2d — results appended below after execution.

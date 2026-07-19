#!/usr/bin/env bash
# AtlasInfer -- 7B validation on Lightning AI (or any single Linux + CUDA GPU).
#
# Run from the repo root. Stages are priority-ordered so you can stop when your
# GPU hours run out and still have the highest-value result.
#
#   bash run_lightning.sh <stage> [MODEL]
#     stages: setup | compare | sweep | downstream | latency | all
#     MODEL default: Qwen/Qwen3.5-9B-Base  (latest open Qwen, Feb 2026, no gating).
#       NOTE Qwen3.5 is a HYBRID arch (Gated Delta Networks + sparse MoE); AtlasInfer
#       targets nn.Linear/Conv1D and is validated on DENSE transformers, so run the
#       0.8B smoke first to confirm it quantizes cleanly. Known-dense fallback:
#       Qwen/Qwen3-8B-Base (2025). Field-standard protocol: meta-llama/Llama-2-7b-hf
#       (needs `huggingface-cli login` + accepting Meta's license on HF).
#
# This runs ONE model. For a CROSS-FAMILY spread (SmolLM3 / Phi-4-mini / Mistral /
# OLMo-2 / Qwen — what makes a quant result credible), loop `compare` over models,
# or run the full suite per family with:
#     python reproduce.py --device-map --models <m1> <m2> ...
# See docs/lightning_7b.md for the recommended dense, ungated family list.
#
# GPU: A100-40GB recommended (7B FP16 baseline ~14 GB + GPTQ Hessian headroom).
#      L4/A10-24GB works but is tight. 13B+ needs >1 GPU: `export DEVICE_MAP=--device-map`.
#
# The external GPTQ/AWQ baselines (gptqmodel/autoawq) are heavy and version-fragile;
# compare_baselines.py SKIPS any baseline that fails to import, so a partial install
# still yields a full table of whatever loaded (bitsandbytes always works here).
set -uo pipefail
STAGE="${1:-all}"
MODEL="${2:-Qwen/Qwen3.5-9B-Base}"   # latest open Qwen (Jul 2026); dense fallback: Qwen/Qwen3-8B-Base
EVAL_TOKENS="${EVAL_TOKENS:-40000}"
DEVICE_MAP="${DEVICE_MAP:-}"      # set to "--device-map" for 13B+ across >1 GPU

echo "== AtlasInfer 7B validation | stage=$STAGE model=$MODEL tokens=$EVAL_TOKENS =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

setup() {
  # Core suite + bitsandbytes (the always-available 4-bit baseline on Linux/CUDA).
  pip install -q -e '.[benchmark,eval]' bitsandbytes
  # Real GPTQ + AWQ reference baselines (optional; skip if the install is painful):
  pip install -q optimum gptqmodel autoawq || echo "(external GPTQ/AWQ baselines optional -- continuing without them)"
  python -c "import torch;print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  # Cheap smoke on the latest small model to confirm env + that AtlasInfer quantizes
  # the (hybrid) arch cleanly BEFORE spending 9B hours:
  python compare_baselines.py --model Qwen/Qwen3.5-0.8B-Base --eval-tokens 4000 --skip awq gptq
}
# Head-to-head vs bitsandbytes (+ real GPTQ/AWQ if installed). THE credibility result.
compare()    { python compare_baselines.py --model "$MODEL" --eval-tokens "$EVAL_TOKENS" --double-quant $DEVICE_MAP; }
# Uniform + mixed-precision perplexity/memory Pareto (the repo's signature result).
sweep()      { python benchmark.py --model "$MODEL" --eval-tokens "$EVAL_TOKENS" --bits 4.5 5 6 7 $DEVICE_MAP; }
# Downstream zero-shot accuracy (ARC/HellaSwag/PIQA/WinoGrande) -- tasks, not just ppl.
downstream() { python eval_downstream.py --model "$MODEL" --limit 1000 $DEVICE_MAP; }
# Decode tok/s + peak GPU memory (single-GPU only; do NOT pass --device-map).
latency()    { python bench_latency.py --model "$MODEL"; }

case "$STAGE" in
  setup)      setup;;
  compare)    compare;;
  sweep)      sweep;;
  downstream) downstream;;
  latency)    latency;;
  all)        setup;      echo "--- compare ---";    compare    || echo "compare FAILED, continuing";
              echo "--- sweep ---";      sweep      || echo "sweep FAILED, continuing";
              echo "--- downstream ---"; downstream || echo "downstream FAILED, continuing";
              echo "--- latency ---";    latency    || echo "latency FAILED, continuing";;
  *) echo "unknown stage: $STAGE (use setup|compare|sweep|downstream|latency|all)"; exit 1;;
esac
echo "== done: $STAGE. Results written under results/ (zip and download them) =="

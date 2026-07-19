#!/usr/bin/env bash
# AtlasInfer -- 7B validation on Lightning AI (or any single Linux + CUDA GPU).
#
# Run from the repo root. Stages are priority-ordered so you can stop when your
# GPU hours run out and still have the highest-value result.
#
#   bash run_lightning.sh <stage> [MODEL]
#     stages: setup | compare | sweep | downstream | latency | all | overnight
#     MODEL default: Qwen/Qwen3.5-9B-Base  (latest open Qwen, Feb 2026, no gating).
#       NOTE Qwen3.5 is a HYBRID arch (Gated Delta Networks + sparse MoE); AtlasInfer
#       targets nn.Linear/Conv1D and is validated on DENSE transformers, so run the
#       0.8B smoke first to confirm it quantizes cleanly. Known-dense fallback:
#       Qwen/Qwen3-8B-Base (2025). Field-standard protocol: meta-llama/Llama-2-7b-hf
#       (needs `huggingface-cli login` + accepting Meta's license on HF).
#
# UNATTENDED / OVERNIGHT -- one command, then walk away:
#     mkdir -p results/_logs
#     nohup bash run_lightning.sh overnight > results/_logs/overnight.log 2>&1 &
#
#   `overnight` runs setup, then `compare` across 3 UNGATED families (nothing can
#   block on a Hugging Face login while you sleep), then sweep + downstream +
#   latency on $MODEL. Every stage gets its own log, a failing stage never kills
#   the run, and a summary table prints at the end.
#   ~5 h total (~12 credits on an L40S). Override the list with FAMILIES="a b c".
#   AUTOSTOP=1 tries to power the machine off when done so idle credits aren't
#   burned (verify it works for your Studio; the Lightning UI idle-timeout is the
#   reliable backstop).
#
# For a CROSS-FAMILY spread you can also run the full suite per family with:
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
# Overnight cross-family list. UNGATED ONLY by default so an HF login prompt can
# never stall an unattended run. Add gated models (Gemma/Llama) only if you ran
# `huggingface-cli login` first.
FAMILIES="${FAMILIES:-mistralai/Mistral-7B-v0.3 Qwen/Qwen3-8B-Base microsoft/Phi-4-mini-instruct}"
AUTOSTOP="${AUTOSTOP:-0}"
SUMMARY="results/_logs/overnight_summary.txt"

echo "== AtlasInfer 7B validation | stage=$STAGE model=$MODEL tokens=$EVAL_TOKENS =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --- preflight: fail loudly HERE rather than 7 stages deep -------------------
# Must run from the repo root (pip install -e . and the harness paths depend on it).
if [ ! -f pyproject.toml ] || [ ! -d atlasinfer ]; then
  echo "ERROR: run this from the AtlasInfer repo root (no pyproject.toml/atlasinfer here)."
  echo "       cd into the cloned repo first:  cd AtlasInfer"
  exit 1
fi
# Some images ship only python3 (no `python` alias) -- that alone fails every stage.
PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then echo "ERROR: no python/python3 on PATH."; exit 1; fi
echo "python: $PY ($("$PY" --version 2>&1))"
# Only one overnight run at a time, or two processes interleave into the summary.
LOCK="results/_logs/.overnight.lock"

setup() {
  # Use the interpreter we resolved, and its own pip -- `pip` may not be on PATH.
  "$PY" -m pip install -q -e '.[benchmark,eval]' bitsandbytes || {
    echo "ERROR: core install failed. Nothing downstream can work."; return 1; }
  # Real GPTQ + AWQ reference baselines (optional; skip if the install is painful):
  "$PY" -m pip install -q optimum gptqmodel autoawq || echo "(external GPTQ/AWQ baselines optional -- continuing without them)"
  "$PY" -c "import torch;print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || return 1
  "$PY" -c "import atlasinfer, transformers, datasets; print('imports OK')" || {
    echo "ERROR: atlasinfer/transformers/datasets not importable after install."; return 1; }
  # Cheap smoke on the latest small model to confirm env + that AtlasInfer quantizes
  # the (hybrid) arch cleanly BEFORE spending 9B hours:
  "$PY" compare_baselines.py --model Qwen/Qwen3.5-0.8B-Base --eval-tokens 4000 --skip awq gptq
}
# Head-to-head vs bitsandbytes (+ real GPTQ/AWQ if installed). THE credibility result.
compare()    { "$PY" compare_baselines.py --model "$MODEL" --eval-tokens "$EVAL_TOKENS" --double-quant $DEVICE_MAP; }
# Uniform + mixed-precision perplexity/memory Pareto (the repo's signature result).
sweep()      { "$PY" benchmark.py --model "$MODEL" --eval-tokens "$EVAL_TOKENS" --bits 4.5 5 6 7 $DEVICE_MAP; }
# Downstream zero-shot accuracy (ARC/HellaSwag/PIQA/WinoGrande) -- tasks, not just ppl.
downstream() { "$PY" eval_downstream.py --model "$MODEL" --limit 1000 $DEVICE_MAP; }
# Decode tok/s + peak GPU memory (single-GPU only; do NOT pass --device-map).
latency()    { "$PY" bench_latency.py --model "$MODEL"; }

# ---- unattended runner -------------------------------------------------------
# Each stage: own log file, timed, failure-tolerant, recorded in a summary table.
_stage() {                      # _stage <label> <fn> [model]
  local label="$1" fn="$2" mdl="${3:-$MODEL}" start=$SECONDS
  mkdir -p results/_logs
  echo "[$(date '+%F %T')] START $label  ($mdl)"
  if MODEL="$mdl" "$fn" >> "results/_logs/${label}.log" 2>&1; then
    printf 'OK    %-34s %4d min\n' "$label" $(( (SECONDS-start)/60 )) >> "$SUMMARY"
    echo "[$(date '+%F %T')] DONE  $label"
  else
    printf 'FAIL  %-34s %4d min  -> results/_logs/%s.log\n' \
           "$label" $(( (SECONDS-start)/60 )) "$label" >> "$SUMMARY"
    echo "[$(date '+%F %T')] FAIL  $label (continuing)"
  fi
}

overnight() {
  local T0=$SECONDS
  mkdir -p results/_logs
  # Refuse to start a second concurrent run -- two would interleave into $SUMMARY.
  if ! ( set -o noclobber; : > "$LOCK" ) 2>/dev/null; then
    echo "ERROR: an overnight run is already in progress (lock: $LOCK)."
    echo "       If that's stale, delete it:  rm $LOCK"
    exit 1
  fi
  trap 'rm -f "$LOCK"' EXIT
  : > "$SUMMARY"
  echo "Overnight run started $(date '+%F %T')"
  echo "Cross-family compare: $FAMILIES"
  echo "Depth model:          $MODEL"

  # setup is a HARD prerequisite: if the env isn't installed, every later stage
  # fails in seconds and the "keep going" policy just burns GPU hours for nothing.
  _stage setup setup
  if grep -q '^FAIL  setup' "$SUMMARY"; then
    echo ""
    echo "ABORTING: setup failed, so nothing downstream can succeed."
    echo "Root cause is in results/_logs/setup.log -- fix that and re-run."
    tail -25 results/_logs/setup.log 2>/dev/null
    return 1
  fi
  # 1) Cross-family headline: `compare` on each family (the credibility result).
  for m in $FAMILIES; do
    _stage "compare_$(echo "$m" | tr '/:' '__')" compare "$m"
  done
  # 2) Depth on the primary model.
  _stage sweep      sweep
  _stage downstream downstream
  _stage latency    latency

  echo ""
  echo "================= OVERNIGHT SUMMARY ================="
  cat "$SUMMARY"
  printf 'TOTAL %-34s %4d min\n' "(wall clock)" $(( (SECONDS-T0)/60 ))
  echo "====================================================="
  echo "Results in results/  ->  zip -r results.zip results"
  if [ "$AUTOSTOP" = "1" ]; then
    echo "AUTOSTOP=1 -- attempting shutdown so idle credits aren't burned..."
    sudo poweroff 2>/dev/null || sudo shutdown -h now 2>/dev/null || \
      echo "(could not power off from inside; stop the Studio in the Lightning UI)"
  fi
}

case "$STAGE" in
  setup)      setup;;
  compare)    compare;;
  sweep)      sweep;;
  downstream) downstream;;
  latency)    latency;;
  overnight)  overnight;;
  all)        setup;      echo "--- compare ---";    compare    || echo "compare FAILED, continuing";
              echo "--- sweep ---";      sweep      || echo "sweep FAILED, continuing";
              echo "--- downstream ---"; downstream || echo "downstream FAILED, continuing";
              echo "--- latency ---";    latency    || echo "latency FAILED, continuing";;
  *) echo "unknown stage: $STAGE (use setup|compare|sweep|downstream|latency|all|overnight)"; exit 1;;
esac
echo "== done: $STAGE. Results written under results/ (zip and download them) =="

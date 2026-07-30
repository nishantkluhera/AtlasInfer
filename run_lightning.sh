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
# DENSE by default. The previous default (Qwen/Qwen3.5-9B-Base) is a hybrid
# Gated-Delta + sparse-MoE architecture: its expert layers only fire for routed
# tokens, which breaks the sensitivity profiler's assumption that every layer
# sees every calibration token. On a paid GPU that is a silently-wrong allocation
# discovered hours in. Pass a model explicitly to override.
MODEL="${2:-Qwen/Qwen3-8B-Base}"     # dense, open, 2025. Hybrid/MoE: smoke-test first.
EVAL_TOKENS="${EVAL_TOKENS:-40000}"
DEVICE_MAP="${DEVICE_MAP:-}"      # set to "--device-map" for 13B+ across >1 GPU
# Overnight cross-family list. UNGATED ONLY by default so an HF login prompt can
# never stall an unattended run. Add gated models (Gemma/Llama) only if you ran
# `huggingface-cli login` first.
FAMILIES="${FAMILIES:-mistralai/Mistral-7B-v0.3 Qwen/Qwen3-8B-Base microsoft/Phi-4-mini-instruct}"
AUTOSTOP="${AUTOSTOP:-0}"
SUMMARY="results/_logs/overnight_summary.txt"
# Install into an isolated venv (see setup()). USE_VENV=0 uses the host env.
USE_VENV="${USE_VENV:-1}"
VENV="${VENV:-.venv}"

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
  # Install into an ISOLATED venv by default. Managed cloud images (Lightning's
  # `cloudspace` conda env in particular) often ship a partly-broken environment:
  # half-installed distributions (`~umpy`), missing dist-info dirs that make pip
  # abort with OSError, and scipy/sklearn compiled against NumPy 1.x while numpy
  # itself is 2.x -- which breaks `import transformers` via sklearn->scipy with
  # "numpy.core.multiarray failed to import". A clean venv sidesteps all of it.
  # Set USE_VENV=0 to install into the host environment instead.
  if [ "$USE_VENV" = "1" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
      echo "Creating isolated venv at $VENV (host env may have broken/ABI-conflicting packages)..."
      "$PY" -m venv "$VENV" || { echo "WARN: venv creation failed; falling back to host env"; USE_VENV=0; }
    fi
    if [ -x "$VENV/bin/python" ]; then
      PY="$(cd "$VENV/bin" && pwd)/python"     # later stages inherit this
      echo "using venv python: $PY"
      "$PY" -m pip install -q -U pip setuptools wheel
      # Fresh CUDA torch (the venv starts empty; ~2-4 min).
      "$PY" -m pip install -q torch || { echo "ERROR: torch install failed."; return 1; }
    fi
  fi
  "$PY" -m pip install -q -e '.[benchmark,eval]' bitsandbytes || {
    echo "ERROR: core install failed. Nothing downstream can work."; return 1; }
  # Real GPTQ reference baseline (optional). Installed separately from autoawq so
  # that autoawq -- deprecated, and pinned to transformers 4.51 -- cannot take the
  # GPTQ baseline down with it when pip resolves them together.
  "$PY" -m pip install -q optimum gptqmodel || \
    echo "(external GPTQ baseline optional -- continuing without it)"
  # autoawq last: most likely to fail, least important (AtlasInfer's own awq-nf4
  # arm covers the mechanism). Failure here must not affect anything above.
  "$PY" -m pip install -q autoawq 2>/dev/null || \
    echo "(autoawq unavailable -- deprecated upstream; using AtlasInfer's awq-nf4 instead)"
  # Report what actually landed, so a missing backend is visible NOW rather than
  # as a cryptic failure hours into the paid run.
  "$PY" - <<'PYEOF'
for mod, why in [("bitsandbytes", "bnb-int8 / bnb-nf4 rows"),
                 ("gptqmodel", "external gptq (auto-gptq) row"),
                 ("awq", "external awq (autoawq) row")]:
    try:
        __import__(mod)
        print(f"  baseline OK      {mod:<14} -> {why}")
    except Exception as e:
        print(f"  baseline MISSING {mod:<14} -> {why} will be SKIPPED ({type(e).__name__})")
PYEOF
  "$PY" -c "import torch;print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || return 1
  # Full import chain, incl. the scipy/sklearn path transformers pulls in. This is
  # the check that would have caught the NumPy 1.x/2.x ABI break immediately.
  "$PY" -c "import numpy, scipy, sklearn, transformers, datasets, atlasinfer; print('imports OK; numpy', numpy.__version__)" || {
    echo "ERROR: import check failed -- classic cause is numpy 2.x with"
    echo "       numpy-1.x-compiled scipy/sklearn. An isolated venv (USE_VENV=1)"
    echo "       avoids it; if you forced USE_VENV=0, pin 'numpy<2' in the host env."
    return 1; }
  # Cheap smoke BEFORE spending 7B hours. Deliberately does NOT skip gptq: `--skip`
  # matches by prefix, so `--skip gptq` would also drop gptq-nf4 AND the new
  # gptq-mixed composition arm -- i.e. the smoke would pass while leaving the most
  # important new code path completely unexercised, and it would fail hours into
  # the paid run instead. Only awq (slow, and the most install-fragile) is skipped.
  # Small DENSE model on purpose: a hybrid/MoE smoke can pass or fail for reasons
  # that say nothing about the dense 7B you are about to run.
  # --out keeps this short-eval run out of results/, so it cannot overwrite a
  # committed full-length comparison for the same model.
  "$PY" compare_baselines.py --model Qwen/Qwen2.5-0.5B --eval-tokens 2000 \
        --skip awq --out results/_smoke || return 1
  # Assert the composition arm actually produced a row. Without this the smoke is
  # green whenever gptq-mixed silently FAILED-and-continued (it is `guarded`).
  "$PY" - <<'PYEOF' || return 1
import json, sys
rows = json.load(open("results/_smoke/comparison_Qwen_Qwen2.5-0.5B.json"))["rows"]
required = ("AtlasInfer gptq-nf4", "AtlasInfer mixed", "AtlasInfer gptq-mixed")
missing = [m for m in required if not any(r["method"].startswith(m) for r in rows)]
if missing:
    print(f"SMOKE FAILED: no rows for {missing} -- those paths errored and were "
          f"swallowed by the `guarded` wrapper. Fix before spending GPU hours.")
    sys.exit(1)
print(f"smoke OK: {len(rows)} rows incl. {', '.join(required)}")
PYEOF
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
  # Truncate, don't append: on a retry a stale log would otherwise still be here,
  # and the abort path below tails setup.log -- showing a PREVIOUS run's error as
  # if it were the current one.
  : > "results/_logs/${label}.log"
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

#!/bin/bash
# Phase 3c production runs, live LLM backend (VALIDATION_PLAN.md).
# Sequential; each mode logs to runs/<mode>.out. Designed to run detached:
#   nohup bash validation/run_production.sh > validation/runs/production.log 2>&1 &
# Resumable: the response cache (.llm_cache) replays completed calls, so
# re-running after an interruption fast-forwards to where it stopped.

set -u
export PYTHONUNBUFFERED=1   # live progress through tee for detached runs
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/../.venv-validation/bin/python"
SWEEP="$DIR/tier2_sweep.py"
BE="${BACKEND:-ollama:llama3.1:8b}"
mkdir -p "$DIR/runs"

echo "=== production start $(date) backend=$BE ==="

# P2 (the M2-killer) first: ground-truth grid near the boundary.
# Config mock-calibrated 2026-06-12: 8/8 within 15%, alpha within 6%.
echo "--- grid ---"
"$PY" "$SWEEP" --backend "$BE" grid \
    --n-seeds 8 --target-n 250 2>&1 | tee "$DIR/runs/grid.out"

# P1 hysteresis loop: ramp past the fold (l0_c=0.6243 at alpha=0.6) with
# enough points above it for collapse to nucleate on the up-leg, and a
# deep floor so the down-leg fully recovers below l0_rec=0.4.
echo "--- hysteresis ---"
"$PY" "$SWEEP" --backend "$BE" hysteresis \
    --alpha 0.6 --l0-min 0.125 --l0-max 0.70 --l0-step 0.025 \
    --dwell 80 --max-commitments 12000 2>&1 | tee "$DIR/runs/hysteresis.out"

# P1 reset-cure at alpha >= 1: start above the fold (deterministic
# collapse), then load-reduction arm vs drain arm.
echo "--- resetcure ---"
"$PY" "$SWEEP" --backend "$BE" resetcure \
    --n-windows 20 --max-commitments 1500 2>&1 | tee "$DIR/runs/resetcure.out"

# Corner cascade statistics (spot-check; the exponent claim stays with
# the 10^5-avalanche sim fits): tight deadline theta=0.2.
echo "--- corner ---"
"$PY" "$SWEEP" --backend "$BE" --theta 0.2 corner \
    --n-seeds 8 --max-commitments 500 2>&1 | tee "$DIR/runs/corner.out"

echo "=== production done $(date) ==="

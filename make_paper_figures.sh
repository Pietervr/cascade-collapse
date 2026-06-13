#!/bin/bash
# Regenerate the paper's figure files and collect them under paper_figures/
# with the exact filenames the LaTeX source references. Seeded; deterministic.
#
#   bash make_paper_figures.sh            # all five
#   FAST=1 bash make_paper_figures.sh     # cheaper avfit preview (noisier)
#
# Run from the repository root inside the venv (pip install -r requirements.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/paper_figures"
mkdir -p "$OUT"

echo "== sr_simulation_validation =="
( cd "$ROOT/derivation" && "$PY" sr_simulation.py )
cp "$ROOT/derivation/figures/sr_simulation_validation.pdf" "$OUT/"

echo "== ising_spin_dissipation =="
( cd "$ROOT/derivation" && "$PY" ising_spin_dissipation.py )
cp "$ROOT/derivation/figures/ising_spin_dissipation.pdf" "$OUT/"

echo "== cascade_hysteresis =="
( cd "$ROOT/sr_rigor_fix" && "$PY" cascade_sim.py hysteresis --theta 5 \
    --alpha 0.5 --l0-min 0.30 --l0-max 0.70 --l0-step 0.025 --dwell 1500 )
cp "$ROOT/sr_rigor_fix/figures/hysteresis.pdf" "$OUT/cascade_hysteresis.pdf"

echo "== cascade_nscaling (SM) =="
( cd "$ROOT/sr_rigor_fix" && "$PY" cascade_sim.py nscaling )
cp "$ROOT/sr_rigor_fix/figures/nscaling.pdf" "$OUT/cascade_nscaling.pdf"

echo "== cascade_avfit (expensive) =="
AVFIT_ARGS=""
if [ "${FAST:-0}" = "1" ]; then AVFIT_ARGS="--t-max 400000"; fi
( cd "$ROOT/sr_rigor_fix" && "$PY" cascade_sim.py avfit $AVFIT_ARGS )
cp "$ROOT/sr_rigor_fix/figures/avfit.pdf" "$OUT/cascade_avfit.pdf"

echo ""
echo "Done. Paper figures collected in: $OUT"
ls -1 "$OUT"

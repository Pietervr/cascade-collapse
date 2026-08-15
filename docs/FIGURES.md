# Figure & result manifest

Every figure in the paper, the exact command that produces it, and the file it
lands in. All commands are seeded; re-running reproduces the committed result.
Run them from the repository root inside the venv (`pip install -r
requirements.txt`).

`make_paper_figures.sh` runs the four main-text/SM generators below and copies
their output into `paper_figures/` under the paper's filenames.

---

## Main text

### `figures/specification_map.pdf`
The intro roadmap figure (paper Fig. 1): cell, star, and AI system split by
the three specification classes (archive / flux / trajectory), with the
certification lever. Pure drawing — no simulation.

```bash
python derivation/make_specification_map.py
# writes derivation/figures/specification_map.pdf
```
Cost: seconds. No network.

### `figures/sr_simulation_validation.pdf`
The α = 0 (open-loop) limit: a direct event-driven M/M/1 simulation confirms
the closed form SR(ρ) = 1/(e^{MΔt} − 1) and the critical-slowing-down scaling
τ_relax ~ [μ(1 − √ρ)²]^{−1}.

```bash
python derivation/sr_simulation.py
# writes derivation/figures/sr_simulation_validation.pdf
# also caches sr_simulation_results.npz; set REPLOT=1 to re-plot from cache
```
Cost: ~1–3 min (vectorised Lindley recursion). No network.

### `figures/cascade_avfit.pdf`
Maximum-likelihood fit of P(s) ~ s^{−τ} e^{−s/s_c} at three points marching
into the self-referential corner. Result: τ = 1.500 at the deep points (the
Otter 3/2 branching exponent), with a cutoff renormalised ~1.25× by the
zero-lag congestion correlation. The three points collapse onto one master
curve.

```bash
python sr_rigor_fix/cascade_sim.py avfit
# writes sr_rigor_fix/figures/avfit.pdf  -> paper figures/cascade_avfit.pdf
```
Cost: the expensive one — 4–12 ×10⁴ avalanches per point, `--t-max 4_000_000`.
Tens of minutes. Reduce `--t-max` for a faster, noisier preview.

### `figures/cascade_hysteresis.pdf`
The first-order hysteresis loop: a slow ℓ₀ ramp up then down at fixed α. The
up-ramp rides the lucid branch and collapses; the down-ramp rides the collapsed
branch x = ℓ₀/(1−α) and stays collapsed below the static recovery spinodal
(the accumulated backlog is a memory variable).

```bash
python sr_rigor_fix/cascade_sim.py hysteresis --theta 5 --alpha 0.5 \
    --l0-min 0.30 --l0-max 0.70 --l0-step 0.025 --dwell 1500
# writes sr_rigor_fix/figures/hysteresis.pdf -> paper figures/cascade_hysteresis.pdf
```
Cost: a few minutes.

## Supplemental Material

### `figures/ising_spin_dissipation.pdf`
The "cost half" of the division of labour: a driven two-level (Ising) system
produces genuine, positive, but **bounded** entropy production — it never
diverges. Two independent EP estimators (medium-entropy sum and
hysteresis-loop area) agree.

```bash
python derivation/ising_spin_dissipation.py
# writes derivation/figures/ising_spin_dissipation.pdf
```
Cost: ~1–2 min. No network.

### `figures/cascade_nscaling.pdf`
N-scaling (the fleet limit): the mean-field bistability sharpens into a
first-order transition as N parallel decoders share the exogenous stream and
spawn pool. At mid-wedge, lifetimes grow faster than polynomially in N; N = 16
is fully censored at the horizon.

```bash
python sr_rigor_fix/cascade_sim.py nscaling
# writes sr_rigor_fix/figures/nscaling.pdf -> paper figures/cascade_nscaling.pdf
```
Cost: several minutes (8 seeds × 5 values of N, `--t-max 200_000`).

---

## Supporting numerical evidence (not figures, but in the paper's prose)

| `cascade_sim.py` subcommand | What it establishes |
|---|---|
| `smoke` | α=0 reduces to exact M/M/1; the ⟨s⟩ = x/ℓ₀ bookkeeping identity; fold/cusp closed forms; entropy-layer exactness |
| `identity` | ⟨s⟩ = x*/ℓ₀ across the lucid branch |
| `phase` | the measured phase boundary tracks the recovery spinodal on long observation times |
| `ews` | saddle-node early-warning scaling (rising lag-1 autocorrelation) and the estimation-race caveat |
| `lagstudy` | the "delay is a gate" congestion-enhancement law vs. spawn lag |
| `escape` | metastable-lifetime scaling with wedge depth and θ |
| `entropy` | Landauer-priced avalanche bursts; collapsed-branch entropy floor |

Each has a seeded invocation; defaults reproduce the values quoted in
`sr_rigor_fix/THEORY_NOTE_README.md` (findings F1–F8).

---

## Empirical-validation figures (Tier 2)

The empirical-validation section's panels are produced by `validation/
tier2_sweep.py` (hysteresis loop, reset-cure two-arm trace) and by the
estimator-recovery grid. Reproduce with:

```bash
BACKEND=ollama:llama3.1:8b bash validation/run_production.sh
```

This writes per-mode tables and figures under `validation/runs/`. The offline
structural version (no LLM, exact ground truth) is `validation/
test_estimator_recovery.py`.

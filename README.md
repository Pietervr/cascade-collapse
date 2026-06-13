# cascade-collapse

Simulation and validation code for the paper

> **First-Order Recoverability Collapse in Self-Referential Information Decoders**
> Pieter van Rooyen, University of Stellenbosch (Dept. of Electrical & Electronic Engineering).

This repository reproduces every figure and numerical result in the paper, end
to end, from a clean checkout. It contains the closed-loop bifurcation
simulator, the two supporting derivation-note simulations, and the two-tier
empirical-validation pipeline (a fully instrumented multi-agent LLM experiment
plus the shared estimator that is then applied to a public corpus).

Everything is seeded. The structural results (mean-field forms, cascade
exponents, the estimator-recovery test) reproduce deterministically with no
network access; only the *live* LLM runs require a model server.

---

## What the paper claims, and where the code is

The paper is a statistical-physics theory of a self-referential decoder: a
system that consumes its own un-certified outputs faster than it can certify
them. The order parameter is the **stability ratio** SR = Ṅ_uncert / Ṅ_cert.
With one feedback parameter — α, the mean number of downstream consumers per
uncertified commitment — the open-loop SR→∞ alarm is **pre-empted by a
first-order collapse**: a fold, a bistable hysteresis wedge, a cusp, and an
irreversible "reset-cure" regime above α = 1.

| Result (paper) | Code |
|---|---|
| Mean-field fold / cusp / recovery spinodal, closed forms | `sr_rigor_fix/cascade_sim.py` (`smoke`) |
| SR divergence at α=0 reduces to exact M/M/1 + critical slowing-down | `derivation/sr_simulation.py` |
| Bounded per-event Landauer cost (driven Ising spin; the "cost half") | `derivation/ising_spin_dissipation.py` |
| Cascade-size law P(s) ~ s^(−3/2) with grounding-set cutoff (`avfit`) | `sr_rigor_fix/cascade_sim.py` (`avfit`) |
| Hysteresis loop / first-order character (`hysteresis`) | `sr_rigor_fix/cascade_sim.py` (`hysteresis`) |
| N-scaling: fleet limit sharpens the transition (SM) | `sr_rigor_fix/cascade_sim.py` (`nscaling`) |
| **Empirical validation (Tier 2):** controlled LLM pipeline where (α, ℓ₀) are ground truth; the shared estimator recovers them | `validation/` |
| **Empirical validation (Tier 1):** the same estimator on an observational corpus | `validation/` (corpus adapters; see protocol) |

The exact command for each paper figure is in [`docs/FIGURES.md`](docs/FIGURES.md).

---

## Layout

```
sr_rigor_fix/           the closed-loop simulator (the theory engine)
  cascade_sim.py          mean-field solvers + event-driven queue with
                          genealogy tracking + entropy layer; 11 subcommands
  THEORY_NOTE_README.md   the working theory note's findings log (F1–F8)
derivation/             the two derivation-note simulations
  sr_simulation.py        M/M/1 SR(ρ) closed form + τ_relax critical slowing
  ising_spin_dissipation.py  driven two-level system: positive, BOUNDED EP
validation/             the two-tier empirical-validation pipeline
  MEASUREMENT_PROTOCOL.md  the pre-registered measurement protocol (frozen)
  llm_backend.py          one generate() over Ollama / Anthropic / Mock
  tier2_pipeline.py       instrumented multi-agent cascade; emits the
                          frozen event-log schema; certification = verifier-
                          or-tool only
  estimator.py            the shared, rate-free estimator (Tier 1 AND Tier 2)
  tier2_sweep.py          production driver: latency/grid/hysteresis/
                          resetcure/corner
  test_estimator_recovery.py  the offline P2 estimator-recovery test (Mock)
  run_production.sh       the full live-backend run sequence
docs/FIGURES.md         figure → exact-command manifest
make_paper_figures.sh   regenerate the paper's figure files by their names
```

> **Note on the `sr_rigor_fix/` name.** It is historical (the module began as
> a rigor patch and grew into the closed-loop theory engine). The imports in
> `validation/` resolve to it by this name, so it is preserved verbatim from
> the development tree. Read it as `theory/`.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Validate the simulator against the exact M/M/1 limit + closed forms
python sr_rigor_fix/cascade_sim.py smoke

# 2. Reproduce the offline estimator-recovery test (the "M2-killer", no LLM)
cd validation && python test_estimator_recovery.py && cd ..

# 3. Regenerate the paper's figures (see docs/FIGURES.md for per-figure cost)
bash make_paper_figures.sh
```

Steps 1–2 run in seconds to a couple of minutes. The avalanche-fit figure in
step 3 is the expensive one (millions of avalanches); budget accordingly —
`docs/FIGURES.md` lists the wall-clock for each.

---

## The two-tier validation (what makes the empirical claim falsifiable)

A reviewer's sharpest objection to a model like this is that it is "falsifiable
only about its own toy model." The answer is **one estimator, two tiers**:

- **Tier 2 (controlled).** `tier2_pipeline.py` runs a real multi-agent LLM
  system whose topology is instrumented so that α and ℓ₀ are *known ground
  truth*. The content (claims, verifier verdicts) is produced by a real model;
  the structural dynamics (re-consumption of uncertified output, verifier
  coverage, deadline race) are controlled. `estimator.py` — using only the
  genealogy and the certified/uncertified flags, no wall-clock rates —
  recovers (α, ℓ₀). Agreement within a pre-registered tolerance validates the
  instrument (prediction **P2**).
- **Tier 1 (observational).** The *same* `estimator.py`, with per-framework
  adapters, is then applied to a public corpus of agent trajectories. Because
  the estimator was validated in Tier 2, its Tier-1 output is trusted.

The measurement protocol (operational definitions, the certification rule,
the pre-registered predictions P1–P5, the risk register) is frozen in
[`validation/MEASUREMENT_PROTOCOL.md`](validation/MEASUREMENT_PROTOCOL.md) —
that file is a pre-registration: §2 (definitions) and §5 (predictions) were
fixed before the observational results were inspected.

### Reproducing the live runs

```bash
# local model server
ollama pull llama3.1:8b

# one point — sizes the per-call cost
python validation/tier2_sweep.py --backend ollama:llama3.1:8b latency

# the full pre-registered sequence (grid → hysteresis → resetcure → corner)
BACKEND=ollama:llama3.1:8b bash validation/run_production.sh
```

The backend is a one-line swap (`--backend anthropic:claude-...` for the
frontier spot-check, `--backend mock:mock@0.2` for an offline structural
dry-run). Every call is cached on disk, so an interrupted run resumes for free.

---

## Citing

If you use this code, please cite the paper (see `CITATION.cff`). A DOI for an
archived release will be added on submission.

## License

Code released under the MIT License (`LICENSE`). The measurement protocol and
documentation are released under CC-BY-4.0.

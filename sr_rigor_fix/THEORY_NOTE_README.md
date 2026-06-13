# sr_rigor_fix — working theory note (kept separate from the live `pre/` build)

What began as a rigor patch for the stability ratio has grown into **the
closed-loop theory note**: a mean-field bifurcation theory of the
self-referential decoder — the statistical-physics theory of the
hallucination-cascade phenomenon documented empirically in
[arXiv:2606.07937](https://arxiv.org/abs/2606.07937) (which has no theory).
The folder name now undersells the content; rename when promoting into the
paper.

## Version history

- **v1** — proposed `SR` as a *general* "ratio of two entropy-production rates
  of one driven population." **Refuted** by adversarial review (Prigogine `Ṡ_e`
  is a signed flux and a true NESS forces `Ṡ_i = −Ṡ_e`; the known-bit dilemma;
  hand-summed terms, no single generator).
- **v2** — two-tier restructure. Tier 1 (general, bulletproof): `SR ≡
  Ṅ_uncert/Ṅ_cert` as order parameter + one-way Landauer bound. Tier 2
  (model-specific): joint queue⊗spin process (service *is* a Glauber spin's
  first passage → M/M/1 derived, not imposed) with the EP-ratio prefactor
  bounded both ways.
- **v3 (current)** — **the feedback closure + bifurcation analysis.** One new
  parameter: α = mean candidates spawned per *uncertified* commitment
  ("ungatedness"). The open-loop paper is the exact α=0 limit. Results, all in
  closed form (dimensionless: x = λ_eff/μ, ℓ₀ = λ₀/μ, θ = μΔt):
  - **Fold (collapse spinodal):** ℓ₀ᶜ = θx_f²/(1+θx_f) with
    α(1+θx_f)e^{−(1−x_f)θ} = 1.
  - **Recovery spinodal:** ℓ₀ = 1−α; **bistable wedge** between them;
    **cusp** at (α*, ℓ₀*) = (1/(1+θ), θ/(1+θ)). Below cusp: continuous
    transition at the *renormalized* boundary ℓ₀ = 1−α.
  - **Collapse pre-empts the divergence:** SR_fold = 1/(α(1+θx_f)−1) is
    *finite* — the open-loop SR→∞ alarm never fires; first-order jump instead.
  - **Reset-cure theorem:** for α ≥ 1 collapse is irreversible under load
    reduction; only gating (lower α) or reset (drain backlog / clear context)
    recover. Runaway candidate growth = the observed token-burn failure mode.
  - **Avalanches:** branching ratio b = αP_u = 1 − ℓ₀/x*; mean size = load
    amplification x*/ℓ₀ (exact bookkeeping identity); Otter s^{−3/2} with
    cutoff s_c ≈ 2(x*/ℓ₀)² — **grounding bounds the cascades**. Fold
    instability factorizes as b·(1+θx) = genealogy × congestion coupling; the
    interacting-avalanche exponent is an open question the simulation settles.
  - **Early warnings:** saddle-node scaling τ, Var ~ (ℓ₀ᶜ−ℓ₀)^{−1/2}; rising
    lag-1 autocorrelation is the correct alarm (grounds the poster's
    early-warning marker).
  - **Entropy layer (v2 retained):** Landauer-priced avalanche bursts;
    collapsed-branch floor Ṡ_syn ≥ k_B ln2 · λ₀/(1−α); SR_EP discontinuity at
    the fold. Gate-as-priced-demon (Sagawa–Ueda) pointer: exiting the wedge
    needs q > 1 − 1/(α(1+θ)).
  - Four falsifiable predictions mapped onto the arXiv:2606.07937 experimental
    class (cascade-size statistics, hysteresis/reset asymmetry, early-warning
    scaling, boundary renormalization).

## Contents
- `sr_rigor_note.tex` — the **v5** note (10 pp; v3 theory + §"Simulation
  findings" F1–F7 + §"Priority and novelty" deep search). Self-contained.
- `cascade_sim.py` — simulator; 10 subcommands (`smoke/identity/phase/
  hysteresis/avalanche/ews/lagstudy/escape/avfit/entropy`). All seeded.
- `run_*_2026-06-12.txt` — seeded production-run records (phase / ews /
  avalanche / lagstudy / escape / avfit / entropy), committed as the evidence
  record. `.txt` because `*.log` is gitignored for LaTeX artifacts.
- `cascade_sim.py` — the discovery simulator (event-driven closed-loop queue
  with genealogy tracking + mean-field solvers + entropy layer). Subcommands:
  `smoke`, `identity`, `phase`, `hysteresis`, `avalanche`, `ews`. Figures land
  in `figures/` (not committed; regenerate with the seeded commands below).

## Build / run
```
pdflatex sr_rigor_note.tex && pdflatex sr_rigor_note.tex
python3 cascade_sim.py smoke                                  # validation
python3 cascade_sim.py hysteresis --theta 5 --alpha 0.5 --l0-min 0.30 \
        --l0-max 0.70 --l0-step 0.025 --dwell 1500            # the loop
python3 cascade_sim.py avalanche --theta 0.2 --alpha 0.9 --l0 0.01 \
        --t-max 1000000                                       # 3/2 law
```

## Simulation status (2026-06-12): validated, first findings in

**Validation (smoke PASS):** α=0 reproduces exact M/M/1 (P_u to 0.001);
the bookkeeping identity ⟨s⟩ = x/ℓ₀ holds to ~1% at every tested point;
fold/cusp closed forms confirmed; entropy layer exact
(SR_EP = (ln2/2βh)·SR to machine precision).

**Finding 1 — congestion-correlation enhancement (beyond mean-field).**
Zero-lag offspring land in the very congestion that killed their parent, so
the simulated feedback is *stronger* than mean-field: x_sim > x_mf everywhere
(e.g. α=0.9, ℓ₀=0.05: amplification 5.4× vs predicted 1.6×). Mechanism
*proved* by the `spawn_lag` knob: lag ≫ queue correlation time → x_sim = x_mf
to 3 decimals. Consequence: **mean-field is anti-conservative** — the true
collapse spinodal sits below ℓ₀ᶜ (measured: collapse at ℓ₀ ≈ 0.50 vs
mean-field 0.653 at θ=5, α=0.5; ~3× shift in the self-referential corner).

**Finding 2 — hysteresis confirmed, dynamically wider than static.** The loop
is wide open (figure `hysteresis.png`): up-ramp rides the lucid branch with a
metastable flicker precursor at ℓ₀=0.425, collapses ≈0.50; down-ramp rides the
collapsed branch x = ℓ₀/(1−α) and *stays collapsed below the static recovery
spinodal* because the accumulated backlog is a memory variable (recovery lags
by the drain time of the debt).

**Finding 3 — the 3/2 exponent is robust; the cutoff renormalizes.**
Avalanche sizes follow s^(−3/2) over ~2 decades (10,090 closed avalanches,
0 censored); the congestion correlation feeds the branching ratio
(b_sim = 0.877 vs mean-field 0.743), stretching the cutoff but not bending
the exponent — a preliminary answer to the note's open question.

**Prop. 2 observed accidentally:** at α=1.1 the system collapsed and
avalanches never closed (supercritical genealogy in collapse) — the
irreversibility theorem in action. The `avalanche` subcommand now
self-diagnoses this.

## Paper constraints (Pieter, 2026-06-12 — binding for the restructure)
- **arXiv:2606.07937 is a passing reference only** (not peer-reviewed; do not
  anchor the paper's motivation on it — the theory stands on its own physics,
  with the cascade literature cited "by the way").
- **Length discipline:** the closed-loop section must be tight; the space
  comes out of the Illustrative Application material, not the physics.
- **Table 1 of the PRE draft (`tab:evolutionary_hierarchy`) STAYS** in the
  paper (overrides the earlier review suggestion to cut it).

## Production runs (2026-06-12) — landed, folded into note v4 §Findings

- **lagstudy** (α=0.6, ℓ₀=0.2, θ=1): the enhancement law measured — excess
  +25.6% at lag 0 decaying monotonically to +1.7% at lag 50/μ; crossover at
  the queue correlation time. "Delay is a gate."
- **phase map** (θ=1, 23×19 grid, T=8000): the measured boundary tracks the
  **recovery spinodal**, not the collapse spinodal — on observation times
  beyond the escape time the whole bistable wedge belongs to the collapsed
  phase (deterministic spinodal = fast-ramp/large-system limit). Detection-
  limited near the continuous boundary (slow backlog growth) — noted.
- **ews** (θ=5, α=0.5): every point with ε ≤ 0.20 of the mean-field fold
  escaped by fluctuation at t ~ (1.1–4.1)×10⁴; sole survivor (ε=0.30) already
  at AC1 = 0.996 → the "estimation race" remark in the note: EW indicators
  must converge faster than the escape; use at distance / in fleets.
- **avalanche** (θ=0.2, α=0.9, ℓ₀=0.01): 60,243 closed avalanches, 0
  censored; s^(−3/2) over ~2 decades; b_eff = 0.881 vs mean-field 0.743 —
  exponent robust, cutoff renormalized.

## Remaining measurements (2026-06-12) — done, in note v5 §Findings F5–F7

- **escape** (α=0.5; θ=3,5,7; 8 seeds; 0/8 censored): median lifetime falls
  with wedge depth AND with θ (f=0.5: 1.70→1.05→0.74 ×10⁴ for θ=3,5,7).
  **Correction to v3:** θ is *not* the sharpening parameter — it deepens the
  wedge but shallows the lucid well. System size **N** is the sharpening axis
  (the one load-bearing run still outstanding).
- **avfit** (ML fit P(s)~s^−τ e^−s/sc, 4–12×10⁴ avalanches/pt): **τ = 1.500
  [1.500,1.500]** at the deep points, 1.52 shallow; cutoffs 185/86/46 vs
  predicted 143/66/37 (uniform ~1.25× prefactor = the F1 correlation); three
  points **collapse** onto one master curve. Interaction renormalizes the
  scale, not the exponent.
- **entropy**: synthetic fraction jumps <0.1 → >0.95 across the wedge
  (sharp order-parameter step); 25,221 synthetic-entropy bursts inherit the
  s^−3/2 tail.

## Deep priority search (2026-06-12) — novelty fixed

**Strongest precedent is peer-reviewed and helps:** Bronson et al.,
**HotOS '21 "Metastable Failures in Distributed Systems"** names/unifies this
exact phenomenon (collapse, hysteresis, reset-cure via "big corrective
action") — but is *engineering and qualitative*: by its own framing **no order
parameter, phase diagram, critical exponents, hysteresis quantification,
bifurcation analysis, or thermodynamic treatment** (confirmed by fetching the
paper). It is the ideal anchor — authoritative, peer-reviewed, and it
documents precisely the gaps this work fills. **This lets 2606.07937 drop to a
one-line pointer** (satisfies Pieter's constraint).
**Sibling, not this:** model collapse / MAD (self-consuming models) is
*training-time* generational drift, not *inference-time* queue saturation —
cite to delimit.
**Mechanism precedents:** first-order congestion collapse (PRE 75 036102; TCP),
Takács feedback queues (constant feedback → no fold); τ=3/2 is textbook
branching (cite, don't claim).
**Genuinely new (survives search):** the deadline-feedback closure + closed-form
fold/cusp; collapse-pre-empts-divergence (Prop 1); the reset-cure *theorem*
(Prop 2); the genealogy×congestion factorization + cutoff law; the whole
thermodynamic layer (Landauer-priced avalanches, SR_EP discontinuity, demon-
priced gate); the "delay is a gate" enhancement law.
**Framing rule:** claim the *theory* (exponents, cusp, entropy pricing), never
the *phenomenon*.

## F8 — N-scaling (2026-06-12, the final load-bearing run)

`nscaling` subcommand (CascadeSim gained `n_servers`: N parallel decoders
sharing the exogenous stream + spawn pool, uniform routing; smoke regression
identical at N=1). At mid-wedge (θ=5, α=0.5, f=0.5, 8 seeds): median lifetime
~1.2×10⁴ for N≤4, 3.5×10⁴ at N=8, **8/8 censored at 2×10⁵ for N=16** —
faster-than-polynomial growth: the mean-field bistability becomes a sharp
first-order transition in the fleet limit. A single tightly-coupled agent is
fragile; a fleet is collectively stable.

## Status
Note v5 + simulator (11 subcommands) + findings F1–F8 + priority search.
**The paper restructure is underway on branch `paper-closed-loop`** (new
§"Closed-Loop Decoders" + Appendix, Tier-1 SR definition, notation sweep
Ṡ_syn/Ṡ_anc, Bronson anchor, three cascade figures, Table 1 intact).

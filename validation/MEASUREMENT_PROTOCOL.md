# Measurement protocol — Tier 1 → Tier 2 validation (PRE-REGISTRATION)

**Status: FROZEN 2026-06-12 (Pieter sign-off).** This commit is the
pre-registration timestamp. §2 (definitions) and §5 (predictions) are now
locked: they must not change after the first Tier-1 analysis result is
computed. §4 estimator code may still be debugged for *correctness* against
§2, but any change to a thresholding/binning *choice* re-opens §5 and requires
a new freeze commit (recorded in the changelog at the bottom). Tier-1 results
(P3–P5) must not be inspected before the estimator is validated on Tier 2.

Governs Phases 2–4 of `../VALIDATION_PLAN.md`. The whole point: **one mapping,
one estimator, applied identically to a controlled experiment (Tier 2, where
α and ℓ₀ are ground truth) and an observational corpus (Tier 1, MAST-Data).**
Tier 2 validates the estimator; the validated estimator is then trusted on
Tier 1. This is what answers reviewer objection M2 ("falsifiable only about its
own toy model").

---

## 1. Freeze rule
- §2 (definitions) and §5 (predictions) are frozen at sign-off.
- §4 (estimator code) may be debugged after freeze **only** for correctness
  against §2; any change to a thresholding/binning *choice* in §4 re-opens §5.
- Tier-2 ground-truth runs and the estimator-validation (§5 P2) may be inspected
  before freeze (they calibrate the instrument). Tier-1 results (P3–P5) must
  not be looked at until freeze.

## 2. Operational definitions (the crux — identical across both tiers)

The theory's objects are abstract; here they are pinned to events in an agent
event log. A **log** is a partially ordered set of *messages* (agent turns,
tool calls, sub-answers), each with: emitter, content, the set of prior
messages in its input context (its *parents*), and a type.

- **Commitment.** A message that conditions downstream work — i.e. one that
  becomes a parent of at least one later message, or that issues an irreversible
  external action (tool call, environment action, final answer). Granularity:
  **message-level** is primary; claim-level (sentence/triple extraction) is a
  pre-registered robustness variant (§6).
- **Certification event.** A message/operation that *validates* a prior
  commitment against something external before it is reused: an explicit
  verifier/critic/reflection turn, a tool execution returning a ground result
  (code run + tests, search hit, calculator), or human confirmation. Per-
  framework adapters enumerate which message types count (§4).
- **Certified vs. uncertified commitment.** A commitment is **certified** if a
  certification event acts on it *before* any child consumes it; **uncertified**
  if a child consumes it first (or no certification event ever acts on it).
- **Cascade (avalanche).** The genealogy subtree rooted at one uncertified
  commitment: all later commitments causally descended from it through the
  parent edges. **Cascade size s** = number of commitments in that subtree
  (root included).
- **Grounded (exogenous) commitment.** A commitment whose parents are *only*
  external/grounded inputs (task prompt, retrieved document, tool result) — not
  descended from an uncertified commitment. These are the ℓ₀ "exogenous
  arrivals."

**Genealogy construction.** Build a DAG: edge A→B iff A ∈ parents(B). For
fixed-pipeline frameworks parents = the framework's message-passing structure;
for free-form group chats (e.g. AG2) parents = messages in B's context window
that B references or that are within the framework's memory horizon. The
adapter is per-framework; the *rule* is identical.

## 3. Parameters and their meaning in the log

- **SR** = Ṅ_uncert / Ṅ_cert  (counts of uncertified vs certified commitments).
- **α (ungatedness)** = mean out-degree of *uncertified* commitments in the
  genealogy (mean number of children that consume an unverified output).
- **ℓ₀ (grounding)** = fraction of commitments that are grounded (exogenous).
  Equivalently via the branching identity, ℓ₀/x* = 1 − b.
- **θ (deadline depth)** = certification-time / decision-horizon ratio.
  **Least identifiable from static logs.** Decision: θ is **estimated only in
  Tier 2** (controlled); in Tier 1 it is treated as a fixed nuisance, and the
  Tier-1 predictions (§5 P3) are framed to depend on α, ℓ₀ — through b = 1 −
  ℓ₀/x* and s_c ≈ 2(x*/ℓ₀)² — at *comparable* θ across frameworks, not on θ
  itself.

## 4. The shared estimator (one codebase, both tiers)

`validation/estimator.py` (to build). Input: a normalized event log
(§2 schema). Output: per-run and per-framework
{N_uncert, N_cert, SR, α̂, ℓ̂₀, cascade-size list, b̂=1−1/⟨s⟩, ŝ_c}.
Pipeline:
1. **Adapter** (per source) → normalized log (emitter, content, parents, type).
   Tier 2: emitted directly by the instrumented pipeline (no inference).
   Tier 1: per-framework parser for MAST raw traces (AG2, MetaGPT, Magentic,
   ChatDev, …). This parsing is the real work item Phase 1 surfaced.
2. **Classify** each commitment certified/uncertified per §2.
3. **Build** the genealogy DAG; extract rooted subtrees → cascade sizes.
4. **Estimate** α̂, ℓ̂₀, SR, ⟨s⟩, b̂, ŝ_c; ML power-law+cutoff fit
   (reuse `sr_rigor_fix/cascade_sim.py::_fit_powerlaw_cutoff`, with the
   selection-bias-aware variant of §6).
**Gold-subset check:** the 19 human-step-labelled MAST traces
(`MAD_human_labelled_dataset.json`) validate the automatic error-root +
certification classification against human annotation before the estimator is
run on the 1,242-trace full set.

## 5. Pre-registered predictions (FROZEN at sign-off)

**Tier 2 (controlled — causal, the base):**
- **P1.** Sweeping (α, ℓ₀) across the cusp α* = 1/(1+θ): a continuous transition
  below the cusp, a *first-order* transition with a hysteresis loop above it
  (collapse near ℓ₀^c, recovery only below ℓ₀ = 1−α). Reset-cure at α ≥ 1
  (load reduction alone does not recover; backlog drain does).
- **P2 (the M2-killer).** The estimator recovers the *known* ground-truth
  (α, ℓ₀) set in the pipeline, within a pre-stated tolerance (target ≤ 15 %
  relative, reported with CI). Calibrates SR_c.

**Tier 1 (observational — corroboration, MAST-Data):**
- **P3.** Cascade sizes are **subcritical with framework-dependent cutoffs**
  (curated failure traces sit far from the corner; we do *not* predict a clean
  −3/2 here). At comparable θ, the cutoff ordering across the 7 frameworks
  tracks estimated grounding: **lower ℓ̂₀ ⇒ larger ŝ_c.** Parameter-free after
  the Tier-2 SR_c calibration.
- **P4 (independent cross-check).** Traces MAST-annotated with inter-agent
  misalignment (category 2) show larger extracted cascades than traces whose
  dominant mode is system-design (cat 1) or verification (cat 3), controlling
  for framework topology (out-degree).
- **P5 (stretch).** Tier-1 per-framework and Tier-2 swept cascade distributions
  collapse onto one P(s)·s^{3/2} vs s/ŝ_c master curve.

## 6. Risk register & null-result handling (pre-registered)

| Risk | Pre-registered handling |
|---|---|
| **Selection bias** (MAST = curated *failures* → P(s) truncated) | Claim cutoff *ordering* (P3), not the exponent, from Tier 1; MLE-with-cutoff on the tail; report truncation explicitly. The exponent is a Tier-2 claim only. |
| **Fixed-topology confound** (a framework's out-degree caps branching) | Report α̂ *and* raw out-degree; compare cutoffs at matched topology; treat topology as a covariate in P3/P4. |
| **θ unidentifiable from static logs** | θ estimated in Tier 2 only; Tier-1 claims depend on α, ℓ₀ (§3). |
| **Certification inference is a modeling choice** | Validate on the 19 gold human-labelled traces (§4); report classifier agreement; run the claim-level robustness variant. |
| **Tier-1 null** (no ordering) | *Informative*, pre-registered: bounds real MAS as well-grounded / far-subcritical. Reported as a result, not suppressed. |
| **Estimator overfit to Tier-2** | P3/P4 use *no* Tier-1 fitting beyond the Tier-2-calibrated SR_c (out-of-sample). |

## 7. Analysis discipline
- Same `estimator.py` for both tiers; adapters differ, core identical.
- Tier-1 results not inspected until freeze (§1).
- All runs seeded; raw logs + extracted genealogies archived; code → the
  public `cascade-collapse` repo at Phase 5 (D2).

## 8. Resolved judgment calls (Pieter, 2026-06-12 — part of the freeze)
1. **Certification in MAST = verifier-or-tool ONLY.** A certification event is
   a designated verifier/critic/reflection turn, OR a tool execution returning
   a ground result (code+tests, search, calculator), OR human confirmation. **A
   peer agent's response is NOT certification — it is consumption** (precisely
   what makes the consumed commitment uncertified). This is the load-bearing
   rule for the certified/uncertified split; it is deliberately strict, which
   biases SR *upward* (more uncertified) — a conservative direction for the
   collapse claims, and stated as such in the paper.
2. **Granularity: message-level primary, claim-level as the §6 robustness
   variant.** Confirmed.
3. **Division of labor confirmed.** MAST probes the subcritical regime: Tier 1
   carries the cross-framework cutoff *ordering* (P3/P4); Tier 2 carries the
   exponent and the phase structure (P1). The exponent is never claimed from
   Tier 1.
4. **P2 tolerance: ≤15 % relative recovery of (α, ℓ₀)** to call the estimator
   validated; reported with CI. Confirmed.

## Changelog
- 2026-06-12: frozen (Pieter sign-off). Defaults accepted; certification rule
  set to verifier-or-tool-only (§8.1).

#!/usr/bin/env python3
"""B3 statistics for the pipeline experiment (referee-grade error bars).

1. Collapse-point ensemble: N independent seeded up-ramps at
   (alpha, theta) = (0.5, 8); per run, the last lucid and first collapsed
   ramp value -> the collapse-point distribution (the up-ramp collapse is a
   stochastic escape, so a single bracket is one draw).
2. Experimental-cascade MLE: the same two-stage profile-likelihood fit used
   for the simulations (sr_rigor_fix/cascade_sim.py:_fit_powerlaw_cutoff)
   applied to the 31,903 measured closed genealogies.
3. Cusp-scan repeats: 3 seeded repetitions of the alpha scan at theta = 1.5.

Writes results/b3_ensemble.json (and leaves the original result files
untouched).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sr_rigor_fix"))

import pipeline_rig as rig                      # noqa: E402
from cascade_sim import _fit_powerlaw_cutoff    # noqa: E402

ALPHA, THETA = 0.5, 8.0
DWELL, L0_MAX, STEPS = 6.0, 0.8, 15
N_RUNS = 8
CUSP_THETA, CUSP_ALPHAS, CUSP_REPS = 1.5, [0.2, 0.35, 0.5, 0.65, 0.8], 3
COLLAPSE_BACKLOG = 2000


def mle_experimental_cascades() -> dict:
    d = json.loads((RESULTS / "cascades_a0.92_l0.15_th1.5.json").read_text())
    sizes = np.concatenate([np.full(int(c), int(s))
                            for s, c in d["sizes_hist"].items()])
    tau, tlo, thi, sc, sclo, schi, n = _fit_powerlaw_cutoff(sizes, s_min=2)
    out = dict(n_cascades=int(len(sizes)), n_fit=int(n),
               tau=float(tau), tau_lo=float(tlo), tau_hi=float(thi),
               s_c=float(sc), s_c_lo=float(sclo), s_c_hi=float(schi),
               s_c_pred_from_mean=2.0 * float(sizes.mean()) ** 2)
    print("[mle]", json.dumps(out), flush=True)
    return out


def collapse_ensemble(mu: float) -> dict:
    l0s = np.linspace(0.05, L0_MAX, STEPS)
    runs = []
    for r in range(N_RUNS):
        rig.RNG = np.random.default_rng(100 + r)
        pipe = rig.Pipeline(mu, ALPHA, THETA, 0.0)
        last_lucid = first_collapsed = None
        t_clock = time.perf_counter()
        for l0 in l0s:
            now = time.perf_counter()
            pipe.schedule_exogenous(l0, max(now, t_clock), now + DWELL)
            w = pipe.run_window(DWELL)
            t_clock = time.perf_counter()
            print(f"[ens {r}] l0={l0:.3f} x={w['x']:.3f} "
                  f"P_u={w['P_u']:.3f} backlog={w['backlog']}", flush=True)
            if w["backlog"] > COLLAPSE_BACKLOG or w["P_u"] > 0.9:
                first_collapsed = float(l0)
                break
            last_lucid = float(l0)
        runs.append(dict(run=r, last_lucid=last_lucid,
                         first_collapsed=first_collapsed))
    fc = [x["first_collapsed"] for x in runs if x["first_collapsed"]]
    out = dict(runs=runs, n=len(fc),
               first_collapsed_min=min(fc) if fc else None,
               first_collapsed_max=max(fc) if fc else None,
               first_collapsed_median=float(np.median(fc)) if fc else None)
    print("[ens]", json.dumps({k: v for k, v in out.items() if k != "runs"}),
          flush=True)
    return out


def cusp_repeats(mu: float) -> dict:
    reps = []
    for rep in range(CUSP_REPS):
        row = {}
        for alpha in CUSP_ALPHAS:
            rig.RNG = np.random.default_rng(500 + 10 * rep + int(alpha * 10))
            pipe = rig.Pipeline(mu, alpha, CUSP_THETA, 0.0)
            xs, l0s = [], np.linspace(0.05, 0.98, STEPS)
            for l0 in l0s:
                now = time.perf_counter()
                pipe.schedule_exogenous(l0, now, now + DWELL)
                w = pipe.run_window(DWELL)
                xs.append(w["x"])
                if w["backlog"] > rig.BACKLOG_CAP:
                    break
            jumps = np.diff(xs)
            row[str(alpha)] = round(float(jumps.max()), 4) if len(jumps) else None
            print(f"[cusp rep {rep}] alpha={alpha}: {row[str(alpha)]}",
                  flush=True)
        reps.append(row)
    summary = {a: dict(mean=float(np.mean([r[str(a)] for r in reps])),
                       lo=float(np.min([r[str(a)] for r in reps])),
                       hi=float(np.max([r[str(a)] for r in reps])))
               for a in CUSP_ALPHAS}
    print("[cusp]", json.dumps(summary), flush=True)
    return dict(alphas=CUSP_ALPHAS, alpha_star=1 / (1 + CUSP_THETA),
                reps=reps, summary=summary)


def main() -> None:
    mu = rig._mu()
    out = dict(alpha=ALPHA, theta=THETA, dwell=DWELL, steps=STEPS,
               mu_hat=mu,
               mle=mle_experimental_cascades(),
               collapse_ensemble=collapse_ensemble(mu),
               cusp_repeats=cusp_repeats(mu))
    (RESULTS / "b3_ensemble.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("wrote results/b3_ensemble.json")


if __name__ == "__main__":
    main()

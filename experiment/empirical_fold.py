#!/usr/bin/env python3
"""Mean-field collapse spinodal computed with the EMPIRICAL service law.

The paper's mean-field spinodal l0^c = 0.724 (alpha=0.5, theta=8) uses the
M/M/1 deadline-overflow probability P_u = exp(-(1-x)theta), i.e. assumes
exponential service. The rig's real service has CV ~= 0.31, whose thinner
sojourn tail weakens the feedback at given load and shifts the predicted
fold. This script computes the honest mean-field prediction:

  1. sample the real service-time distribution (same solver as the rig);
  2. estimate P_u(x) = P(sojourn > theta) for a stationary M/G/1 queue at
     utilization x by Lindley recursion, with service times bootstrapped
     from the measured samples (time in units of the mean service time,
     matching the rig's deadline convention Delta t = theta / mu_hat);
  3. locate the fold as the maximum of l0(x) = x (1 - alpha P_u(x)) on the
     lucid branch;
  4. validate the estimator by running the same pipeline with Exp(1)
     service, which must reproduce the analytic M/M/1 fold.

Writes results/empirical_fold.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pipeline_rig import do_real_work

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ALPHA = 0.5
THETA = 8.0
N_SAMPLES = 4000
N_JOBS = 400_000
N_WARMUP = 20_000
SEEDS = (1, 2, 3)
X_GRID = np.round(np.arange(0.70, 0.985, 0.01), 3)


def sample_service(n: int = N_SAMPLES) -> np.ndarray:
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        do_real_work()
        ts[i] = time.perf_counter() - t0
    return ts


def p_u_mg1(service_pool: np.ndarray, x: float, seed: int) -> float:
    """P(sojourn > THETA) for stationary M/G/1 at utilization x.

    Service times in units of their mean (E[S] = 1), Poisson arrivals at
    rate x, FCFS, Lindley recursion for the waiting time.
    """
    rng = np.random.default_rng(seed)
    s = rng.choice(service_pool, size=N_JOBS)
    a = rng.exponential(1.0 / x, size=N_JOBS)
    w = 0.0
    hits = 0
    counted = 0
    for i in range(N_JOBS):
        t = w + s[i]                      # sojourn of job i
        if i >= N_WARMUP:
            counted += 1
            if t > THETA:
                hits += 1
        w = t - a[i]
        if w < 0.0:
            w = 0.0
    return hits / counted


def fold_from_curve(xs: np.ndarray, l0s: np.ndarray) -> float:
    """Fold = max of l0(x); refined by a local quadratic through the peak."""
    k = int(np.argmax(l0s))
    if 0 < k < len(xs) - 1:
        c = np.polyfit(xs[k - 1:k + 2], l0s[k - 1:k + 2], 2)
        xv = -c[1] / (2 * c[0])
        return float(np.polyval(c, xv))
    return float(l0s[k])


def scan(service_pool: np.ndarray, tag: str) -> dict:
    per_seed_folds = []
    pu_mean = np.zeros(len(X_GRID))
    for seed in SEEDS:
        pus = np.array([p_u_mg1(service_pool, x, seed * 1000 + i)
                        for i, x in enumerate(X_GRID)])
        l0s = X_GRID * (1.0 - ALPHA * pus)
        per_seed_folds.append(fold_from_curve(X_GRID, l0s))
        pu_mean += pus / len(SEEDS)
    l0_mean = X_GRID * (1.0 - ALPHA * pu_mean)
    fold = fold_from_curve(X_GRID, l0_mean)
    print(f"[{tag}] fold = {fold:.4f}  (seeds: "
          + ", ".join(f"{f:.4f}" for f in per_seed_folds) + ")")
    return dict(fold=fold, per_seed=per_seed_folds,
                x_grid=X_GRID.tolist(), p_u=pu_mean.tolist(),
                l0_curve=l0_mean.tolist())


def mm1_fold_analytic() -> float:
    g = lambda xf: ALPHA * (1 + THETA * xf) * np.exp(-(1 - xf) * THETA) - 1
    lo, hi = 1e-6, 1 - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    xf = 0.5 * (lo + hi)
    return float(xf - ALPHA * xf * np.exp(-(1 - xf) * THETA))


def main() -> None:
    print("sampling real service times ...")
    ts = sample_service()
    mu_hat = 1.0 / ts.mean()
    cv = float(ts.std() / ts.mean())
    print(f"n={len(ts)}  mu_hat={mu_hat:.1f}/s  CV={cv:.3f}")
    pool = ts / ts.mean()                 # E[S] = 1

    rng = np.random.default_rng(7)
    exp_pool = rng.exponential(1.0, size=200_000)

    out = dict(alpha=ALPHA, theta=THETA,
               calibration=dict(n=len(ts), mu_hat=float(mu_hat), cv=cv),
               mm1_fold_analytic=mm1_fold_analytic(),
               validation_exponential=scan(exp_pool, "Exp(1) validation"),
               empirical=scan(pool, "empirical service"))
    (RESULTS / "empirical_fold.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"analytic M/M/1 fold: {out['mm1_fold_analytic']:.4f}")
    print("wrote results/empirical_fold.json")


if __name__ == "__main__":
    main()

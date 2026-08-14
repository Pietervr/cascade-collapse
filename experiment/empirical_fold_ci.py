#!/usr/bin/env python3
"""Bootstrap confidence interval for the empirical-service collapse spinodal.

empirical_fold.py reports the point estimate l0^c_emp ~= 0.803 from one pool
of 4000 measured service times and three Lindley seeds. This script
quantifies that number's uncertainty with the IDENTICAL estimator:

  1. sample a fresh 4000-sample service pool on the same solver (so
     session-to-session calibration drift is included by construction);
  2. compute the point fold for the fresh pool;
  3. draw B bootstrap resamples of the pool (with replacement); each yields
     a fold via the same M/G/1-Lindley pipeline (one seed per replicate, so
     the interval also absorbs the Lindley Monte-Carlo noise);
  4. report percentile intervals of the replicate folds.

Writes results/empirical_fold_ci.json, archiving the normalized service
pool alongside the replicate folds for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from empirical_fold import ALPHA, X_GRID, fold_from_curve, p_u_mg1, sample_service

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

B = 20


def fold_for_pool(pool: np.ndarray, seed: int) -> float:
    pus = np.array([p_u_mg1(pool, x, seed * 1000 + i)
                    for i, x in enumerate(X_GRID)])
    return fold_from_curve(X_GRID, X_GRID * (1.0 - ALPHA * pus))


def main() -> None:
    print("sampling fresh service pool ...")
    ts = sample_service()
    mu_hat = 1.0 / ts.mean()
    cv = float(ts.std() / ts.mean())
    print(f"n={len(ts)}  mu_hat={mu_hat:.1f}/s  CV={cv:.3f}")
    pool = ts / ts.mean()

    point = fold_for_pool(pool, seed=1)
    print(f"point fold (fresh pool) = {point:.4f}")

    rng = np.random.default_rng(2026)
    folds = []
    for b in range(B):
        bp = rng.choice(pool, size=pool.size, replace=True)
        f = fold_for_pool(bp, seed=100 + b)
        folds.append(f)
        print(f"  bootstrap {b + 1:2d}/{B}: fold = {f:.4f}")

    fa = np.array(folds)
    lo, hi = np.percentile(fa, [2.5, 97.5])
    out = dict(alpha=ALPHA,
               calibration=dict(n=len(ts), mu_hat=float(mu_hat), cv=cv),
               point_fold=point,
               bootstrap_folds=fa.tolist(),
               ci95=[float(lo), float(hi)],
               spread=[float(fa.min()), float(fa.max())],
               service_pool_normalized=pool.tolist())
    (RESULTS / "empirical_fold_ci.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"95% bootstrap interval: [{lo:.4f}, {hi:.4f}]  "
          f"(spread [{fa.min():.4f}, {fa.max():.4f}])")
    print("wrote results/empirical_fold_ci.json")


if __name__ == "__main__":
    main()

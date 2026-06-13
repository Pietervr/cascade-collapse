#!/usr/bin/env python3
"""Shared estimator (Tier 2 AND Tier 1) -- the M2-killer.

Reads a normalized event log (validation/MEASUREMENT_PROTOCOL.md section 2
schema) and recovers the theory's structural parameters using ONLY the
genealogy + certified/uncertified flags -- no wall-clock rates -- so the
identical code applies to the controlled pipeline (Tier 2, ground truth known)
and to MAST static traces (Tier 1).

Recovered (all rate-free):
  N_uncert, N_cert, SR = N_uncert/N_cert
  alpha_hat   = mean out-degree of uncertified commitments
  P_u         = N_uncert / (N_uncert + N_cert)
  grounding   = N_exo / N_total           (= 1 - b at steady state; the
                                            protocol's l0_hat)
  b_size      = 1 - 1/<s>                  (from cascade sizes)
  b_identity  = alpha_hat * P_u            (bookkeeping)
  consistency = |b_identity - (1 - grounding)|   -> should be ~0 on real logs
  cascade_sizes (per exogenous-root subtree) + ML power-law+cutoff fit

The two independent routes to b (b_identity vs b_size=1-grounding) agreeing is
the on-real-data analog of cascade_sim's <s> = x/l0 internal check.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# reuse the validated ML fit from the simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "sr_rigor_fix"))
try:
    from cascade_sim import _fit_powerlaw_cutoff
except Exception:                                   # pragma: no cover
    _fit_powerlaw_cutoff = None


def _children_index(commits):
    """parent_id -> list of child ids, and id -> record."""
    by_id = {c["id"]: c for c in commits}
    kids = {c["id"]: [] for c in commits}
    for c in commits:
        for p in c.get("parents", []):
            if p in kids:
                kids[p].append(c["id"])
    return by_id, kids


def _subtree_size(root, kids):
    seen, stack, n = set(), [root], 0
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        n += 1
        stack.extend(kids.get(x, []))
    return n


def estimate(commits, fit=True):
    """commits: list of section-2 commitment records. Returns dict."""
    by_id, kids = _children_index(commits)
    N = len(commits)
    exo = [c for c in commits if not c.get("parents")]
    unc = [c for c in commits if c.get("uncertified_fired")]
    cert = [c for c in commits if c.get("certified")]
    n_u, n_c = len(unc), len(cert)

    alpha_hat = (float(np.mean([len(kids[c["id"]]) for c in unc]))
                 if unc else float("nan"))
    P_u = n_u / (n_u + n_c) if (n_u + n_c) else float("nan")
    grounding = len(exo) / N if N else float("nan")
    SR = (n_u / n_c) if n_c else float("inf")

    sizes = np.array([_subtree_size(c["id"], kids) for c in exo], dtype=float)
    mean_s = float(sizes.mean()) if sizes.size else float("nan")
    b_size = 1.0 - 1.0 / mean_s if mean_s and mean_s > 0 else float("nan")
    b_identity = alpha_hat * P_u if alpha_hat == alpha_hat else float("nan")
    consistency = (abs(b_identity - (1.0 - grounding))
                   if b_identity == b_identity else float("nan"))

    out = dict(N=N, N_exo=len(exo), N_uncert=n_u, N_cert=n_c, SR=SR,
               alpha_hat=alpha_hat, P_u=P_u, grounding=grounding,
               mean_cascade=mean_s, b_size=b_size, b_identity=b_identity,
               b_from_grounding=1.0 - grounding, consistency=consistency,
               n_cascades=int(sizes.size))
    if fit and _fit_powerlaw_cutoff is not None and sizes.size >= 200:
        tau, tlo, thi, sc, slo, shi, nfit = _fit_powerlaw_cutoff(sizes)
        out.update(tau=tau, tau_lo=tlo, tau_hi=thi, s_c=sc, n_fit=nfit)
    return out


def estimate_log(log, **kw):
    return estimate(log["commitments"], **kw)


def _fmt(d):
    keys = ["N", "N_exo", "N_uncert", "N_cert", "SR", "alpha_hat", "P_u",
            "grounding", "mean_cascade", "b_identity", "b_from_grounding",
            "consistency", "n_cascades"]
    return "  ".join(f"{k}={d[k]:.3f}" if isinstance(d[k], float)
                     else f"{k}={d[k]}" for k in keys if k in d)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        log = json.load(open(path))
        d = estimate_log(log)
        gt = log.get("params", {})
        print(f"== {os.path.basename(path)} ==")
        if "alpha" in gt:
            print(f"   ground truth: alpha={gt['alpha']} l0={gt['l0']} "
                  f"theta={gt['theta']}")
        print("   " + _fmt(d))
        if "tau" in d:
            print(f"   tau={d['tau']:.3f} [{d['tau_lo']:.3f},{d['tau_hi']:.3f}]"
                  f" s_c={d['s_c']:.0f}")

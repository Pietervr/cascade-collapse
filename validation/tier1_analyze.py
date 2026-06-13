#!/usr/bin/env python3
"""Tier-1 analysis (VALIDATION_PLAN Phase 4): run the SHARED estimator over the
adapter output and test the pre-registered observational predictions.

  P3  cross-framework cutoff ORDERING: lower grounding l0_hat => larger cascade
      cutoff. (Tier 1 claims the ordering, NOT the -3/2 exponent -- curated
      failure traces are truncated; risk register.)
  P4  traces MAST-annotated with inter-agent misalignment (category 2) show
      larger extracted cascades than category 1 (system-design) or 3
      (verification), controlling for framework topology.
  P5  Tier-1 per-framework + Tier-2 cascade distributions collapse onto one
      P(s)*s^{3/2} vs s/s_c master curve.

Reads validation/runs/tier1/tier1_<fw>.json; writes runs/tier1/analysis/.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from estimator import estimate, _children_index, _subtree_size   # noqa: E402

TIER1 = os.path.join(_HERE, "runs", "tier1")
OUT = os.path.join(TIER1, "analysis")
# generic (lower-confidence) adapters -- reported but flagged
GENERIC = {"OpenManus", "AppWorld", "HyperAgent"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _reid(traces):
    """Pool all traces' commitments with globally unique ids (offset per trace
    so subtree roots never collide). Returns (all_commits, per_trace_sizes)."""
    allc, off = [], 0
    per_trace = []
    for t in traces:
        cs = t["commitments"]
        if not cs:
            continue
        for c in cs:
            c2 = dict(c)
            c2["id"] = c["id"] + off
            c2["parents"] = [p + off for p in c.get("parents", [])]
            allc.append(c2)
        # per-trace cascade sizes (subtrees under this trace's exo roots)
        _, kids = _children_index([dict(c, parents=c.get("parents", []))
                                   for c in cs])
        exo = [c["id"] for c in cs if not c.get("parents")]
        per_trace.append([_subtree_size(r, kids) for r in exo])
        off += max(c["id"] for c in cs) + 1
    return allc, per_trace


def _spearman(x, y):
    """Spearman rank correlation (no scipy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TIER1, "tier1_*.json")))
    rows, all_sizes = [], {}
    for f in files:
        fw = os.path.basename(f)[len("tier1_"):-len(".json")]
        traces = json.load(open(f))
        allc, per_trace = _reid(traces)
        # fit=False: Tier-1 claims cutoff ORDERING via robust proxies
        # (mean/p95/max), not the -3/2 exponent (curated-failure truncation).
        d = estimate(allc, fit=False)
        sizes = np.array([s for ts in per_trace for s in ts], dtype=float)
        all_sizes[fw] = sizes
        # robust cutoff proxies (ordering, not exponent): mean, p95, ML s_c
        p95 = float(np.percentile(sizes, 95)) if sizes.size else float("nan")
        smax = float(sizes.max()) if sizes.size else float("nan")
        rows.append(dict(
            framework=fw, n_traces=len(traces), N=d["N"],
            grounding=d["grounding"], SR=d["SR"], alpha_hat=d["alpha_hat"],
            mean_cascade=d["mean_cascade"], b_size=d["b_size"],
            p95=p95, s_max=smax, s_c=d.get("s_c"), tau=d.get("tau"),
            n_cascades=d["n_cascades"], generic=(fw in GENERIC)))

    rows.sort(key=lambda r: r["grounding"])
    print(f"{'framework':<11}{'gen':>4}{'N':>7}{'l0_hat':>8}{'SR':>7}"
          f"{'<s>':>7}{'p95':>6}{'s_max':>7}{'s_c':>7}")
    for r in rows:
        sc = f"{r['s_c']:.0f}" if r["s_c"] else "-"
        print(f"{r['framework']:<11}{'*' if r['generic'] else ' ':>4}"
              f"{r['N']:>7}{r['grounding']:>8.3f}{r['SR']:>7.2f}"
              f"{r['mean_cascade']:>7.2f}{r['p95']:>6.0f}{r['s_max']:>7.0f}"
              f"{sc:>7}")

    # IDENTITY DISCLOSURE: each claim has exactly one parent, so the exo-rooted
    # subtrees partition all commitments => mean_cascade = N/N_exo = 1/grounding
    # EXACTLY. This is the Tier-2 bookkeeping identity <s>=x/l0 again -- a
    # self-consistency check, NOT a prediction test. P3 must therefore be judged
    # on the cascade TAIL/cutoff (p95, s_max, distribution shape), which is NOT
    # mechanically fixed by the grounding count.
    hi = [r for r in rows if not r["generic"]]
    g = [r["grounding"] for r in hi]
    gall = [r["grounding"] for r in rows]
    ident_err = max(abs(r["mean_cascade"] - 1.0 / r["grounding"])
                    for r in rows if r["grounding"] > 0)
    print(f"\nIdentity check: max|<s> - 1/l0_hat| = {ident_err:.3g} "
          f"(<s>=1/l0_hat by construction; consistency, not a test)")
    P3 = {}
    for key in ("p95", "s_max"):
        P3[key] = _spearman(g, [r[key] for r in hi])
        P3["all7_" + key] = _spearman(gall, [r[key] for r in rows])
    print(f"P3 -- the NON-trivial test (cutoff/tail ordering, expect NEGATIVE):")
    print(f"   high-confidence (n={len(hi)}): grounding vs p95   rho="
          f"{P3['p95']:+.3f}   vs s_max rho={P3['s_max']:+.3f}")
    print(f"   all 7 frameworks          : grounding vs p95   rho="
          f"{P3['all7_p95']:+.3f}   vs s_max rho={P3['all7_s_max']:+.3f}")

    # P4: misalignment (cat 2) traces -> larger cascades, within framework.
    print("\nP4 (cat-2 misalignment vs not, mean max-cascade per trace, "
          "within framework):")
    p4 = []
    for f in files:
        fw = os.path.basename(f)[len("tier1_"):-len(".json")]
        traces = json.load(open(f))
        cat2, other = [], []
        for t in traces:
            cs = t["commitments"]
            if not cs:
                continue
            _, kids = _children_index([dict(c, parents=c.get("parents", []))
                                       for c in cs])
            exo = [c["id"] for c in cs if not c.get("parents")]
            mx = max([_subtree_size(r, kids) for r in exo], default=1)
            (cat2 if t["mast"].get("2") else other).append(mx)
        if cat2 and other:
            m2, mo = float(np.mean(cat2)), float(np.mean(other))
            p4.append((fw, m2, mo, len(cat2), len(other)))
            print(f"   {fw:<11} cat2 <s_max>={m2:6.2f} (n={len(cat2):>3})  "
                  f"other={mo:6.2f} (n={len(other):>3})  "
                  f"ratio={m2/mo if mo else float('nan'):.2f}")
    p4_pass = sum(1 for _, m2, mo, _, _ in p4 if m2 > mo)
    print(f"   -> cat2 larger in {p4_pass}/{len(p4)} frameworks")

    # P5 master curve: Tier-1 per-fw + Tier-2 corner, P(s)*s^{3/2} vs s/s_c
    _plot_master(rows, all_sizes)

    summary = dict(rows=rows, P3=P3,
                   P4=[dict(fw=f, cat2=m2, other=mo, n2=n2, no=no)
                       for f, m2, mo, n2, no in p4],
                   P4_pass=f"{p4_pass}/{len(p4)}")
    with open(os.path.join(OUT, "tier1_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\nwrote", OUT)
    return 0


def _plot_master(rows, all_sizes):
    plt = _mpl()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    # left: P3 -- grounding vs cutoff ordering
    hi = [r for r in rows if not r["generic"]]
    axL.scatter([r["grounding"] for r in hi], [r["p95"] for r in hi],
                c="C0", zorder=3)
    for r in hi:
        axL.annotate(r["framework"], (r["grounding"], r["p95"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    axL.set_xlabel(r"estimated grounding $\hat\ell_0$")
    axL.set_ylabel(r"cascade cutoff (95th pct)")
    axL.set_title("(a) P3: lower grounding $\\Rightarrow$ larger cutoff",
                  fontsize=9)
    # right: P5 master curve (rescaled tails)
    for r in rows:
        if r["generic"]:
            continue
        s = all_sizes[r["framework"]]
        sc = r["p95"] if r["p95"] and r["p95"] > 1 else 1.0
        s = s[s >= 1]
        if s.size < 30:
            continue
        bins = np.unique(np.round(np.logspace(0, np.log10(s.max()+1), 16)))
        h, e = np.histogram(s, bins=bins, density=True)
        ctr = np.sqrt(e[:-1] * e[1:])
        m = h > 0
        axR.loglog((ctr/sc)[m], (h * ctr**1.5)[m], "o-", ms=3,
                   label=r["framework"])
    axR.set_xlabel(r"$s/s_c$")
    axR.set_ylabel(r"$P(s)\,s^{3/2}$")
    axR.set_title("(b) P5: rescaled cascade tails", fontsize=9)
    axR.legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"tier1_master.{ext}"), dpi=160)


if __name__ == "__main__":
    raise SystemExit(main())

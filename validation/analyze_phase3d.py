#!/usr/bin/env python3
"""Phase 3d analysis (VALIDATION_PLAN.md): turn the live Tier-2 production runs
into the paper's empirical-validation figures + the SR_c calibration.

Reads validation/runs/{grid,resetcure,corner}; writes figures + a
phase3d_summary.json under validation/runs/analysis/.

Panels:
  A  P2 recovery (the M2-killer): estimator alpha_hat vs ground-truth alpha and
     grounding vs gf*, with the +-15% pre-registered tolerance band.
  B  reset-cure irreversibility: backlog vs time, load-reduction arm (runs
     away) vs drain arm (recovers) at alpha=1.2 >= 1.
  C  corner cascade-size distribution P(s) with the ML power-law+cutoff fit.
  D  SR_c calibration: measured SR = N_uncert/N_cert per grid point vs escape
     fraction, against the closed-form SR_fold(alpha,theta) = 1/(alpha(1+theta
     x_f) - 1).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "sr_rigor_fix"))
from estimator import estimate                              # noqa: E402
from cascade_sim import fold_point                          # noqa: E402

RUNS = os.path.join(_HERE, "runs")
OUT = os.path.join(RUNS, "analysis")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def sr_fold(alpha, theta):
    """Closed-form critical SR at the fold (finite — the pre-empted divergence)."""
    fp = fold_point(alpha, theta)
    if fp is None:
        return None
    x_f = fp[0]
    denom = alpha * (1.0 + theta * x_f) - 1.0
    return (1.0 / denom) if denom > 0 else float("inf")


# ----------------------------------------------------------------------
# A + D: grid (recovery + SR_c calibration)
# ----------------------------------------------------------------------

def analyze_grid():
    summ = _load(os.path.join(RUNS, "grid", "grid_summary.json"))
    theta = summ["theta"]
    rows = [r for r in summ["rows"] if r.get("n_lucid")]
    # recompute pooled SR per point from the raw per-seed logs (gitignored but
    # present locally); SR isn't in the summary.
    cal = []
    for r in rows:
        a, f = r["alpha"], r["f"]
        logs = sorted(glob.glob(os.path.join(
            RUNS, "grid", f"grid_a{a}_f{f}_s*.json")))
        n_u = n_c = 0
        for p in logs:
            d = estimate(_load(p)["commitments"], fit=False)
            n_u += d["N_uncert"]
            n_c += d["N_cert"]
        SR = (n_u / n_c) if n_c else float("inf")
        esc_frac = r["n_escaped"] / (r["n_escaped"] + r["n_lucid"])
        cal.append(dict(alpha=a, f=f, l0=r["l0"], SR=SR,
                        SR_fold=sr_fold(a, theta), esc_frac=esc_frac,
                        rel_alpha=r["rel_alpha"],
                        rel_grounding=r["rel_grounding"],
                        alpha_hat=r["alpha_hat"], grounding=r["grounding"],
                        gf_true=r["gf_true"]))
    return theta, rows, cal


def fig_recovery(rows, path):
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    a_true = np.array([r["alpha"] for r in rows])
    a_hat = np.array([r["alpha_hat"] for r in rows])
    g_true = np.array([r["gf_true"] for r in rows])
    g_hat = np.array([r["grounding"] for r in rows])
    lo, hi = 0.2, 1.0
    xs = np.linspace(lo, hi, 2)
    ax1.fill_between(xs, 0.85 * xs, 1.15 * xs, color="0.85",
                     label="±15% band")
    ax1.plot(xs, xs, "k--", lw=0.8)
    ax1.scatter(a_true, a_hat, c="C0", zorder=3)
    ax1.set_xlabel(r"ground-truth $\alpha$")
    ax1.set_ylabel(r"estimated $\hat\alpha$")
    ax1.set_title("P2: feedback recovery")
    ax1.legend(fontsize=7)
    gl = np.linspace(0.8, 1.0, 2)
    ax2.fill_between(gl, 0.85 * gl, 1.15 * gl, color="0.85")
    ax2.plot(gl, gl, "k--", lw=0.8)
    ax2.scatter(g_true, g_hat, c="C1", zorder=3)
    ax2.set_xlabel(r"ground-truth grounded fraction $\ell_0/x^*$")
    ax2.set_ylabel(r"estimated grounding $\hat\ell_0$-frac")
    ax2.set_title("P2: grounding recovery")
    fig.suptitle("Tier-2 estimator recovery (live llama3.1:8b, 8 seeds/pt)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path[:-4] + ".pdf")
    return path


def fig_srcal(cal, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    sc = ax.scatter([c["esc_frac"] for c in cal], [c["SR"] for c in cal],
                    c=[c["alpha"] for c in cal], cmap="viridis", zorder=3)
    for c in cal:
        if c["SR_fold"] and np.isfinite(c["SR_fold"]):
            ax.axhline(c["SR_fold"], color="0.8", lw=0.5, zorder=0)
    ax.set_xlabel("escape fraction (collapsed seeds / total)")
    ax.set_ylabel(r"measured $\mathrm{SR}=N_{\rm uncert}/N_{\rm cert}$")
    ax.set_title(r"SR$_c$ calibration: SR rises into the collapse boundary")
    fig.colorbar(sc, label=r"$\alpha$")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path[:-4] + ".pdf")
    return path


# ----------------------------------------------------------------------
# B: reset-cure
# ----------------------------------------------------------------------

def fig_resetcure(path):
    plt = _mpl()
    files = sorted(glob.glob(os.path.join(
        RUNS, "resetcure", "resetcure_a*_seed*.json")))
    if not files:
        return None
    d = _load(files[0])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    styles = {"load_reduction": ("C3", "o-", "load reduction only"),
              "drain": ("C0", "s-", "backlog drain (reset)")}
    t_int = None
    for arm, tr in ((a, d["results"][a]["trace"]) for a in d["results"]):
        col, ls, lab = styles[arm]
        t = [w["t"] for w in tr]
        bl = [w["backlog"] for w in tr]
        ax.plot(t, bl, ls, color=col, ms=3, label=lab)
        for w in tr:
            if w.get("intervention"):
                t_int = w["t"]
    if t_int is not None:
        ax.axvline(t_int, color="0.6", ls=":", lw=1)
        ax.text(t_int, ax.get_ylim()[1] * 0.9, " intervene", fontsize=7,
                color="0.4")
    ax.set_yscale("symlog")
    ax.set_xlabel("time")
    ax.set_ylabel("backlog (pending commitments)")
    ax.set_title(rf"Reset-cure at $\alpha={d['alpha']}\geq1$ "
                 rf"($\ell_0:{d['l0']}\to{d['l0_post']}$, live)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path[:-4] + ".pdf")
    return dict(path=path, cured=d.get("cured"),
                final={a: d["results"][a]["trace"][-1]["backlog"]
                       for a in d["results"]})


# ----------------------------------------------------------------------
# C: corner P(s)
# ----------------------------------------------------------------------

def fig_corner(path):
    plt = _mpl()
    summ_files = sorted(glob.glob(os.path.join(
        RUNS, "corner", "corner_summary_*.json")))
    seed_files = sorted(glob.glob(os.path.join(RUNS, "corner", "corner_s*.json")))
    seed_files = [f for f in seed_files if "summary" not in f]
    if not seed_files:
        return None
    # pooled cascade sizes via the estimator's subtree extraction
    from estimator import _children_index, _subtree_size
    sizes = []
    for p in seed_files:
        commits = _load(p)["commitments"]
        _, kids = _children_index(commits)
        exo = [c for c in commits if not c.get("parents")]
        sizes.extend(_subtree_size(c["id"], kids) for c in exo)
    sizes = np.array(sizes, dtype=float)
    summ = _load(summ_files[0]) if summ_files else {}
    pooled = summ.get("pooled", {})
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    smax = int(sizes.max())
    bins = np.unique(np.round(np.logspace(0, np.log10(smax + 1), 24)))
    hist, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.loglog(centers[hist > 0], hist[hist > 0], "o", ms=4, label="P(s) live")
    if "tau" in pooled:
        tau, sc = pooled["tau"], pooled["s_c"]
        ss = np.logspace(0, np.log10(smax), 100)
        pl = ss ** (-tau) * np.exp(-ss / sc)
        m = (centers > 1) & (hist > 0)
        norm = hist[m][0] / (centers[m][0] ** (-tau)
                             * np.exp(-centers[m][0] / sc))
        ax.loglog(ss, norm * pl, "k-", lw=1,
                  label=rf"ML fit $\tau={tau:.2f}$, $s_c={sc:.0f}$")
    ax.set_xlabel("cascade size $s$")
    ax.set_ylabel("P(s)")
    b_size = pooled.get("b_size")
    btxt = rf" ($b \approx {b_size:.2f}$)" if isinstance(b_size, (int, float)) \
        and np.isfinite(b_size) else ""
    ax.set_title("Corner cascade statistics (live)" + btxt)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path[:-4] + ".pdf")
    return dict(path=path, n_cascades=int(sizes.size),
                mean=float(sizes.mean()), tau=pooled.get("tau"),
                s_c=pooled.get("s_c"), b_size=pooled.get("b_size"))


# ----------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    print("== A/D: grid recovery + SR_c calibration ==")
    theta, rows, cal = analyze_grid()
    out["P2"] = dict(theta=theta, n_points=len(rows),
                     max_abs_rel_alpha=max(abs(r["rel_alpha"]) for r in rows),
                     max_abs_rel_grounding=max(abs(r["rel_grounding"])
                                               for r in rows),
                     pass_15pct=all(abs(r["rel_alpha"]) <= 15
                                    and abs(r["rel_grounding"]) <= 15
                                    for r in rows))
    fig_recovery(rows, os.path.join(OUT, "p2_recovery.png"))
    fig_srcal(cal, os.path.join(OUT, "src_calibration.png"))
    # SR_c estimate: the SR where the escape fraction first exceeds 1/2
    lucid = [c for c in cal if c["esc_frac"] < 0.5]
    coll = [c for c in cal if c["esc_frac"] >= 0.5]
    src = None
    if lucid and coll:
        src = 0.5 * (max(c["SR"] for c in lucid)
                     + min(c["SR"] for c in coll))
    out["SR_c"] = dict(estimate=src,
                       calibration=[{k: c[k] for k in
                                     ("alpha", "f", "SR", "SR_fold",
                                      "esc_frac")} for c in cal])
    print(f"   P2: {out['P2']['n_points']} pts, max|rel alpha|="
          f"{out['P2']['max_abs_rel_alpha']:.1f}%, "
          f"PASS={out['P2']['pass_15pct']}")
    print(f"   SR_c ~ {src}")

    print("== B: reset-cure ==")
    rc = fig_resetcure(os.path.join(OUT, "resetcure.png"))
    out["reset_cure"] = rc
    if rc:
        print(f"   cured={rc['cured']} final={rc['final']}")

    print("== C: corner P(s) ==")
    cc = fig_corner(os.path.join(OUT, "corner_ps.png"))
    out["corner"] = cc
    if cc:
        print(f"   {cc['n_cascades']} cascades, <s>={cc['mean']:.2f}, "
              f"tau={cc['tau']}, s_c={cc['s_c']}")
    else:
        print("   (corner data not present yet)")

    with open(os.path.join(OUT, "phase3d_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

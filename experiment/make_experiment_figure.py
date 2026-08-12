#!/usr/bin/env python3
"""Figure for the real-pipeline experiment (paper Sec. IV E).

(A) measured hysteresis loop vs mean-field lucid branch, spinodals, and the
    collapsed branch x = min(1, l0/(1-alpha));
(B) closed-cascade size distribution vs P(s) ~ s^{-3/2} e^{-s/s_c} with
    s_c = 2/(1-b_eff)^2, b_eff = alpha * P_u measured;
(C) cusp scan: maximal single-step jump of x on an up-ramp vs alpha, with
    the mean-field cusp alpha* = 1/(1+theta).

Reads experiment/results/*.json; writes experiment/figs/pipeline_experiment.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
BLUE, ORANGE, RED, GREY = "#1F4E79", "#E07B00", "#C00000", "0.45"


def lucid_branch(alpha: float, theta: float, xf: float) -> tuple:
    xs = np.linspace(1e-3, xf, 300)
    l0 = xs - alpha * xs * np.exp(-(1 - xs) * theta)
    return l0, xs


def fold_point(alpha: float, theta: float) -> tuple:
    g = lambda x: alpha * (1 + theta * x) * np.exp(-(1 - x) * theta) - 1
    lo, hi = 1e-6, 1 - 1e-9
    for _ in range(80):                      # bisection; g increasing in x
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    xf = 0.5 * (lo + hi)
    l0c = xf - alpha * xf * np.exp(-(1 - xf) * theta)
    return xf, l0c


def panel_a(ax) -> dict:
    d = json.loads((RES / "hysteresis_a0.5_th8.0_q0.0.json").read_text())
    a, th = d["alpha"], d["theta"]
    up = [(r["l0"], r["x"]) for r in d["rows"] if r["direction"] == "up"]
    dn = [(r["l0"], r["x"]) for r in d["rows"] if r["direction"] == "down"]
    xf, l0c = fold_point(a, th)
    l0s, xs = lucid_branch(a, th, xf)
    ax.plot(l0s, xs, "k-", lw=1.6, label="mean-field lucid")
    l0g = np.linspace(0.01, 0.85, 100)
    ax.plot(l0g, np.minimum(1.0, l0g / (1 - a)), "k--", lw=1.2,
            label="collapsed branch (capped)")
    ax.axvline(l0c, color=RED, ls=":", lw=1.4)
    ax.text(l0c - 0.012, 0.44, r"$\ell_0^{c}$ (M/M/1)", rotation=90,
            fontsize=7.0, color=RED, ha="right", va="bottom")
    emp = json.loads((RES / "empirical_fold.json").read_text())
    l0c_emp = emp["empirical"]["fold"]
    ax.axvline(l0c_emp, color="#7A0000", ls="--", lw=1.4)
    ax.text(l0c_emp + 0.008, 0.30, r"$\ell_{0,\rm emp}^{c}$ (empirical service)",
            rotation=90, fontsize=7.0, color="#7A0000", va="bottom")
    ax.axvline(1 - a, color=BLUE, ls=":", lw=1.4)
    ax.text(1 - a + 0.008, 0.60, r"recovery $\ell_0=1-\alpha$", rotation=90,
            fontsize=7.0, color=BLUE, va="bottom")
    ax.plot(*zip(*up), "o-", color=BLUE, ms=4, lw=1.2,
            label="up-ramp (measured)")
    ax.plot(*zip(*dn), "s-", color=ORANGE, ms=4, lw=1.2,
            label="down-ramp (measured)")
    ax.set_xlabel(r"exogenous load $\ell_0$")
    ax.set_ylabel(r"effective load $x$")
    ax.set_title(f"(A) measured hysteresis "
                 rf"($\alpha={a}$, $\theta={th:.0f}$)", fontsize=9.5,
                 loc="left")
    ax.legend(fontsize=6.6, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="none", borderaxespad=0.3)
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.25)
    return dict(l0c=l0c, xf=xf)


def panel_b(ax) -> dict:
    f = sorted(RES.glob("cascades_*.json"))[-1]
    d = json.loads(f.read_text())
    hist = {int(k): v for k, v in d["sizes_hist"].items()}
    sizes = np.array(sorted(hist))
    counts = np.array([hist[s] for s in sizes], float)
    total = counts.sum()
    pdf = counts / total
    # b_eff from the mean cascade size <s> = 1/(1-b)
    mean_s = float((sizes * counts).sum() / total)
    b_eff = 1.0 - 1.0 / mean_s
    s_c = 2.0 / (1.0 - b_eff) ** 2
    ax.loglog(sizes, pdf, "o", color=BLUE, ms=4, alpha=0.75,
              label=f"measured ({int(total)} cascades)")
    ss = np.linspace(1, sizes.max(), 400)
    theory = ss ** -1.5 * np.exp(-ss / s_c)
    theory *= pdf[0] / theory[0]
    ax.loglog(ss, theory, "-", color=RED, lw=1.6,
              label=rf"$s^{{-3/2}}e^{{-s/s_c}}$, $s_c=2/(1-b_{{\rm eff}})^2$")
    ax.set_xlabel("cascade size $s$")
    ax.set_ylabel("$P(s)$")
    ax.set_title(rf"(B) cascades ($b_{{\rm eff}}={b_eff:.2f}$)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    return dict(file=f.name, n=int(total), b_eff=b_eff, s_c=s_c,
                mean_s=mean_s)


def panel_c(ax) -> dict:
    f = sorted(RES.glob("cusp_*.json"))[-1]
    d = json.loads(f.read_text())
    astar = d["alpha_star"]
    alphas = [r["alpha"] for r in d["runs"]]
    jumps = [r["max_jump"] for r in d["runs"]]
    ax.plot(alphas, jumps, "o-", color=BLUE, ms=5, lw=1.4,
            label="max single-step jump in $x$")
    ax.axvline(astar, color=RED, ls=":", lw=1.4)
    ax.text(astar + 0.01, max(jumps) * 0.55,
            rf"$\alpha^*=1/(1+\theta)={astar:.2f}$", rotation=90,
            fontsize=7.5, color=RED)
    ax.set_xlabel(r"ungatedness $\alpha$")
    ax.set_ylabel("largest up-ramp step jump")
    ax.set_title(rf"(C) cusp scan ($\theta={d['theta']}$)", fontsize=9.5,
                 loc="left")
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25)
    return dict(alpha_star=astar, alphas=alphas, jumps=jumps)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    info = dict(A=panel_a(axes[0]), B=panel_b(axes[1]), C=panel_c(axes[2]))
    fig.tight_layout()
    out = FIGS / "pipeline_experiment.pdf"
    fig.savefig(out, bbox_inches="tight")
    (FIGS / "pipeline_experiment_info.json").write_text(
        json.dumps(info, indent=2, default=float))
    print(f"wrote {out}")
    print(json.dumps(info, indent=2, default=float))


if __name__ == "__main__":
    main()

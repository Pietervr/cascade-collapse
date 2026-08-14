#!/usr/bin/env python3
"""Quantitative closed-loop phase-structure figure (paper Fig. 2).

(A) Response curves x(l0) from Eq. (10), x = l0 + alpha*x*exp(-(1-x)*theta),
    at theta=1 for alpha below and above the cusp alpha* = 1/(1+theta):
    the continuous response and the S-curve with fold, unstable branch,
    capped collapsed branch, and the hysteresis jumps.
(B) The (alpha, l0) phase diagram at theta=1, computed (not schematic):
    collapse spinodal l0^c(alpha) [Eq. (11)], recovery spinodal
    l0 = 1-alpha, continuous boundary below the cusp, the cusp point,
    shaded LUCID / BISTABLE / COLLAPSED regions, and the reset-only
    regime alpha >= 1.

Writes figures/phase_diagram.pdf (from the repo root's figures/ convention:
saved next to this script, copied by the paper build).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# CM/usetex: match the manuscript body font (paper figures)
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}"})
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, RED, GREEN = "#1F4E79", "#E07B00", "#C00000", "#2E7D32"
THETA = 1.0


def fold_x(alpha: float, theta: float = THETA) -> float:
    """Solve alpha*(1+theta*x)*exp(-(1-x)*theta) = 1 by bisection."""
    g = lambda x: alpha * (1 + theta * x) * np.exp(-(1 - x) * theta) - 1
    lo, hi = 1e-9, 1 - 1e-9
    if g(hi) < 0:          # no interior fold
        return np.nan
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def l0_of_x(x: np.ndarray, alpha: float, theta: float = THETA) -> np.ndarray:
    return x - alpha * x * np.exp(-(1 - x) * theta)


def panel_a(ax) -> None:
    astar = 1 / (1 + THETA)
    for alpha, color, tag in ((0.30, BLUE, r"$\alpha=0.3<\alpha^*$"),
                              (0.70, RED, r"$\alpha=0.7>\alpha^*$")):
        xf = fold_x(alpha)
        if np.isnan(xf):                      # continuous response
            xs = np.linspace(1e-3, 1.0, 400)
            ax.plot(l0_of_x(xs, alpha), xs, "-", color=color, lw=1.8,
                    label=tag + " (continuous)")
        else:
            l0c = float(l0_of_x(np.array([xf]), alpha)[0])
            rec = 1 - alpha
            # stable lucid branch
            xs = np.linspace(1e-3, xf, 300)
            ax.plot(l0_of_x(xs, alpha), xs, "-", color=color, lw=1.8,
                    label=tag + " (S-curve)")
            # unstable middle branch (parametric continuation to x=1)
            xu = np.linspace(xf, 1.0, 200)
            ax.plot(l0_of_x(xu, alpha), xu, "--", color=color, lw=1.2)
            # collapsed branch: fluid balance x = l0/(1-alpha), l0 >= 1-alpha
            lc = np.linspace(rec, 0.85, 100)
            ax.plot(lc, lc / (1 - alpha), "-", color=color, lw=1.8)
            # hysteresis jumps (down-jump lands on the lucid root at rec)
            xr = 0.5
            for _ in range(200):
                xr = rec + alpha * xr * np.exp(-(1 - xr) * THETA)
            ax.annotate("", xy=(l0c, l0c / (1 - alpha)), xytext=(l0c, xf),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.1, ls=":"))
            ax.annotate("", xy=(rec, xr), xytext=(rec, 1.0),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.1, ls=":"))
            ax.plot([l0c], [xf], "o", color=color, ms=5)
            ax.plot([rec], [1.0], "s", color=color, ms=5)
    ax.set_xlim(0, 0.88)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel(r"grounding $\ell_0$")
    ax.set_ylabel(r"effective load $x$")
    ax.set_title(rf"(A) response of Eq. (10) at $\theta={THETA:.0f}$"
                 rf"  ($\alpha^*={1/(1+THETA):.1f}$)", fontsize=9.5,
                 loc="left")
    ax.legend(fontsize=7.2, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="none")
    ax.grid(alpha=0.25)


def panel_b(ax) -> None:
    astar, l0star = 1 / (1 + THETA), THETA / (1 + THETA)
    al = np.linspace(astar, 1.25, 300)
    l0c = np.array([l0_of_x(np.array([fold_x(a)]), a)[0]
                    if not np.isnan(fold_x(a)) else np.nan for a in al])
    a_cont = np.linspace(0, astar, 100)
    a_rec = np.linspace(astar, 1.0, 100)

    # region shading
    ax.fill_between(np.concatenate([a_cont, al]),
                    np.concatenate([1 - a_cont, l0c]), 1.08,
                    color=RED, alpha=0.10)
    ax.fill_between(a_rec, 1 - a_rec,
                    np.interp(a_rec, al, l0c), color=ORANGE, alpha=0.18)
    ax.fill_between(np.concatenate([a_cont, a_rec]),
                    0, np.concatenate([1 - a_cont, 1 - a_rec]),
                    color=BLUE, alpha=0.10)
    ax.fill_between([1.0, 1.25], 0, np.interp([1.0, 1.25], al, l0c),
                    color=ORANGE, alpha=0.18)

    ax.plot(a_cont, 1 - a_cont, "-.", color="k", lw=1.4,
            label=r"continuous boundary $\ell_0=1-\alpha$")
    ax.plot(al, l0c, "-", color=RED, lw=1.8,
            label=r"collapse spinodal $\ell_0^{c}(\alpha)$, Eq. (11)")
    ax.plot(a_rec, 1 - a_rec, "--", color=BLUE, lw=1.8,
            label=r"recovery spinodal $\ell_0=1-\alpha$")
    ax.plot([astar], [l0star], "ko", ms=6)
    ax.annotate("cusp", xy=(astar, l0star),
                xytext=(astar - 0.20, l0star - 0.14), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8))
    ax.axvline(1.0, color="0.4", lw=0.9, ls=":")

    ax.text(0.16, 0.22, "LUCID", fontsize=9, color=BLUE, weight="bold")
    ax.text(0.30, 0.78, "COLLAPSED", fontsize=9,
            color="#7A0000", weight="bold", zorder=6)
    ax.text(0.71, 0.255, "bistable", fontsize=8, color=ORANGE,
            rotation=-33)
    ax.text(1.02, 0.08, "reset-only\n" r"($\alpha\geq1$)", fontsize=7.5,
            color="0.25")
    ax.text(0.03, 0.885, r"$\alpha\to0$: open-loop $\mathrm{CR}=1$",
            fontsize=7, color="0.35", va="top", rotation=-38)

    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"ungatedness $\alpha$")
    ax.set_ylabel(r"grounding $\ell_0$")
    ax.set_title(rf"(B) phase diagram at $\theta={THETA:.0f}$ (computed)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.0, loc="upper right", frameon=True,
              framealpha=0.9, edgecolor="none")
    ax.grid(alpha=0.25)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout()
    out = HERE / "phase_diagram.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

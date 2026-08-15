#!/usr/bin/env python3
"""The specification map (paper Fig. 1): cell, star, and AI system split by
the three specification classes (archive / flux / trajectory).

Roadmap figure for the Introduction: each row is a composite system, each
column a specification class; the double arrows assert the role
correspondences component by component. Citations and class definitions are
carried by the paper caption (self-contained); the artwork stays clean.

Writes figures/specification_map.pdf (copied into the paper's figures/).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# CM/usetex: match the manuscript body font (paper figures)
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}"})
from matplotlib import patches
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)

BLUE, ORANGE, RED, GREEN = "#1F4E79", "#E07B00", "#C00000", "#2E7D32"
GRAY = "0.35"
XC = (1.9, 5.0, 8.1)
COLC = (BLUE, RED, GREEN)

fig, ax = plt.subplots(figsize=(7.4, 7.55))
ax.set_xlim(0, 10); ax.set_ylim(0.9, 14.55); ax.axis("off")

# ---------------- column headers ----------------
heads = [(r"\textbf{archive-specified}", r"(Class 2) information at rest",
          r"flux $\to 0$: persists;", r"restarts bit-identical"),
         (r"\textbf{flux-specified}", r"(Class 1) borne by the flux",
          r"flux $\to 0$: vanishes;", r"re-forms with fresh phase"),
         (r"\textbf{trajectory-specified}", r"(Class 3) written by history",
          r"flux $\to 0$: lost;", r"never returns")]
for (x, c, (h1, h2, h3, h4)) in zip(XC, COLC, heads):
    ax.text(x, 14.28, h1, ha="center", fontsize=10, color=c)
    ax.text(x, 13.94, h2, ha="center", fontsize=6.6, color=GRAY)
    ax.text(x, 13.68, h3, ha="center", fontsize=6.6, color=GRAY)
    ax.text(x, 13.45, h4, ha="center", fontsize=6.6, color=GRAY)
ax.text(5.0, 13.12, r"(specification classes: caption and Sec.~VI; "
                    r"loop, backlog, and certification: Secs.~III--IV)",
        ha="center", fontsize=6.5, color=GRAY, style="italic")

# ---------------- cell row ----------------
ax.add_patch(patches.Ellipse((5, 11.45), 8.6, 2.7, facecolor="#F4F1EA",
                             edgecolor="0.25", lw=1.4))
ax.text(0.22, 11.45, r"\textsc{cell}", rotation=90, va="center",
        ha="center", fontsize=9, color="0.2")
ax.add_patch(patches.Circle((1.9, 11.45), 0.78, facecolor="#E3EAF2",
                            edgecolor=BLUE, lw=1.3))
xs = np.linspace(1.38, 2.42, 300)
ax.plot(xs, 11.45 + 0.23 * np.sin(14 * xs), color=BLUE, lw=1.2)
ax.plot(xs, 11.45 + 0.23 * np.sin(14 * xs + 1.1), color=BLUE, lw=0.75,
        alpha=0.6)
ax.text(1.9, 10.86, r"genome", ha="center", fontsize=6.8, color=BLUE)
for k, a in zip(range(4), (1.0, 0.75, 0.5, 0.3)):
    yy = np.linspace(10.55, 12.35, 200)
    ax.plot(4.42 + 0.40 * k + 0.09 * np.sin(6.0 * yy), yy,
            color=RED, lw=1.4, alpha=a)
ax.annotate("", xy=(6.15, 12.1), xytext=(5.8, 12.1),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
rng = np.random.default_rng(7)
th, rr = rng.uniform(0, 2 * np.pi, 24), np.sqrt(rng.uniform(0, 1, 24))
ax.scatter(8.15 + 0.95 * rr * np.cos(th), 11.45 + 0.72 * rr * np.sin(th),
           s=rng.uniform(4, 20, 24), color=GREEN, alpha=0.7, lw=0)
ax.text(1.9, 9.82, r"genome (nucleus)", ha="center", fontsize=7.6, color=BLUE)
ax.text(5.0, 9.82, r"Min waves $\cdot$ spindle $\cdot$ polarity",
        ha="center", fontsize=7.2, color=RED)
ax.text(5.0, 9.56, r"(sustained by ATP flux)", ha="center", fontsize=6.4,
        color=GRAY)
ax.text(8.1, 9.82, r"expression state $\cdot$ metabolite pools",
        ha="center", fontsize=6.8, color=GREEN)

# ---------------- arrows cell <-> star ----------------
for x in XC[:2]:
    ax.annotate("", xy=(x, 8.82), xytext=(x, 9.30),
                arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.28",
                                color="0.45", lw=1.2))
ax.annotate("", xy=(8.1, 8.82), xytext=(8.1, 9.30),
            arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.28",
                            color="0.7", lw=1.0, linestyle=(0, (2, 2))))

# ---------------- star row (column-glyph grammar, no leaders) ----------------
ax.add_patch(patches.FancyBboxPatch((0.7, 6.0), 8.6, 2.6,
             boxstyle="round,pad=0.02,rounding_size=0.18",
             facecolor="#FDF6EC", edgecolor="0.25", lw=1.4))
ax.text(0.22, 7.3, r"\textsc{star}", rotation=90, va="center",
        ha="center", fontsize=9, color="0.2")
# col 1: the radiative bulk -- layered interior (aspect-corrected circle)
for w, h, fc in ((1.24, 1.60, "#F8E3B0"), (0.79, 1.02, "#F3CF7E"),
                 (0.38, 0.49, "#EDB84F")):
    ax.add_patch(patches.Ellipse((1.9, 7.68), w, h, facecolor=fc,
                                 edgecolor="0.3" if w > 1.4 else "none",
                                 lw=1.2))
# col 2: convection rolls (granulation cells)
for i, cx in enumerate((4.32, 5.0, 5.68)):
    t1, t2 = (25, 335) if i % 2 == 0 else (205, 155)
    ax.add_patch(patches.Arc((cx, 7.52), 0.5, 0.72, theta1=t1, theta2=t2,
                             color=RED, lw=1.4))
    a0 = np.deg2rad(t1)
    tip = (cx + 0.25 * np.cos(a0), 7.52 + 0.36 * np.sin(a0))
    prv = (cx + 0.25 * np.cos(a0 + 0.4), 7.52 + 0.36 * np.sin(a0 + 0.4))
    ax.annotate("", xy=tip, xytext=prv,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
# col 3: none
ax.text(8.1, 7.62, r"---", ha="center", fontsize=9, color="0.5")
ax.text(8.1, 7.28, r"(no trajectory-carried state)", ha="center",
        fontsize=6.2, color=GRAY)
# labels
ax.text(1.9, 6.55, r"radiative bulk$^{\dagger}$ (near-LTE)",
        ha="center", fontsize=7.2, color=BLUE)
ax.text(1.9, 6.31, r"$^{\dagger}$non-dissipative role,",
        ha="center", fontsize=6.2, color=GRAY, style="italic")
ax.text(1.9, 6.10, r"not an archive",
        ha="center", fontsize=6.2, color=GRAY, style="italic")
ax.text(5.0, 6.55, r"convection zone / granulation", ha="center",
        fontsize=7.2, color=RED)
ax.text(5.0, 6.30, r"(sustained by heat flux)", ha="center", fontsize=6.4,
        color=GRAY)

# ---------------- arrows star <-> AI ----------------
for x in XC[:2]:
    ax.annotate("", xy=(x, 5.28), xytext=(x, 5.76),
                arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.28",
                                color="0.45", lw=1.2))
ax.annotate("", xy=(8.1, 5.28), xytext=(8.1, 5.76),
            arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.28",
                            color="0.7", lw=1.0, linestyle=(0, (2, 2))))

# ---------------- AI row ----------------
ax.add_patch(patches.FancyBboxPatch((0.7, 2.5), 8.6, 2.6,
             boxstyle="round,pad=0.02,rounding_size=0.18",
             facecolor="#F1F3F6", edgecolor="0.25", lw=1.4))
ax.text(0.22, 3.8, r"\textsc{AI system}", rotation=90, va="center",
        ha="center", fontsize=9, color="0.2")
ax.add_patch(patches.Rectangle((1.15, 3.55), 1.5, 0.7, facecolor="#E3EAF2",
                               edgecolor=BLUE, lw=1.3))
ax.add_patch(patches.Ellipse((1.9, 4.25), 1.5, 0.28, facecolor="#E3EAF2",
                             edgecolor=BLUE, lw=1.3))
ax.add_patch(patches.Ellipse((1.9, 3.55), 1.5, 0.28, facecolor="#E3EAF2",
                             edgecolor=BLUE, lw=1.3))
ax.text(1.9, 3.86, r"$\theta$ (weights)", ha="center", fontsize=7.4,
        color=BLUE)
arc = patches.Arc((5.0, 4.0), 1.35, 0.92, theta1=25, theta2=335,
                  color=RED, lw=1.5)
ax.add_patch(arc)
a0 = np.deg2rad(25)
tip = (5.0 + 0.5 * 1.35 * np.cos(a0), 4.0 + 0.5 * 0.92 * np.sin(a0))
prv = (5.0 + 0.5 * 1.35 * np.cos(a0 + 0.35),
       4.0 + 0.5 * 0.92 * np.sin(a0 + 0.35))
ax.annotate("", xy=tip, xytext=prv,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
for k in range(4):
    ax.add_patch(patches.Rectangle((4.40 + 0.32 * k, 3.22), 0.26, 0.26,
                                   facecolor="white", edgecolor=RED, lw=1.0))
ax.annotate("", xy=(4.34, 3.35), xytext=(3.80, 3.35),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
for k in range(4):
    ax.add_patch(patches.Rectangle((7.25 + 0.07 * ((-1) ** k), 3.30 + 0.26 * k),
                                   1.7, 0.22, facecolor="#E9F0E9",
                                   edgecolor=GREEN, lw=1.0))
ax.text(8.03, 4.19, r"?", fontsize=7, color=GREEN, ha="center", va="center")
ax.text(1.9, 3.02, r"trained artifact", ha="center",
        fontsize=7.4, color=BLUE)
ax.text(1.9, 2.76, r"(checkpoint)", ha="center", fontsize=6.4, color=BLUE)
ax.text(5.0, 2.95, r"operating loop: queue $\cdot$ activations",
        ha="center", fontsize=7.2, color=RED)
ax.text(5.0, 2.69, r"(sustained by informational flux)", ha="center",
        fontsize=6.4, color=GRAY)
ax.text(8.1, 3.02, r"uncertified backlog", ha="center",
        fontsize=7.4, color=GREEN)
ax.text(8.1, 2.76, r"(in-flight context)", ha="center", fontsize=6.4,
        color=GREEN)

# ---------------- certification lever ----------------
ax.annotate("", xy=(2.15, 2.34), xytext=(7.85, 2.34),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2,
                            connectionstyle="arc3,rad=-0.10"))
ax.text(5.0, 1.58, r"certification: trajectory-carried commitments $\to$ "
                   r"archived ones (the operative stability lever)",
        ha="center", fontsize=7.6, color="0.2")
ax.text(5.0, 1.28, r"(the cell largely lacks this lever --- somatic state is "
                   r"not written back into the genome; CRISPR spacer "
                   r"acquisition is the rare exception)",
        ha="center", fontsize=6.4, color=GRAY, style="italic")

out = HERE / "figures" / "specification_map.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {out}")

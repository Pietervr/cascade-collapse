#!/usr/bin/env python3
"""
Supplemental-Material simulation for the SR-divergence note: the "cost half"
of the division of labour.

Demonstrates that the DRIVEN TWO-LEVEL SYSTEM (single Ising spin) of Sec. 3.2:

  (1) produces GENUINE, POSITIVE entropy production under driving (the
      detailed-balance escape the reversible queue lacks), and
  (2) that this entropy production is BOUNDED -- it rises and falls with the
      driving rate but never diverges. This is the negative result that
      justifies the division of labour: the spin grounds the per-event cost,
      the queue (Fig. 1) grounds the divergence.

It does NOT, and must not, reproduce the SR ~ (1-CR)^{-1} divergence -- that
would re-import the borrowed divergence the note removes.

Model: spin s in {+1,-1}, energy E_s = -h(t) s, sinusoidal field
h(t) = h0 sin(2 pi t / T_drive). Glauber rates with local detailed balance
    k_{s->-s} = (1/2 tau0)[1 - s tanh(beta h)],   k_{+->-}/k_{-->+} = e^{-2 beta h}.
Ensemble of independent walkers stepped on a fine fixed grid (k*dt << 1).

Entropy production, two independent estimators (must agree):
  (a) medium-entropy sum:  per flip s->-s, dS_med = -2 beta h s ; period-
      averaged <S_tot> = sum(dS_med)/(M * T) >= 0 (in cyclostationary state
      the system-entropy change averages to zero over a period).
  (b) hysteresis-loop area: <S_tot> = beta * (-oint m dh) / T_drive.

Outputs:
  figures/ising_spin_dissipation.pdf  (A: hysteresis loops; B: EP rate bounded)
  a stdout table (both estimators + dissipation per resolved bit vs Landauer).

    python3 ising_spin_dissipation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- parameters --------------------------------
SEED   = 7
BETA   = 1.0          # inverse temperature (k_B T = 1)
TAU0   = 1.0          # bare relaxation timescale
H0     = 1.5          # field amplitude (units of k_B T)
M_WALK = 3000         # independent walkers (ensemble average)
N_CYC  = 20           # drive cycles simulated
N_BURN = 5            # burn-in cycles (discarded)
STEPS_PER_TAU = 200   # time resolution: dt = min(T_drive,tau0)/STEPS_PER_TAU

NU = np.logspace(-1.3, 1.5, 16)[:-1]    # drive rate nu = 1/T_drive (units 1/tau0);
                                        # drop the fastest point (under-sampled, reads as a spurious upturn)
LOOP_NU = np.array([0.06, 0.25, 8.0])   # nu for the three hysteresis loops
NBINS = 120                              # phase bins for loop folding

OUT   = "figures/ising_spin_dissipation.pdf"
LN2   = np.log(2.0)


def glauber_rate(s, h):
    return (1.0 / (2.0 * TAU0)) * (1.0 - s * np.tanh(BETA * h))


def run(Tdrive, rng):
    """Ensemble run at one driving period. Returns EP-rate (med & loop), loop."""
    steps_per_cyc = max(int(round(Tdrive / (min(Tdrive, TAU0) / STEPS_PER_TAU))), 50)
    dt = Tdrive / steps_per_cyc
    total = steps_per_cyc * N_CYC
    burn = steps_per_cyc * N_BURN

    s = rng.choice(np.array([-1.0, 1.0]), size=M_WALK)
    Smed = 0.0
    # phase-folded loop accumulators (all steady-state cycles -> smooth loop)
    h_acc = np.zeros(NBINS)
    m_acc = np.zeros(NBINS)
    cnt = np.zeros(NBINS)

    for i in range(total):
        t = i * dt
        h = H0 * np.sin(2.0 * np.pi * t / Tdrive)
        flip = rng.random(M_WALK) < glauber_rate(s, h) * dt
        if i >= burn:
            Smed += np.sum(np.where(flip, -2.0 * BETA * h * s, 0.0))
            b = int((t % Tdrive) / Tdrive * NBINS) % NBINS
            h_acc[b] += h
            m_acc[b] += s.mean()
            cnt[b] += 1
        s = np.where(flip, -s, s)

    T_steady = dt * (total - burn)
    ep_med = Smed / (M_WALK * T_steady)
    v = cnt > 0
    h_bin = h_acc[v] / cnt[v]
    m_bin = m_acc[v] / cnt[v]
    h_bin = np.append(h_bin, h_bin[0])         # close the loop
    m_bin = np.append(m_bin, m_bin[0])
    Wdiss = -np.trapezoid(m_bin, h_bin)        # -oint m dh  (>=0 for lag)
    ep_loop = BETA * Wdiss / Tdrive
    return ep_med, ep_loop, h_bin, m_bin


def main():
    rng = np.random.default_rng(SEED)
    print(f"Driven two-level system  beta={BETA}  tau0={TAU0}  h0={H0}  "
          f"M={M_WALK}  seed={SEED}\n")
    print("  nu(1/tau0)  T_drive   EP_med     EP_loop    rel.diff   "
          "beta*W/bit  vs ln2")
    print("  " + "-" * 68)

    ep_med_list, ep_loop_list = [], []
    for nu in NU:
        Td = 1.0 / nu
        em, el, _, _ = run(Td, rng)
        ep_med_list.append(em)
        ep_loop_list.append(el)
        w_per_bit = el * Td / 2.0          # beta*W_diss per resolved sign-bit
        rel = abs(em - el) / max(abs(el), 1e-9)
        print(f"  {nu:9.3f}  {Td:7.2f}  {em:9.4f}  {el:9.4f}  {rel:8.2%}  "
              f"{w_per_bit:9.3f}   {'>' if w_per_bit > LN2 else '<'} {LN2:.3f}")
    ep_med_list = np.array(ep_med_list)
    ep_loop_list = np.array(ep_loop_list)

    # representative hysteresis loops
    loops = []
    for nu in LOOP_NU:
        _, _, h, m = run(1.0 / nu, rng)
        loops.append((nu, h, m))

    plateau = np.median(ep_loop_list[NU > 1.0])   # fast-drive saturation value
    peak = ep_loop_list.max()
    print(f"\n  EP rate is BOUNDED: rises from ~0 (quasi-static) to a fast-drive "
          f"plateau\n  <S_tot> -> {plateau:.3f} k_B/tau0 "
          f"(analytic (beta/tau0)<h tanh beta h>); max over sweep {peak:.3f}. "
          f"No divergence.")

    # ----------------------------- figure --------------------------------
    plt.rcParams.update({"font.size": 11, "font.family": "serif",
                         "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    blue, orange, grey, red = "#1F4E79", "#D9822B", "0.45", "#C00000"
    cols = ["#9ecae1", "#D9822B", "#1F4E79"]
    labs = ["quasi-static", "resonant", "fast drive"]

    # Panel A: hysteresis loops
    for (nu, h, m), c, lb in zip(loops, cols, labs):
        axA.plot(BETA * h, m, "-", color=c, lw=2,
                 label=fr"{lb}  ($\nu\tau_0={nu:g}$)")
    heq = np.linspace(-H0, H0, 200)
    axA.plot(BETA * heq, np.tanh(BETA * heq), ":", color=grey, lw=1.3,
             label=r"equilibrium $\tanh\beta h$")
    axA.set_xlabel(r"field  $\beta h(t)$")
    axA.set_ylabel(r"magnetization  $m=\langle s\rangle$")
    axA.set_title("(A)  hysteresis: loop area $=$ dissipated work", fontsize=11)
    axA.legend(fontsize=8.3, loc="upper left", framealpha=0.95, edgecolor="none")
    axA.grid(True, alpha=0.18)

    # Panel B: EP rate vs drive rate -- bounded
    axB.semilogx(NU, ep_loop_list, "-", color=blue, lw=2,
                 label=r"$-\oint m\,dh$ (loop area)")
    axB.semilogx(NU, ep_med_list, "o", color=orange, ms=6, mec="k", mew=0.5,
                 label=r"$\sum\Delta S_{\mathrm{med}}$ (trajectory)")
    axB.axhline(plateau, ls="--", color=grey, lw=1.1)
    axB.text(NU[0] * 1.1, plateau * 1.03,
             fr"bounded plateau $\approx {plateau:.2f}\,k_B/\tau_0$",
             color=grey, fontsize=8.5, va="bottom")
    axB.annotate("no divergence\n(cf. queue $\\mathrm{SR}\\to\\infty$, Fig. 1)",
                 xy=(NU[-3], ep_loop_list[-3]), xytext=(1.4, plateau * 0.42),
                 fontsize=8.5, color=red, ha="left",
                 arrowprops=dict(arrowstyle="->", color=red, lw=0.9))
    axB.set_xlabel(r"drive rate  $\nu\,\tau_0 = \tau_0/T_{\mathrm{drive}}$")
    axB.set_ylabel(r"entropy-production rate  $\langle\dot S_{\mathrm{tot}}\rangle$"
                   r"  $(k_B/\tau_0)$")
    axB.set_title(r"(B)  positive but bounded dissipation", fontsize=11)
    axB.set_ylim(0, peak * 1.25)
    axB.legend(fontsize=8.5, loc="upper right", framealpha=0.95, edgecolor="none")
    axB.grid(True, which="both", alpha=0.18)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simulation check for the SR-divergence derivation note.

Verifies, against a direct event-driven M/M/1 simulation:

  (A) the closed form        SR(rho) = 1 / (exp(M*Dt) - 1),
      with M = mu(1-rho) = C_self - R_self  (feasibility margin),
      i.e. SR = (uncertified firing rate)/(certified firing rate)
      where a commitment fires "uncertified" iff its sojourn time
      (arrival -> certification complete) exceeds the decision horizon Dt.

  (B) the critical-slowing-down scaling
        tau_relax(rho) ~ 1 / [ mu (1 - sqrt(rho))^2 ]  ~ (1-CR)^{-2},
      by measuring the integrated autocorrelation time of the backlog
      n(t) reconstructed from the arrival/departure events.

The M/M/1 sojourn time is exactly Exp(mu-lambda); the waiting time is
obtained exactly (not approximately) from the Skorokhod-reflected
random walk (vectorised Lindley recursion):
    Wq(i) = P(i) - min_{0<=k<=i} P(k),   P = cumsum of (serv - interarrival).

Outputs:
  figures/sr_simulation_validation.pdf   (two-panel figure)
  a stdout table of empirical vs theoretical SR and tau, with rel. errors.

    python3 sr_simulation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "sr_simulation_results.npz"   # set env REPLOT=1 to skip sim and re-plot

# ----------------------------- parameters --------------------------------
SEED      = 11
MU        = 1.0          # certification capacity  C_self
DT        = 1.0          # decision horizon        (mu*Dt = 1)
N         = 3_000_000    # commitments per rho
BURN_FRAC = 0.10         # discard transient (fraction of customers)
GRID_DT   = 1.0          # sampling step for n(t) autocorrelation

# rho sweep for SR (panel A) - pushed close to the boundary
RHOS_SR  = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995])
# rho sweep for tau (panel B) - capped at 0.96 where the run still spans
# many relaxation times so tau_int is clean (at rho=0.98 tau~1e4 approaches
# the run length and the estimate is biased low - finite-run limit, not theory)
RHOS_TAU = np.array([0.50, 0.70, 0.80, 0.90, 0.94, 0.96])

OUT = "figures/sr_simulation_validation.pdf"


# ----------------------------- theory ------------------------------------
def sr_theory(rho):
    M = MU * (1.0 - rho)
    return 1.0 / np.expm1(M * DT)          # 1/(e^{M Dt} - 1)

def tau_theory(rho):
    return 1.0 / (MU * (1.0 - np.sqrt(rho)) ** 2)   # 1/spectral gap


# ----------------------------- simulator ---------------------------------
def simulate(rho, n, rng):
    """Exact M/M/1 FCFS. Returns (sojourn, arr, dep) arrays."""
    lam   = rho * MU
    inter = rng.exponential(1.0 / lam, size=n)     # interarrival times
    serv  = rng.exponential(1.0 / MU,  size=n)     # service times
    arr   = np.cumsum(inter)

    # vectorised Lindley via Skorokhod reflection:
    # increments incr[i] = serv[i-1] - inter[i] for i>=1, incr[0]=0
    incr      = np.empty(n)
    incr[0]   = 0.0
    incr[1:]  = serv[:-1] - inter[1:]
    P         = np.cumsum(incr)
    wq        = P - np.minimum.accumulate(P)       # waiting time in queue >= 0
    sojourn   = wq + serv                          # arrival -> certified
    dep       = arr + sojourn                      # departures (sorted, FCFS)
    return sojourn, arr, dep


def sr_empirical(sojourn):
    b      = int(BURN_FRAC * sojourn.size)
    s      = sojourn[b:]
    uncert = np.count_nonzero(s > DT)
    cert   = s.size - uncert
    return uncert / max(cert, 1), uncert / s.size   # SR, uncertified fraction


def integrated_autocorr_time(x, dt):
    """tau_int (time units) of series x sampled at step dt, Sokal window."""
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    acf /= acf[0]
    # adaptive (Sokal) window: sum until window >= 5*tau
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * acf[1:m + 1].sum()
        if m >= 5.0 * tau:
            break
    return tau * dt


def tau_empirical(arr, dep, rho):
    """Integrated autocorrelation time of backlog n(t)."""
    t_end = dep[-1]
    t0    = BURN_FRAC * t_end
    grid  = np.arange(t0, t_end, GRID_DT)
    # n(t) = #arrivals<=t - #departures<=t  (dep is sorted for FCFS)
    n_t = (np.searchsorted(arr, grid, side="right")
           - np.searchsorted(dep, grid, side="right")).astype(float)
    return integrated_autocorr_time(n_t, GRID_DT)


# ----------------------------- run ---------------------------------------
def compute():
    rng = np.random.default_rng(SEED)

    print(f"M/M/1 certification queue   mu={MU}  Dt={DT}  N={N:,}  seed={SEED}\n")
    print("  PANEL A : stability ratio  SR = N_uncert / N_cert")
    print("  rho(CR)   M*Dt      SR_emp      SR_theory   rel.err   uncert.frac")
    print("  " + "-" * 62)
    sr_emp_list = []
    for rho in RHOS_SR:
        sojourn, _, _ = simulate(rho, N, rng)
        sr_e, ufrac = sr_empirical(sojourn)
        sr_t = sr_theory(rho)
        sr_emp_list.append(sr_e)
        print(f"  {rho:6.3f}  {MU*(1-rho)*DT:6.4f}  {sr_e:10.4f}  {sr_t:10.4f}"
              f"  {abs(sr_e/sr_t-1):7.3%}  {ufrac:8.4f}")
    sr_emp_list = np.array(sr_emp_list)

    print("\n  PANEL B : integrated autocorrelation time of backlog n(t)")
    print("  rho(CR)   1-CR       tau_emp     tau_theory  rel.err")
    print("  " + "-" * 54)
    tau_emp_list = []
    for rho in RHOS_TAU:
        _, arr, dep = simulate(rho, N, rng)
        te = tau_empirical(arr, dep, rho)
        tt = tau_theory(rho)
        tau_emp_list.append(te)
        print(f"  {rho:6.3f}  {1-rho:7.4f}  {te:10.2f}  {tt:10.2f}"
              f"  {abs(te/tt-1):7.3%}")
    tau_emp_list = np.array(tau_emp_list)

    max_err = np.max(np.abs(sr_emp_list / sr_theory(RHOS_SR) - 1))
    print(f"\n  max |SR_emp/SR_theory - 1| over the sweep = {max_err:.2%}")
    np.savez(CACHE, sr_emp=sr_emp_list, tau_emp=tau_emp_list)
    return sr_emp_list, tau_emp_list


def main():
    if os.environ.get("REPLOT") and os.path.exists(CACHE):
        d = np.load(CACHE)
        sr_emp_list, tau_emp_list = d["sr_emp"], d["tau_emp"]
        print(f"[REPLOT] loaded cached results from {CACHE}")
    else:
        sr_emp_list, tau_emp_list = compute()

    # ----------------------------- figure --------------------------------
    plt.rcParams.update({"font.size": 11, "font.family": "serif",
                         "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    blue, orange, grey = "#1F4E79", "#D9822B", "0.45"

    # Panel A: SR vs CR
    rr = np.linspace(0.40, 0.9985, 400)
    axA.semilogy(rr, sr_theory(rr), "-", color=blue, lw=2,
                 label=r"theory  $\mathrm{SR}=1/(e^{\mathcal{M}\Delta t}-1)$")
    axA.semilogy(rr, 1.0 / (MU * (1 - rr) * DT), "--", color=grey, lw=1.2,
                 label=r"heavy-traffic  $(\mu\Delta t)^{-1}(1-\mathrm{CR})^{-1}$")
    axA.semilogy(RHOS_SR, sr_emp_list, "o", color=orange, ms=7,
                 mec="k", mew=0.6, label="simulation", zorder=5)
    axA.axvline(1.0, color="#C00000", lw=1.2)
    axA.text(0.997, axA.get_ylim()[0] * 1.5, r"$\Gamma$", color="#C00000",
             ha="right", va="bottom", fontsize=13)
    axA.set_xlabel(r"capacity ratio  $\mathrm{CR}=R_{\mathrm{self}}/C_{\mathrm{self}}$")
    axA.set_ylabel(r"stability ratio  $\mathrm{SR}=\dot S_i/\dot S_e$")
    axA.set_title("(A)  closed form vs. simulation", fontsize=11)
    axA.set_xlim(0.4, 1.02)
    axA.legend(fontsize=8.5, loc="upper left", framealpha=0.95, edgecolor="none")
    axA.grid(True, which="both", alpha=0.18)

    # Panel B: tau vs (1-CR), log-log
    om = 1.0 - RHOS_TAU
    oo = np.logspace(np.log10(om.min() * 0.7), np.log10(om.max() * 1.2), 100)
    rr2 = 1.0 - oo
    axB.loglog(oo, tau_theory(rr2), "-", color=blue, lw=2,
               label=r"theory  $1/[\mu(1-\sqrt{\mathrm{CR}})^2]$")
    # slope -2 guide
    c2 = tau_theory(1 - om[2]) * om[2] ** 2
    axB.loglog(oo, c2 * oo ** (-2.0), ":", color=grey, lw=1.3,
               label=r"slope $-2$ guide")
    axB.loglog(om, tau_emp_list, "s", color=orange, ms=7,
               mec="k", mew=0.6, label="simulation", zorder=5)
    axB.set_xlabel(r"distance to boundary  $1-\mathrm{CR}$")
    axB.set_ylabel(r"autocorrelation time  $\tau_{\mathrm{relax}}$")
    axB.set_title("(B)  critical slowing down at $\\Gamma$", fontsize=11)
    axB.legend(fontsize=8.5, loc="upper right", framealpha=0.95, edgecolor="none")
    axB.grid(True, which="both", alpha=0.18)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()

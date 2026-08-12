#!/usr/bin/env python3
"""Closed-loop certification-queue simulator (theory note v3).

Event-driven simulation of the RSC certification queue with feedback.
Exogenous candidates arrive Poisson(lambda0). A single FCFS server certifies
at rate mu (service Exp(mu); equivalently spin-borne service -- a two-state
Glauber spin's first passage, which is exactly exponential). A job whose
sojourn exceeds the deadline dt fires UNCERTIFIED at its deadline and spawns
Poisson(alpha) offspring candidates (the cascade). Jobs remain in the queue
until served either way (certification work is consumed even when the
commitment has already fired), matching the mean-field closure of the note:

    x = l0 + alpha * x * exp(-(1 - x) * theta),
    x = lambda_eff / mu,   l0 = lambda0 / mu,   theta = mu * dt.

NOTE the discovery angle: with feedback, arrivals are NOT Poisson (spawn
bursts correlate with congestion), so the mean-field -- which assumes the
stationary M/M/1 sojourn tail -- is an approximation the simulation tests.

Entropy layer (minimal exact wiring, note v3 Sec. 6, kB = 1):
  certified event   -> sigma_anc = 2*beta_h   (one Glauber flip to alignment)
  uncertified event -> sigma_syn = ln 2       (erasure of the unresolved bit)

Subcommands:
  smoke       quick correctness checks against closed forms
  identity    <s> = x*/l0 bookkeeping identity across parameter points
  phase       (alpha, l0) sweep at fixed theta; collapse map + spinodals
  hysteresis  slow l0 ramp up then down; bistability loop figure
  avalanche   cascade-size distribution; 3/2 line + cutoff
  ews         early-warning scaling (AC1, variance) approaching the fold

All randomness is seeded (--seed). Figures land in ./figures/.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from collections import deque

import numpy as np

LN2 = math.log(2.0)

# ----------------------------------------------------------------------
# Mean-field closed forms (note v3, Sec. 2-3)
# ----------------------------------------------------------------------


def g_map(x: float, l0: float, alpha: float, theta: float) -> float:
    return l0 + alpha * x * math.exp(-(1.0 - x) * theta)


def F_of_x(x: float, alpha: float, theta: float) -> float:
    """F(x) = x - alpha x e^{-(1-x)theta}; steady states solve F(x) = l0."""
    return x * (1.0 - alpha * math.exp(-(1.0 - x) * theta))


def _bisect(f, a: float, b: float, tol: float = 1e-12, it: int = 200) -> float:
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("bisection: no sign change")
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5 * (a + b)


def fold_point(alpha: float, theta: float):
    """Return (x_f, l0_c) of the collapse spinodal, or None if no fold."""
    if alpha * (1.0 + theta) <= 1.0:
        return None

    def h(x):
        return alpha * (1.0 + theta * x) * math.exp(-(1.0 - x) * theta) - 1.0

    if h(0.0) >= 0.0:  # fold pushed to x<=0: lucid branch never stable
        return (0.0, 0.0)
    x_f = _bisect(h, 0.0, 1.0)
    l0_c = theta * x_f * x_f / (1.0 + theta * x_f)
    return (x_f, l0_c)


def cusp(theta: float):
    return 1.0 / (1.0 + theta), theta / (1.0 + theta)


def lucid_x(l0: float, alpha: float, theta: float):
    """Smallest root of F(x)=l0 (the lucid branch), or None past the fold."""
    fp = fold_point(alpha, theta)
    hi = fp[0] if fp is not None else 1.0 - 1e-12
    if hi <= 0.0:
        return None
    if F_of_x(hi, alpha, theta) < l0 - 1e-15:
        return None  # past the fold / boundary: no lucid state
    if l0 <= 0.0:
        return 0.0
    return _bisect(lambda x: F_of_x(x, alpha, theta) - l0, 0.0, hi)


# ----------------------------------------------------------------------
# Event-driven engine
# ----------------------------------------------------------------------

ARR, DLN, SVC = 0, 1, 2  # event kinds


class CascadeSim:
    """Single-server FCFS certification queue with deadline-fired feedback."""

    def __init__(self, mu=1.0, dt=1.0, alpha=0.0, l0=0.5, beta_h=1.0,
                 spawn_lag=0.0, n_servers=1, seed=1,
                 store_avalanches=2_000_000):
        self.mu, self.dt, self.alpha = mu, dt, alpha
        self.l0 = l0
        self.beta_h = beta_h
        # spawn_lag: mean of an exponential delay between an uncertified
        # fire and its offspring's arrival. 0 = worst case (offspring land
        # in the very congestion that killed the parent); large values
        # decorrelate spawns from queue state and approach the mean-field.
        self.spawn_lag = spawn_lag
        # n_servers: N parallel decoders sharing the exogenous stream and
        # the spawn pool (uniform random routing). Aggregate mean-field is
        # identical to N=1 per-server; collective fluctuations in the
        # feedback loop average down ~1/sqrt(N) -- N is the sharpening
        # parameter of the first-order transition (note v5, F5).
        self.n_servers = n_servers
        self.rng = np.random.default_rng(seed)
        self.store_avalanches = store_avalanches
        self.reset()

    # -- state -------------------------------------------------------
    def reset(self):
        self.t = 0.0
        self.heap = []
        self.seq = 0
        self.queues = [deque() for _ in range(self.n_servers)]
        self.serving = [None] * self.n_servers
        self._waiting = 0             # jobs queued, not in service
        self._busy = 0                # servers occupied
        self.jobs = {}                # jid -> [root, t_arr, fired(bool), qi]
        self.next_jid = 0
        self.roots = {}               # rid -> [created, fired, n_uncert]
        self.avalanche_sizes = []
        self.avalanche_uncert = []
        # cumulative counters
        self.n_arr = 0                # all arrivals (exo + spawned)
        self.n_exo = 0
        self.n_cert = 0
        self.n_unc = 0
        self.S_syn = 0.0
        self.S_anc = 0.0
        self._exo_token = 0           # invalidates stale exo-arrival events
        self.samples_t = []
        self.samples_backlog = []
        self.collapsed_at = None

    def _push(self, t, kind, payload):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, payload))

    # -- exogenous arrivals -------------------------------------------
    def _schedule_exo(self):
        if self.l0 <= 0.0:
            return
        lam0 = self.l0 * self.mu * self.n_servers   # aggregate exo stream
        dt = self.rng.exponential(1.0 / lam0)
        self._push(self.t + dt, ARR, ("exo", self._exo_token))

    def waiting(self):
        return self._waiting

    def set_l0(self, l0: float):
        """Change exogenous load (ramps); invalidates the pending exo event."""
        self.l0 = l0
        self._exo_token += 1
        self._schedule_exo()

    # -- job mechanics -------------------------------------------------
    def _new_job(self, root):
        """root=None: exogenous (opens its own avalanche). Otherwise a
        spawned member whose `created` count was already incremented at
        spawn-decision time in _on_deadline (closure-race safety)."""
        jid = self.next_jid
        self.next_jid += 1
        if root is None:               # exogenous: becomes its own root
            root = jid
            self.roots[root] = [1, 0, 0]
            self.n_exo += 1
        qi = (int(self.rng.integers(self.n_servers))
              if self.n_servers > 1 else 0)
        self.jobs[jid] = [root, self.t, False, qi]
        self.n_arr += 1
        self.queues[qi].append(jid)
        self._waiting += 1
        self._push(self.t + self.dt, DLN, jid)
        self._try_start(qi)
        return jid

    def _try_start(self, qi):
        if self.serving[qi] is None and self.queues[qi]:
            jid = self.queues[qi].popleft()
            self._waiting -= 1
            self._busy += 1
            self.serving[qi] = jid
            svc = self.rng.exponential(1.0 / self.mu)
            self._push(self.t + svc, SVC, jid)

    def _root_fire(self, root, uncert):
        rec = self.roots[root]
        rec[1] += 1
        if uncert:
            rec[2] += 1
        if rec[1] == rec[0]:           # avalanche closed
            if len(self.avalanche_sizes) < self.store_avalanches:
                self.avalanche_sizes.append(rec[0])
                self.avalanche_uncert.append(rec[2])
            del self.roots[root]

    # -- event handlers -------------------------------------------------
    def _on_arrival(self, payload):
        if payload[0] == "exo":
            if payload[1] != self._exo_token:
                return                 # stale (rate was changed)
            self._new_job(None)
            self._schedule_exo()
        else:                          # spawned offspring
            self._new_job(payload[1])

    def _on_deadline(self, jid):
        job = self.jobs.get(jid)
        if job is None or job[2]:
            return                     # already certified (and/or cleaned up)
        job[2] = True                  # fires uncertified NOW
        self.n_unc += 1
        self.S_syn += LN2
        root = job[0]
        # Spawn BOOKKEEPING must precede the fire-closure check: increment
        # `created` for all offspring now, then fire, then emit arrivals.
        k = int(self.rng.poisson(self.alpha))
        if k:
            self.roots[root][0] += k
        self._root_fire(root, uncert=True)
        for _ in range(k):
            lag = (self.rng.exponential(self.spawn_lag)
                   if self.spawn_lag > 0.0 else 0.0)
            self._push(self.t + lag, ARR, ("spawn", root))

    def _on_service(self, jid):
        job = self.jobs.pop(jid)
        qi = job[3]
        assert self.serving[qi] == jid
        self.serving[qi] = None
        self._busy -= 1
        if not job[2]:                 # certified before its deadline
            job[2] = True
            self.n_cert += 1
            self.S_anc += 2.0 * self.beta_h
            self._root_fire(job[0], uncert=False)
        # else: wasted certification of an already-fired commitment
        self._try_start(qi)

    # -- main loop -------------------------------------------------------
    def run(self, t_max, sample_dt=5.0, max_events=20_000_000,
            collapse_backlog=None, stop_on_collapse=False):
        if not self.heap:
            self._schedule_exo()
        next_sample = self.t + sample_dt
        t_end = self.t + t_max
        ev = 0
        while self.heap and ev < max_events:
            t, sq, kind, payload = heapq.heappop(self.heap)
            if t > t_end:
                # re-queue for the next run() segment -- discarding here
                # would wedge the server (lost SVC) or kill the exo stream
                heapq.heappush(self.heap, (t, sq, kind, payload))
                self.t = t_end
                break
            self.t = t
            ev += 1
            if kind == ARR:
                self._on_arrival(payload)
            elif kind == DLN:
                self._on_deadline(payload)
            else:
                self._on_service(payload)
            while t >= next_sample:
                self.samples_t.append(next_sample)
                self.samples_backlog.append(self._waiting + self._busy)
                next_sample += sample_dt
            if collapse_backlog is not None and self.collapsed_at is None:
                if self._waiting > collapse_backlog:
                    self.collapsed_at = self.t
                    if stop_on_collapse:
                        break
        return self

    # -- measurement -------------------------------------------------------
    def snapshot(self):
        return dict(t=self.t, n_arr=self.n_arr, n_exo=self.n_exo,
                    n_cert=self.n_cert, n_unc=self.n_unc,
                    S_syn=self.S_syn, S_anc=self.S_anc)


def window_rates(s0, s1, mu):
    """Stats between two snapshots."""
    T = s1["t"] - s0["t"]
    if T <= 0:
        return None
    n_u = s1["n_unc"] - s0["n_unc"]
    n_c = s1["n_cert"] - s0["n_cert"]
    out = dict(
        T=T,
        x=(s1["n_arr"] - s0["n_arr"]) / (mu * T),
        Pu=n_u / max(n_u + n_c, 1),
        SR=(n_u / n_c) if n_c > 0 else float("inf"),
        SR_EP=((s1["S_syn"] - s0["S_syn"]) / (s1["S_anc"] - s0["S_anc"]))
        if s1["S_anc"] > s0["S_anc"] else float("inf"),
        n_exo=s1["n_exo"] - s0["n_exo"],
        n_arr=s1["n_arr"] - s0["n_arr"],
    )
    return out


# ----------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = os.path.join(os.path.dirname(__file__), "..", "styles",
                         "pre.mplstyle")
    if os.path.exists(style):
        try:
            plt.style.use(style)
        except Exception:
            pass
    return plt


def _figdir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------

def cmd_smoke(args):
    """Validate against closed forms where they are exact or near-exact."""
    ok = True

    # (1) alpha = 0: exact M/M/1. P_u = e^{-(1-x)theta}, x = l0.
    sim = CascadeSim(alpha=0.0, l0=0.5, dt=1.0, seed=args.seed)
    sim.run(t_max=5_000)               # warmup
    a = sim.snapshot()
    sim.run(t_max=45_000)
    b = sim.snapshot()
    st = window_rates(a, b, 1.0)
    pu_pred = math.exp(-(1.0 - 0.5) * 1.0)
    print(f"[alpha=0] x = {st['x']:.4f} (pred 0.5)   "
          f"Pu = {st['Pu']:.4f} (pred {pu_pred:.4f})")
    ok &= abs(st["x"] - 0.5) < 0.02 and abs(st["Pu"] - pu_pred) < 0.02

    # (2) closed loop, lucid. The bookkeeping identity <s> = x/l0 is EXACT
    # and tested tightly. x_sim vs mean-field is NOT exact: zero-lag spawns
    # land in the congestion that killed their parent, so the simulated
    # feedback is stronger than mean-field (x_sim >= x_mf expected).
    alpha, l0, theta = 0.6, 0.2, 1.0
    xs = lucid_x(l0, alpha, theta)
    sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, seed=args.seed + 1)
    sim.run(t_max=5_000)
    a = sim.snapshot()
    sim.run(t_max=95_000)
    b = sim.snapshot()
    st = window_rates(a, b, 1.0)
    s_mean = st["n_arr"] / max(st["n_exo"], 1)
    print(f"[closed]  x = {st['x']:.4f} (mean-field {xs:.4f}; "
          f"enhancement expected)   <s> = {s_mean:.3f} "
          f"(identity x/l0 = {st['x']/l0:.3f})")
    ok &= abs(s_mean - st["x"] / l0) < 0.05      # exact identity, tight
    ok &= st["x"] >= xs - 0.02                   # enhancement direction

    # (2b) decorrelation check: with spawn lag >> queue correlation time,
    # the mean-field becomes accurate.
    sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, spawn_lag=50.0,
                     seed=args.seed + 2)
    sim.run(t_max=5_000)
    a = sim.snapshot()
    sim.run(t_max=95_000)
    b = sim.snapshot()
    st2 = window_rates(a, b, 1.0)
    print(f"[lagged]  x = {st2['x']:.4f} (mean-field {xs:.4f}; "
          "should now agree)")
    ok &= abs(st2["x"] - xs) < 0.03

    # (3) fold + cusp closed forms at theta = 1.
    a_star, l_star = cusp(1.0)
    fp = fold_point(1.0, 1.0)
    print(f"[forms]   cusp = ({a_star:.3f}, {l_star:.3f})  "
          f"fold(alpha=1): x_f = {fp[0]:.4f}, l0_c = {fp[1]:.4f} "
          "(note: 0.557, 0.199)")
    ok &= abs(fp[0] - 0.557) < 0.01 and abs(fp[1] - 0.199) < 0.005

    # (4) entropy layer: SR_EP = (ln2 / 2 beta_h) * SR.
    pref = LN2 / 2.0
    ratio = st["SR_EP"] / st["SR"] if math.isfinite(st["SR"]) else float("nan")
    print(f"[entropy] SR_EP/SR = {ratio:.4f} (pred {pref:.4f})")
    ok &= abs(ratio - pref) < 1e-9

    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_identity(args):
    """<s> = x*/l0 across a spread of lucid-branch points."""
    theta = args.theta
    pts = [(0.3, 0.3), (0.6, 0.2), (0.8, 0.15), (0.5, 0.4), (0.9, 0.05)]
    print(f"theta={theta}:  alpha   l0     x_sim   x_mf    <s>_sim  x_sim/l0")
    for alpha, l0 in pts:
        xs = lucid_x(l0, alpha, theta)
        if xs is None:
            print(f"        {alpha:.2f}  {l0:.3f}  (past fold -- skipped)")
            continue
        sim = CascadeSim(alpha=alpha, l0=l0, dt=theta / 1.0, seed=args.seed)
        sim.run(t_max=4_000)
        a = sim.snapshot()
        sim.run(t_max=args.t_max)
        b = sim.snapshot()
        st = window_rates(a, b, 1.0)
        s_mean = st["n_arr"] / max(st["n_exo"], 1)
        print(f"        {alpha:.2f}  {l0:.3f}  {st['x']:.4f}  {xs:.4f}"
              f"  {s_mean:7.3f}  {st['x']/l0:7.3f}")
    return 0


def cmd_hysteresis(args):
    """Slow l0 ramp up then down at fixed alpha; the bistability loop."""
    plt = _mpl()
    theta, alpha = args.theta, args.alpha
    fp = fold_point(alpha, theta)
    l0_rec = 1.0 - alpha
    print(f"mean-field: collapse spinodal l0_c = "
          f"{fp[1]:.4f} (x_f = {fp[0]:.4f}); recovery l0 = {l0_rec:.4f}")

    grid_up = np.arange(args.l0_min, args.l0_max + 1e-9, args.l0_step)
    grid = np.concatenate([grid_up, grid_up[::-1]])
    sim = CascadeSim(alpha=alpha, l0=grid[0], dt=theta, seed=args.seed)
    xs, backlogs, srs = [], [], []
    for l0 in grid:
        sim.set_l0(float(l0))
        a = sim.snapshot()
        sim.run(t_max=args.dwell, max_events=4_000_000)
        b = sim.snapshot()
        st = window_rates(a, b, 1.0)
        xs.append(st["x"])
        srs.append(st["SR"])
        backlogs.append(sim.waiting())
        print(f"  l0={l0:.3f}  x={st['x']:.3f}  Pu={st['Pu']:.3f}  "
              f"backlog={sim.waiting()}")

    n = len(grid_up)
    fig, ax = plt.subplots(figsize=(4.9, 3.55))
    ax.plot(grid[:n], xs[:n], "o-", ms=3, label="ramp up")
    ax.plot(grid[n:], xs[n:], "s-", ms=3, label="ramp down")
    # mean-field lucid branch
    ll = np.linspace(1e-3, fp[1], 200)
    ax.plot(ll, [lucid_x(v, alpha, theta) for v in ll], "k-", lw=1,
            label="mean-field lucid")
    coll = np.linspace(l0_rec, max(grid), 100)
    ax.plot(coll, coll / (1.0 - alpha), "k--", lw=1,
            label=r"collapsed $x=\ell_0/(1{-}\alpha)$")
    ax.axvline(fp[1], color="r", ls=":", lw=1)
    ax.axvline(l0_rec, color="b", ls=":", lw=1)
    ax.set_xlabel(r"$\ell_0$")
    ax.set_ylabel(r"$x=\lambda_{\rm eff}/\mu$")
    ax.set_title(rf"(B) hysteresis ($\alpha={alpha}$, $\theta={theta:g}$)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none")
    out = os.path.join(_figdir(), "hysteresis.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_avalanche(args):
    """Cascade-size distribution near criticality (b -> 1)."""
    plt = _mpl()
    theta, alpha, l0 = args.theta, args.alpha, args.l0
    xs = lucid_x(l0, alpha, theta)
    if xs is None:
        print("point is past the fold; choose smaller l0/alpha")
        return 1
    b = 1.0 - l0 / xs
    s_c = 2.0 / (1.0 - b) ** 2
    print(f"mean-field: x* = {xs:.4f}, b = {b:.4f}, <s> = {xs/l0:.2f}, "
          f"s_c ~ {s_c:.0f}")
    sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, seed=args.seed)
    sim.run(t_max=args.t_max, max_events=30_000_000,
            collapse_backlog=3000)
    sizes = np.array(sim.avalanche_sizes, dtype=float)
    n_open = len(sim.roots)
    print(f"collected {sizes.size} closed avalanches "
          f"({n_open} still open / censored); <s>_sim = {sizes.mean():.2f}")
    if sim.collapsed_at is not None:
        print(f"WARNING: system collapsed at t = {sim.collapsed_at:.0f} -- "
              "for alpha >= 1 avalanches never close in collapse (Prop. 2);"
              " choose a point deeper in the lucid region")
    if sizes.size < 1000:
        print("WARNING: thin statistics; increase --t-max or move the point")
    # log-binned pdf
    edges = np.unique(np.round(np.logspace(0, math.log10(max(sizes.max(), 10)),
                                           40)).astype(int))
    hist, _ = np.histogram(sizes, bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = hist / (widths * sizes.size)
    m = pdf > 0
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.loglog(centers[m], pdf[m], "o", ms=4, label="simulation")
    ref = pdf[m][0] * (centers[m] / centers[m][0]) ** (-1.5)
    ax.loglog(centers[m], ref, "k--", lw=1, label=r"$s^{-3/2}$")
    ax.axvline(s_c, color="r", ls=":", lw=1, label=r"$s_c=2/(1{-}b)^2$")
    ax.set_xlabel("avalanche size $s$")
    ax.set_ylabel("$P(s)$")
    ax.set_title(rf"$\alpha={alpha}$, $\ell_0={l0}$, $\theta={theta}$"
                 rf"  ($b={b:.3f}$)")
    ax.legend(fontsize=7)
    out = os.path.join(_figdir(), "avalanche.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_ews(args):
    """Early-warning scaling approaching the fold from the lucid side."""
    plt = _mpl()
    theta, alpha = args.theta, args.alpha
    fp = fold_point(alpha, theta)
    l0c = fp[1]
    eps_list = np.array([0.30, 0.20, 0.12, 0.08, 0.05, 0.03])
    ac1, var, eps_kept = [], [], []
    for eps in eps_list:
        l0 = l0c * (1.0 - eps)
        sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, seed=args.seed)
        sim.run(t_max=args.t_max, sample_dt=args.sample_dt,
                collapse_backlog=3000, stop_on_collapse=True)
        if sim.collapsed_at is not None:
            print(f"  eps={eps:.2f}: collapsed at t={sim.collapsed_at:.0f}"
                  " (fluctuation past separatrix) -- skipped")
            continue
        s = np.array(sim.samples_backlog[len(sim.samples_backlog) // 4:],
                     dtype=float)
        if s.size < 100:
            continue
        s = s - s.mean()
        v = float(np.mean(s * s))
        a1 = float(np.mean(s[:-1] * s[1:]) / v) if v > 0 else float("nan")
        ac1.append(a1)
        var.append(v)
        eps_kept.append(eps)
        print(f"  eps={eps:.2f} (l0={l0:.4f}): AC1={a1:.3f}  var={v:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    axes[0].semilogx(eps_kept, ac1, "o-")
    axes[0].set_xlabel(r"$(\ell_0^c-\ell_0)/\ell_0^c$")
    axes[0].set_ylabel("lag-1 autocorrelation")
    axes[1].loglog(eps_kept, var, "o-")
    ek = np.array(eps_kept)
    axes[1].loglog(ek, var[0] * (ek / ek[0]) ** (-0.5), "k--", lw=1,
                   label=r"$\epsilon^{-1/2}$")
    axes[1].set_xlabel(r"$(\ell_0^c-\ell_0)/\ell_0^c$")
    axes[1].set_ylabel("backlog variance")
    axes[1].legend(fontsize=7)
    fig.suptitle(rf"early warnings: $\alpha={alpha}$, $\theta={theta}$",
                 fontsize=9)
    out = os.path.join(_figdir(), "ews.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_escape(args):
    """Multi-seed metastable escape times across the wedge + theta scaling.

    For each theta and each fractional depth f into the bistable wedge
    (l0 = l0_rec + f*(l0_c - l0_rec)), run several seeds from a lucid start
    and record the time at which the backlog first exceeds the collapse
    threshold. Censored runs (no collapse within t_max) enter at t_max.
    """
    plt = _mpl()
    alpha = args.alpha
    thetas = [float(v) for v in args.thetas.split(",")]
    fracs = [float(v) for v in args.fracs.split(",")]
    curves = {}
    for theta in thetas:
        fp = fold_point(alpha, theta)
        l0_rec = 1.0 - alpha
        meds, lows, highs, cens = [], [], [], []
        for f in fracs:
            l0 = l0_rec + f * (fp[1] - l0_rec)
            times, censored = [], 0
            for s in range(args.seeds):
                sim = CascadeSim(alpha=alpha, l0=l0, dt=theta,
                                 seed=args.seed + 1000 * s)
                sim.run(t_max=args.t_max, collapse_backlog=args.backlog,
                        stop_on_collapse=True, max_events=30_000_000)
                if sim.collapsed_at is None:
                    censored += 1
                    times.append(args.t_max)
                else:
                    times.append(sim.collapsed_at)
            arr = np.array(times)
            meds.append(float(np.median(arr)))
            lows.append(float(np.percentile(arr, 25)))
            highs.append(float(np.percentile(arr, 75)))
            cens.append(censored)
            print(f"theta={theta:g} f={f:.2f} l0={l0:.4f}: "
                  f"median T_esc={meds[-1]:.0f} "
                  f"[{lows[-1]:.0f},{highs[-1]:.0f}] "
                  f"({censored}/{args.seeds} censored)")
        curves[theta] = (fracs, meds, lows, highs, cens)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for theta, (fr, meds, lows, highs, cens) in curves.items():
        ax.errorbar(fr, meds,
                    yerr=[np.array(meds) - np.array(lows),
                          np.array(highs) - np.array(meds)],
                    marker="o", ms=4, capsize=2,
                    label=rf"$\theta={theta:g}$")
    ax.axhline(args.t_max, color="gray", ls=":", lw=1)
    ax.text(0.02, args.t_max * 0.92, "censoring horizon", fontsize=6,
            color="gray", va="top")
    ax.set_yscale("log")
    ax.set_xlabel(r"wedge depth $f$  "
                  r"($\ell_0=\ell_0^{\rm rec}+f(\ell_0^c-\ell_0^{\rm rec})$)")
    ax.set_ylabel(r"median escape time $T_{\rm esc}$")
    ax.set_title(rf"metastable lifetimes, $\alpha={alpha}$"
                 rf" ({args.seeds} seeds)")
    ax.legend(fontsize=7)
    out = os.path.join(_figdir(), "escape.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def _fit_powerlaw_cutoff(sizes, s_min=2):
    """Discrete ML fit of P(s) ~ s^-tau exp(-s/s_c) on s >= s_min.

    Two-stage grid-search ML (coarse global, then fine local) so the
    ~1-sigma profile intervals from the lnL - 0.5 contour are not
    grid-limited. Returns (tau, tau_lo, tau_hi, sc, sc_lo, sc_hi, n_fit).
    """
    data = sizes[sizes >= s_min].astype(float)
    n = data.size
    sum_ln = float(np.sum(np.log(data)))
    sum_s = float(np.sum(data))

    def _scan(taus, scs):
        support = np.arange(s_min, int(max(10 * scs.max(), data.max())) + 1,
                            dtype=float)
        ln_s = np.log(support)
        lnL = np.empty((taus.size, scs.size))
        for i, tau in enumerate(taus):
            for j, sc in enumerate(scs):
                z = np.exp(-tau * ln_s - support / sc)
                lnL[i, j] = (-tau * sum_ln - sum_s / sc
                             - n * math.log(float(z.sum())))
        return lnL

    # stage 1: coarse global
    taus = np.linspace(1.10, 2.00, 46)
    scs = np.logspace(1.0, 3.8, 43)
    k = np.unravel_index(np.argmax(_scan(taus, scs)), (taus.size, scs.size))
    t0, s0 = taus[k[0]], scs[k[1]]
    # stage 2: fine local (step ~0.002 in tau, ~1% in s_c)
    taus = np.linspace(max(1.0, t0 - 0.06), t0 + 0.06, 61)
    scs = s0 * np.logspace(-0.18, 0.18, 37)
    lnL = _scan(taus, scs)
    k = np.unravel_index(np.argmax(lnL), lnL.shape)
    best = lnL[k]
    keep = lnL >= best - 0.5
    tau_band = taus[np.any(keep, axis=1)]
    sc_band = scs[np.any(keep, axis=0)]
    return (taus[k[0]], tau_band.min(), tau_band.max(),
            scs[k[1]], sc_band.min(), sc_band.max(), n)


def cmd_avfit(args):
    """ML exponent fits + cutoff-scaling collapse across parameter points."""
    plt = _mpl()
    pts = []
    for chunk in args.points.split(","):
        a, l0, th = (float(v) for v in chunk.split(":"))
        pts.append((a, l0, th))
    fig, ax = plt.subplots(figsize=(4.9, 3.55))
    rows = []
    for (alpha, l0, theta) in pts:
        sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, seed=args.seed)
        sim.run(t_max=args.t_max, max_events=40_000_000,
                collapse_backlog=3000)
        if sim.collapsed_at is not None:
            print(f"  ({alpha},{l0},{theta}): collapsed -- skipped")
            continue
        sizes = np.array(sim.avalanche_sizes, dtype=float)
        b_eff = 1.0 - 1.0 / sizes.mean()
        sc_pred = 2.0 / (1.0 - b_eff) ** 2
        tau, tlo, thi, sc, slo, shi, nfit = _fit_powerlaw_cutoff(sizes)
        rows.append((alpha, l0, theta, sizes.size, b_eff, tau, tlo, thi,
                     sc, slo, shi, sc_pred))
        print(f"  a={alpha} l0={l0} th={theta}: N={sizes.size}  "
              f"b_eff={b_eff:.3f}  tau={tau:.3f} [{tlo:.3f},{thi:.3f}]  "
              f"s_c={sc:.0f} [{slo:.0f},{shi:.0f}]  "
              f"(pred 2/(1-b)^2={sc_pred:.0f})")
        # collapse panel: P(s) * s^{3/2} vs s / s_c_pred
        edges = np.unique(np.round(np.logspace(
            0, math.log10(max(sizes.max(), 10)), 36)).astype(int))
        hist, _ = np.histogram(sizes, bins=edges)
        widths = np.diff(edges)
        centers = np.sqrt(edges[:-1] * edges[1:])
        pdf = hist / (widths * sizes.size)
        m = pdf > 0
        ax.loglog(centers[m] / sc_pred, pdf[m] * centers[m] ** 1.5, "o",
                  ms=3, label=rf"$\alpha={alpha},\ \ell_0={l0},"
                  rf"\ \theta={theta}$")
    xx = np.logspace(-2.5, 0.8, 60)
    ax.loglog(xx, 0.35 * np.exp(-xx), "k--", lw=1,
              label=r"$\propto e^{-s/s_c}$")
    ax.set_xlabel(r"$s/s_c^{\rm pred}$, $s_c^{\rm pred}=2/(1-b_{\rm eff})^2$")
    ax.set_ylabel(r"$P(s)\,s^{3/2}$")
    ax.set_title("(A) cutoff-scaling collapse", fontsize=9.5, loc="left")
    ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="none")
    out = os.path.join(_figdir(), "avfit.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_entropy(args):
    """Entropy-layer production figures.

    (a) synthetic fraction S_syn/(S_syn+S_anc) and SR_EP vs l0 across the
        wedge (short horizon, lucid start) -- the discontinuity at collapse;
    (b) per-avalanche synthetic-entropy burst distribution (ln2 x uncert
        count), inheriting the avalanche tail.
    """
    plt = _mpl()
    theta, alpha = args.theta, args.alpha
    fp = fold_point(alpha, theta)
    l0_rec = 1.0 - alpha
    l0s = np.linspace(l0_rec - 0.12, fp[1] + 0.08, 21)
    fracs, sreps = [], []
    for l0 in l0s:
        sim = CascadeSim(alpha=alpha, l0=float(l0), dt=theta, seed=args.seed)
        sim.run(t_max=args.t_horizon, max_events=4_000_000)
        tot = sim.S_syn + sim.S_anc
        fr = sim.S_syn / tot if tot > 0 else float("nan")
        fracs.append(fr)
        sreps.append(sim.S_syn / sim.S_anc if sim.S_anc > 0 else float("inf"))
        print(f"  l0={l0:.3f}: syn fraction={fr:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    axes[0].plot(l0s, fracs, "o-", ms=3)
    axes[0].axvline(fp[1], color="r", ls=":", lw=1, label=r"$\ell_0^c$")
    axes[0].axvline(l0_rec, color="b", ls=":", lw=1,
                    label=r"$\ell_0^{\rm rec}$")
    axes[0].set_xlabel(r"$\ell_0$")
    axes[0].set_ylabel(r"$\dot S_{\rm syn}/\dot S_{\rm tot}$")
    axes[0].set_title(rf"synthetic fraction, $T={args.t_horizon:g}$")
    axes[0].legend(fontsize=7)
    # (b) burst distribution from a corner run
    sim = CascadeSim(alpha=args.b_alpha, l0=args.b_l0, dt=args.b_theta,
                     seed=args.seed)
    sim.run(t_max=args.b_t_max, max_events=40_000_000, collapse_backlog=3000)
    u = np.array(sim.avalanche_uncert, dtype=float)
    bursts = LN2 * u[u > 0]
    print(f"  burst run: {bursts.size} avalanches with nonzero "
          f"synthetic entropy")
    edges = np.unique(np.round(np.logspace(
        0, math.log10(max(u.max(), 10)), 32)).astype(int))
    hist, _ = np.histogram(u[u > 0], bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = hist / (widths * max(bursts.size, 1))
    m = pdf > 0
    axes[1].loglog(centers[m] * LN2, pdf[m], "o", ms=3)
    ref = pdf[m][0] * (centers[m] / centers[m][0]) ** (-1.5)
    axes[1].loglog(centers[m] * LN2, ref, "k--", lw=1, label=r"$\propto"
                   r"\ \Sigma^{-3/2}$")
    axes[1].set_xlabel(r"$\Sigma_{\rm syn}=u\,k_B\ln 2$  $(k_B{=}1)$")
    axes[1].set_ylabel(r"$P(\Sigma_{\rm syn})$")
    axes[1].set_title(rf"synthetic-entropy bursts ($\alpha={args.b_alpha}$,"
                      rf" $\ell_0={args.b_l0}$, $\theta={args.b_theta}$)")
    axes[1].legend(fontsize=7)
    out = os.path.join(_figdir(), "entropy.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_nscaling(args):
    """Escape-time growth with system size N (the sharpening axis, F5).

    N parallel decoders share the exogenous stream and the spawn pool at
    fixed per-server load (mid-wedge by default). The aggregate mean-field
    is N-independent; collective fluctuations average ~1/sqrt(N), so the
    metastable lifetime should grow rapidly (exponentially) with N --
    promoting the mean-field bistability to a sharp first-order transition.
    """
    plt = _mpl()
    theta, alpha, f = args.theta, args.alpha, args.frac
    fp = fold_point(alpha, theta)
    l0 = (1.0 - alpha) + f * (fp[1] - (1.0 - alpha))
    ns = [int(v) for v in args.ns.split(",")]
    print(f"alpha={alpha}, theta={theta}, f={f} -> l0={l0:.4f} "
          f"(wedge: {1-alpha:.3f}..{fp[1]:.4f})")
    meds, lows, highs, cens = [], [], [], []
    for n in ns:
        times, censored = [], 0
        for s in range(args.seeds):
            sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, n_servers=n,
                             seed=args.seed + 1000 * s + n)
            sim.run(t_max=args.t_max, collapse_backlog=args.backlog * n,
                    stop_on_collapse=True, max_events=40_000_000)
            if sim.collapsed_at is None:
                censored += 1
                times.append(args.t_max)
            else:
                times.append(sim.collapsed_at)
        arr = np.array(times)
        meds.append(float(np.median(arr)))
        lows.append(float(np.percentile(arr, 25)))
        highs.append(float(np.percentile(arr, 75)))
        cens.append(censored)
        print(f"  N={n:3d}: median T_esc={meds[-1]:.0f} "
              f"[{lows[-1]:.0f},{highs[-1]:.0f}] "
              f"({censored}/{args.seeds} censored)")
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.errorbar(ns, meds,
                yerr=[np.array(meds) - np.array(lows),
                      np.array(highs) - np.array(meds)],
                marker="o", ms=4, capsize=2)
    for n, m, c in zip(ns, meds, cens):
        if c > 0:
            ax.annotate(f"{c}/{args.seeds} cens.", (n, m), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
    ax.axhline(args.t_max, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"system size $N$ (parallel decoders)")
    ax.set_ylabel(r"median escape time $T_{\rm esc}$")
    ax.set_title(rf"$N$-scaling: $\alpha={alpha}$, $\theta={theta}$,"
                 rf" $f={f}$ ({args.seeds} seeds)")
    out = os.path.join(_figdir(), "nscaling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_lagstudy(args):
    """Enhancement vs spawn lag: interpolate worst case -> mean-field.

    The lag is the latency between an uncertified fire and its offspring's
    arrival. lag -> 0 is the worst case (offspring land in the congestion
    that killed the parent); lag >> queue correlation time decorrelates and
    recovers the mean-field. Physically: how quickly unverified outputs are
    re-consumed downstream -- a design lever distinct from gating.
    """
    plt = _mpl()
    theta, alpha, l0 = args.theta, args.alpha, args.l0
    xs = lucid_x(l0, alpha, theta)
    if xs is None:
        print("point past the mean-field fold; pick smaller alpha/l0")
        return 1
    lags = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    out_x, out_pu = [], []
    print(f"alpha={alpha}, l0={l0}, theta={theta}: mean-field x* = {xs:.4f}")
    for lag in lags:
        sim = CascadeSim(alpha=alpha, l0=l0, dt=theta, spawn_lag=lag,
                         seed=args.seed)
        sim.run(t_max=args.warmup)
        a = sim.snapshot()
        sim.run(t_max=args.t_max)
        b = sim.snapshot()
        st = window_rates(a, b, 1.0)
        out_x.append(st["x"])
        out_pu.append(st["Pu"])
        print(f"  lag={lag:6.1f}  x={st['x']:.4f}  Pu={st['Pu']:.4f}  "
              f"excess={(st['x']-xs)/xs*100:+.1f}%")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(lags, out_x, "o-", ms=4, label="simulation")
    ax.axhline(xs, color="k", ls="--", lw=1, label="mean-field $x^*$")
    ax.set_xscale("symlog", linthresh=0.2)
    ax.set_xlabel(r"spawn lag $\tau_{\rm lag}$ (units of $1/\mu$)")
    ax.set_ylabel(r"$x=\lambda_{\rm eff}/\mu$")
    ax.set_title(rf"feedback latency: $\alpha={alpha}$, $\ell_0={l0}$,"
                 rf" $\theta={theta}$")
    ax.legend(fontsize=7)
    out = os.path.join(_figdir(), "lagstudy.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


def cmd_phase(args):
    """Coarse (alpha, l0) collapse map at fixed theta + spinodal overlays."""
    plt = _mpl()
    theta = args.theta
    alphas = np.linspace(args.a_min, args.a_max, args.na)
    l0s = np.linspace(args.l_min, args.l_max, args.nl)
    coll = np.zeros((args.nl, args.na))
    for i, l0 in enumerate(l0s):
        for j, alpha in enumerate(alphas):
            sim = CascadeSim(alpha=float(alpha), l0=float(l0), dt=theta,
                             seed=args.seed)
            sim.run(t_max=args.t_max, collapse_backlog=1500,
                    stop_on_collapse=True, max_events=2_000_000)
            coll[i, j] = 1.0 if sim.collapsed_at is not None else 0.0
        print(f"  l0={l0:.3f}: collapsed at "
              f"{[f'{a:.2f}' for a, c in zip(alphas, coll[i]) if c]}")
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.pcolormesh(alphas, l0s, coll, cmap="Reds", shading="auto", vmax=1.4)
    aa = np.linspace(max(args.a_min, 1.0 / (1.0 + theta) + 1e-3),
                     args.a_max, 120)
    folds = [fold_point(a, theta) for a in aa]
    ax.plot(aa, [f[1] for f in folds], "k--", lw=1.2,
            label=r"collapse spinodal $\ell_0^c$")
    ar = np.linspace(args.a_min, min(args.a_max, 1.0), 60)
    ax.plot(ar, 1.0 - ar, "b:", lw=1.2, label=r"recovery $\ell_0=1-\alpha$")
    a_st, l_st = cusp(theta)
    ax.plot([a_st], [l_st], "k*", ms=10, label="cusp")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\ell_0$")
    ax.set_title(rf"collapse map from lucid start, $\theta={theta}$,"
                 rf" $T={args.t_max:g}$")
    ax.legend(fontsize=7, loc="upper right")
    out = os.path.join(_figdir(), "phase.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(out[:-4] + ".pdf")
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=1)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke")

    q = sub.add_parser("identity")
    q.add_argument("--theta", type=float, default=1.0)
    q.add_argument("--t-max", type=float, default=60_000)

    q = sub.add_parser("hysteresis")
    q.add_argument("--theta", type=float, default=5.0)
    q.add_argument("--alpha", type=float, default=0.5)
    q.add_argument("--l0-min", type=float, default=0.40)
    q.add_argument("--l0-max", type=float, default=0.70)
    q.add_argument("--l0-step", type=float, default=0.02)
    q.add_argument("--dwell", type=float, default=800.0)

    q = sub.add_parser("avalanche")
    q.add_argument("--theta", type=float, default=0.2)
    q.add_argument("--alpha", type=float, default=1.1)
    q.add_argument("--l0", type=float, default=0.010)
    q.add_argument("--t-max", type=float, default=60_000)

    q = sub.add_parser("ews")
    q.add_argument("--theta", type=float, default=5.0)
    q.add_argument("--alpha", type=float, default=0.5)
    q.add_argument("--t-max", type=float, default=30_000)
    q.add_argument("--sample-dt", type=float, default=5.0)

    q = sub.add_parser("lagstudy")
    q.add_argument("--theta", type=float, default=1.0)
    q.add_argument("--alpha", type=float, default=0.6)
    q.add_argument("--l0", type=float, default=0.2)
    q.add_argument("--warmup", type=float, default=5_000)
    q.add_argument("--t-max", type=float, default=150_000)

    q = sub.add_parser("nscaling")
    q.add_argument("--theta", type=float, default=5.0)
    q.add_argument("--alpha", type=float, default=0.5)
    q.add_argument("--frac", type=float, default=0.5)
    q.add_argument("--ns", type=str, default="1,2,4,8,16")
    q.add_argument("--seeds", type=int, default=8)
    q.add_argument("--t-max", type=float, default=200_000)
    q.add_argument("--backlog", type=int, default=1500)

    q = sub.add_parser("escape")
    q.add_argument("--alpha", type=float, default=0.5)
    q.add_argument("--thetas", type=str, default="3,5,7")
    q.add_argument("--fracs", type=str, default="0.25,0.5,0.75")
    q.add_argument("--seeds", type=int, default=8)
    q.add_argument("--t-max", type=float, default=150_000)
    q.add_argument("--backlog", type=int, default=1500)

    q = sub.add_parser("avfit")
    q.add_argument("--points", type=str,
                   default="0.9:0.01:0.2,0.85:0.02:0.2,0.8:0.03:0.2")
    q.add_argument("--t-max", type=float, default=4_000_000)

    q = sub.add_parser("entropy")
    q.add_argument("--theta", type=float, default=5.0)
    q.add_argument("--alpha", type=float, default=0.5)
    q.add_argument("--t-horizon", type=float, default=8_000)
    q.add_argument("--b-alpha", type=float, default=0.9)
    q.add_argument("--b-l0", type=float, default=0.01)
    q.add_argument("--b-theta", type=float, default=0.2)
    q.add_argument("--b-t-max", type=float, default=3_000_000)

    q = sub.add_parser("phase")
    q.add_argument("--theta", type=float, default=1.0)
    q.add_argument("--a-min", type=float, default=0.1)
    q.add_argument("--a-max", type=float, default=1.2)
    q.add_argument("--na", type=int, default=12)
    q.add_argument("--l-min", type=float, default=0.05)
    q.add_argument("--l-max", type=float, default=0.95)
    q.add_argument("--nl", type=int, default=10)
    q.add_argument("--t-max", type=float, default=4_000)

    args = p.parse_args(argv)
    return {"smoke": cmd_smoke, "identity": cmd_identity,
            "hysteresis": cmd_hysteresis, "avalanche": cmd_avalanche,
            "ews": cmd_ews, "phase": cmd_phase, "lagstudy": cmd_lagstudy,
            "escape": cmd_escape, "avfit": cmd_avfit,
            "entropy": cmd_entropy, "nscaling": cmd_nscaling}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

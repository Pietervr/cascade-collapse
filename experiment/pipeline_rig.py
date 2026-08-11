#!/usr/bin/env python3
"""Real-workload closed-loop pipeline experiment.

This is NOT a Monte-Carlo of the mean-field model. The server performs real
computation (dense linear solves, wall-clock timed; empirical, non-exponential
service distribution), arrivals are scheduled in wall-clock time, deadlines
are wall-clock deadlines, and the feedback loop re-injects offspring tasks at
the moment a task fires uncertified. The model's assumptions (Poisson/Exp,
mean-field decorrelation) are therefore NOT built in; the measured fold
location, hysteresis loop, cascade statistics, and cusp boundary are genuine
tests of the theory's universality claims on a real computational system.

Mechanics (mirrors the paper's closed loop, Sec. IV):
  - exogenous tasks arrive as a Poisson stream at rate l0 * mu_hat
    (mu_hat = calibrated real service rate);
  - the single FCFS worker solves a real linear system per task
    (size jittered -> heterogeneous service times);
  - a task whose sojourn (completion - arrival) exceeds Dt = theta/mu_hat is
    classified UNCERTIFIED; it still consumed real capacity (jobs are served
    either way, as in the paper), and at its deadline instant it spawns
    Poisson(alpha_eff) offspring, alpha_eff = (1-q)*alpha, genealogy-tagged;
  - certified tasks spawn nothing.

Observables per dwell window: effective load x = completions/(mu_hat*T),
uncertified fraction P_u, stability ratio SR, backlog. Cascades: sizes of
closed genealogies.

Protocols:
  calibrate            measure mu_hat and the service-time CV
  hysteresis           ramp l0 up then down at fixed (alpha, theta)
  cascades             long run in the subcritical corner; collect sizes
  cusp                 up-ramps across alpha values; classify jump vs smooth

Usage:
  python3 pipeline_rig.py calibrate
  python3 pipeline_rig.py hysteresis --alpha 0.5 --theta 5
  python3 pipeline_rig.py cascades  --alpha 0.9 --l0 0.05 --theta 0.2
  python3 pipeline_rig.py cusp     --theta 1.0
Outputs JSON/CSV into experiment/results/.
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

MATRIX_SIZE = 300          # base size of the real work unit (service ~2-3 ms
                           # so deadlines sit far above OS scheduling jitter)
SIZE_JITTER = 0.25         # +/- fraction -> heterogeneous service times
BACKLOG_CAP = 60000        # sanity cap: declare collapsed and stop the dwell
RNG = np.random.default_rng()


# ----------------------------------------------------------------------
# the real work unit
# ----------------------------------------------------------------------
def do_real_work() -> None:
    n = int(MATRIX_SIZE * (1.0 + SIZE_JITTER * (2 * RNG.random() - 1)))
    a = RNG.standard_normal((n, n))
    a = a @ a.T + n * np.eye(n)          # SPD
    b = RNG.standard_normal(n)
    np.linalg.solve(a, b)


def calibrate(n_samples: int = 3000) -> dict:
    """Measure the real service-time distribution."""
    ts = np.empty(n_samples)
    for i in range(n_samples):
        t0 = time.perf_counter()
        do_real_work()
        ts[i] = time.perf_counter() - t0
    out = dict(mu_hat=1.0 / float(ts.mean()),
               mean_s=float(ts.mean()), cv=float(ts.std() / ts.mean()),
               n=n_samples)
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------
# the closed-loop pipeline
# ----------------------------------------------------------------------
class Pipeline:
    """Single-worker FCFS real-computation pipeline with deadline feedback.

    Deadline violation is detected in REAL TIME (a deadline heap scanned by
    the dispatcher), not at service completion: a task that exceeds its
    deadline fires uncertified at that instant and spawns its offspring
    immediately, while remaining enqueued -- it still consumes real service
    capacity, exactly as in the paper's closed loop.
    """

    def __init__(self, mu_hat: float, alpha: float, theta: float,
                 q: float = 0.0):
        self.mu = mu_hat
        self.alpha_eff = (1.0 - q) * alpha
        self.dt = theta / mu_hat                 # wall-clock deadline
        self.arrivals: list[tuple[float, int, int]] = []   # (t, id, cascade)
        self.queue: deque[tuple[float, int, int]] = deque()
        self.deadlines: list[tuple[float, int]] = []       # (t_deadline, id)
        self.pending: dict[int, int] = {}                  # id -> cascade
        self.fired: set[int] = set()
        self.next_id = 0
        self.open_cascades: dict[int, int] = defaultdict(int)  # id -> open ct
        self.cascade_size: dict[int, int] = defaultdict(int)
        self.closed_sizes: list[int] = []
        self.stats = dict(cert=0, uncert=0, done=0)

    def schedule_exogenous(self, l0: float, t_from: float, t_to: float) -> None:
        rate = l0 * self.mu
        t = t_from
        while True:
            t += RNG.exponential(1.0 / rate)
            if t >= t_to:
                break
            cid = self.next_id
            heapq.heappush(self.arrivals, (t, self.next_id, cid))
            self.open_cascades[cid] += 1
            self.cascade_size[cid] += 1
            self.next_id += 1

    def _ingest_due(self, now: float) -> None:
        while self.arrivals and self.arrivals[0][0] <= now:
            t_arr, tid, cascade = heapq.heappop(self.arrivals)
            self.queue.append((t_arr, tid, cascade))
            self.pending[tid] = cascade
            heapq.heappush(self.deadlines, (t_arr + self.dt, tid))

    def _fire_due(self, now: float) -> None:
        """Fire deadline violations in real time: spawn offspring NOW,
        keep the job enqueued (it still consumes capacity)."""
        while self.deadlines and self.deadlines[0][0] <= now:
            _t, tid = heapq.heappop(self.deadlines)
            if tid not in self.pending or tid in self.fired:
                continue
            self.fired.add(tid)
            cascade = self.pending[tid]
            self.stats["uncert"] += 1
            for _ in range(RNG.poisson(self.alpha_eff)):
                heapq.heappush(self.arrivals, (now, self.next_id, cascade))
                self.open_cascades[cascade] += 1
                self.cascade_size[cascade] += 1
                self.next_id += 1

    def _complete(self, tid: int, cascade: int, t_arr: float,
                  t_done: float) -> None:
        if tid not in self.fired and (t_done - t_arr) <= self.dt:
            self.stats["cert"] += 1
        elif tid not in self.fired:
            # completed past deadline before the heap fired it: fire now
            self.fired.add(tid)
            self.stats["uncert"] += 1
            for _ in range(RNG.poisson(self.alpha_eff)):
                heapq.heappush(self.arrivals, (t_done, self.next_id, cascade))
                self.open_cascades[cascade] += 1
                self.cascade_size[cascade] += 1
                self.next_id += 1
        self.stats["done"] += 1
        self.pending.pop(tid, None)
        self.fired.discard(tid)
        self.open_cascades[cascade] -= 1
        if self.open_cascades[cascade] == 0:
            self.closed_sizes.append(self.cascade_size.pop(cascade))
            del self.open_cascades[cascade]

    def run_window(self, duration: float) -> dict:
        """Serve for `duration` wall seconds; return window observables."""
        t0 = time.perf_counter()
        t_end = t0 + duration
        served = 0
        s0 = dict(self.stats)
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            self._ingest_due(now)
            self._fire_due(now)
            if not self.queue:
                if self.arrivals:
                    gap = self.arrivals[0][0] - now
                    if gap > 0.0015:
                        time.sleep(gap - 0.001)
                    # spin the remainder: sub-ms waits must not add latency
                    while time.perf_counter() < self.arrivals[0][0]:
                        pass
                    continue
                break
            t_arr, tid, cascade = self.queue.popleft()
            do_real_work()
            self._complete(tid, cascade, t_arr, time.perf_counter())
            served += 1
            if len(self.queue) > BACKLOG_CAP:
                break
        elapsed = time.perf_counter() - t0
        cert = self.stats["cert"] - s0["cert"]
        unc = self.stats["uncert"] - s0["uncert"]
        return dict(x=served / (self.mu * elapsed),
                    P_u=unc / max(cert + unc, 1),
                    SR=(unc / cert) if cert else float("inf"),
                    backlog=len(self.queue) + len(self.arrivals),
                    served=served, elapsed=elapsed)


# ----------------------------------------------------------------------
# protocols
# ----------------------------------------------------------------------
def _mu() -> float:
    cal = RESULTS / "calibration.json"
    if not cal.exists():
        return calibrate()["mu_hat"]
    return json.loads(cal.read_text())["mu_hat"]


def hysteresis(alpha: float, theta: float, q: float, dwell: float,
               l0_max: float, steps: int) -> None:
    mu = _mu()
    ups = np.linspace(0.05, l0_max, steps)
    ramp = list(ups) + list(ups[::-1][1:])
    directions = ["up"] * len(ups) + ["down"] * (len(ups) - 1)
    pipe = Pipeline(mu, alpha, theta, q)
    rows, t_clock = [], time.perf_counter()
    for l0, direction in zip(ramp, directions):
        now = time.perf_counter()
        pipe.schedule_exogenous(l0, max(now, t_clock), now + dwell)
        w = pipe.run_window(dwell)
        rows.append(dict(l0=round(float(l0), 4), direction=direction, **{
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in w.items()}))
        t_clock = time.perf_counter()
        print(f"[hys] l0={l0:.3f} {direction}: x={w['x']:.3f} "
              f"P_u={w['P_u']:.3f} backlog={w['backlog']}", flush=True)
    out = RESULTS / f"hysteresis_a{alpha}_th{theta}_q{q}.json"
    out.write_text(json.dumps(dict(alpha=alpha, theta=theta, q=q, mu_hat=mu,
                                   dwell=dwell, rows=rows), indent=2))
    print(f"wrote {out}")


def cascades(alpha: float, l0: float, theta: float, duration: float) -> None:
    mu = _mu()
    pipe = Pipeline(mu, alpha, theta, 0.0)
    now = time.perf_counter()
    pipe.schedule_exogenous(l0, now, now + duration)
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        w = pipe.run_window(min(10.0, t_end - time.perf_counter()))
        print(f"[casc] served={w['served']} closed={len(pipe.closed_sizes)} "
              f"P_u={w['P_u']:.3f}", flush=True)
        if w["served"] == 0:
            break
    sizes = np.array(pipe.closed_sizes)
    out = RESULTS / f"cascades_a{alpha}_l{l0}_th{theta}.json"
    out.write_text(json.dumps(dict(
        alpha=alpha, l0=l0, theta=theta, mu_hat=mu,
        n_cascades=int(len(sizes)),
        sizes_hist={int(s): int(c) for s, c in
                    zip(*np.unique(sizes, return_counts=True))}), indent=2))
    print(f"wrote {out} ({len(sizes)} closed cascades)")


def cusp(theta: float, alphas: list[float], dwell: float, steps: int) -> None:
    mu = _mu()
    results = []
    for alpha in alphas:
        pipe = Pipeline(mu, alpha, theta, 0.0)
        xs, l0s = [], np.linspace(0.05, 0.98, steps)
        for l0 in l0s:
            now = time.perf_counter()
            pipe.schedule_exogenous(l0, now, now + dwell)
            w = pipe.run_window(dwell)
            xs.append(w["x"])
            if w["backlog"] > BACKLOG_CAP:
                break
        jumps = np.diff(xs)
        results.append(dict(alpha=alpha,
                            max_jump=round(float(jumps.max()), 4)
                            if len(jumps) else None,
                            xs=[round(float(v), 4) for v in xs],
                            l0s=[round(float(v), 4) for v in
                                 l0s[:len(xs)]]))
        print(f"[cusp] alpha={alpha}: max step jump "
              f"{results[-1]['max_jump']}", flush=True)
    out = RESULTS / f"cusp_th{theta}.json"
    out.write_text(json.dumps(dict(theta=theta, mu_hat=mu,
                                   alpha_star=1.0 / (1.0 + theta),
                                   runs=results), indent=2))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", choices=["calibrate", "hysteresis",
                                         "cascades", "cusp"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--l0", type=float, default=0.05)
    ap.add_argument("--l0-max", type=float, default=0.75)
    ap.add_argument("--dwell", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--duration", type=float, default=180.0)
    a = ap.parse_args()
    if a.protocol == "calibrate":
        calibrate()
    elif a.protocol == "hysteresis":
        hysteresis(a.alpha, a.theta, a.q, a.dwell, a.l0_max, a.steps)
    elif a.protocol == "cascades":
        cascades(a.alpha, a.l0, a.theta, a.duration)
    elif a.protocol == "cusp":
        cusp(a.theta, [0.2, 0.35, 0.5, 0.65, 0.8], a.dwell, a.steps)


if __name__ == "__main__":
    main()

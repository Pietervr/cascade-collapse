#!/usr/bin/env python3
"""Tier-2 production runs (VALIDATION_PLAN.md Phase 3c).

Drives the instrumented LLM pipeline (tier2_pipeline.py) through the
pre-registered Tier-2 experiments of validation/MEASUREMENT_PROTOCOL.md:

  latency     one live point; per-call latency -> sizes the sweep budget
  grid        P2 ground-truth grid pushed toward the fold boundary
              (l0 = f * l0_c(alpha, theta)); estimator recovery vs truth
  hysteresis  P1: slow l0 ramp up then down at fixed alpha > alpha*;
              window-rate x(l0) loop (collapse at l0_c, recovery at 1-alpha)
  resetcure   P1: alpha >= 1; collapse, then arm A (load reduction only)
              vs arm B (load reduction + backlog drain); backlog + SR traces
  corner      cascade-size statistics near b -> 1 (P(s) + ML tau fit)

All runs seeded; every run writes its full frozen-schema event log plus a
summary JSON under validation/runs/. Mean-field reference values come from
sr_rigor_fix/cascade_sim.py (fold_point, lucid_x, cusp).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "sr_rigor_fix"))

from llm_backend import get_backend                      # noqa: E402
from tier2_pipeline import CascadePipeline               # noqa: E402
from estimator import estimate                           # noqa: E402
from cascade_sim import fold_point, lucid_x, cusp        # noqa: E402

RUNS = os.path.join(_HERE, "runs")
CACHE = os.path.join(_HERE, ".llm_cache")


def _outdir(tag):
    d = os.path.join(RUNS, tag)
    os.makedirs(d, exist_ok=True)
    return d


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _pipe(args, **over):
    be = get_backend(args.backend, cache_dir=CACHE)
    kw = dict(alpha=args.alpha, l0=args.l0, theta=args.theta,
              verifier_coverage=1.0, seed=args.seed,
              max_commitments=getattr(args, "max_commitments", 400))
    kw.update(over)
    return CascadePipeline(be, **kw)


def _window(pipe, dwell):
    """Advance one dwell; return window rates (cascade_sim semantics)."""
    a = pipe.snapshot()
    pipe.run_until(pipe.t + dwell)
    b = pipe.snapshot()
    T = b["t"] - a["t"]
    n_u = b["n_unc"] - a["n_unc"]
    n_c = b["n_cert"] - a["n_cert"]
    return dict(T=T,
                x=(b["n_arr"] - a["n_arr"]) / (pipe.mu * T) if T > 0
                else float("nan"),
                Pu=n_u / max(n_u + n_c, 1),
                SR=(n_u / n_c) if n_c > 0 else float("inf"),
                n_arr=b["n_arr"] - a["n_arr"], backlog=b["backlog"],
                n_calls=b["n_calls"] - a["n_calls"])


# ----------------------------------------------------------------------
# latency: one live point -> sizes everything else
# ----------------------------------------------------------------------

def cmd_latency(args):
    out = _outdir("latency")
    t0 = time.monotonic()
    pipe = _pipe(args)
    log = pipe.run()
    wall = time.monotonic() - t0
    n = len(log["commitments"])
    calls = log["n_llm_calls"]
    per = wall / max(calls, 1)
    d = estimate(log["commitments"], fit=False)
    print(f"point: alpha={args.alpha} l0={args.l0} theta={args.theta} "
          f"backend={args.backend}")
    print(f"  {n} commitments, {calls} LLM calls, wall={wall:.1f}s "
          f"-> {per:.2f}s/call")
    print(f"  estimator: alpha_hat={d['alpha_hat']:.3f} "
          f"grounding={d['grounding']:.3f} consistency={d['consistency']:.3f}")
    for label, ncalls in [("grid 8pts x 6 seeds x 300", 8 * 6 * 600),
                          ("hysteresis 24 pts, dwell 60", 24 * 60 * 2),
                          ("corner 2000 commitments", 4000)]:
        print(f"  projection [{label}]: ~{ncalls} calls "
              f"~= {ncalls * per / 3600:.1f} h")
    _dump(os.path.join(out, f"latency_seed{args.seed}.json"),
          dict(log=log, wall_s=wall, per_call_s=per, estimator=d))
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------
# grid: P2 production -- recovery near the boundary
# ----------------------------------------------------------------------

def _alpha_hat_censored(log, dt):
    """Mean out-degree of uncertified commitments, restricted to those whose
    deadline fired before the commitment cap first censored a spawn (and
    before t_max, automatically: unfired nodes are excluded). Out-degree is
    Poisson(alpha) on BOTH branches, so this pools escaped and lucid seeds
    without survivor conditioning."""
    cap_t = log.get("cap_hit_t")
    kids = {}
    for c in log["commitments"]:
        for p in c.get("parents", []):
            kids[p] = kids.get(p, 0) + 1
    sample = [kids.get(c["id"], 0) for c in log["commitments"]
              if c.get("uncertified_fired")
              and (cap_t is None or c["t_created"] + dt < cap_t)]
    return (float(np.mean(sample)), len(sample)) if sample \
        else (float("nan"), 0)


def cmd_grid(args):
    out = _outdir("grid")
    alphas = [float(v) for v in args.alphas.split(",")]
    fracs = [float(v) for v in args.fracs.split(",")]
    theta = args.theta
    rows = []
    print(f"theta={theta}  cusp alpha*={cusp(theta)[0]:.3f}  "
          f"backend={args.backend}  seeds={args.n_seeds}  "
          f"target_n={args.target_n}")
    print(f"{'alpha':>6} {'f':>5} {'l0':>6} | {'a_hat':>7} {'rel%':>6} | "
          f"{'grnd':>6} {'gf*':>6} {'rel%':>6} | {'cons':>6} {'N':>6}")
    for alpha in alphas:
        fp = fold_point(alpha, theta)
        if fp is None or fp[1] <= 0:
            print(f"{alpha:6.2f}  -- no fold; skipped")
            continue
        for f in fracs:
            l0 = f * fp[1]
            xs = lucid_x(l0, alpha, theta)
            gf_true = l0 / xs if xs else float("nan")   # grounded fraction
            # time-based stop (E[N] = x* mu t): a binding commitment cap
            # correlates with spawn bursts and biases grounding low
            t_max = args.target_n / max(xs, 1e-6)
            ahats, grnds, cons, Ns, esc = [], [], [], 0, 0
            for s in range(args.n_seeds):
                pipe = _pipe(args, alpha=alpha, l0=l0,
                             seed=args.seed + 100 * s,
                             max_commitments=int(2.5 * args.target_n),
                             t_max=t_max)
                log = pipe.run()
                # escape flag (pre-registered, set BEFORE live runs): a
                # lucid-branch run keeps an O(1) backlog; a seed that
                # escaped to the collapsed branch (P1 metastability) has a
                # linearly growing one. Escaped seeds are reported but
                # excluded from the lucid-branch ground-truth comparison.
                escaped = pipe.backlog() > 15
                log["escaped"] = bool(escaped)
                _dump(os.path.join(
                    out, f"grid_a{alpha}_f{f}_s{args.seed + 100 * s}.json"),
                    log)
                # alpha_hat: ALL seeds (out-degree is branch-independent;
                # conditioning on lucid survival biases it low at high alpha),
                # censoring-filtered against the commitment cap
                ah_s, n_s = _alpha_hat_censored(log, pipe.dt)
                if n_s:
                    ahats.append((ah_s, n_s))
                if escaped:
                    esc += 1
                    continue
                # grounding: lucid seeds only (gf* = l0/x* is a lucid-branch
                # quantity); full shared estimator for the record
                d = estimate(log["commitments"], fit=False)
                grnds.append((d["grounding"], d["N"]))
                cons.append(d["consistency"])
                Ns += d["N"]
            if not ahats or not grnds:
                print(f"{alpha:6.2f} {f:5.2f} {l0:6.3f} | insufficient lucid "
                      f"seeds ({esc} escaped); point not scored")
                rows.append(dict(alpha=alpha, f=f, l0=l0, gf_true=gf_true,
                                 n_escaped=esc, n_lucid=0))
                continue
            av, aw = np.array([v for v, _ in ahats]), \
                np.array([w for _, w in ahats], dtype=float)
            gv, gw = np.array([v for v, _ in grnds]), \
                np.array([w for _, w in grnds], dtype=float)
            ah = float(np.average(av, weights=aw))      # pooled out-degree
            gr = float(np.average(gv, weights=gw))      # pooled grounding
            ra = 100 * (ah - alpha) / alpha
            rg = 100 * (gr - gf_true) / gf_true
            sd = float(np.std(av) / max(len(av) - 1, 1) ** 0.5)
            rows.append(dict(alpha=alpha, f=f, l0=l0, gf_true=gf_true,
                             alpha_hat=ah, alpha_hat_sem=sd, rel_alpha=ra,
                             grounding=gr, rel_grounding=rg,
                             n_uncert=int(aw.sum()),
                             consistency=float(np.nanmean(cons)), N=Ns,
                             n_escaped=esc, n_lucid=len(grnds)))
            flag = "" if abs(ra) <= 15 and abs(rg) <= 15 else "  <-- >15%"
            print(f"{alpha:6.2f} {f:5.2f} {l0:6.3f} | {ah:7.3f} {ra:+6.1f} | "
                  f"{gr:6.3f} {gf_true:6.3f} {rg:+6.1f} | "
                  f"{float(np.nanmean(cons)):6.3f} {Ns:6d} "
                  f"esc={esc}/{args.n_seeds}{flag}")
    _dump(os.path.join(out, "grid_summary.json"),
          dict(theta=theta, backend=args.backend, seeds=args.n_seeds,
               rows=rows))
    scored = [r for r in rows if r.get("n_lucid")]
    bad = [r for r in scored
           if abs(r["rel_alpha"]) > 15 or abs(r["rel_grounding"]) > 15]
    print(f"\nP2 production grid: {'PASS' if not bad else 'FAIL'} "
          f"({len(scored) - len(bad)}/{len(scored)} lucid points within 15%)")
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------
# hysteresis: P1 loop
# ----------------------------------------------------------------------

def cmd_hysteresis(args):
    out = _outdir("hysteresis")
    alpha, theta = args.alpha, args.theta
    fp = fold_point(alpha, theta)
    l0_rec = 1.0 - alpha
    l0_max = args.l0_max if args.l0_max > 0 else fp[1] + 0.05
    print(f"alpha={alpha} theta={theta}: l0_c={fp[1]:.4f} "
          f"l0_rec={l0_rec:.4f}  ramp {args.l0_min}->{l0_max} "
          f"step {args.l0_step} dwell {args.dwell}  backend={args.backend}")
    grid_up = np.arange(args.l0_min, l0_max + 1e-9, args.l0_step)
    grid = np.concatenate([grid_up, grid_up[::-1]])
    pipe = _pipe(args, alpha=alpha, l0=float(grid[0]),
                 max_commitments=args.max_commitments)
    pipe.start()
    trace = []
    for i, l0 in enumerate(grid):
        pipe.set_l0(float(l0))
        w = _window(pipe, args.dwell)
        w.update(l0=float(l0), leg="up" if i < len(grid_up) else "down")
        trace.append(w)
        print(f"  [{w['leg']:>4}] l0={l0:.3f}  x={w['x']:.3f}  "
              f"SR={w['SR'] if w['SR'] != float('inf') else 'inf'}  "
              f"backlog={w['backlog']}  ({w['n_calls']} calls)")
        if len(pipe.events) >= args.max_commitments:
            print("  !! global commitment cap hit; ramp truncated")
            break
    res = dict(alpha=alpha, theta=theta, l0_c=fp[1], l0_rec=l0_rec,
               dwell=args.dwell, backend=args.backend, seed=args.seed,
               trace=trace, n_llm_calls=pipe.n_calls)
    _dump(os.path.join(out, f"hysteresis_a{alpha}_seed{args.seed}.json"), res)
    _dump(os.path.join(out, f"hysteresis_a{alpha}_seed{args.seed}_events.json"),
          pipe.event_log())
    _plot_hysteresis(res, out, args)
    print("wrote", out)
    return 0


def _plot_hysteresis(res, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = res["trace"]
    up = [(w["l0"], w["x"]) for w in tr if w["leg"] == "up"]
    dn = [(w["l0"], w["x"]) for w in tr if w["leg"] == "down"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if up:
        ax.plot(*zip(*up), "o-", ms=3, label="ramp up")
    if dn:
        ax.plot(*zip(*dn), "s-", ms=3, label="ramp down")
    ll = np.linspace(1e-3, res["l0_c"], 200)
    ax.plot(ll, [lucid_x(v, res["alpha"], res["theta"]) for v in ll],
            "k-", lw=1, label="mean-field lucid")
    coll = np.linspace(res["l0_rec"], max(w["l0"] for w in tr), 100)
    ax.plot(coll, coll / (1.0 - res["alpha"]), "k--", lw=1,
            label=r"collapsed $x=\ell_0/(1{-}\alpha)$")
    ax.axvline(res["l0_c"], color="r", ls=":", lw=1)
    ax.axvline(res["l0_rec"], color="b", ls=":", lw=1)
    ax.set_xlabel(r"$\ell_0$")
    ax.set_ylabel(r"$x$")
    ax.set_title(rf"LLM pipeline hysteresis: $\alpha={res['alpha']}$, "
                 rf"$\theta={res['theta']}$ ({res['backend']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out, f"hysteresis_a{res['alpha']}_seed{res['seed']}.png")
    fig.savefig(p, dpi=160)
    print("wrote", p)


# ----------------------------------------------------------------------
# resetcure: P1 at alpha >= 1
# ----------------------------------------------------------------------

def cmd_resetcure(args):
    out = _outdir("resetcure")
    alpha, theta = args.alpha, args.theta
    assert alpha >= 1.0, "resetcure probes alpha >= 1"
    print(f"alpha={alpha} theta={theta} l0={args.l0} "
          f"l0_post={args.l0_post}  collapse at backlog>{args.backlog} "
          f" backend={args.backend}")
    results = {}
    for arm in ("load_reduction", "drain"):
        pipe = _pipe(args, alpha=alpha, l0=args.l0,
                     max_commitments=args.max_commitments)
        pipe.start()
        trace, intervened, drained_n = [], False, 0
        for k in range(args.n_windows):
            w = _window(pipe, args.dwell)
            if not intervened and w["backlog"] > args.backlog:
                pipe.set_l0(args.l0_post)        # both arms reduce load
                if arm == "drain":
                    drained_n = pipe.drain()     # only this arm drains
                intervened = True
                w["intervention"] = arm
            w.update(window=k, t=pipe.t)
            trace.append(w)
            print(f"  [{arm:>14}] w{k:02d} t={pipe.t:7.1f} x={w['x']:.3f} "
                  f"backlog={w['backlog']:5d}"
                  + ("  <- INTERVENE" + (f" (drained {drained_n})"
                                         if arm == "drain" else "")
                     if w.get("intervention") else ""))
            if len(pipe.events) >= args.max_commitments:
                print("  !! commitment cap hit")
                break
        results[arm] = dict(trace=trace, n_llm_calls=pipe.n_calls,
                            intervened=intervened, drained=drained_n)
        _dump(os.path.join(out, f"resetcure_{arm}_seed{args.seed}_events.json"),
              pipe.event_log())
    final = {a: results[a]["trace"][-1]["backlog"] for a in results}
    cured = (final["drain"] < args.backlog / 4
             and final["load_reduction"] >= args.backlog)
    print(f"\nfinal backlog: load_reduction={final['load_reduction']} "
          f"drain={final['drain']}  -> reset-cure "
          f"{'CONFIRMED' if cured else 'NOT confirmed'}")
    _dump(os.path.join(out, f"resetcure_a{alpha}_seed{args.seed}.json"),
          dict(alpha=alpha, theta=theta, l0=args.l0, l0_post=args.l0_post,
               backend=args.backend, seed=args.seed, results=results,
               cured=bool(cured)))
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------
# corner: cascade stats near b -> 1
# ----------------------------------------------------------------------

def cmd_corner(args):
    out = _outdir("corner")
    xs = lucid_x(args.l0, args.alpha, args.theta)
    if xs is None:
        print("point is past the fold; choose smaller l0/alpha")
        return 1
    b = 1.0 - args.l0 / xs
    print(f"alpha={args.alpha} l0={args.l0} theta={args.theta}: "
          f"mean-field x*={xs:.4f} b={b:.4f} s_c~{2 / (1 - b) ** 2:.0f} "
          f" backend={args.backend}")
    pooled = []
    for s in range(args.n_seeds):
        pipe = _pipe(args, seed=args.seed + 100 * s)
        log = pipe.run()
        _dump(os.path.join(out, f"corner_s{args.seed + 100 * s}.json"), log)
        d = estimate(log["commitments"], fit=False)
        pooled.append((log, d))
        print(f"  seed {args.seed + 100 * s}: N={d['N']} "
              f"cascades={d['n_cascades']} <s>={d['mean_cascade']:.2f} "
              f"b_size={d['b_size']:.3f}")
    # pooled fit across seeds (re-id to keep roots distinct)
    allc, off = [], 0
    for log, _ in pooled:
        for c in log["commitments"]:
            c2 = dict(c)
            c2["id"] = c["id"] + off
            c2["parents"] = [p + off for p in c["parents"]]
            allc.append(c2)
        off += max(c["id"] for c in log["commitments"]) + 1
    d = estimate(allc, fit=True)
    print(f"pooled: N={d['N']} cascades={d['n_cascades']} "
          f"<s>={d['mean_cascade']:.2f} b_size={d['b_size']:.3f} "
          f"(mean-field b={b:.3f})")
    if "tau" in d:
        print(f"  ML fit: tau={d['tau']:.3f} [{d['tau_lo']:.3f},"
              f"{d['tau_hi']:.3f}]  s_c={d['s_c']:.0f}")
    _dump(os.path.join(out, f"corner_summary_a{args.alpha}_l{args.l0}.json"),
          dict(alpha=args.alpha, l0=args.l0, theta=args.theta,
               backend=args.backend, mean_field=dict(x=xs, b=b),
               pooled=d))
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="ollama:llama3.1:8b")
    p.add_argument("--theta", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("latency")
    q.add_argument("--alpha", type=float, default=0.6)
    q.add_argument("--l0", type=float, default=0.3)
    q.add_argument("--max-commitments", type=int, default=100)

    q = sub.add_parser("grid")
    q.add_argument("--alphas", default="0.3,0.5,0.7,0.9")
    q.add_argument("--fracs", default="0.5,0.85")
    q.add_argument("--n-seeds", type=int, default=6)
    q.add_argument("--target-n", type=int, default=300)
    q.add_argument("--alpha", type=float, default=0.0)   # unused; _pipe API
    q.add_argument("--l0", type=float, default=0.0)

    q = sub.add_parser("hysteresis")
    q.add_argument("--alpha", type=float, default=0.6)
    q.add_argument("--l0-min", type=float, default=0.20)
    q.add_argument("--l0-max", type=float, default=-1.0)
    q.add_argument("--l0-step", type=float, default=0.025)
    q.add_argument("--dwell", type=float, default=60.0)
    q.add_argument("--max-commitments", type=int, default=6000)
    q.add_argument("--l0", type=float, default=0.0)      # unused

    q = sub.add_parser("resetcure")
    q.add_argument("--alpha", type=float, default=1.2)
    # must start ABOVE the fold (l0_c = 0.516 at alpha=1.2, theta=5) so the
    # collapse is deterministic, not a metastable escape
    q.add_argument("--l0", type=float, default=0.6)
    q.add_argument("--l0-post", type=float, default=0.05)
    q.add_argument("--backlog", type=int, default=60)
    q.add_argument("--dwell", type=float, default=30.0)
    q.add_argument("--n-windows", type=int, default=24)
    q.add_argument("--max-commitments", type=int, default=4000)

    q = sub.add_parser("corner")
    # b -> 1 needs a tight deadline: run with --theta 0.2 (cf. the committed
    # sim corner runs: a=0.85-0.9, l0=0.01-0.03, th=0.2 -> b_eff 0.77-0.88)
    q.add_argument("--alpha", type=float, default=0.9)
    q.add_argument("--l0", type=float, default=0.02)
    q.add_argument("--n-seeds", type=int, default=6)
    q.add_argument("--max-commitments", type=int, default=500)

    a = p.parse_args(argv)
    return {"latency": cmd_latency, "grid": cmd_grid,
            "hysteresis": cmd_hysteresis, "resetcure": cmd_resetcure,
            "corner": cmd_corner}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

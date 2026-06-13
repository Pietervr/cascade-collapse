#!/usr/bin/env python3
"""Tier-2 controlled cascade pipeline (instrumented multi-agent system).

Mirrors the closed-loop dynamics of sr_rigor_fix/cascade_sim.py -- the same
physics -- but commitments and certifications are real LLM `generate()` calls
through the model-agnostic backend, and the run emits the FROZEN event-log
schema of validation/MEASUREMENT_PROTOCOL.md so the shared estimator recovers
the *set* (alpha, l0) as ground truth (prediction P2).

Topology is CONTROLLED (we route each uncertified commitment to Poisson(alpha)
downstream workers; exogenous grounded fraction sets l0; verifier coverage and
deadline theta set the certification race). The BACKEND provides content and
genuine error propagation: a worker building on an erroneous, uncertified
parent tends to produce an erroneous child, so the realized cascade is the set
topology pruned by real propagation -- exactly b = alpha * P(propagate).

Event-driven (heap), bounded by max_commitments. Certification = a designated
VERIFIER agent or a TOOL only (protocol section 8.1: a peer/worker response is
consumption, not certification).

Event-log record per commitment (the section 2 schema):
  {id, parents:[...], emitter, type:'exo'|'claim', content, t_created,
   certified:bool, t_certified, uncertified_fired:bool, root, generation,
   ground_truth_error:bool|None}
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import numpy as np

from llm_backend import get_backend

ARR, DEADLINE, SERVICE = 0, 1, 2

WORKER_SYS = ("You are a reasoning worker in a multi-agent pipeline. Given a "
              "parent statement, produce ONE short derived claim that builds "
              "on it. Begin with 'CLAIM:'. Be concise (one sentence).")
VERIFIER_SYS = ("You are a designated verifier. Decide whether the CLAIM is "
                "adequately supported by its PARENT and basic knowledge. "
                "Answer on the first line exactly 'VERDICT: SUPPORTED' or "
                "'VERDICT: REFUTED', then a one-line reason.")


class CascadePipeline:
    def __init__(self, backend, alpha=0.5, l0=0.4, theta=5.0, mu=1.0,
                 verifier_coverage=1.0, seed=1, max_commitments=400,
                 t_max=5000.0, temperature=0.7,
                 propagate_only_errors=True):
        self.be = backend
        self.alpha, self.l0, self.theta, self.mu = alpha, l0, theta, mu
        self.dt = theta / mu                    # decision horizon
        self.qcov = verifier_coverage           # fraction of commits a
        #                                         verifier even attempts
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.max_commitments = max_commitments
        self.t_max = t_max
        self.temperature = temperature
        self.prop_err = propagate_only_errors
        self.reset()

    def reset(self):
        self.t = 0.0
        self.heap = []
        self.seq = 0
        self.events = {}        # id -> record
        self.queue = []         # ids awaiting verification (FCFS)
        self.serving = None
        self.next_id = 0
        self._exo_token = 0
        self.n_calls = 0
        # cumulative counters (window_rates-style measurement during ramps)
        self.n_arr = 0
        self.n_exo = 0
        self.n_cert_cum = 0
        self.n_unc_cum = 0
        self.cap_hit_t = None   # first time the commitment cap censored a spawn

    def _push(self, t, kind, payload):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, payload))

    def _gen(self, system, user):
        self.n_calls += 1
        r = self.be.generate(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=self.temperature, max_tokens=80,
            seed=int(self.rng.integers(1 << 31)))
        return r.text.strip()

    # -- commitment creation -------------------------------------------
    def _new_commitment(self, parent_id, exo=False):
        if len(self.events) >= self.max_commitments:
            if self.cap_hit_t is None:
                self.cap_hit_t = self.t   # spawns censored from here on
            return None
        cid = self.next_id
        self.next_id += 1
        if exo:
            parents, root, gen, perr = [], cid, 0, False
            content = self._gen(WORKER_SYS,
                                "Produce a grounded factual CLAIM.")
        else:
            p = self.events[parent_id]
            root, gen = p["root"], p["generation"] + 1
            # genuine error propagation: child inherits/with-prob produces
            # error if parent was erroneous (real LLM marks via [ERROR] or
            # we sample propagation for non-marking models)
            content = self._gen(
                WORKER_SYS,
                f"PARENT: {p['content']}\nProduce ONE derived CLAIM.")
            perr = ("[ERROR]" in content.upper())
            if self.prop_err and p.get("ground_truth_error"):
                # propagate with high prob through unverified erroneous parent
                perr = perr or (self.rng.random() < 0.8)
        rec = dict(id=cid, parents=parents if exo else [parent_id],
                   emitter="worker", type="exo" if exo else "claim",
                   content=content, t_created=self.t, certified=False,
                   t_certified=None, uncertified_fired=False, root=root,
                   generation=gen, ground_truth_error=perr)
        self.events[cid] = rec
        self.n_arr += 1
        if exo:
            self.n_exo += 1
        # enter verification queue iff a verifier attempts it (coverage)
        if self.rng.random() < self.qcov:
            self.queue.append(cid)
            self._try_serve()
        self._push(self.t + self.dt, DEADLINE, cid)
        return cid

    def _try_serve(self):
        if self.serving is None and self.queue:
            cid = self.queue.pop(0)
            self.serving = cid
            svc = self.rng.exponential(1.0 / self.mu)
            self._push(self.t + svc, SERVICE, cid)

    # -- exogenous arrivals (grounded, rate l0*mu) ----------------------
    def _schedule_exo(self):
        if self.l0 <= 0:
            return
        rate = self.l0 * self.mu
        self._push(self.t + self.rng.exponential(1.0 / rate),
                   ARR, ("exo", self._exo_token))

    # -- handlers -------------------------------------------------------
    def _on_arrival(self, payload):
        if payload[0] == "exo" and payload[1] == self._exo_token:
            self._new_commitment(None, exo=True)
            self._schedule_exo()

    def _on_deadline(self, cid):
        rec = self.events.get(cid)
        if (rec is None or rec["certified"] or rec["uncertified_fired"]
                or rec.get("drained")):
            return
        rec["uncertified_fired"] = True       # fires UNCERTIFIED now
        self.n_unc_cum += 1
        # feedback: spawn Poisson(alpha) downstream workers that consume it
        k = int(self.rng.poisson(self.alpha))
        for _ in range(k):
            self._new_commitment(cid, exo=False)

    def _on_service(self, cid):
        if self.serving == cid:
            self.serving = None
        rec = self.events.get(cid)
        if (rec and not rec["uncertified_fired"] and not rec["certified"]
                and not rec.get("drained")):
            verdict = self._gen(
                VERIFIER_SYS,
                f"PARENT: {self._parent_text(rec)}\nCLAIM: {rec['content']}")
            rec["certified"] = True           # a verifier acted in time
            rec["t_certified"] = self.t
            rec["verdict"] = ("REFUTED" if "REFUT" in verdict.upper()
                              else "SUPPORTED")
            self.n_cert_cum += 1
        self._try_serve()

    def _parent_text(self, rec):
        if not rec["parents"]:
            return "(grounded)"
        return self.events[rec["parents"][0]]["content"]

    # -- ramp / reset controls (CascadeSim semantics) --------------------
    def set_l0(self, l0):
        """Change exogenous grounding rate; invalidates the pending exo
        event (hysteresis ramps)."""
        self.l0 = float(l0)
        self._exo_token += 1
        self._schedule_exo()

    def set_alpha(self, alpha):
        """Change feedback branching (load-reduction arm of reset-cure)."""
        self.alpha = float(alpha)

    def drain(self):
        """Backlog drain (the reset cure): every pending commitment --
        neither certified nor uncertified-fired yet -- is discarded from
        the certification race and never fires feedback."""
        n = 0
        for rec in self.events.values():
            if not rec["certified"] and not rec["uncertified_fired"] \
                    and not rec.get("drained"):
                rec["drained"] = True
                n += 1
        self.queue = []
        self.serving = None   # stale SERVICE events no-op via drained flag
        return n

    def backlog(self):
        return len(self.queue) + (1 if self.serving is not None else 0)

    def snapshot(self):
        return dict(t=self.t, n_arr=self.n_arr, n_exo=self.n_exo,
                    n_cert=self.n_cert_cum, n_unc=self.n_unc_cum,
                    backlog=self.backlog(), n_calls=self.n_calls)

    # -- run ------------------------------------------------------------
    def start(self):
        """Schedule the exo stream + seed one commitment (idempotent)."""
        if not self.heap:
            self._schedule_exo()
            self._new_commitment(None, exo=True)

    def run_until(self, t_end, max_events=200000):
        """Advance the event loop to sim-time t_end; events beyond t_end are
        re-queued for the next segment (ramp dwells). Stops early if the
        global max_commitments cap is hit."""
        ev = 0
        while self.heap and ev < max_events:
            t, sq, kind, payload = heapq.heappop(self.heap)
            if t > t_end:
                heapq.heappush(self.heap, (t, sq, kind, payload))
                self.t = t_end
                break
            if len(self.events) >= self.max_commitments:
                # re-queue (a dropped SVC would wedge the server next segment)
                heapq.heappush(self.heap, (t, sq, kind, payload))
                break
            self.t = t
            ev += 1
            if kind == ARR:
                self._on_arrival(payload)
            elif kind == DEADLINE:
                self._on_deadline(payload)
            else:
                self._on_service(payload)
        return self

    def run(self):
        self.start()
        self.run_until(self.t_max)
        return self.event_log()

    def event_log(self):
        return dict(
            params=dict(alpha=self.alpha, l0=self.l0, theta=self.theta,
                        mu=self.mu, verifier_coverage=self.qcov,
                        seed=self.seed, backend=f"{self.be.name}:{self.be.model}",
                        max_commitments=self.max_commitments),
            commitments=[self.events[k] for k in sorted(self.events)],
            cap_hit_t=self.cap_hit_t, n_llm_calls=self.n_calls)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="mock:mock@0.35")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--l0", type=float, default=0.3)
    p.add_argument("--theta", type=float, default=5.0)
    p.add_argument("--coverage", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-commitments", type=int, default=400)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    be = get_backend(a.backend,
                     cache_dir=os.path.join(os.path.dirname(__file__),
                                            ".llm_cache"))
    pipe = CascadePipeline(be, alpha=a.alpha, l0=a.l0, theta=a.theta,
                           verifier_coverage=a.coverage, seed=a.seed,
                           max_commitments=a.max_commitments)
    log = pipe.run()
    n = len(log["commitments"])
    nu = sum(c["uncertified_fired"] for c in log["commitments"])
    nc = sum(c["certified"] for c in log["commitments"])
    print(f"backend={a.backend} alpha={a.alpha} l0={a.l0} theta={a.theta}: "
          f"{n} commitments, {nu} uncertified, {nc} certified, "
          f"{log['n_llm_calls']} LLM calls")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(log, fh)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

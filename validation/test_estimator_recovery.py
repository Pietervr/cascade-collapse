#!/usr/bin/env python3
"""P2 dry-run: does the shared estimator recover the SET alpha from pipeline
event logs? Uses the mock backend (no LLM, exact structural ground truth) so
the Tier-2 plumbing + estimator are validated offline. Pooling many seeds per
(alpha,l0) to beat finite-run boundary truncation.
"""
import numpy as np
from llm_backend import get_backend
from tier2_pipeline import CascadePipeline
from estimator import estimate

GRID = [(0.3, 0.4), (0.5, 0.3), (0.7, 0.25), (0.9, 0.2)]
SEEDS = 12
THETA = 5.0

be = get_backend("mock:mock@0.0")   # fail_rate irrelevant for structure
print(f"{'alpha':>6} {'l0':>5} | {'alpha_hat':>9} {'rel%':>6} | "
      f"{'b_id':>6} {'1-grnd':>6} {'consist':>7} | cascades")
ok = True
for alpha, l0 in GRID:
    pooled = []
    for s in range(SEEDS):
        pipe = CascadePipeline(be, alpha=alpha, l0=l0, theta=THETA,
                               seed=1000 + s, max_commitments=300,
                               propagate_only_errors=False)
        pooled.extend(pipe.run()["commitments"])
        # relabel ids to keep them globally unique across seeds
    # re-id pooled logs so subtrees don't collide
    offset = 0
    fixed = []
    base = 0
    seen_roots = 0
    # simplest: estimate per-seed then average alpha_hat (cleaner than pooling)
    ahats, bids, grnds, cons = [], [], [], []
    for s in range(SEEDS):
        pipe = CascadePipeline(be, alpha=alpha, l0=l0, theta=THETA,
                               seed=2000 + s, max_commitments=300,
                               propagate_only_errors=False)
        d = estimate(pipe.run()["commitments"], fit=False)
        if d["alpha_hat"] == d["alpha_hat"]:
            ahats.append(d["alpha_hat"]); bids.append(d["b_identity"])
            grnds.append(d["grounding"]); cons.append(d["consistency"])
    ah = float(np.mean(ahats))
    rel = 100 * (ah - alpha) / alpha
    bid = float(np.mean(bids)); gr = float(np.mean(grnds))
    con = float(np.mean(cons))
    flag = "" if abs(rel) <= 15 else "  <-- >15%"
    print(f"{alpha:6.2f} {l0:5.2f} | {ah:9.3f} {rel:+6.1f} | "
          f"{bid:6.3f} {1-gr:6.3f} {con:7.3f} | n~{len(ahats)}{flag}")
    ok &= abs(rel) <= 15
print("\nP2 dry-run:", "PASS (estimator recovers set alpha within 15%)"
      if ok else "FAIL")

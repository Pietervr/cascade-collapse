#!/usr/bin/env python3
"""Gold-subset validation (VALIDATION_PLAN Phase 4, protocol section 4).

The published gold set (MAD_human_labelled_dataset.json) is TRACE-level: 19
traces, each with 3 annotators voting TRUE/FALSE on each MAST failure mode --
there are NO per-step certification labels. So the protocol's "validate the
automatic certification classifier per step" becomes a trace-level SIGNAL check:

  * traces humans flag with a VERIFICATION failure (category 3: no/incomplete/
    incorrect verification) should show a LOWER adapter certified-fraction;
  * traces humans flag with INTER-AGENT MISALIGNMENT (category 2) should show
    LARGER adapter-extracted cascades.

A positive sign in both is the trace-level analog of classifier agreement.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from tier1_adapters import SEGMENTERS, seg_generic, build_records   # noqa: E402
from estimator import estimate, _children_index, _subtree_size      # noqa: E402

GOLD = os.path.join(_HERE, "..", "tier1_data", "MAD_human_labelled_dataset.json")


def cats_from_annotations(annotations):
    """Majority vote (>=2 of 3 annotators) -> which MAST categories present."""
    cat = {1: 0, 2: 0, 3: 0}
    for a in annotations:
        code = re.match(r"\s*([123])\.", a.get("failure mode", ""))
        if not code:
            continue
        votes = sum(bool(a.get(f"annotator_{i}")) for i in (1, 2, 3))
        if votes >= 2:
            cat[int(code.group(1))] = 1
    return cat


def main():
    gold = json.load(open(GOLD))
    rows = []
    for r in gold:
        fw = r["mas_name"]
        seg = SEGMENTERS.get(fw, seg_generic)
        text = r["trace"] if isinstance(r["trace"], str) \
            else r["trace"]["trajectory"]
        recs = build_records(seg(text))
        d = estimate(recs, fit=False)
        _, kids = _children_index(recs)
        exo = [c["id"] for c in recs if not c.get("parents")]
        smax = max([_subtree_size(x, kids) for x in exo], default=1)
        cert_frac = d["N_cert"] / d["N"] if d["N"] else 0.0
        cats = cats_from_annotations(r["annotations"])
        rows.append(dict(fw=fw, N=d["N"], grounding=d["grounding"],
                         cert_frac=cert_frac, s_max=smax, **{f"cat{k}": v
                         for k, v in cats.items()}))

    print(f"{'framework':<11}{'N':>5}{'l0':>6}{'cert%':>7}{'s_max':>6}"
          f"  cat1 cat2 cat3")
    for r in rows:
        print(f"{r['fw']:<11}{r['N']:>5}{r['grounding']:>6.2f}"
              f"{100*r['cert_frac']:>6.0f}%{r['s_max']:>6}"
              f"   {r['cat1']}    {r['cat2']}    {r['cat3']}")

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    # signal 1: verification-failure traces -> lower certified fraction
    v_yes = [r["cert_frac"] for r in rows if r["cat3"]]
    v_no = [r["cert_frac"] for r in rows if not r["cat3"]]
    # signal 2: misalignment traces -> larger cascades
    m_yes = [r["s_max"] for r in rows if r["cat2"]]
    m_no = [r["s_max"] for r in rows if not r["cat2"]]
    print(f"\nGold signal 1 (verification cat3 -> LOWER cert fraction):")
    print(f"   cert_frac: cat3={mean(v_yes):.3f} (n={len(v_yes)})  "
          f"no-cat3={mean(v_no):.3f} (n={len(v_no)})  "
          f"-> {'OK' if mean(v_yes) < mean(v_no) else 'NOT in direction'}")
    print(f"Gold signal 2 (misalignment cat2 -> LARGER cascades):")
    print(f"   s_max: cat2={mean(m_yes):.2f} (n={len(m_yes)})  "
          f"no-cat2={mean(m_no):.2f} (n={len(m_no)})  "
          f"-> {'OK' if mean(m_yes) > mean(m_no) else 'NOT in direction'}")
    out = os.path.join(_HERE, "runs", "tier1", "analysis", "gold_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dict(rows=rows,
                       cert_cat3=mean(v_yes), cert_nocat3=mean(v_no),
                       smax_cat2=mean(m_yes), smax_nocat2=mean(m_no)),
                  fh, indent=2, default=float)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tier-1 adapters: MAST raw trajectories -> the frozen section-2 event-log
schema the shared estimator consumes (validation/MEASUREMENT_PROTOCOL.md).

The corpus (mcemri/MAST-Data, 1242 traces, 7 frameworks) stores each trace as
a single free-text log string whose format is framework-specific. Each adapter
SEGMENTS the log into an ordered list of messages, tagging each with a `kind`:

    prompt      the task / human requirement (an exogenous, grounded input)
    tool        a tool / code-execution result (a grounded input)
    certifier   a designated verifier/critic/reviewer/tester turn  (section 8.1:
                a peer agent's response is NOT certification -- it is consumption)
    agent       a regular peer-agent commitment (a claim/output)

A single, framework-independent rule (build_records) then turns the tagged
message stream into section-2 commitment records:

  * prompt / tool messages are GROUNDED (exogenous): type='exo', parents=[].
  * an `agent` message's parent is the most recent prior OPEN (uncertified,
    non-grounded) agent commitment it builds on; if none is open (the previous
    commitment was grounded or just certified) it is itself grounded -- a
    grounded restart.  ==> grounding fraction l0_hat = N_exo / N.
  * a `certifier` message certifies the most recent open agent commitment
    (verifier-or-tool-only rule); the certified commitment does not fire.
  * an agent commitment that is consumed by a later child while still open
    fires UNCERTIFIED.  ==> cascades = subtrees under exogenous roots.

The same estimator.py then reads these records for both tiers. This is the
observational (Tier-1) half of the validation; the rule is identical to the one
the instrumented Tier-2 pipeline emits by construction.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _brace_segments(text: str):
    """Yield top-level {...} substrings, respecting quoted strings (Python
    repr uses ' or "); used to parse AG2's concatenated dict-repr trajectories."""
    depth = 0
    start = None
    quote = None
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None
        i += 1

# ----------------------------------------------------------------------
# role classification (applied to a segmented agent name, framework-aware)
# ----------------------------------------------------------------------

# section 8.1 (frozen): certification = a designated verifier/critic/reflection
# turn OR a tool execution returning a ground result. A test *writer* that never
# executes is a peer (consumption), so 'tester' is deliberately NOT here -- only
# explicit critic/review/verify/judge roles. Executed tests reach us as 'tool'.
CERTIFIER_RE = re.compile(
    r"review|critic|verif|\bqa\b|judge|evaluat|inspector|reflect", re.I)
# names that are coordinators/managers -- peers, NOT certifiers (consumption)
COORDINATOR_RE = re.compile(
    r"orchestrat|chief executive|manager|host|planner|user_proxy", re.I)


def _is_certifier(name: str) -> bool:
    return bool(CERTIFIER_RE.search(name or ""))


# ----------------------------------------------------------------------
# per-framework segmenters -> list[(role_name, kind, content)]
# kind in {prompt, tool, certifier, agent}
# ----------------------------------------------------------------------

def _ag2_turns(text: str):
    """AG2 stores trajectories two ways: (a) concatenated Python dict reprs
    {'content': [...], 'role': ..., 'name': ...}, or (b) a YAML-ish block.
    Return list[(name, content)] for whichever is present."""
    turns = []
    # (a) dict-repr stream (the majority format)
    for seg in _brace_segments(text):
        if "'content'" not in seg and '"content"' not in seg:
            continue
        try:
            d = ast.literal_eval(seg)
        except Exception:
            continue
        if not isinstance(d, dict) or "name" not in d:
            continue
        content = d.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        turns.append((str(d.get("name", "agent")), str(content)))
    if turns:
        return turns
    # (b) YAML-ish fallback
    body = text.split("trajectory:", 1)
    body = body[1] if len(body) > 1 else text
    block_re = re.compile(
        r"content:\s*(?P<content>.*?)\n\s*role:\s*\w+\s*\n\s*name:\s*(?P<name>\S+)",
        re.S)
    for m in block_re.finditer(body):
        turns.append((m.group("name").strip(), m.group("content").strip()))
    return turns


def seg_ag2(text: str):
    """AG2: mathproxyagent's first (long) message = the prompt scaffolding; its
    short, code-free messages = code-execution results (tool, grounded); the
    'assistant' name = the solver agent producing claims."""
    msgs = [("task", "prompt", "problem statement")]
    for i, (name, content) in enumerate(_ag2_turns(text)):
        if i == 0:
            msgs.append((name, "prompt", content))
            continue
        is_exec = (name == "mathproxyagent"
                   and "```" not in content and len(content) < 400)
        msgs.append((name, "tool" if is_exec else "agent", content))
    return msgs


def seg_metagpt(text: str):
    """MetaGPT: 'FROM: Human ...' requirement, then 'NEW MESSAGES:' blocks each
    starting 'AgentName: <content>', dash-delimited."""
    msgs = []
    chunks = re.split(r"-{20,}", text)
    name_re = re.compile(r"^\s*([A-Z][A-Za-z0-9_ ]{1,40}?):\s", re.M)
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if "UserRequirement" in ch or "FROM: Human" in ch:
            msgs.append(("Human", "prompt", ch[:2000]))
            continue
        nm = name_re.search(ch)
        if not nm:
            continue
        name = nm.group(1).strip()
        content = ch[nm.end():].strip()
        kind = "certifier" if _is_certifier(name) else "agent"
        # explicit test execution / pytest output in the chunk == grounded tool
        if re.search(r"\bpassed\b|\bfailed\b|PASSED|FAILED|Traceback", ch) \
                and not _is_certifier(name):
            kind = "tool"
        msgs.append((name, kind, content[:2000]))
    return msgs


def seg_magentic(text: str):
    """Magentic-One: '---------- Name ----------' delimited turns (after the
    pip/docker preamble). Orchestrator = coordinator (peer); Executor/WebSurfer/
    FileSurfer tool actions = grounded; user = prompt."""
    parts = re.split(r"-{6,}\s*([A-Za-z0-9_ ]+?)\s*-{6,}", text)
    # parts: [pre, name1, body1, name2, body2, ...]
    msgs = []
    started = False
    it = iter(range(1, len(parts) - 1, 2))
    for i in it:
        name = parts[i].strip()
        body = parts[i + 1].strip()
        if name.lower() == "user":
            started = True
            msgs.append(("user", "prompt", body[:2000]))
            continue
        if not started:
            continue
        low = name.lower()
        if "executor" in low or "computerterminal" in low:
            kind = "tool"
        elif _is_certifier(name):
            kind = "certifier"
        else:
            kind = "agent"      # Orchestrator, WebSurfer, FileSurfer, Coder, Assistant
        msgs.append((name, kind, body[:2000]))
    return msgs


def seg_chatdev(text: str):
    """ChatDev: timestamped '[ts INFO] Role: **Role A<->Role B on : Phase**'
    turns, '[Seminar Conclusion]', and '[Execute Detail]'/'[Update Codes]'
    (grounded code execution)."""
    msgs = [("task", "prompt", "software requirement")]
    line_re = re.compile(
        r"^\[[\d :\-]+INFO\]\s+(?P<spk>[A-Za-z ]+?):\s+\*\*(?P<tag>.*?)\*\*",
        re.M)
    last = 0
    for m in line_re.finditer(text):
        spk = m.group("spk").strip()
        tag = m.group("tag")
        if spk.lower() == "system":
            continue
        # content is the text following this header up to the next header
        kind = "certifier" if _is_certifier(spk) else "agent"
        msgs.append((spk, kind, tag[:300]))
    # code execution markers -> grounded tool events interspersed
    for _m in re.finditer(r"\*\*\[(Execute Detail|Update Codes|Test Reports?)\]\*\*",
                          text):
        msgs.append(("system_exec", "tool", "code execution"))
    return msgs


def seg_generic(text: str):
    """Fallback for OpenManus / AppWorld / HyperAgent.

    OpenManus & AppWorld are single-loop code agents: an agent step (code +
    rationale) alternates with a 'Code Execution Output' (grounded tool result).
    HyperAgent is a hierarchical SWE agent logged as
    '... - INFO - <Role>'s Response: Thought:/Action:/Observation:' where
    Observation = grounded tool result, Planner/Navigator/Editor = agent steps."""
    msgs = [("task", "prompt", "task")]

    if "Executing step" in text and "Manus" in text:
        # OpenManus loguru: 'Executing step N/M' agent steps + tool results
        for m in re.finditer(
                r"(?P<step>Executing step \d+/\d+)|"
                r"(?P<tool>completed its mission!|Observed output of cmd)", text):
            if m.group("step"):
                msgs.append(("Manus", "agent", "step"))
            else:
                msgs.append(("tool", "tool", "result"))
        return msgs

    if "Code Execution Output" in text:
        # alternation: text before each exec output = an agent step
        parts = re.split(r"Code Execution Output", text)
        for j, p in enumerate(parts):
            p = p.strip()
            if p:
                msgs.append(("agent", "agent", p[:400]))
            if j < len(parts) - 1:
                msgs.append(("executor", "tool", "code execution output"))
        return msgs

    if "'s Response:" in text or "Intern Name:" in text:
        anchor = re.compile(
            r"(?:- INFO - )?(?P<who>[\w ()\-]+?)'s Response:\s*"
            r"(?P<kind>Thought|Action|Observation)?", re.I)
        for m in anchor.finditer(text):
            who = m.group("who").strip()
            tag = (m.group("kind") or "").lower()
            if tag == "observation":
                msgs.append((who, "tool", "observation"))
            elif _is_certifier(who):
                msgs.append((who, "certifier", "review"))
            else:
                msgs.append((who, "agent", who))
        return msgs

    # last-resort ReAct anchors
    for m in re.finditer(r"^(?P<k>Thought|Action|Observation)\b[:\s]", text, re.M):
        k = m.group("k").lower()
        msgs.append((k, "tool" if k == "observation" else "agent", k))
    return msgs


SEGMENTERS = {
    "AG2": seg_ag2, "MetaGPT": seg_metagpt, "Magentic": seg_magentic,
    "ChatDev": seg_chatdev, "OpenManus": seg_generic, "AppWorld": seg_generic,
    "HyperAgent": seg_generic,
}
# confidence: bespoke parsers vs the generic ReAct fallback
HIGH_CONFIDENCE = {"AG2", "MetaGPT", "Magentic", "ChatDev"}


# ----------------------------------------------------------------------
# uniform genealogy + certification rule  ->  section-2 commitment records
# ----------------------------------------------------------------------

def build_records(msgs, base=0):
    """msgs: list[(name, kind, content)] in temporal order.
    Returns list of section-2 commitment records (ids offset by `base`)."""
    recs = []
    open_id = None        # most recent OPEN (uncertified, non-grounded) agent
    nid = base

    def add(parents, emitter, typ, certified=False):
        nonlocal nid
        root = nid if not parents else recs[parents[0] - base]["root"]
        gen = 0 if not parents else recs[parents[0] - base]["generation"] + 1
        recs.append(dict(id=nid, parents=parents, emitter=emitter, type=typ,
                         certified=certified, t_certified=None,
                         uncertified_fired=False, root=root, generation=gen))
        nid += 1
        return nid - 1

    for name, kind, _content in msgs:
        if kind == "prompt":
            add([], name, "exo")            # grounded task input
            open_id = None
        elif kind == "tool":
            # tool/code execution: certifies the open agent (verifier-or-tool)
            # AND is itself a grounded arrival the next step builds on
            if open_id is not None:
                recs[open_id - base]["certified"] = True
                open_id = None
            add([], name, "exo")
        elif kind == "certifier":
            # critic/review turn: a certification EVENT, not a commitment
            # (Tier-2 parity: a verifier acting flips the flag, adds no node)
            if open_id is not None:
                recs[open_id - base]["certified"] = True
                open_id = None
        else:  # agent commitment
            if open_id is None:
                open_id = add([], name, "exo")          # grounded restart
            else:
                recs[open_id - base]["uncertified_fired"] = True
                open_id = add([open_id], name, "claim")  # consumes open parent
    return recs


def adapt_trace(rec):
    """One corpus record -> (records, meta)."""
    fw = rec["mas_name"]
    seg = SEGMENTERS.get(fw, seg_generic)
    text = rec["trace"]["trajectory"]
    msgs = seg(text)
    recs = build_records(msgs)
    return recs, dict(mas_name=fw, benchmark=rec["benchmark_name"],
                      llm=rec["llm_name"], trace_id=rec["trace_id"],
                      n_msgs=len(msgs), high_conf=fw in HIGH_CONFIDENCE)


# ----------------------------------------------------------------------
# MAST annotation category helpers (for P4)
# ----------------------------------------------------------------------

def mast_categories(ann):
    """Return which of the 3 MAST categories are flagged in this trace.
    1.x = specification/system-design, 2.x = inter-agent misalignment,
    3.x = task verification."""
    cat = {1: 0, 2: 0, 3: 0}
    for k, v in ann.items():
        if v and "." in k:
            cat[int(k.split(".")[0])] = 1
    return cat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=os.path.join(
        _HERE, "..", "tier1_data", "MAD_full_dataset.json"))
    p.add_argument("--out", default=os.path.join(_HERE, "runs", "tier1"))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    with open(a.data) as fh:
        corpus = json.load(fh)
    if a.limit:
        corpus = corpus[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    per_fw = {}
    for rec in corpus:
        recs, meta = adapt_trace(rec)
        fw = meta["mas_name"]
        per_fw.setdefault(fw, []).append(dict(meta=meta, commitments=recs,
                                              mast=mast_categories(
                                                  rec["mast_annotation"])))
    for fw, traces in per_fw.items():
        with open(os.path.join(a.out, f"tier1_{fw}.json"), "w") as fh:
            json.dump(traces, fh)
        tot = sum(len(t["commitments"]) for t in traces)
        print(f"{fw:<11} {len(traces):>4} traces  {tot:>7} commitments  "
              f"({'bespoke' if fw in HIGH_CONFIDENCE else 'generic'})")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

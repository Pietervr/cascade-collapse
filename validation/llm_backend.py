#!/usr/bin/env python3
"""Model-agnostic LLM backend for the Tier-2 cascade experiment.

One `generate()` interface, swappable adapters:
  - OllamaBackend     : local server (the sweep workhorse)
  - AnthropicBackend  : frontier spot-check (needs ANTHROPIC_API_KEY)

Design goals (validation/MEASUREMENT_PROTOCOL.md):
  - The Tier-2 pipeline calls only `Backend.generate(...)`; model choice is a
    one-line swap, so the estimator validation is independent of model.
  - Stdlib only (urllib) -- runnable with no pip installs.
  - Optional on-disk response cache keyed by (model, messages, temperature,
    seed) for reproducibility and cheap re-runs of a sweep.

Spec strings: "ollama:llama3.1:8b", "anthropic:claude-sonnet-4-6", ...
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Result + base
# ----------------------------------------------------------------------

@dataclass
class GenResult:
    text: str
    model: str
    cached: bool = False
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


class Backend:
    """Abstract chat backend. Subclasses implement _raw_generate."""

    name = "abstract"

    def __init__(self, model: str, cache_dir: str | None = None):
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- cache key over the full request --------------------------------
    def _key(self, messages, temperature, max_tokens, seed):
        blob = json.dumps(
            {"m": self.model, "msgs": messages, "t": temperature,
             "mx": max_tokens, "s": seed},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, messages, temperature=0.7, max_tokens=512,
                 seed=None) -> GenResult:
        """messages: list of {"role": "system"|"user"|"assistant",
        "content": str}. Returns GenResult."""
        ckey = None
        if self.cache_dir:
            ckey = self._key(messages, temperature, max_tokens, seed)
            path = os.path.join(self.cache_dir, ckey + ".json")
            if os.path.exists(path):
                with open(path) as fh:
                    d = json.load(fh)
                return GenResult(text=d["text"], model=self.model,
                                 cached=True, meta=d.get("meta", {}))
        t0 = _now()
        text, meta = self._raw_generate(messages, temperature, max_tokens,
                                        seed)
        dt = _now() - t0
        if self.cache_dir and ckey:
            with open(os.path.join(self.cache_dir, ckey + ".json"), "w") as fh:
                json.dump({"text": text, "meta": meta}, fh)
        return GenResult(text=text, model=self.model, cached=False,
                         latency_s=dt, meta=meta)

    def _raw_generate(self, messages, temperature, max_tokens, seed):
        raise NotImplementedError


def _now():
    # Date.now()-style wall clock is fine here (not a workflow script).
    return time.monotonic()


def _post_json(url, payload, timeout=300, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------------------------------------------------
# Ollama (local)
# ----------------------------------------------------------------------

class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model="llama3.1:8b",
                 host="http://localhost:11434", cache_dir=None,
                 num_ctx=8192):
        super().__init__(model, cache_dir)
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx

    def _raw_generate(self, messages, temperature, max_tokens, seed):
        opts = {"temperature": float(temperature), "num_ctx": self.num_ctx}
        if max_tokens:
            opts["num_predict"] = int(max_tokens)
        if seed is not None:
            opts["seed"] = int(seed)
        payload = {"model": self.model, "messages": messages,
                   "stream": False, "options": opts}
        last = None
        for attempt in range(4):
            try:
                d = _post_json(f"{self.host}/api/chat", payload, timeout=600)
                return d["message"]["content"], {
                    "eval_count": d.get("eval_count"),
                    "prompt_eval_count": d.get("prompt_eval_count")}
            except (urllib.error.URLError, KeyError, TimeoutError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Ollama generate failed after retries: {last}")

    def ensure_model(self):
        """Pull the model if absent (returns True if present/ready)."""
        try:
            d = _post_json(f"{self.host}/api/show", {"name": self.model},
                           timeout=30)
            return "modelfile" in d or "details" in d
        except Exception:
            return False


# ----------------------------------------------------------------------
# Anthropic (frontier spot-check)
# ----------------------------------------------------------------------

class AnthropicBackend(Backend):
    name = "anthropic"
    API = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(self, model="claude-sonnet-4-6", cache_dir=None,
                 api_key=None):
        super().__init__(model, cache_dir)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def _raw_generate(self, messages, temperature, max_tokens, seed):
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        # split out system messages (Anthropic takes system separately)
        sys = "\n\n".join(m["content"] for m in messages
                          if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m["role"] in ("user", "assistant")]
        payload = {"model": self.model, "max_tokens": int(max_tokens or 512),
                   "temperature": float(temperature), "messages": turns}
        if sys:
            payload["system"] = sys
        headers = {"x-api-key": self.api_key,
                   "anthropic-version": self.VERSION}
        d = _post_json(self.API, payload, timeout=300, headers=headers)
        text = "".join(b.get("text", "") for b in d.get("content", [])
                       if b.get("type") == "text")
        return text, {"usage": d.get("usage")}


# ----------------------------------------------------------------------
# Mock (no network; for pipeline+estimator development with exact ground
# truth). Content is templated; the *structural* dynamics (alpha, l0, theta,
# verification) are owned by the pipeline, so estimator validation (P2) needs
# no real model. Deterministic given (messages, seed).
# ----------------------------------------------------------------------

class MockBackend(Backend):
    name = "mock"

    def __init__(self, model="mock", cache_dir=None, fail_rate=0.0):
        # fail_rate: probability a generated claim self-declares an error
        # marker; lets us exercise content-level error parsing offline.
        super().__init__(model, cache_dir)
        self.fail_rate = fail_rate

    def _raw_generate(self, messages, temperature, max_tokens, seed):
        h = int(hashlib.sha256(
            (json.dumps(messages, sort_keys=True) + str(seed)).encode()
        ).hexdigest(), 16)
        last = messages[-1]["content"] if messages else ""
        # crude role detection from the system/user content
        sys = " ".join(m["content"] for m in messages
                       if m["role"] == "system").lower()
        if "verifier" in sys or "verify" in last.lower():
            verdict = "SUPPORTED" if (h % 100) / 100.0 >= self.fail_rate \
                else "REFUTED"
            return f"VERDICT: {verdict}\nReason: mock check.", {"mock": True}
        bad = (h % 1000) / 1000.0 < self.fail_rate
        tag = " [ERROR]" if bad else ""
        return (f"CLAIM: mock derived statement {h % 100000}{tag}",
                {"mock": True, "injected_error": bad})


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def get_backend(spec: str, cache_dir=None) -> Backend:
    """spec = 'ollama:llama3.1:8b' | 'anthropic:claude-sonnet-4-6'
    | 'mock:mock' | 'mock:mock@0.3' (fail_rate 0.3)."""
    if ":" not in spec:
        raise ValueError(f"backend spec needs provider: prefix, got {spec!r}")
    provider, model = spec.split(":", 1)
    if provider == "ollama":
        return OllamaBackend(model=model, cache_dir=cache_dir)
    if provider == "anthropic":
        return AnthropicBackend(model=model, cache_dir=cache_dir)
    if provider == "mock":
        fr = 0.0
        if "@" in model:
            model, fr = model.split("@", 1)
            fr = float(fr)
        return MockBackend(model=model, cache_dir=cache_dir, fail_rate=fr)
    raise ValueError(f"unknown provider {provider!r}")


if __name__ == "__main__":
    import sys
    spec = sys.argv[1] if len(sys.argv) > 1 else "ollama:llama3.1:8b"
    be = get_backend(spec)
    r = be.generate(
        [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0.0, max_tokens=16, seed=1)
    print(f"[{be.name}:{be.model}] cached={r.cached} {r.latency_s:.2f}s "
          f"-> {r.text!r}")

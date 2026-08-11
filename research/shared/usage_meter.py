"""
research/shared/usage_meter.py

Per-query LLM usage accounting, shared by all five systems.

Nothing in the codebase recorded token consumption, LLM call counts, or turn
counts, so cost and effort could not be compared across architectures. Adding
counters to each agent would have meant touching five systems and dozens of call
sites, and would have drifted as prompts changed.

Instead this patches the single point every system funnels through —
`openai.resources.chat.completions.Completions.create` — and attributes each
call to whichever query is currently open. That makes the measurement uniform by
construction: the MAS, the three MAS baselines, and the single agent are all
counted the same way, including calls made inside cheap gates
(`_llm_bool`, `_needs_profile`) that a per-agent counter would likely miss.

Definitions
-----------
llm_calls   Chat completion requests issued while the query was open.
turns       Distinct agent activations. Recorded by the runner via note_turn();
            for the single agent this is its ReAct iteration count. This is the
            one figure that is not architecture-neutral — each system defines an
            activation differently — so it is reported alongside llm_calls
            rather than in place of it.
tokens      prompt/completion/total, summed from the API's own usage field.
            Calls whose response carries no usage block are counted in
            calls_without_usage rather than silently contributing zero.

Usage:
    from usage_meter import METER
    METER.install()
    with METER.query("N7"):
        ...run the query...
    METER.dump("logs/usage_mas.jsonl")
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._installed = False
        self._current: str | None = None
        self.records: dict[str, dict] = {}

    # ── record shape ──────────────────────────────────────────────────────────

    def _blank(self, qid: str) -> dict:
        return {
            "query_id": qid,
            "llm_calls": 0,
            "turns": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls_without_usage": 0,
            "models": {},
        }

    def _rec(self, qid: str) -> dict:
        if qid not in self.records:
            self.records[qid] = self._blank(qid)
        return self.records[qid]

    # ── instrumentation ───────────────────────────────────────────────────────

    def install(self) -> None:
        """Patch the OpenAI SDK's chat-completions entry point. Idempotent."""
        if self._installed:
            return

        from openai.resources.chat import completions as _c

        original = _c.Completions.create
        meter = self

        def create(self, *args, **kwargs):  # noqa: ANN001 — SDK signature
            resp = original(self, *args, **kwargs)
            meter._record(kwargs.get("model", "?"), resp)
            return resp

        _c.Completions.create = create
        self._installed = True

    def _record(self, model: str, resp) -> None:
        with self._lock:
            qid = self._current
            if qid is None:
                return  # call made outside any query (e.g. agent construction)
            r = self._rec(qid)
            r["llm_calls"] += 1
            r["models"][model] = r["models"].get(model, 0) + 1

            usage = getattr(resp, "usage", None)
            if usage is None:
                r["calls_without_usage"] += 1
                return
            r["prompt_tokens"]     += getattr(usage, "prompt_tokens", 0) or 0
            r["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            r["total_tokens"]      += getattr(usage, "total_tokens", 0) or 0

    # ── scoping ───────────────────────────────────────────────────────────────

    @contextmanager
    def query(self, qid: str):
        with self._lock:
            self._current = qid
            self._rec(qid)
        try:
            yield
        finally:
            with self._lock:
                self._current = None

    def note_turn(self, n: int = 1) -> None:
        """Record n agent activations against the open query."""
        with self._lock:
            if self._current is not None:
                self._rec(self._current)["turns"] += n

    def set_turns(self, n: int) -> None:
        with self._lock:
            if self._current is not None:
                self._rec(self._current)["turns"] = n

    # ── output ────────────────────────────────────────────────────────────────

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for qid, rec in self.records.items():
                fh.write(json.dumps(rec) + "\n")


METER = UsageMeter()

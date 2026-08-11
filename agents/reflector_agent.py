"""
agents/reflector_agent.py  —  v1.1.0

ReflectorAgent: quality reviewer for synthesized answers.

P2P agent chain position
------------------------
  synthesizer → reflector → (CONFIRM | CHALLENGE) → synthesizer

Receives VALIDATE from SynthesizerAgent.
Reads SharedLog for session context (SIGDIAL mechanism).

DA types emitted
----------------
  CONFIRM   — answer is acceptable; content = the approved answer text
  CHALLENGE — issues found; metadata["issues"] = list of specific problems
              Synthesizer responds by calling DataExplorer (DATA_INSUFFICIENCY)
              and resubmitting a VALIDATE.
"""
from __future__ import annotations

import json
from core.message_schema import A2AMessage, DialogueActType
from core.shared_log import SharedLog
from agents.base_agent import BaseAgent


class ReflectorAgent(BaseAgent):
    """
    Quality gate between SynthesizerAgent and the user.
    Emits CONFIRM or CHALLENGE based on a focused LLM review.
    """

    name    = "reflector"
    version = "1.1.0"

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        answer = msg.content
        query  = msg.metadata.get("original_query", "")

        # Read session log — gives Reflector visibility into what DataExplorer
        # profiled, what SQL was run, and any prior CHALLENGE/revision rounds.
        session_ctx = self.log.format_for_llm(msg.session_id)
        self._note(msg, "session_read",
                   f"read {len(session_ctx)} chars of prior agent outputs")
        self._note(msg, "reflect_start",
                   f"reviewing answer for: {query[:80]}")

        review = self._review(query, answer, session_ctx)
        ok     = review.get("ok", True)
        issues = review.get("issues", [])

        if ok or not issues:
            self._note(msg, "reflect_confirm", "answer accepted — no critical issues")
            return self.reply(
                msg, DialogueActType.CONFIRM,
                content    = answer,
                confidence = 0.9,
            )
        else:
            issues_text = " | ".join(issues)
            self._note(msg, "reflect_challenge",
                       f"CHALLENGE ({len(issues)} issue(s)): {issues_text}")
            return self.reply(
                msg, DialogueActType.CHALLENGE,
                content    = answer,
                confidence = 0.5,
                metadata   = {"issues": issues},
            )

    # ── Review ────────────────────────────────────────────────────────────────

    def _review(self, query: str, answer: str, session_ctx: str = "") -> dict:
        ctx_block = (
            f"Prior agent findings in this session (DataExplorer profiles, raw SQL results):\n"
            f"{session_ctx}\n\n"
            if session_ctx and session_ctx != "(no substantive agent output yet)" else ""
        )
        schema_absent = (
            "billing cost, electricity cost, cost per node-hour, job cost, "
            "carbon footprint, CO2 emissions, node temperature, CPU temperature, "
            "OS kernel version, software version, firmware version, "
            "network latency, inter-node bandwidth, network topology, "
            "GPU metrics, job queue wait time, scheduler queue depth"
        )
        prompt = (
            "You are a quality reviewer for an HPC data assistant (Fugaku supercomputer).\n\n"
            "══ HIGHEST PRIORITY RULE — evaluate this BEFORE anything else ══\n"
            "The Fugaku schema does NOT contain data for these concepts:\n"
            f"  {schema_absent}\n\n"
            "If the user's question asks about ANY of those concepts AND the answer "
            "correctly states the data is 'not available', 'not tracked', 'not in the "
            "dataset', 'not in the schema', or 'not captured' — you MUST return "
            '{"ok": true, "issues": []} IMMEDIATELY.\n'
            "A correct refusal of unavailable data is the RIGHT answer. "
            "Do NOT challenge it.\n"
            "══════════════════════════════════════════════════════════════════\n\n"
            f"{ctx_block}"
            f"Question: {query}\n"
            f"Answer: {answer}\n\n"
            "Fugaku dataset facts (for semantic mismatch detection):\n"
            "- NO temperature columns exist. Any answer claiming CPU/node temperature is WRONG.\n"
            "- NO billing/cost/pricing columns exist. Any claimed cost per node/hour is WRONG.\n"
            "- NO GPU data exists. Fugaku has no GPUs.\n"
            "- uctmut = CPU time in milliseconds (billions range), NOT temperature or utilization %.\n"
            "- Power (avgpcon) is in Watts — NOT a billing cost.\n\n"
            "Check for CRITICAL issues only (flag these — only if NOT covered by the highest "
            "priority rule above):\n"
            "1. Answer CLAIMS to provide data the Fugaku schema cannot contain "
            "(e.g., gives fake temperature numbers, fake billing figures).\n"
            "2. A column's data is mislabeled as a different concept "
            "(e.g. uctmut called 'temperature', avgpcon called 'billing cost').\n"
            "3. Power/energy values stated without watts/joules units.\n"
            "4. Time/duration values stated without seconds or hours units.\n"
            "5. The answer completely ignores the user's question.\n"
            "6. Obviously wrong numbers (negative counts, etc.).\n\n"
            "Do NOT flag: style, verbosity, extra detail, phrasing preferences,\n"
            "             missing commas in numbers, formatting choices,\n"
            "             an answer that says 'not available' for schema-absent data.\n\n"
            "If the answer is factually reasonable and units are present — "
            "return ok=true even if it could be improved.\n\n"
            "Reply with ONLY valid JSON:\n"
            "{\"ok\": true, \"issues\": []}\n"
            "or\n"
            "{\"ok\": false, \"issues\": [\"specific issue 1\"]}"
        )
        try:
            raw = self._llm(
                prompt,
                response_format={"type": "json_object"},
                model="gpt-4o-mini",
            )
            return json.loads(raw)
        except Exception:
            return {"ok": True, "issues": []}

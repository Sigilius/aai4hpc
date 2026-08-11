"""
agents/synthesizer_agent.py  —  v1.1.0

SynthesizerAgent: converts raw agent results into user-facing prose.

P2P agent chain position
------------------------
  sql_agent/doc_agent → synthesizer → reflector
                      ↖               ↓
                        ← data_explorer  (DATA_INSUFFICIENCY trigger)

SIGDIAL trigger semantics
-------------------------
  VALIDATE        — synthesizer asking reflector to check the answer
  CHALLENGE       — reflector flagging a problem back to synthesizer
  DATA_INSUFFICIENCY — synthesizer asks data_explorer for column ranges/units
                       after receiving a CHALLENGE about missing units

DA types emitted
----------------
  SYNTHESIZE — formatted answer (approved by Reflector or no Reflector wired)
  CAVEAT     — formatted answer with uncertainty flags attached
"""
from __future__ import annotations

from core.message_schema import (
    A2AMessage,
    DelegationTrigger,
    DialogueActType,
    UncertaintyFlag,
)
from core.shared_log import SharedLog
from agents.base_agent import BaseAgent


_BYPASS_SENTINEL       = "\x00UNCONFIRMED\x00"
_SCHEMA_REJECT_SENTINEL = "\x00SCHEMAREJ\x00"


class SynthesizerAgent(BaseAgent):
    """
    Formats raw SQL/doc results into user-facing prose.

    Reflector loop (P2P):
      1. Send VALIDATE to Reflector.
      2. If CHALLENGE: call DataExplorer (DATA_INSUFFICIENCY) for column
         profiles to enrich the revision. Then re-send VALIDATE once more.
      3. If CONFIRM (first or second pass): use Reflector's content.

    The DataExplorer call after CHALLENGE is the SIGDIAL-measurable
    DATA_INSUFFICIENCY trigger — agents self-navigate without orchestrator.
    """

    name    = "synthesizer"
    version = "1.1.0"

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        query  = msg.metadata.get("original_query", msg.content)
        raw    = msg.content
        sql    = msg.metadata.get("sql_query", "")
        domain = msg.metadata.get("domain_info", "")
        conf   = msg.metadata.get("confidence")
        flags  = msg.metadata.get("uncertainty_flags", [])

        self._note(msg, "synthesize_start",
                   f"formatting answer for: {query[:80]}")

        answer = self._format(query, raw, sql, domain)
        self._note(msg, "synthesize_draft",
                   f"draft answer ({len(answer)} chars)")

        # ── Forward to Reflector for validation ───────────────────────────────
        reflector_bypassed = False
        schema_rejected = False
        if self.has_peer("reflector"):
            answer = await self._reflector_loop(msg, query, raw, sql, domain, answer)
            if answer.startswith(_SCHEMA_REJECT_SENTINEL):
                answer = answer[len(_SCHEMA_REJECT_SENTINEL):]
                schema_rejected = True
            elif answer.startswith(_BYPASS_SENTINEL):
                answer = answer[len(_BYPASS_SENTINEL):]
                reflector_bypassed = True

        unc_flags = [UncertaintyFlag(f) for f in flags
                     if f in {e.value for e in UncertaintyFlag}]
        if reflector_bypassed:
            unc_flags.append(UncertaintyFlag.UNCONFIRMED_REFLECTOR)
        if schema_rejected:
            unc_flags.append(UncertaintyFlag.NOT_FOUND)
        da = DialogueActType.CAVEAT if unc_flags else DialogueActType.SYNTHESIZE
        return self.reply(
            msg, da,
            content    = answer,
            confidence = conf,
            flags      = unc_flags,
        )

    # ── Reflector loop ────────────────────────────────────────────────────────

    async def _reflector_loop(
        self,
        msg: A2AMessage,
        query: str,
        raw: str,
        sql: str,
        domain: str,
        answer: str,
    ) -> str:
        """
        VALIDATE → (CONFIRM | CHALLENGE → DATA_INSUFFICIENCY → VALIDATE → CONFIRM)

        At most two Reflector calls per session to prevent infinite loops.
        The DATA_INSUFFICIENCY trigger fires on the DataExplorer call after
        the first CHALLENGE, giving the paper its second trigger type.
        """
        # ── First Reflector check (VALIDATE) ──────────────────────────────────
        refl_req = self.make_request(
            "reflector", msg,
            DialogueActType.VALIDATE,      # asks Reflector to verify — not SYNTHESIZE
            answer,
            metadata={"original_query": query, "raw_result": raw},
        )
        self._note(msg, "reflector_dispatch",
                   "sending VALIDATE to reflector (first check)")
        refl_resp = await self.ask_peer("reflector", refl_req)

        if refl_resp.da_type != DialogueActType.CHALLENGE:
            # CONFIRM — accept
            self._note(msg, "reflector_confirm", "answer approved on first check")
            return refl_resp.content

        # ── CHALLENGE received → probe DataExplorer (DATA_INSUFFICIENCY) ──────
        issues = refl_resp.metadata.get("issues", [])
        self._note(msg, "reflector_challenge",
                   f"CHALLENGE received: {issues}")

        unit_profile = ""
        if self.has_peer("data_explorer"):
            # Pass query + reviewer issues so DataExplorer can infer relevant columns
            probe_content = f"{query}\nReviewer issues: {'; '.join(issues)}"
            prof_req = self.make_request(
                "data_explorer", msg,
                DialogueActType.REQUEST,
                probe_content,
                trigger  = DelegationTrigger.DATA_INSUFFICIENCY,  # SIGDIAL key trigger
                metadata = {"full_categorical_scan": False},
            )
            self._note(msg, "data_explorer_dispatch",
                       "DATA_INSUFFICIENCY → probing column ranges/units via data_explorer")
            prof_resp = await self.ask_peer("data_explorer", prof_req)
            unit_profile = prof_resp.content
            self._note(msg, "data_explorer_resp",
                       f"received {len(unit_profile)} chars of column profiles")
            if prof_resp.da_type == DialogueActType.REJECT:
                # Schema-absent query: data_explorer found no relevant columns.
                # Revise once for good form, then short-circuit — a second
                # reflector call would only repeat the same CHALLENGE.
                answer = self._revise(query, raw, sql, domain, answer, issues, "")
                self._note(msg, "schema_reject_shortcircuit",
                           "data_explorer REJECT — schema-absent query; confirming without second VALIDATE")
                return _SCHEMA_REJECT_SENTINEL + answer

        # ── Revise with the issues + any column profile data ──────────────────
        answer = self._revise(query, raw, sql, domain, answer, issues, unit_profile)
        self._note(msg, "synthesis_revised",
                   f"revised answer ({len(answer)} chars)")

        # ── Second Reflector check (VALIDATE) — final pass ────────────────────
        refl_req2 = self.make_request(
            "reflector", msg,
            DialogueActType.VALIDATE,
            answer,
            metadata={"original_query": query, "raw_result": raw},
        )
        self._note(msg, "reflector_redispatch",
                   "re-submitting revised answer to reflector (second check)")
        refl_resp2 = await self.ask_peer("reflector", refl_req2)
        self._note(msg, "reflector_final",
                   f"final verdict: {refl_resp2.da_type.value}")
        if refl_resp2.da_type == DialogueActType.CHALLENGE:
            # Reflector issued two CHALLENGEs and still not satisfied — bypass
            # with the best-effort revised answer, but mark it as unconfirmed.
            self._note(msg, "reflector_bypass",
                       "second CHALLENGE unresolved — bypassing reflector (UNCONFIRMED)")
            return _BYPASS_SENTINEL + refl_resp2.content
        return refl_resp2.content

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format(self, query: str, raw: str, sql: str, domain: str) -> str:
        prompt = (
            "You are an HPC analyst assistant for the Fugaku supercomputer.\n\n"
            f"The user asked: {query}\n\n"
            + (f"SQL query used:\n{sql}\n\n" if sql else "")
            + f"Raw result:\n{raw}\n\n"
            + (f"Domain context:\n{domain}\n\n" if domain else "")
            + "Units context:\n"
            "- Power values (avgpcon, minpcon, maxpcon) are in Watts\n"
            "- duration is in seconds\n"
            "- nnumr / nnuma = node counts (not CPU cores)\n"
            "- econ = total energy in Joules\n"
            "- uctmut = user CPU time in MILLISECONDS (not seconds)\n"
            "- cnumut = number of CPU cores used (a count, not a percentage)\n\n"
            "Write a clear, direct answer to the user's question using the result.\n"
            "IMPORTANT rules:\n"
            "1. If the user asked multiple questions, address EVERY one — do not omit any part.\n"
            "2. If the raw result contains prediction output (lines with 'Failure probability:', "
            "'Risk level:', 'Expected runtime:', 'Expected energy:'), include those values "
            "directly in the answer — they are the answer to the prediction question.\n"
            "3. Include units. Do not mention SQL. Do not add commentary beyond what the data shows.\n"
            "4. CRITICAL — Schema-absent data: The Fugaku dataset does NOT contain billing costs, "
            "electricity rates, node temperatures, CPU temperatures, OS kernel versions, "
            "network latency, inter-node bandwidth, carbon footprint, or GPU metrics. "
            "If any part of the question asks for these, explicitly state: "
            "'[topic] is not available in the Fugaku dataset' — do NOT invent or estimate values.\n"
            "5. CRITICAL — Historical rate vs predictor output: When the raw result contains BOTH "
            "a SQL-derived historical failure rate (e.g. '25.93% of 64-node jobs failed') AND "
            "a predictor failure probability (e.g. 'Failure probability: 0.6%'), these are "
            "DIFFERENT numbers answering DIFFERENT sub-questions. NEVER report the predictor "
            "probability as the historical rate or vice versa. Label each clearly.\n"
            "6. CRITICAL — Multiple SQL data sources: When the raw result contains BOTH a "
            "'Database-derived prediction parameters' section AND a 'Historical context from "
            "database' section, treat them differently. The 'Database-derived prediction "
            "parameters' section provides input values for the predictor (avg node count, avg "
            "walltime) and its failure rate reflects a BROADER filter that may not match the "
            "user's specific historical question. The 'Historical context from database' section "
            "contains the SPECIFIC historical rate that directly answers the user's historical "
            "sub-question (with the exact filters the user asked for). When answering 'what is "
            "the historical failure rate for X' sub-questions, PREFER the value from "
            "'Historical context from database' over any rate in 'Database-derived prediction "
            "parameters'."
        )
        return self._llm(prompt)

    # ── Revision after Reflector CHALLENGE ───────────────────────────────────

    def _revise(
        self,
        query: str,
        raw: str,
        sql: str,
        domain: str,
        current_answer: str,
        issues: list,
        unit_profile: str = "",
    ) -> str:
        issues_str = "\n".join(f"- {i}" for i in issues)
        profile_block = (
            f"\nColumn profiles from DataExplorer (use for units/ranges):\n{unit_profile}\n"
            if unit_profile else ""
        )
        prompt = (
            "You are an HPC analyst assistant for the Fugaku supercomputer.\n\n"
            f"The user asked: {query}\n\n"
            f"Your previous answer:\n{current_answer}\n\n"
            f"A reviewer flagged these issues:\n{issues_str}\n"
            f"{profile_block}\n"
            + (f"Original data:\n{raw}\n\n" if raw else "")
            + "Units: Power in Watts, duration in seconds, energy in Joules, "
            "node counts are nodes (not cores).\n\n"
            "Rewrite the answer to fix every flagged issue. Be concise and factual."
        )
        return self._llm(prompt)

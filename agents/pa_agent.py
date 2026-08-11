"""
agents/pa_agent.py  —  v1.0.0

PAAgent: Predictive Analytics agent for Fugaku job risk assessment.

P2P agent chain position
------------------------
  gateway → pa_agent → sql_agent   (DATA_INSUFFICIENCY: needs historical user/system stats)
                     → doc_agent   (KNOWLEDGE_GAP: needs system policy/procedure context)
                     → synthesizer (final formatting + reflector validation)

SIGDIAL trigger semantics
-------------------------
  DATA_INSUFFICIENCY — pa_agent → sql_agent when the query asks for historical
                       context alongside the prediction (e.g. user's past job count,
                       system-wide average stats to contextualise the prediction)
  KNOWLEDGE_GAP      — pa_agent → doc_agent when the query asks for policy or
                       procedural context (e.g. walltime limits, job script guidance,
                       "what does the doc say about failures like this")

DA types emitted
----------------
  INFORM  — prediction result forwarded to synthesizer (or returned directly)
  CAVEAT  — prediction with uncertainty flags (new user cold-start, etc.)
  REJECT  — parameter extraction failed or all prediction attempts errored
"""
from __future__ import annotations

import json
import sys
import os

from core.message_schema import (
    A2AMessage,
    DelegationTrigger,
    DialogueActType,
    UncertaintyFlag,
)
from agents.base_agent import BaseAgent

# Make analytics/ importable from hpc/
_ANALYTICS = os.path.join(os.path.dirname(__file__), "..", "analytics")
if _ANALYTICS not in sys.path:
    sys.path.insert(0, os.path.abspath(_ANALYTICS))


class PAAgent(BaseAgent):
    """
    Predictive Analytics Agent.

    Flow:
      1. Extract all job specs from the query (LLM, handles multi-job queries).
      2. Run Predictor for each spec — 4-stage ML pipeline.
      3. If query needs historical SQL context → DATA_INSUFFICIENCY → sql_agent.
      4. If query needs doc policy context    → KNOWLEDGE_GAP      → doc_agent.
      5. Compose prediction + optional enrichment into a single raw result block.
      6. Forward to synthesizer if wired, otherwise reply INFORM directly.
    """

    name    = "pa_agent"
    version = "1.0.0"

    def __init__(self, log, verbose: bool = False) -> None:
        super().__init__(log, verbose)
        self._predictor = None      # lazy-load: Predictor loads ~5 pkl files on first call

    # ── Lazy predictor ─────────────────────────────────────────────────────────

    @property
    def predictor(self):
        if self._predictor is None:
            from predict import Predictor
            self._predictor = Predictor()
        return self._predictor

    # ── Main handle ───────────────────────────────────────────────────────────

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        query = msg.content
        # Use full original query when delegated from doc_agent
        original_query = msg.metadata.get("original_query", query)
        self._note(msg, "pa_start", f"query={query[:120]}")

        # Incoming doc_context when doc_agent delegates forward to us
        incoming_doc_context = msg.metadata.get("doc_context", "")
        if incoming_doc_context:
            self._note(msg, "doc_context_received",
                       f"doc_agent delegated forward — doc_context={len(incoming_doc_context)} chars")

        # ── Classify needs upfront (one LLM call each, cheap) ─────────────────
        needs_sql_params  = self.has_peer("sql_agent") and self._needs_sql_input_params(query)
        needs_sql_compare = self.has_peer("sql_agent") and self._needs_sql_comparison(query)
        # context enrichment: skip only if a sql_compare call already covers all SQL needs
        needs_sql_enrich  = (not needs_sql_compare
                             and self.has_peer("sql_agent") and self._needs_sql_context(query))
        # Guard: skip doc sub-call if doc already delegated to us
        needs_doc         = (not incoming_doc_context
                             and self.has_peer("doc_agent") and self._needs_doc_context(query))

        self._note(msg, "pa_classify",
                   f"sql_params={needs_sql_params} sql_compare={needs_sql_compare} "
                   f"sql_enrich={needs_sql_enrich} doc={needs_doc}")

        # ── Phase 1: pre-prediction SQL — get job parameters from DB ──────────
        sql_params_context = ""
        sql_derived_overrides: dict = {}
        all_flags: list[UncertaintyFlag] = []
        if needs_sql_params:
            sql_input_q = self._extract_sql_input_query(query)
            self._note(msg, "sql_params_dispatch",
                       f"DATA_INSUFFICIENCY (pre-predict) → sql_agent: {sql_input_q[:80]}")
            sql_req = self.make_request(
                "sql_agent", msg,
                DialogueActType.REQUEST,
                sql_input_q,
                trigger  = DelegationTrigger.DATA_INSUFFICIENCY,
                metadata = {"original_query": query, "subquery_for": "pa_input_params"},
            )
            sql_resp = await self.ask_peer("sql_agent", sql_req)
            if sql_resp.da_type not in (DialogueActType.REJECT,):
                sql_params_context = sql_resp.content
                sql_derived_overrides = self._parse_sql_params(query, sql_params_context)
                self._note(msg, "sql_params_parsed",
                           f"overrides from SQL: {sql_derived_overrides}")
            else:
                # SQL rejected: required input data unavailable — prediction uses defaults
                all_flags.append(UncertaintyFlag.PARTIALLY_FOUND)
                self._note(msg, "sql_params_rejected",
                           "sql_agent REJECT → prediction will use default parameters")

        # ── Step 2: extract job specs, override with SQL-derived params ────────
        specs = self._extract_all_job_specs(query)
        if sql_derived_overrides:
            for spec in specs:
                spec.update(sql_derived_overrides)
        self._note(msg, "param_extract",
                   f"extracted {len(specs)} job spec(s): "
                   + ", ".join(s.get("label", "?") for s in specs)
                   + (" [SQL-overridden]" if sql_derived_overrides else ""))

        # ── Step 3: run predictor for each spec ───────────────────────────────
        prediction_blocks: list[str] = []
        raw_results: list[dict] = []

        for spec in specs:
            label = spec.pop("label", "Job")
            job = {k: v for k, v in spec.items()
                   if k in ("nnumr", "elpl", "pclass", "usr", "jnam", "qdt")}
            job["nnumr"] = int(job.get("nnumr", 1))
            job["elpl"]  = int(job.get("elpl", 3600))
            job["pclass"] = str(job.get("pclass", "compute-bound"))

            self._note(msg, "predict_call",
                       f"{label}: nnumr={job['nnumr']} elpl={job['elpl']} "
                       f"pclass={job['pclass']} usr={job.get('usr','?')}"
                       + (" [params from SQL]" if sql_derived_overrides else ""))
            try:
                result = self.predictor.predict(job)
                raw_results.append({"label": label, "job": job, "result": result})
                block = self._format_prediction(label, result)
                prediction_blocks.append(block)

                usr    = job.get("usr", "__unknown__")
                p_fail = result["p_fail"]

                # CONFIDENCE_LOW: fire only when a specific user was requested but
                # is NOT in the model's training data — not when no user was given.
                # No-user queries use global rates which is normal/expected behavior.
                usr_was_specified = usr not in (None, "", "__unknown__")
                usr_is_known      = usr in self.predictor.user_stats
                if usr_was_specified and not usr_is_known:
                    all_flags.append(UncertaintyFlag.CONFIDENCE_LOW)

                # Low-sample check: count similar jobs and their historical fail rate.
                # Model predictions based on ≤5 data points are unreliable regardless
                # of what p_fail says — the model may be confidently wrong.
                _LOW_SAMPLE_N      = 5     # ≤ 5 similar historical jobs
                _HIST_FAIL_THRESH  = 0.30  # ≥ 30% historical failure rate
                n_similar, hist_fail_rate = self._count_similar_jobs(job)

                if (n_similar >= 0                           # DB reachable
                        and n_similar <= _LOW_SAMPLE_N
                        and hist_fail_rate >= _HIST_FAIL_THRESH):
                    all_flags.append(UncertaintyFlag.LOW_SAMPLE)
                    self._note(msg, "low_sample_caution",
                               f"n_similar={n_similar} ≤ {_LOW_SAMPLE_N}, "
                               f"hist_fail={hist_fail_rate:.1%} ≥ {_HIST_FAIL_THRESH:.0%}, "
                               f"model_p_fail={p_fail:.1%} → LOW_SAMPLE flag set")
                    result["_n_similar"]      = n_similar
                    result["_hist_fail_rate"] = hist_fail_rate

                self._note(msg, "predict_done",
                           f"{label}: risk={result['risk_level']} "
                           f"p_fail={result['p_fail']:.3f} "
                           f"n_similar={n_similar}")
            except Exception as exc:
                self._note(msg, "predict_error", f"{label}: {exc}")
                prediction_blocks.append(f"**{label}:** Prediction failed — {exc}")

        prediction_summary = "\n\n".join(prediction_blocks)

        # ── Phase 4: post-prediction SQL comparison (gated on risk level) ─────
        sql_comparison = ""
        if needs_sql_compare and raw_results:
            # Always fire — the query explicitly asked for a cross-check.
            # Pass predicted p_fail as context so SQL and synthesizer can frame the comparison.
            p_fail_primary = raw_results[0]["result"]["p_fail"]
            risk_primary   = raw_results[0]["result"]["risk_level"]
            comp_q = self._extract_sql_comparison_query(query, p_fail_primary)
            self._note(msg, "sql_compare_dispatch",
                       f"DATA_INSUFFICIENCY (post-predict, p_fail={p_fail_primary:.3f}) "
                       f"→ sql_agent: {comp_q[:80]}")
            sql_req = self.make_request(
                "sql_agent", msg,
                DialogueActType.REQUEST,
                comp_q,
                trigger  = DelegationTrigger.DATA_INSUFFICIENCY,
                metadata = {
                    "original_query":   query,
                    "subquery_for":     "pa_comparison",
                    "predicted_p_fail": str(round(p_fail_primary, 4)),
                    "predicted_risk":   risk_primary,
                },
            )
            sql_resp = await self.ask_peer("sql_agent", sql_req)
            if sql_resp.da_type not in (DialogueActType.REJECT,):
                sql_comparison = sql_resp.content
                self._note(msg, "sql_compare_resp",
                           f"received {len(sql_comparison)} chars for comparison "
                           f"(model said p_fail={p_fail_primary:.3f})")

        # ── Step 5: optional SQL context enrichment (existing behaviour) ──────
        sql_context = ""
        if needs_sql_enrich:
            sql_query = self._extract_sql_subquestion(query)
            self._note(msg, "sql_dispatch",
                       f"DATA_INSUFFICIENCY → sql_agent: {sql_query[:80]}")
            sql_req = self.make_request(
                "sql_agent", msg,
                DialogueActType.REQUEST,
                sql_query,
                trigger  = DelegationTrigger.DATA_INSUFFICIENCY,
                metadata = {"original_query": query, "subquery_for": "pa_enrichment"},
            )
            sql_resp = await self.ask_peer("sql_agent", sql_req)
            if sql_resp.da_type not in (DialogueActType.REJECT,):
                sql_context = sql_resp.content
                self._note(msg, "sql_resp",
                           f"received {len(sql_context)} chars from sql_agent")

        # ── Step 6: doc enrichment — use forwarded context or sub-call ──────────
        doc_context = incoming_doc_context  # already set if doc delegated to us
        if needs_doc:
            doc_query = self._extract_doc_subquestion(query)
            self._note(msg, "doc_dispatch",
                       f"KNOWLEDGE_GAP → doc_agent: {doc_query[:80]}")
            doc_req = self.make_request(
                "doc_agent", msg,
                DialogueActType.REQUEST,
                doc_query,
                trigger  = DelegationTrigger.KNOWLEDGE_GAP,
                metadata = {"original_query": query, "subquery_for": "pa_enrichment"},
            )
            doc_resp = await self.ask_peer("doc_agent", doc_req)
            if doc_resp.da_type not in (DialogueActType.REJECT,):
                doc_context = doc_resp.content
                self._note(msg, "doc_resp",
                           f"received {len(doc_context)} chars from doc_agent")

        # ── Step 7: compose final raw result block ────────────────────────────
        raw_parts = [prediction_summary]
        if sql_params_context:
            raw_parts.append(
                f"Database-derived prediction parameters:\n{sql_params_context}\n"
                f"(Parameters used: {sql_derived_overrides})"
            )
        if sql_comparison:
            raw_parts.append(f"Historical cross-check from database:\n{sql_comparison}")
        if sql_context:
            raw_parts.append(f"Historical context from database:\n{sql_context}")
        if doc_context:
            raw_parts.append(f"System documentation context:\n{doc_context}")
        raw_result = "\n\n---\n\n".join(raw_parts)

        self._note(msg, "pa_compose",
                   f"raw_result={len(raw_result)} chars "
                   f"(sql_params={bool(sql_params_context)} "
                   f"sql_compare={bool(sql_comparison)} "
                   f"sql_enrich={bool(sql_context)} "
                   f"doc={bool(doc_context)})")

        # ── Step 8: forward to synthesizer or reply directly ──────────────────
        conf = 0.90 if not all_flags else 0.70

        # When sql_agent delegated to us, it passes sql_context in metadata.
        # Merge it into the combined content so synthesizer has both sql + prediction.
        sql_context_from_caller = msg.metadata.get("sql_context", "")
        combined_for_synth = raw_result
        if sql_context_from_caller:
            combined_for_synth = (
                f"SQL result:\n{sql_context_from_caller}\n\n"
                f"Prediction result:\n{raw_result}"
            )
            self._note(msg, "sql_pa_combined",
                       f"merged sql ({len(sql_context_from_caller)} chars) + prediction")

        if self.has_peer("synthesizer"):
            synth_req = self.make_request(
                "synthesizer", msg,
                DialogueActType.INFORM,
                combined_for_synth,
                metadata={
                    "original_query":    original_query,
                    "sql_query":         "",
                    "domain_info":       doc_context or sql_context or sql_params_context,
                    "confidence":        conf,
                    "uncertainty_flags": [f.value for f in all_flags],
                    "prediction_count":  len(raw_results),
                    "sql_params_used":   bool(sql_derived_overrides),
                    "sql_comparison":    bool(sql_comparison),
                    "low_sample_caution": self._build_low_sample_caution(
                        raw_results, all_flags
                    ),
                },
            )
            self._note(msg, "synthesizer_dispatch",
                       "forwarding prediction result to synthesizer")
            return await self.ask_peer("synthesizer", synth_req)

        da = DialogueActType.CAVEAT if all_flags else DialogueActType.INFORM
        return self.reply(
            msg, da,
            content    = raw_result,
            confidence = conf,
            flags      = all_flags,
        )

    # ── Job parameter extraction ───────────────────────────────────────────────

    def _extract_all_job_specs(self, query: str) -> list[dict]:
        """
        LLM extracts all distinct job configurations from the query.
        Handles single or multi-job queries. Always returns at least one spec.
        """
        prompt = f"""Extract ALL distinct HPC job configurations mentioned in this query.
A distinct job = any combination of node count, walltime, job class, OR submission time that needs separate evaluation.

Return a JSON object with key "jobs" containing an array. Return at least ONE job — never return empty.

Rules:
- nnumr default=1, elpl default=3600, pclass default="compute-bound"
- Convert time: "2 hours"→7200, "24hr"→86400, "30min"→1800, "overnight"→28800
- Preserve usr if mentioned in the query (e.g. "usr_1234" or "user_1234")
- label: short human description of the job (e.g. "512-node compute-bound 24h")
- If a specific submission day/time is mentioned, include it as qdt in ISO format "YYYY-MM-DD HH:MM:SS".
  Use 2024 as the reference year. Day mapping: Monday=2024-01-01, Tuesday=2024-01-02,
  Wednesday=2024-01-03, Thursday=2024-01-04, Friday=2024-01-05, Saturday=2024-01-06, Sunday=2024-01-07.
  If multiple submission windows are compared, create one job entry per window with its own qdt.
- Output ONLY the JSON object.

Example output (two submission windows compared):
{{"jobs": [
  {{"label": "192-node CB 2h Friday 9am", "nnumr": 192, "elpl": 7200, "pclass": "compute-bound", "usr": "usr_2111", "qdt": "2024-01-05 09:00:00"}},
  {{"label": "192-node CB 2h Sunday midnight", "nnumr": 192, "elpl": 7200, "pclass": "compute-bound", "usr": "usr_2111", "qdt": "2024-01-07 00:00:00"}}
]}}

Query: {query}"""

        try:
            raw = self._llm(
                prompt,
                response_format={"type": "json_object"},
                model="gpt-4o-mini",
            )
            parsed = json.loads(raw)
            specs = parsed.get("jobs", [])
            if not specs:
                raise ValueError("empty jobs array")
            return specs
        except Exception as exc:
            self._note_plain(f"param_extract_fallback: {exc}")
            # fallback: minimal single spec
            return [{"label": "Job", "nnumr": 1, "elpl": 3600, "pclass": "compute-bound"}]

    def _note_plain(self, text: str) -> None:
        """Note without a msg object — used in sync helpers."""
        pass   # silent fallback; _note requires A2AMessage

    # ── Context need gates ─────────────────────────────────────────────────────

    def _needs_sql_input_params(self, query: str) -> bool:
        """True when SQL must run BEFORE prediction to supply the job parameters themselves."""
        prompt = (
            "Decide if this HPC prediction query requires querying the DATABASE FIRST "
            "to obtain the actual job parameters (node count, walltime) that will be "
            "used AS INPUTS to the prediction model.\n\n"
            "Answer YES only if ALL of these are true:\n"
            "1. The query asks to use database-computed values (averages, medians) as "
            "   the prediction inputs — e.g. 'use the average node count of failed jobs "
            "   as the parameters for prediction', 'predict using those exact averages'\n"
            "2. The prediction cannot proceed until those database values are known\n\n"
            "Answer NO if:\n"
            "- The job parameters are explicitly stated in the query (e.g. '512 nodes, 24h')\n"
            "- The database is only needed for historical context or comparison after prediction\n"
            "- The query just asks for enrichment or verification, not parameter derivation\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _needs_sql_comparison(self, query: str) -> bool:
        """True when the query asks to cross-check / validate the prediction against historical data."""
        prompt = (
            "Decide if this prediction query also asks to CROSS-CHECK or COMPARE "
            "the model's prediction against ACTUAL HISTORICAL DATA from a database.\n\n"
            "Answer YES only if the query explicitly asks:\n"
            "- To compare the predicted failure rate/risk against observed historical rates\n"
            "- Whether the model's prediction agrees with real data\n"
            "- To validate or verify the prediction using actual job history\n"
            "- 'Do the model and data agree?', 'cross-check against reality', etc.\n\n"
            "Answer NO if:\n"
            "- The query only asks for a prediction (no comparison requested)\n"
            "- The database is only for enrichment context, not explicit comparison\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _needs_sql_context(self, query: str) -> bool:
        prompt = (
            "Decide if this prediction query also needs HISTORICAL DATABASE STATISTICS "
            "from a Fugaku HPC job database to give a complete answer.\n\n"
            "Answer YES only if the query explicitly asks for:\n"
            "- Historical counts, averages, or stats (e.g. 'how many jobs like this', "
            "  'what percentage fail', 'average runtime for similar jobs')\n"
            "- A specific user's past job statistics (e.g. 'how many jobs has usr_1234 submitted')\n"
            "- System-wide trends or comparisons alongside the prediction\n\n"
            "Answer NO if the query is ONLY about:\n"
            "- Predicting failure probability, runtime, or energy for a specific job\n"
            "- Risk assessment or recommendation for a job\n"
            "- Explaining model output or risk factors\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _needs_doc_context(self, query: str) -> bool:
        prompt = (
            "Decide if this prediction query also needs SYSTEM DOCUMENTATION context "
            "to give a complete answer.\n\n"
            "Answer YES only if the query also asks about:\n"
            "- What the documentation says about failure causes or risk factors\n"
            "- Walltime limits, scheduling policies, or submission rules\n"
            "- How to configure or modify the job (node shape, compiler flags)\n"
            "- What happens when a job fails or exceeds resource limits\n"
            "- System-level procedures alongside the prediction\n\n"
            "Answer NO if the query is ONLY about:\n"
            "- Predicting failure probability, runtime, or energy\n"
            "- Risk scores or model-based recommendations\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    # ── Sub-question extraction ────────────────────────────────────────────────

    def _extract_sql_subquestion(self, query: str) -> str:
        prompt = (
            "You are extracting a database sub-question from a mixed HPC query.\n\n"
            "The Fugaku job database contains historical job telemetry: "
            "counts, averages, user stats, node usage, power, failure rates, etc.\n\n"
            "From the query below, extract the part that can be answered by querying "
            "the Fugaku job database (counts, totals, averages, user statistics).\n"
            "Return a single specific SQL-answerable question. "
            "If the query asks about a specific user (e.g. usr_1898), "
            "include that user ID in the sub-question.\n"
            "If the query mentions a specific node count (e.g. '192 nodes', '48-node'), "
            "include that node count as a constraint in the sub-question.\n"
            "CRITICAL: Do NOT include walltime or job duration as a filter — walltime in "
            "the query describes the user's planned job, not a historical data filter.\n"
            "CRITICAL: Do NOT add year or date constraints unless the query explicitly "
            "mentions a specific year (e.g., 'in 2022', 'from 2023 to 2024'). "
            "For queries about all-time or global rates, query ALL years.\n"
            "Return ONLY the question, no explanation.\n\n"
            f"Query: {query}\n\n"
            "SQL sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=80).strip()

    def _extract_sql_input_query(self, query: str) -> str:
        """Extract what to ask SQL so we can derive the prediction input parameters."""
        prompt = (
            "You are extracting a database question from an HPC query where the database "
            "result will become the INPUT PARAMETERS for a prediction model.\n\n"
            "The Fugaku job database has columns: nnumr (node count), elpl (walltime in seconds), "
            "pclass (job class), usr (user ID), outcome (FAILED/SUCCEEDED), year.\n\n"
            "From the query below, extract a precise SQL-answerable question that will return "
            "the numeric values needed as prediction inputs (typically AVG of nnumr and/or elpl "
            "for a specific job subset, OR failure rate for a specific node/class/user combo).\n"
            "Be specific: include filters for node count (nnumr), job class, user if mentioned.\n"
            "CRITICAL: Do NOT include walltime or job duration (elpl) as a filter — "
            "walltime in the query describes the user's planned job, not a historical filter.\n"
            "CRITICAL: Do NOT add year or date constraints unless the query explicitly "
            "mentions a specific year (e.g., 'in 2022', 'from 2023 to 2024'). "
            "For general rate queries with no year mentioned, query ALL years.\n"
            "Return ONLY the question, no explanation.\n\n"
            f"Query: {query}\n\n"
            "SQL input-parameter question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=100).strip()

    def _parse_sql_params(self, query: str, sql_response: str) -> dict:
        """
        Parse SQL prose response into numeric prediction overrides.
        Returns a dict with any of: nnumr, elpl, pclass — only what was found.
        """
        prompt = (
            "A SQL query returned the following result. Extract numeric values for "
            "HPC job prediction parameters.\n\n"
            "Return a JSON object with ONLY the keys that were found in the result:\n"
            "  nnumr  — integer node count (round to nearest integer)\n"
            "  elpl   — integer walltime in seconds (round to nearest integer)\n"
            "  pclass — string job class (e.g. 'compute-bound', 'memory-bound')\n\n"
            "Column name mapping — ANY of these map to nnumr (node count):\n"
            "  nnumr, avg_nnumr, avg(nnumr), avg_node_count, node_count, nodes, "
            "  average_nodes, avg_nodes\n"
            "Column name mapping — ANY of these map to elpl (walltime in seconds):\n"
            "  elpl, avg_elpl, avg(elpl), avg_walltime, walltime, avg_walltime_seconds, "
            "  avg_duration, duration\n\n"
            "Rules:\n"
            "- Only include a key if the SQL result contains a clear numeric value for it\n"
            "- If the result is NULL or unavailable, omit that key\n"
            "- Convert times to seconds if needed (1 hour = 3600)\n"
            "- Output ONLY the JSON object, no explanation\n\n"
            f"Original question: {query}\n\n"
            f"SQL result:\n{sql_response[:800]}\n\n"
            "JSON:"
        )
        try:
            raw = self._llm(prompt, response_format={"type": "json_object"}, model="gpt-4o-mini")
            parsed = json.loads(raw)
            overrides = {}
            if "nnumr" in parsed and parsed["nnumr"] is not None:
                overrides["nnumr"] = int(parsed["nnumr"])
            if "elpl" in parsed and parsed["elpl"] is not None:
                overrides["elpl"] = int(parsed["elpl"])
            if "pclass" in parsed and parsed["pclass"]:
                overrides["pclass"] = str(parsed["pclass"])
            return overrides
        except Exception:
            return {}

    def _extract_sql_comparison_query(self, query: str, p_fail: float) -> str:
        """Extract a SQL question for cross-checking the prediction against historical data."""
        prompt = (
            "You are extracting a database question for validating an ML prediction.\n\n"
            "The prediction model estimated a failure probability of "
            f"{p_fail:.4f} ({100*p_fail:.2f}%).\n\n"
            "The Fugaku job database has historical job records with outcomes (FAILED/SUCCEEDED), "
            "node counts (nnumr), walltimes (elpl), job classes (pclass), and years.\n\n"
            "From the query below, extract a precise SQL-answerable question that retrieves "
            "the ACTUAL HISTORICAL failure rate for comparable jobs — so the prediction can be "
            "cross-checked against reality.\n"
            "Be specific: include filters for node range, job class, year if mentioned.\n"
            "Return ONLY the question, no explanation.\n\n"
            f"Query: {query}\n\n"
            "SQL comparison question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=100).strip()

    def _extract_doc_subquestion(self, query: str) -> str:
        prompt = (
            "You are extracting a documentation sub-question from a mixed HPC query.\n\n"
            "The Fugaku documentation covers system commands, policies, job scripts, "
            "configuration, procedures, and technical guidance.\n\n"
            "From the query below, extract ONLY the part that asks about "
            "system documentation, commands, policies, or procedures.\n"
            "Return a single specific documentable question. No explanation.\n\n"
            f"Query: {query}\n\n"
            "Documentation sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=80).strip()

    # ── Sample count + caution ─────────────────────────────────────────────────

    @staticmethod
    def _count_similar_jobs(job: dict) -> tuple[int, float]:
        """
        Count historical Fugaku jobs with the same pclass and exact nnumr,
        and compute their historical failure rate.

        Returns (n_similar, hist_fail_rate).
        Returns (-1, 0.0) if the DB is unreachable.
        """
        try:
            from shared.db import get_connection
            pclass = str(job.get("pclass", ""))
            nnumr  = int(job.get("nnumr", 1))
            db     = get_connection()
            row    = db.execute(
                "SELECT COUNT(*), "
                "AVG(CASE WHEN \"exit state\" = 'failed' THEN 1.0 ELSE 0.0 END) "
                "FROM jobs WHERE pclass = ? AND nnumr = ?",
                [pclass, nnumr],
            ).fetchone()
            db.close()
            n    = int(row[0])
            rate = float(row[1]) if row[1] is not None else 0.0
            return n, rate
        except Exception:
            return -1, 0.0

    @staticmethod
    def _build_low_sample_caution(
        raw_results: list[dict],
        all_flags: list,
    ) -> str:
        """
        Build a specific caution string when LOW_SAMPLE flag is set.
        Returns empty string if the flag is not present.
        """
        if UncertaintyFlag.LOW_SAMPLE not in all_flags:
            return ""
        parts = []
        for entry in raw_results:
            result  = entry["result"]
            n_similar = result.get("_n_similar")
            p_fail  = result["p_fail"]
            label   = entry.get("label", "This job")
            hist_fail = result.get("_hist_fail_rate")
            if n_similar is not None and n_similar <= 5 and hist_fail is not None:
                hist_str = f"{100*hist_fail:.0f}% historically failed"
                parts.append(
                    f"{label}: only {n_similar} similar job(s) exist in the Fugaku dataset "
                    f"({hist_str}). "
                    f"The model's {100*p_fail:.1f}% failure estimate is based on very "
                    f"limited evidence — proceed carefully and consider running a smaller "
                    f"test first."
                )
        return " | ".join(parts) if parts else (
            "This prediction is based on very limited historical data for this "
            "configuration. The failure estimate may be unreliable — proceed with caution."
        )

    @staticmethod
    def _format_prediction(label: str, result: dict) -> str:
        risk    = result["risk_level"]
        p_fail  = result["p_fail"]
        runtime = result["expected_runtime"]
        energy  = result["expected_energy"]
        fail_t  = result["fail_type_if_fails"]
        wasted  = result["wasted_node_hrs_if_slow"]
        reasons = result.get("top_reasons", [])
        raw     = result.get("_raw", {})

        # Numeric values for synthesizer to work with
        exp_dur_s    = raw.get("exp_dur_s", 0)
        exp_energy_j = raw.get("exp_energy_j", 0)
        wasted_nh    = raw.get("wasted_nh", 0)

        lines = [
            f"**{label}**",
            f"Risk level: {risk}",
            f"Failure probability: {p_fail:.4f} ({100*p_fail:.2f}%)",
            f"Expected runtime (if successful): {runtime} ({exp_dur_s:.1f} seconds)",
            f"Expected energy consumption: {energy} ({exp_energy_j:.0f} joules)",
            f"If it fails: {fail_t}",
            f"Expected wasted node-hours on slow failure: {wasted} ({wasted_nh:.2f} node-hours)",
            "",
            "Risk factors:",
        ]
        for r in reasons:
            lines.append(f"  - {r}")

        return "\n".join(lines)

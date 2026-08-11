"""
agents/sql_agent.py  —  v2.0.0

SQLAgent: structured data retrieval for the Fugaku HPC DuckDB.

P2P agent chain position
------------------------
  gateway → sql_agent → data_explorer   (sub-agent: profiling)
                     → synthesizer_agent (next in chain: formatting)

SIGDIAL 2026 — Session-Log Injection
-------------------------------------
1. Read SharedLog BEFORE SQL generation — injects any prior INFORM/CAVEAT
   from DataExplorer into the SQL prompt without bespoke forwarding.
2. RE-READ after DataExplorer call — column profiles are now in the log;
   the updated context reaches the SQL prompt automatically.
3. After producing a result, forward to SynthesizerAgent if wired.
   If no synthesizer is registered, reply INFORM directly (backward compat).

Every decision is recorded via _note() for measurable SIGDIAL analysis.

DA types emitted
----------------
  INFORM  — raw result forwarded to synthesizer (or returned directly)
  REJECT  — SQL generation failed or all retries exhausted
"""
from __future__ import annotations

from core.message_schema import A2AMessage, DelegationTrigger, DialogueActType, UncertaintyFlag
from core.shared_log import SharedLog
from agents.base_agent import BaseAgent
from shared.db import get_connection, get_schema_str
from shared.schema import FUGAKU_SCHEMA_NOTES
import config


GROUP_BY_KEYWORDS = {
    "each", "per", "by", "group", "breakdown", "distribution",
    "classes", "types", "category", "categories", "kind", "kinds",
}


class SQLAgent(BaseAgent):
    """
    Database retrieval agent for the Fugaku HPC DuckDB.

    Receives a bare REQUEST from gateway (or any peer).
    Works entirely from msg.content — no gateway-injected sub_questions.
    Decides its own peer calls based on the query content.
    """

    name    = "sql_agent"
    version = "2.0.0"

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        """
        SIGDIAL session-log injection flow:

        1. Use msg.content as the query — no gateway decomposition needed.
        2. Read session log (first read).
        3. Gate: does this query need column profiling?
        4. If yes AND data_explorer wired → call it, then RE-READ log.
           (DataExplorer's INFORM is now in the log — injected automatically.)
        5. Generate SQL with full session context.
        6. Execute with retry loop.
        7. Forward result to synthesizer if wired, otherwise reply directly.
        """
        query = msg.content
        self._note(msg, "query_resolved", f"query={query[:120]}")

        # ── Read session log (first read) ─────────────────────────────────────
        session_ctx = self.log.format_for_llm(msg.session_id)
        self._note(msg, "session_read",
                   f"read {len(session_ctx)} chars of prior dialog")

        # ── Detect doc sub-question early to isolate the SQL portion ──────────
        # When a query mixes SQL statistics with documentation questions,
        # the doc phrases can cause the LLM to return CANNOT_GENERATE for the
        # whole query. Extract only the SQL sub-question for SQL generation.
        # Guard: if doc already delegated to us (doc_context in metadata), skip
        # re-detection to prevent circular delegation.
        needs_doc = (
            not msg.metadata.get("doc_context")
            and self.has_peer("doc_agent")
            and self._needs_doc_context(query)
        )
        needs_predict = (
            not msg.metadata.get("subquery_for")           # not already a sub-call
            and self.has_peer("pa_agent")
            and self._needs_predict_context(query)
        )
        sql_query = query
        if needs_doc or needs_predict:
            sql_query = self._extract_sql_subquestion(query)
            self._note(msg, "sql_subquery_extracted",
                       f"mixed query → SQL sub-question: {sql_query[:100]}")

        # ── Profile gate: does this query need column profiling? ──────────────
        needs_profile = self._needs_profile(sql_query)
        self._note(msg, "gate_profile", f"needs_profile={needs_profile}")

        # ── Call DataExplorer (SQL agent's own decision, not gateway's) ───────
        if needs_profile and self.has_peer("data_explorer"):
            full_scan = self._has_groupby_intent(sql_query)
            self._note(msg, "explorer_dispatch",
                       f"calling data_explorer, full_categorical_scan={full_scan}")

            req = self.make_request(
                "data_explorer",
                msg,
                DialogueActType.REQUEST,
                sql_query,
                trigger  = DelegationTrigger.SEMANTIC_AMBIGUITY,
                metadata = {"full_categorical_scan": full_scan},
            )
            explorer_resp = await self.ask_peer("data_explorer", req)
            self._note(msg, "explorer_resp",
                       f"DA={explorer_resp.da_type.value}, "
                       f"cols={explorer_resp.metadata.get('columns_profiled')}")

            # ── SIGDIAL KEY STEP: Re-read log ─────────────────────────────────
            # DataExplorer's INFORM/CAVEAT is now appended to the SharedLog.
            # Re-reading injects column profiles into the SQL prompt without
            # any bespoke message-passing — the log IS the communication channel.
            session_ctx = self.log.format_for_llm(msg.session_id)
            self._note(msg, "session_reread",
                       "updated context includes DataExplorer INFORM/CAVEAT")

        # ── Generate and execute (decompose if multi-sub-question) ──────────
        result = await self._generate_and_execute(msg, sql_query, session_ctx)

        # ── Route result ──────────────────────────────────────────────────────
        # REJECT bubbles up directly to the caller (gateway).
        if result["da_type"] == DialogueActType.REJECT:
            return self.reply(
                msg, DialogueActType.REJECT,
                content  = result["content"],
                metadata = result.get("metadata", {}),
            )

        flags = result.get("flags", [])
        da    = DialogueActType.CAVEAT if flags else DialogueActType.INFORM

        # ── Sub-call path: return raw data, skip synthesizer/reflector ────────
        # When called by another agent (pa_agent, sql_agent itself), subquery_for
        # is set in metadata. The caller wants structured data, not formatted prose.
        # Synthesizer/reflector are for user-facing output only.
        if msg.metadata.get("subquery_for"):
            self._note(msg, "subcall_return",
                       f"subquery_for={msg.metadata['subquery_for']} → raw data, "
                       f"skip synthesizer")
            return self.reply(
                msg, da,
                content    = result["content"],
                confidence = result.get("confidence"),
                flags      = flags,
                sql_query  = result.get("sql_query"),
                metadata   = result.get("metadata", {}),
            )

        # ── Top-level path: delegate to pa_agent if predict sub-question found ──
        # pa_agent receives the sql result as context so it can reference it
        # in the combined answer. pa returns raw prediction (subquery_for guard
        # in pa_agent skips its own synthesizer dispatch), then sql combines
        # sql_result + pa_result and sends both to synthesizer.
        if needs_predict and self.has_peer("pa_agent"):
            predict_subq = self._extract_predict_subquestion(query)
            self._note(msg, "pa_delegate",
                       f"KNOWLEDGE_GAP → delegating to pa_agent: {predict_subq[:80]}")
            pa_req = self.make_request(
                "pa_agent", msg,
                DialogueActType.REQUEST,
                predict_subq,
                trigger  = DelegationTrigger.KNOWLEDGE_GAP,
                metadata = {
                    "original_query": query,
                    "sql_context":    result["content"],
                },
            )
            pa_resp = await self.ask_peer("pa_agent", pa_req)
            if pa_resp.da_type != DialogueActType.REJECT:
                return pa_resp

            self._note(msg, "pa_fallback",
                       "pa_agent rejected — continuing with SQL-only result")

        # ── Top-level path: delegate forward to doc if mixed, else synthesize ───
        # When query mixes sql + doc, sql answers its part then delegates FORWARD
        # to doc_agent (sql_context passed in metadata). Doc combines both answers
        # and sends to synthesizer. No round-trip back to sql.
        if needs_doc and self.has_peer("doc_agent"):
            doc_subq = self._extract_doc_subquestion(query)
            self._note(msg, "doc_delegate",
                       f"KNOWLEDGE_GAP → delegating forward to doc_agent: {doc_subq[:80]}")
            doc_req = self.make_request(
                "doc_agent", msg,
                DialogueActType.REQUEST,
                doc_subq,
                trigger  = DelegationTrigger.KNOWLEDGE_GAP,
                metadata = {
                    "original_query":    query,
                    "sql_context":       result["content"],
                    "sql_query":         result.get("sql_query", ""),
                    "uncertainty_flags": [f.value for f in flags],
                },
            )
            self._note(msg, "doc_wait",
                       "sql answer ready — doc_agent will combine and send to synthesizer")
            doc_resp = await self.ask_peer("doc_agent", doc_req)

            # doc handled synthesizer itself — return the final answer
            if doc_resp.da_type != DialogueActType.REJECT:
                return doc_resp

            # doc rejected — fall back to SQL-only result via synthesizer
            self._note(msg, "doc_fallback",
                       "doc_agent rejected — falling back to SQL-only synthesizer path")

        # If doc_agent delegated forward to us, combine doc context + sql result
        doc_context = msg.metadata.get("doc_context", "")
        original_query = msg.metadata.get("original_query", query)
        combined_content = result["content"]
        if doc_context:
            combined_content = (
                f"Documentation context:\n{doc_context}\n\n"
                f"SQL query result:\n{result['content']}"
            )
            self._note(msg, "doc_sql_combined",
                       f"combined doc ({len(doc_context)} chars) + sql ({len(result['content'])} chars)")

        if self.has_peer("synthesizer"):
            synth_req = self.make_request(
                "synthesizer", msg,
                DialogueActType.INFORM,
                combined_content,
                metadata={
                    "original_query":    original_query,
                    "sql_query":         result.get("sql_query", ""),
                    "rows":              result.get("rows", []),
                    "columns":           result.get("columns", []),
                    "confidence":        result.get("confidence"),
                    "uncertainty_flags": [f.value for f in flags],
                    "domain_info":       result.get("domain_info", ""),
                },
            )
            self._note(msg, "synthesizer_dispatch",
                       f"forwarding to synthesizer (doc_delegated={bool(doc_context)})")
            return await self.ask_peer("synthesizer", synth_req)

        return self.reply(
            msg, da,
            content    = result["content"],
            confidence = result.get("confidence"),
            flags      = flags,
            sql_query  = result.get("sql_query"),
            metadata   = result.get("metadata", {}),
        )

    # ── Multi-sub-question decomposition ────────────────────────────────────

    def _extract_subquestions(self, query: str) -> list[str]:
        """
        Split a query into individual sub-questions when it contains 2 or more
        distinct questions that each require separate database lookups.
        Returns a single-element list if the query is already atomic.
        """
        import re
        prompt = (
            "Does this query contain 2 or more DISTINCT sub-questions that each "
            "require a SEPARATE SQL query against the Fugaku job telemetry database?\n\n"
            "IMPORTANT: Only count sub-questions that need SQL — job counts, failure rates, "
            "averages, node counts, energy, etc. Do NOT count sub-questions about:\n"
            "- HPC commands or documentation (pjsub, pjstat, pjdel, pjshowrsc, etc.)\n"
            "- Job submission guidelines or policies\n"
            "- Predicted failure risk (those go to the prediction model, not SQL)\n\n"
            "NOTE: If the query compares compute-bound vs memory-bound with DIFFERENT node/time "
            "filters for each class, that is SINGLE — use GROUP BY pclass in one query.\n"
            "If the query asks for stats on two entirely different entities (e.g. two different "
            "node counts, two different users, two different years), that may be MULTIPLE.\n\n"
            f"Query: {query}\n\n"
            "If YES (2+ SQL sub-questions): list ONLY the SQL sub-questions, one per line, "
            "numbered 1., 2., etc. Each must be self-contained. No explanation.\n"
            "If NO (single SQL question, or no SQL needed): reply with SINGLE.\n\n"
            "Response:"
        )
        result = self._llm(prompt, model="gpt-4o-mini", max_tokens=200).strip()
        if result.upper().startswith("SINGLE"):
            return [query]
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        subqs = []
        for line in lines:
            m = re.match(r'^\d+\.\s*(.+)$', line)
            if m:
                subqs.append(m.group(1).strip())
        return subqs if len(subqs) >= 2 else [query]

    async def _generate_and_execute(
        self, msg: A2AMessage, query: str, session_ctx: str
    ) -> dict:
        """
        Single-question: generate SQL + execute (existing path).
        Multi-sub-question: decompose → run each independently →
          combine with PARTIALLY_FOUND if some are unanswerable,
          REJECT if all are unanswerable.
        """
        subquestions = self._extract_subquestions(query)

        if len(subquestions) <= 1:
            sql = self._generate_sql(query, session_ctx)
            self._note(msg, "sql_gen", f"attempt 1: {sql[:120]}")
            return await self._run_with_retry(msg, query, session_ctx, sql)

        self._note(msg, "sql_decompose",
                   f"decomposed into {len(subquestions)} sub-questions")

        answered: list[dict] = []
        unanswered: list[str] = []

        for i, sq in enumerate(subquestions, 1):
            sql = self._generate_sql(sq, session_ctx)
            self._note(msg, f"sql_gen_sub_{i}", f"{sq[:60]} → {sql[:80]}")
            if sql == "CANNOT_GENERATE":
                unanswered.append(sq)
                self._note(msg, f"sql_cannot_sub_{i}", f"not in schema: {sq[:60]}")
            else:
                sub = await self._run_with_retry(msg, sq, session_ctx, sql)
                if sub["da_type"] == DialogueActType.REJECT:
                    unanswered.append(sq)
                else:
                    answered.append({"question": sq, "result": sub})

        if not answered:
            return {
                "da_type": DialogueActType.REJECT,
                "content": "Cannot generate valid SQL for this query.",
                "metadata": {"domain_info": session_ctx[:500]},
            }

        parts = []
        for item in answered:
            parts.append(f"{item['question']}\n{item['result']['content']}")
        for sq in unanswered:
            parts.append(
                f"{sq}\n"
                f"⚠ This information is not available in the Fugaku telemetry dataset "
                f"— the required data dimension is not tracked."
            )

        all_flags: list[UncertaintyFlag] = []
        for item in answered:
            all_flags.extend(item["result"].get("flags", []))
        if unanswered:
            all_flags.append(UncertaintyFlag.PARTIALLY_FOUND)

        seen: set = set()
        unique_flags = [f for f in all_flags if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

        confidence = 0.95 if not unanswered else 0.70
        combined_sqls = "; ".join(
            item["result"].get("sql_query", "") for item in answered
        )
        return {
            "da_type": DialogueActType.CAVEAT if unique_flags else DialogueActType.INFORM,
            "content": "\n\n".join(parts),
            "confidence": confidence,
            "flags": unique_flags,
            "sql_query": combined_sqls,
            "rows": [],
            "columns": [],
            "domain_info": session_ctx[:500],
            "metadata": {
                "answered_count":   len(answered),
                "unanswered_count": len(unanswered),
                "domain_info":      session_ctx[:500],
            },
        }

    # ── Doc context gate (KNOWLEDGE_GAP) ─────────────────────────────────────

    def _needs_predict_context(self, query: str) -> bool:
        """
        Decide if the query also contains a prediction sub-question that requires
        the PA agent — failure risk, failure probability, runtime estimate, or
        energy estimate for a NEW prospective job.
        Only fires when sql_agent has pa_agent registered as a peer.
        """
        prompt = (
            "Decide whether this query ALSO asks for a ML-based prediction about a NEW job:\n"
            "failure risk, failure probability, runtime estimate, or energy estimate\n"
            "for a job the user is about to submit.\n\n"
            "Answer YES if the query asks for:\n"
            "- Failure risk or failure probability for a new/specific job\n"
            "- What is the risk for a new user running a job\n"
            "- Will a job fail / is it safe to submit\n"
            "- Predicted runtime or energy for a new job\n"
            "- Risk assessment for a job with given parameters\n\n"
            "Answer NO if the query ONLY asks about historical failure RATES, counts,\n"
            "averages, or statistics from past jobs.\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _extract_predict_subquestion(self, query: str) -> str:
        prompt = (
            "From this mixed query, extract ONLY the prediction sub-question.\n"
            "Prediction questions ask: failure risk, failure probability, runtime estimate,\n"
            "or energy estimate for a new/prospective job.\n"
            "Preserve all job parameters mentioned (node count, walltime, user, job class).\n"
            "Return a single specific prediction question. No explanation.\n\n"
            f"Query: {query}\n\nPrediction sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=100).strip()

    def _needs_doc_context(self, query: str) -> bool:
        """
        Decide if the query also contains a documentation sub-question that
        cannot be answered by SQL — commands, policies, procedures, directives.
        Only fires when sql_agent has doc_agent registered as a peer.
        """
        prompt = (
            "Decide whether this query ALSO asks about Fugaku system documentation:\n"
            "commands, job-script directives, policies, procedures, or guidance.\n\n"
            "Answer YES only if the query explicitly asks for:\n"
            "- A specific system command (pjstat, pjsub, pjshowrsc, etc.)\n"
            "- How to do something on Fugaku (submit jobs, configure, compile)\n"
            "- What a directive does ('#PJM', '--mpi', etc.)\n"
            "- A policy or rule (walltime limits, node limits, power caps)\n"
            "- What happens when a system condition is triggered\n"
            "- Any question answerable from Fugaku user manuals\n\n"
            "Answer NO if the query is ONLY about historical data (counts, averages, trends).\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _extract_doc_subquestion(self, query: str) -> str:
        prompt = (
            "From this mixed query, extract ONLY the documentation sub-question.\n"
            "Documentation questions ask: how to use a command, what a directive does, "
            "what a policy/limit is, or how to configure something.\n"
            "Do NOT extract questions about historical data, counts, averages, or statistics.\n"
            "Preserve the EXACT command name (e.g. pjsub, pjdel, pjstat) or policy term.\n"
            "Return a single specific documentation question. No explanation.\n\n"
            f"Query: {query}\n\nDocumentation sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=80).strip()

    def _extract_sql_subquestion(self, query: str) -> str:
        prompt = (
            "From this mixed query, extract ONLY the SQL/database part.\n"
            "The database part asks for job counts, failure rates, averages, top-N rankings, "
            "distributions, or other statistics from the Fugaku HPC job telemetry.\n\n"
            "Rules:\n"
            "- Keep the EXACT question structure (e.g. 'top 5 users by job count' stays as-is)\n"
            "- Do NOT generalize or paraphrase into vague summaries\n"
            "- Do NOT include questions about HPC commands (pjsub, pjstat, pjdel)\n"
            "- Do NOT include prediction/risk questions\n"
            "- Return ONE specific, self-contained database question. No explanation.\n\n"
            f"Query: {query}\n\nDatabase sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=80).strip()

    # ── Profile gate ──────────────────────────────────────────────────────────

    def _needs_profile(self, query: str) -> bool:
        prompt = (
            "You are helping an SQL agent decide whether to profile database columns.\n\n"
            "Profiling IS needed when:\n"
            "- Query involves an UNKNOWN categorical column whose values are uncertain\n\n"
            "Profiling is NOT needed when:\n"
            "- Uses well-known columns: pclass, usr, nnumr, avgpcon, duration, 'exit state'\n"
            "- Asks for aggregates by job class (GROUP BY pclass is safe)\n"
            "- Asks for MAX/MIN/AVG on known numeric columns\n\n"
            f"Query: {query}\n\nReply with exactly one word: yes or no."
        )
        return self._llm_bool(prompt)

    def _has_groupby_intent(self, query: str) -> bool:
        q_lower = query.lower()
        return any(kw in q_lower for kw in GROUP_BY_KEYWORDS)

    # ── SQL generation ────────────────────────────────────────────────────────

    def _generate_sql(self, query: str, session_ctx: str, feedback: str = "") -> str:
        feedback_block = (
            f"\nPrevious attempt feedback (fix this):\n{feedback}\n"
            if feedback else ""
        )

        prompt = f"""You are an HPC telemetry analyst for the Fugaku supercomputer.

Generate a single valid DuckDB SQL query to answer the user's question.

Rules:
- Use ONLY tables and columns listed in the schema below
- Return ONLY the raw SQL — no explanation, no markdown, no backticks
- Return CANNOT_GENERATE if the requested CONCEPT is absent from this dataset —
  temperature, billing/cost, GPU data, network metrics, personal user data.
  Do NOT proxy-calculate missing concepts (e.g. don't use power as billing cost).
  Do NOT relabel a column's value as a different concept (e.g. don't call uctmut "temperature").
- If the query has BOTH SQL-answerable parts AND predictor/model parts, generate SQL for the
  SQL-answerable parts ONLY (failure rates, job counts, averages). Do NOT return CANNOT_GENERATE
  just because the query also asks about predictor outputs, failure probabilities, or risk scores.
- NEVER cast a CATEGORICAL column to a numeric type
- When domain_info lists categorical column profiles, use those column names directly
- For distribution/bucket queries: use CASE WHEN, GROUP BY the alias (not raw column)
- "over N", "more than N" → col > N  |  "at least N" → col >= N  |  "under N" → col < N
- HARD CONSTRAINT: Exactly ONE SQL statement. No semicolons. No multi-statements.

FAILURE RATE COMPUTATION — ALWAYS use this pattern:
  ROUND(100.0 * SUM(CASE WHEN "exit state"='failed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fail_pct
  Never divide failed_count / failed_count (that always equals 1.0). Denominator must be COUNT(*).

PCLASS COMPARISON — when query compares compute-bound vs memory-bound:
  Use WHERE pclass IN ('compute-bound','memory-bound') GROUP BY pclass
  NEVER filter to a single pclass when both classes are asked about.
  This ensures both classes appear in ONE result set.

DOMAIN LABEL TRAP — Fugaku has NO domain/application columns:
  jnam values are anonymized codes like 'jnam_12345' — NOT scientific domains.
  NEVER filter by jnam for "genomics", "DFT", "quantum", "nuclear", "CFD", etc.
  Interpret domain-framed queries by job class (pclass) and node count (nnumr) only.

WALLTIME vs ACTUAL RUNTIME:
  "average walltime" or "average walltime of failed jobs" → AVG(CASE WHEN "exit state"='failed' THEN elpl ELSE NULL END)
  "average walltime" across all jobs → AVG(elpl)
  "actual runtime" or "how long did jobs actually run" → AVG(duration)
  CRITICAL: "average walltime of failed jobs" must filter to failed only inside AVG.
  Use: AVG(CASE WHEN "exit state"='failed' THEN elpl ELSE NULL END) AS avg_failed_walltime_s

ENERGY QUERIES:
  When computing average energy per job, use SELECT AVG(econ) directly — econ is already
  total Joules per job. Do NOT divide by nnumr, elpl, or any other column.
  Example: "average energy for memory-bound jobs" → SELECT AVG(econ) FROM jobs WHERE pclass='memory-bound'

TEMPORAL FILTERING:
  Do NOT add a year filter (WHERE YEAR(...)=X or WHERE year=X) unless the query
  EXPLICITLY mentions a specific year or time period (e.g. "in 2023", "from 2022 to 2023").
  For general failure rate or average queries with no year mentioned, query ALL years.

WALLTIME FILTER IN FAILURE RATE QUERIES:
  A walltime mentioned in the query often describes the USER'S NEW JOB being planned,
  NOT a filter criterion for historical data. Do NOT add an elpl filter in that case.
  Only add an elpl filter when the user EXPLICITLY asks for "historical jobs that ran for X hours"
  or "jobs with walltime limit of approximately X" as a search dimension (not just mentioning
  their own planned job's walltime alongside a node count or pclass question).
  When uncertain whether to filter on elpl: OMIT the elpl filter entirely.

GLOBAL vs USER-SPECIFIC SCOPE:
  "global", "system-wide", "overall", "across all users", "all users" → do NOT filter by usr.
  "usr_X", "user usr_X", "for user X", "for usr_X" → filter WHERE usr='usr_X'.
  When a single query asks for BOTH a global rate AND a user-specific rate, you need
  TWO separate aggregations — use UNION ALL or a GROUP BY that includes a CASE-based
  user dimension. Never report a user-specific rate as the global rate or vice versa.

MEDIAN / PERCENTILE queries:
  "median", "p50", "50th percentile", "middle value" → use PERCENTILE_CONT:
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY <col>) AS median_<col> FROM jobs WHERE ...
  DuckDB supports PERCENTILE_CONT natively. Always use it for median requests.

{feedback_block}
User query: {query}

Prior dialog context (DataExplorer column profiles appear here if called):
{session_ctx}

Database schema:
{get_schema_str()}

Additional schema notes:
{FUGAKU_SCHEMA_NOTES}

SQL:"""

        result = self._llm(prompt)
        if result.strip().upper().startswith("CANNOT_GENERATE"):
            return "CANNOT_GENERATE"
        return result.strip()

    # ── Execution + retry loop ────────────────────────────────────────────────

    async def _run_with_retry(
        self,
        msg: A2AMessage,
        query: str,
        session_ctx: str,
        sql: str,
    ) -> dict:
        """
        Execute SQL with up to config.MAX_REFLECT_ROUNDS attempts.

        Returns a plain dict — does NOT log any A2AMessage.
        Caller (handle) decides the final DA type and routing.

        Return keys: da_type, content, confidence, flags, sql_query,
                     rows, columns, domain_info, metadata
        """
        max_rounds = config.MAX_REFLECT_ROUNDS

        for attempt in range(1, max_rounds + 1):

            if sql.strip().upper().startswith("CANNOT_GENERATE"):
                self._note(msg, f"sql_cannot_gen_{attempt}",
                           "LLM returned CANNOT_GENERATE")
                return {
                    "da_type": DialogueActType.REJECT,
                    "content": "Cannot generate valid SQL for this query.",
                    "metadata": {"domain_info": session_ctx[:500]},
                }

            try:
                rows, cols = self._run_sql(sql)
                self._note(msg, f"sql_exec_{attempt}",
                           f"{len(rows)} rows, cols={cols[:4]}")

                sufficient, reason = self._is_sufficient(query, rows, cols)
                self._note(msg, f"sufficiency_{attempt}",
                           f"sufficient={sufficient}: {reason[:80]}")

                if sufficient or attempt == max_rounds:
                    flags     = self._detect_flags(rows, cols)
                    formatted = self._format_result(rows, cols)
                    return {
                        "da_type":     DialogueActType.CAVEAT if flags else DialogueActType.INFORM,
                        "content":     formatted,
                        "confidence":  0.95 if attempt == 1 else 0.75,
                        "flags":       flags,
                        "sql_query":   sql,
                        "rows":        [list(r) for r in rows[:50]],
                        "columns":     cols,
                        "domain_info": session_ctx[:500],
                        "metadata": {
                            "rows":        [list(r) for r in rows[:50]],
                            "columns":     cols,
                            "domain_info": session_ctx[:500],
                            "attempts":    attempt,
                        },
                    }

                if not rows:
                    feedback = (
                        f"Attempt {attempt}: query returned 0 rows.\n"
                        f"SQL: {sql}\n"
                        f"The WHERE clause is too restrictive. Remove filters one at a time:\n"
                        f"1. Remove any year/date filter (YEAR, strftime, date_trunc)\n"
                        f"2. Remove any elpl/walltime filter unless explicitly required\n"
                        f"3. Relax node count constraints (use >= instead of =)\n"
                        f"Rewrite with broader WHERE clause to return actual rows."
                    )
                else:
                    feedback = (
                        f"Attempt {attempt} result incomplete.\n"
                        f"SQL: {sql}\nRows: {len(rows)}, Cols: {cols}\n"
                        f"Reason: {reason}\nRewrite SQL to fix this."
                    )
                sql = self._generate_sql(query, session_ctx, feedback)
                self._note(msg, f"sql_retry_{attempt}",
                           f"regenerating: {sql[:120]}")

            except Exception as exc:
                self._note(msg, f"sql_error_{attempt}", str(exc)[:200])
                if attempt == max_rounds:
                    return {
                        "da_type": DialogueActType.REJECT,
                        "content": f"SQL execution failed after {max_rounds} attempts: {exc}",
                        "metadata": {"sql": sql, "error": str(exc)},
                    }
                feedback = f"SQL caused an error:\n{exc}\nFix the SQL."
                sql = self._generate_sql(query, session_ctx, feedback)
                self._note(msg, f"sql_retry_{attempt}",
                           f"after error, regenerating: {sql[:120]}")

        return {
            "da_type": DialogueActType.REJECT,
            "content": "Cannot generate valid SQL for this query.",
            "metadata": {"domain_info": session_ctx[:500]},
        }

    # ── Sufficiency check ─────────────────────────────────────────────────────

    def _is_sufficient(self, query, rows, cols) -> tuple[bool, str]:
        if not rows:
            return False, "0 rows returned"
        if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] is None:
            return False, "NULL result"
        # Sanity: failure rate of exactly 1.0 (100%) is almost certainly a SQL bug —
        # the denominator was wrong (e.g. failed/failed instead of failed/all).
        # Catch it before the LLM sufficiency check and force a retry.
        for col_idx, col in enumerate(cols):
            if any(kw in col.lower() for kw in ("fail", "rate", "pct", "pct_")):
                val = rows[0][col_idx] if rows else None
                if val is not None:
                    try:
                        v = float(val)
                        if v >= 99.9:   # 100% failure rate — suspicious
                            return False, (
                                f"Suspicious: failure rate={v:.1f}% suggests "
                                "denominator error (failed/failed instead of "
                                "failed/total). Rewrite: denominator must be "
                                "COUNT(*) not COUNT(failed)."
                            )
                    except (TypeError, ValueError):
                        pass
        preview = rows[:20]  # show enough rows for the LLM to judge completeness
        prompt = (
            f"Query: {query}\n"
            f"Columns: {cols}\n"
            f"Full result ({len(rows)} rows total, showing all):\n{preview}\n\n"
            "Does this result fully answer the query?\n"
            "- YES if the data present is a complete, valid answer (even if you'd expect more rows)\n"
            "- NO only if there are 0 rows, a NULL result, or the result is clearly wrong\n"
            "Reply: YES or NO + one line reason."
        )
        answer = self._llm(prompt, model="gpt-4o-mini", max_tokens=60)
        sufficient = answer.strip().upper().startswith("YES")
        parts  = answer.strip().split(None, 1)
        reason = parts[1] if len(parts) > 1 else answer
        return sufficient, reason

    # ── SQL execution ─────────────────────────────────────────────────────────

    def _run_sql(self, sql: str) -> tuple[list[tuple], list[str]]:
        conn   = get_connection()
        result = conn.execute(sql)
        rows   = result.fetchall()
        cols   = [d[0] for d in result.description]
        return rows, cols

    # ── Uncertainty flags ─────────────────────────────────────────────────────

    def _detect_flags(self, rows, cols) -> list[UncertaintyFlag]:
        if not rows:
            return [UncertaintyFlag.NOT_FOUND]
        # Single-cell COUNT result of 0 → no matching data exists
        if (len(rows) == 1 and len(rows[0]) == 1
                and isinstance(rows[0][0], (int, float))
                and rows[0][0] == 0):
            return [UncertaintyFlag.NOT_FOUND]
        total_cells = len(rows) * len(rows[0]) if rows else 0
        if total_cells > 0:
            null_count = sum(1 for row in rows for val in row if val is None)
            if null_count / total_cells > 0.20:
                return [UncertaintyFlag.NULL_VALUES]
        return []

    # ── Result formatting ─────────────────────────────────────────────────────

    def _format_result(self, rows: list[tuple], cols: list[str]) -> str:
        if not rows:
            return "Query returned no results."

        UNIT_HINTS = {
            "avgpcon":  "avgpcon (W)",
            "minpcon":  "minpcon (W)",
            "maxpcon":  "maxpcon (W)",
            "econ":     "econ (J)",
            "duration": "duration (s)",
            "uctmut":   "uctmut (ms)",   # user CPU time in milliseconds
            "cnumut":   "cnumut (cores used)",
        }
        header_cols = [UNIT_HINTS.get(c, c) for c in cols]
        header = " | ".join(header_cols)

        lines = [header]
        for row in rows[:50]:
            lines.append(" | ".join(str(v) for v in row))

        if len(rows) > 50:
            lines.append(f"... and {len(rows) - 50} more rows")

        return "\n".join(lines)

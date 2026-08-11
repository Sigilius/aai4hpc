"""
agents/doc_agent.py  —  v1.0.0

DocAgent: Fugaku documentation Q&A with hybrid retrieval.

P2P chain position
------------------
  gateway → doc_agent → synthesizer (formatting + reflection)

Retrieval strategy
------------------
1. Session-log read (SIGDIAL mechanism — same as SQLAgent)
2. Query expansion  — LLM rephrases query in 2-3 ways for broader recall
3. Hybrid search    — DocRetriever: BM25 + text-embedding-3-small, RRF fusion
4. Deduplication    — merge results across variants, keep top-8 by RRF score
5. Relevance gate   — gpt-4o-mini filters irrelevant chunks before synthesis
6. Grounded synthesis — answer with breadcrumb citations

DA types emitted
----------------
  INFORM  — answer forwarded to synthesizer (or returned directly)
  REJECT  — no relevant documentation found after relevance gate
"""
from __future__ import annotations

import json
from typing import Optional

from core.message_schema import (
    A2AMessage,
    DelegationTrigger,
    DialogueActType,
    UncertaintyFlag,
)
from agents.base_agent import BaseAgent
from shared.doc_retriever import DocRetriever


TOP_K_RETRIEVE = 10   # chunks fetched per query variant
TOP_K_FUSE     = 8    # chunks kept after dedup across variants
MIN_RELEVANT   = 1    # minimum relevant chunks to attempt synthesis


class DocAgent(BaseAgent):
    """
    Documentation Q&A agent for the Fugaku supercomputer.

    Hybrid BM25+dense retrieval with query expansion and relevance
    filtering before grounded LLM synthesis.
    """

    name    = "doc_agent"
    version = "1.0.0"

    def __init__(self, log, verbose: bool = False) -> None:
        super().__init__(log, verbose)
        self._retriever: Optional[DocRetriever] = None   # lazy-loaded

    @property
    def retriever(self) -> DocRetriever:
        if self._retriever is None:
            self._retriever = DocRetriever(verbose=self.verbose)
        return self._retriever

    # ── Main handler ──────────────────────────────────────────────────────────

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        """
        SIGDIAL doc-retrieval flow:

        1. Read session log (prior context — same injection pattern as SQLAgent).
        2. Expand query into 2-3 variants for better BM25+dense recall.
        3. Hybrid search each variant; deduplicate by chunk_index.
        4. Relevance gate: gpt-4o-mini scores each chunk against query.
        5. Synthesize grounded answer with breadcrumb citations.
        6. Forward to synthesizer if wired, else reply INFORM directly.
        7. REJECT if no relevant chunks survive the gate.
        """
        query = msg.content
        self._note(msg, "query_received", f"query={query[:120]}")

        # ── Read session log ──────────────────────────────────────────────────
        session_ctx = self.log.format_for_llm(msg.session_id)
        self._note(msg, "session_read",
                   f"read {len(session_ctx)} chars of prior dialog")

        # ── Query expansion ───────────────────────────────────────────────────
        expansions = self._expand_query(query, session_ctx)
        self._note(msg, "query_expansion",
                   f"{len(expansions)} variants: {str(expansions)[:200]}")

        # ── Hybrid retrieval + deduplication ──────────────────────────────────
        all_chunks: dict[int, dict] = {}   # array idx → best-scoring chunk
        for variant in expansions:
            hits = self.retriever.search(variant, top_k=TOP_K_RETRIEVE)
            for chunk in hits:
                idx = chunk["_idx"]   # unique array position added by DocRetriever
                if idx not in all_chunks or chunk["rrf_score"] > all_chunks[idx]["rrf_score"]:
                    all_chunks[idx] = chunk

        fused = sorted(all_chunks.values(), key=lambda c: -c["rrf_score"])[:TOP_K_FUSE]
        self._note(msg, "retrieval_done",
                   f"{len(fused)} unique chunks from {len(expansions)} variants; "
                   f"top breadcrumb: {fused[0]['breadcrumb'] if fused else 'none'}")

        if not fused:
            self._note(msg, "no_chunks", "retriever returned 0 chunks → REJECT")
            return self.reply(
                msg, DialogueActType.REJECT,
                content="No relevant Fugaku documentation found for this query.",
                metadata={"domain_info": "doc retrieval returned 0 chunks"},
            )

        # ── Relevance gate ────────────────────────────────────────────────────
        relevant = self._filter_relevant(query, fused)
        self._note(msg, "relevance_gate",
                   f"{len(relevant)}/{len(fused)} chunks passed")

        if len(relevant) < MIN_RELEVANT:
            relevant = fused[:2]
            self._note(msg, "relevance_fallback",
                       "gate filtered all chunks; keeping top-2 by RRF score")

        # ── Synthesize grounded answer ─────────────────────────────────────────
        answer, has_answer = self._synthesize(query, relevant, session_ctx)
        self._note(msg, "synthesis_done",
                   f"has_answer={has_answer}, len={len(answer)}")

        if not has_answer:
            return self.reply(
                msg, DialogueActType.REJECT,
                content=answer,
                metadata={"domain_info": self._chunks_summary(relevant)},
            )

        # ── Sub-call path: return raw answer, skip synthesizer/reflector ────────
        # When called by pa_agent with subquery_for set — caller wants raw text.
        if msg.metadata.get("subquery_for"):
            self._note(msg, "subcall_return",
                       f"subquery_for={msg.metadata['subquery_for']} → raw answer, "
                       f"skip synthesizer")
            return self.reply(
                msg, DialogueActType.INFORM,
                content    = answer,
                confidence = 0.85,
                metadata   = {"source_chunks": [c["breadcrumb"] for c in relevant]},
            )

        # ── Top-level path (direct from gateway OR delegated from sql_agent) ───
        # When sql_agent delegates forward, it passes sql_context + original_query.
        # Doc combines both and sends the full answer to synthesizer once.
        sql_context    = msg.metadata.get("sql_context", "")
        sql_query_used = msg.metadata.get("sql_query", "")
        original_query = msg.metadata.get("original_query", query)

        combined_content = answer
        if sql_context:
            combined_content = (
                f"SQL query result:\n{sql_context}\n\n"
                f"Documentation context:\n{answer}"
            )
            self._note(msg, "sql_doc_combined",
                       f"combined sql ({len(sql_context)} chars) + doc ({len(answer)} chars)")

        # ── Forward delegation to pa_agent (doc→pa chain) ─────────────────────
        # PA takes priority over sql: it can call sql internally if it needs data.
        # Guard: skip when sql already delegated to us (sql_context present) —
        # that means sql was the entry agent, not doc.
        needs_pa = (
            not sql_context
            and self.has_peer("pa_agent")
            and self._needs_pa_context(query)
        )
        if needs_pa:
            pa_subq = self._extract_pa_subquestion(query)
            self._note(msg, "pa_delegate",
                       f"KNOWLEDGE_GAP → delegating forward to pa_agent: {pa_subq[:80]}")
            pa_req = self.make_request(
                "pa_agent", msg,
                DialogueActType.REQUEST,
                pa_subq,
                trigger  = DelegationTrigger.KNOWLEDGE_GAP,
                metadata = {
                    "original_query": original_query,
                    "doc_context":    combined_content,
                },
            )
            pa_resp = await self.ask_peer("pa_agent", pa_req)

            if pa_resp.da_type != DialogueActType.REJECT:
                return pa_resp

            self._note(msg, "pa_fallback",
                       "pa_agent rejected — falling back to sql/synthesizer path")

        # ── Forward delegation to sql_agent (doc→sql chain) ──────────────────
        # Guard: skip if sql already gave us context, or pa is handling it.
        needs_sql = (
            not sql_context
            and not needs_pa
            and self.has_peer("sql_agent")
            and self._needs_sql_context(query)
        )
        if needs_sql:
            sql_subq = self._extract_sql_subquestion(query)
            self._note(msg, "sql_delegate",
                       f"DATA_INSUFFICIENCY → delegating forward to sql_agent: {sql_subq[:80]}")
            sql_req = self.make_request(
                "sql_agent", msg,
                DialogueActType.REQUEST,
                sql_subq,
                trigger  = DelegationTrigger.DATA_INSUFFICIENCY,
                metadata = {
                    "original_query": original_query,
                    "doc_context":    combined_content,
                },
            )
            sql_resp = await self.ask_peer("sql_agent", sql_req)

            if sql_resp.da_type != DialogueActType.REJECT:
                return sql_resp

            self._note(msg, "sql_fallback",
                       "sql_agent rejected — sending doc-only result to synthesizer")

        if self.has_peer("synthesizer"):
            synth_req = self.make_request(
                "synthesizer", msg,
                DialogueActType.INFORM,
                combined_content,
                metadata={
                    "original_query": original_query,
                    "sql_query":      sql_query_used,
                    "domain_info":    self._chunks_summary(relevant),
                    "source_chunks":  [
                        {"breadcrumb": c["breadcrumb"], "rrf_score": c["rrf_score"]}
                        for c in relevant
                    ],
                    "confidence": 0.85,
                },
            )
            self._note(msg, "synthesizer_dispatch",
                       f"forwarding to synthesizer ({len(relevant)} chunks, "
                       f"has_sql_context={bool(sql_context)})")
            return await self.ask_peer("synthesizer", synth_req)

        return self.reply(
            msg, DialogueActType.INFORM,
            content    = combined_content,
            confidence = 0.85,
            metadata   = {
                "source_chunks":  [c["breadcrumb"] for c in relevant],
                "original_query": original_query,
            },
        )

    # ── Query expansion ────────────────────────────────────────────────────────

    def _expand_query(self, query: str, session_ctx: str) -> list[str]:
        """
        Generate 2-3 rephrasings for broader retrieval coverage.

        Different phrasings hit different BM25 token matches and different
        dense embedding neighbourhoods — fusing them improves recall.
        """
        prompt = (
            "You are helping retrieve Fugaku supercomputer documentation.\n\n"
            f"User query: {query}\n\n"
            f"Recent dialog context:\n{session_ctx[:400]}\n\n"
            "Generate 2 alternative phrasings of this query to improve "
            "documentation retrieval. Each should:\n"
            "- Use different keywords or technical terms (command names, policy terms)\n"
            "- Target a different angle (e.g. policy vs procedure vs example)\n"
            "- Stay focused on the same underlying information need\n\n"
            "Return ONLY a JSON array of strings (the original + 2 alternatives):\n"
            "[\"original query\", \"variant 1\", \"variant 2\"]\n\n"
            "JSON:"
        )
        try:
            raw = self._llm(prompt, model="gpt-4o-mini", max_tokens=200)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            variants = json.loads(raw)
            if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                seen = {query}
                result = [query]
                for v in variants:
                    if v not in seen and v.strip():
                        result.append(v)
                        seen.add(v)
                return result[:3]
        except Exception:
            pass
        return [query]

    # ── Relevance gate ─────────────────────────────────────────────────────────

    def _filter_relevant(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        gpt-4o-mini gate: return only chunks that contain useful information.

        Single LLM call for all chunks to minimise latency.
        Falls back to all chunks if JSON parsing fails.
        """
        if not chunks:
            return []

        chunk_list = "\n\n".join(
            f"[{i}] {c['breadcrumb']}\n{c['text'][:350]}"
            for i, c in enumerate(chunks)
        )

        prompt = (
            f"User query: {query}\n\n"
            "Below are Fugaku documentation chunks. "
            "Return the indices of chunks that contain information useful for answering the query.\n\n"
            f"{chunk_list}\n\n"
            "Return a JSON array of 0-based indices. "
            "Include a chunk if it has ANY relevant detail. "
            "Return [] only if none are relevant.\n"
            "JSON array:"
        )
        try:
            raw = self._llm(prompt, model="gpt-4o-mini", max_tokens=80)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            indices = json.loads(raw)
            if isinstance(indices, list):
                valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
                if valid:
                    return [chunks[i] for i in valid]
        except Exception:
            pass
        return chunks   # fallback: keep all

    # ── Grounded synthesis ─────────────────────────────────────────────────────

    def _synthesize(
        self,
        query: str,
        chunks: list[dict],
        session_ctx: str,
    ) -> tuple[str, bool]:
        """
        Generate a grounded answer from relevant documentation chunks.

        Returns (answer_text, has_answer: bool).
        Returns (explanation, False) when docs don't cover the query.
        """
        docs_block = "\n\n---\n\n".join(
            f"[Source: {c['breadcrumb']}]\n{c['text']}"
            for c in chunks
        )

        prompt = (
            "You are an expert assistant for the Fugaku supercomputer.\n"
            "Answer the user's question using ONLY the documentation excerpts below.\n\n"
            f"Question: {query}\n\n"
            f"Prior conversation:\n{session_ctx[:300]}\n\n"
            "Documentation excerpts:\n"
            f"{docs_block}\n\n"
            "Instructions:\n"
            "1. Answer directly and specifically from the excerpts.\n"
            "2. Cite sections in parentheses after each fact, "
            "   e.g. (Job Scheduling Guide > Walltime Limits).\n"
            "3. Use bullet points for multi-part answers.\n"
            "4. If the excerpts do NOT contain enough information, "
            "   start your reply with exactly: NOT_IN_DOCS: followed by "
            "   one sentence explaining what is missing.\n"
            "5. Never invent facts not present in the excerpts.\n\n"
            "Answer:"
        )
        answer = self._llm(prompt)

        if answer.strip().startswith("NOT_IN_DOCS:"):
            explanation = answer.strip()[len("NOT_IN_DOCS:"):].strip()
            return (
                f"The Fugaku documentation does not contain sufficient information "
                f"to answer this question. {explanation}",
                False,
            )

        # Append deduplicated source breadcrumbs as a footer
        sources = list(dict.fromkeys(c["breadcrumb"] for c in chunks))
        source_lines = "\n".join(f"  • {s}" for s in sources[:6])
        answer = answer.rstrip() + f"\n\n**Sources:**\n{source_lines}"

        return answer, True

    # ── PA delegation detection ───────────────────────────────────────────────

    def _needs_pa_context(self, query: str) -> bool:
        """Detect if a doc-classified query also asks for job failure/risk prediction."""
        prompt = (
            "Decide whether this query ALSO asks for a failure risk prediction, "
            "runtime estimate, or safety assessment for a specific HPC job.\n\n"
            "Answer YES only if the query explicitly asks:\n"
            "- Failure probability or risk level for a specific job configuration\n"
            "- Whether a job 'will fail', 'is safe', or 'is risky'\n"
            "- Expected runtime or energy for a new job\n"
            "- Prediction at a specific or maximum configuration\n\n"
            "Answer NO if the query is only about documentation, historical counts, "
            "or system policies without asking for a specific job prediction.\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _extract_pa_subquestion(self, query: str) -> str:
        prompt = (
            "From this mixed query, extract ONLY the prediction sub-question.\n"
            "This is the part asking for failure risk, runtime estimate, or safety "
            "assessment for a specific HPC job.\n"
            "Preserve any job parameters mentioned (node count, walltime, job class).\n"
            "Return a single specific prediction question. No explanation.\n\n"
            f"Query: {query}\n\nPrediction sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=100).strip()

    # ── SQL delegation detection ──────────────────────────────────────────────

    def _needs_sql_context(self, query: str) -> bool:
        """Detect if a doc-classified query also asks for historical statistics."""
        prompt = (
            "Decide whether this query ALSO asks for historical statistics from "
            "the Fugaku HPC job database.\n\n"
            "Answer YES only if the query explicitly asks for:\n"
            "- Counts, averages, percentages, or trends over past jobs\n"
            "- 'How many', 'what percentage', 'how often', 'what is the average'\n"
            "- Distribution or breakdown of jobs by class, user, duration, etc.\n\n"
            "Answer NO if the query is ONLY about commands, policies, or procedures.\n\n"
            f"Query: {query}\n\nReply with exactly one word: YES or NO."
        )
        return self._llm_bool(prompt)

    def _extract_sql_subquestion(self, query: str) -> str:
        prompt = (
            "From this mixed query, extract ONLY the part asking for historical "
            "statistics from the Fugaku HPC job database.\n"
            "Preserve exact numbers, column names, or filters mentioned.\n"
            "Return a single specific database-answerable question. No explanation.\n\n"
            f"Query: {query}\n\nDatabase sub-question:"
        )
        return self._llm(prompt, model="gpt-4o-mini", max_tokens=80).strip()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _chunks_summary(self, chunks: list[dict]) -> str:
        return "; ".join(c["breadcrumb"] for c in chunks[:5])

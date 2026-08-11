"""
research/shared/data_explorer.py

Shared Data Explorer capability for the baseline systems.

Rationale (SIGDIAL 2026 ablation fairness)
------------------------------------------
The Full MAS has seven agents; the baselines were built with six roles
(gateway/sql/pa/doc/synthesizer/reflector) and no column profiler. That made
DataExplorer a *capability* advantage for the MAS rather than a consequence of
the ablation variable (typed A2A messaging vs. shared state), confounding FR
and UAA.

This module gives every baseline the same profiling capability. The profiling
logic, the trigger gate, and the rendered output are ported verbatim from
`agents/data_explorer_agent.py` and `agents/sql_agent.py` so the capability is
byte-identical across systems. What deliberately differs per system is only how
the profile is *transported*:

  Blackboard          → written to an `explorer_result` slot (untyped dict slot)
  Unstructured        → appended to the causal log as plain text
  Unstructured-A2A    → answered as a free-form peer response
  Single Agent        → exposed as a `profile_columns` tool

None of these carry a DA type, an UncertaintyFlag, or a DelegationTrigger, so
the ablation variable stays intact: the baselines gain the capability, not the
typed schema.

No LLM is used for profiling itself — pure DuckDB introspection.
"""
from __future__ import annotations

from typing import Optional

from shared.db import get_connection

# Ported verbatim from agents/sql_agent.py — decides full-categorical-scan mode.
GROUP_BY_KEYWORDS = {
    "each", "per", "by", "group", "breakdown", "distribution",
    "classes", "types", "category", "categories", "kind", "kinds",
}

# Ported verbatim from agents/sql_agent.py::_needs_profile — the LLM gate that
# decides whether profiling is warranted at all.
NEEDS_PROFILE_PROMPT = (
    "You are helping an SQL agent decide whether to profile database columns.\n\n"
    "Profiling IS needed when:\n"
    "- Query involves an UNKNOWN categorical column whose values are uncertain\n\n"
    "Profiling is NOT needed when:\n"
    "- Uses well-known columns: pclass, usr, nnumr, avgpcon, duration, 'exit state'\n"
    "- Asks for aggregates by job class (GROUP BY pclass is safe)\n"
    "- Asks for MAX/MIN/AVG on known numeric columns\n\n"
    "Query: {query}\n\nReply with exactly one word: yes or no."
)


class DataExplorer:
    """
    Column profiler for the Fugaku jobs table.

    Functionally identical to agents/data_explorer_agent.py::DataExplorerAgent,
    minus the A2AMessage/DA wrapper. Reports the same fields, in the same text
    layout, so a baseline receives exactly what the MAS SQL agent receives.
    """

    def __init__(self) -> None:
        self._db = get_connection()

    # ── Column selection ──────────────────────────────────────────────────────

    def _all_columns(self) -> list[str]:
        return [r[0] for r in self._db.execute("DESCRIBE jobs").fetchall()]

    def _all_categorical_columns(self) -> list[str]:
        rows = self._db.execute("DESCRIBE jobs").fetchall()
        return [
            r[0] for r in rows
            if any(t in r[1].upper() for t in ("VARCHAR", "TEXT", "CHAR"))
        ]

    def _infer_columns_from_query(self, query: str) -> list[str]:
        """Return columns whose names appear literally in the query string."""
        q_lower = query.lower()
        return [c for c in self._all_columns() if c.lower() in q_lower]

    def select_columns(self, query: str, full_categorical_scan: bool = False) -> list[str]:
        if full_categorical_scan:
            return self._all_categorical_columns()
        return self._infer_columns_from_query(query)

    # ── Profiling ─────────────────────────────────────────────────────────────

    def profile(self, col: str) -> dict:
        """Profile a single column. Ported verbatim from DataExplorerAgent._profile."""
        try:
            dtype_row = self._db.execute(
                "SELECT column_type FROM (DESCRIBE jobs) WHERE column_name = ?",
                [col],
            ).fetchone()
            if not dtype_row:
                return {"error": f"Column '{col}' not found in schema."}

            dtype = dtype_row[0].upper()
            total = self._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            nulls = self._db.execute(
                f'SELECT COUNT(*) FROM jobs WHERE "{col}" IS NULL'
            ).fetchone()[0]

            profile: dict = {"dtype": dtype, "total_rows": total, "null_count": nulls}

            if any(t in dtype for t in ("VARCHAR", "TEXT", "CHAR")):
                distinct = self._db.execute(
                    f'SELECT DISTINCT "{col}" FROM jobs WHERE "{col}" IS NOT NULL LIMIT 20'
                ).fetchall()
                n_distinct = self._db.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM jobs'
                ).fetchone()[0]
                profile["type"]       = "categorical"
                profile["distinct"]   = [r[0] for r in distinct]
                profile["n_distinct"] = n_distinct
            else:
                stats = self._db.execute(
                    f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}") FROM jobs'
                ).fetchone()
                profile["type"] = "numeric"
                profile["min"]  = stats[0]
                profile["max"]  = stats[1]
                profile["avg"]  = round(stats[2], 4) if stats[2] is not None else None

            return profile

        except Exception as exc:
            return {"error": str(exc)}

    # ── Formatting ────────────────────────────────────────────────────────────

    def format(self, profiles: dict[str, dict]) -> str:
        """Ported verbatim from DataExplorerAgent._format."""
        lines = ["Column profiles from the Fugaku jobs table:\n"]
        for col, p in profiles.items():
            if "error" in p:
                lines.append(f"  {col}: ERROR — {p['error']}")
                continue

            lines.append(f"  {col} ({p['dtype']})")
            lines.append(f"    nulls: {p['null_count']} / {p['total_rows']}")

            if p.get("type") == "categorical":
                lines.append(
                    "    TYPE: CATEGORICAL — do NOT use with AVG, SUM, or any "
                    "numeric operation. Never cast to numeric."
                )
                lines.append(
                    f"    distinct values ({p['n_distinct']} total): {p['distinct']}"
                )
            elif p.get("type") == "numeric":
                lines.append(
                    f"    range: {p['min']} → {p['max']},  avg: {p['avg']}"
                )
            lines.append("")

        return "\n".join(lines)

    # ── Public entry point ────────────────────────────────────────────────────

    def explore(self, query: str, full_categorical_scan: bool = False) -> str:
        """
        Profile the columns relevant to `query`.

        Returns the rendered profile text, or a plain-text "could not identify"
        message when no column matches — the untyped analogue of the MAS
        DataExplorer's REJECT.
        """
        columns = self.select_columns(query, full_categorical_scan)
        if not columns:
            return "Could not identify any relevant columns from the query."

        profiles = {col: self.profile(col) for col in columns}
        summary = self.format(profiles)

        # The MAS agent raises NULL_VALUES as a typed flag here. Baselines have
        # no flag carrier, so the same observation is appended as prose — this
        # is the ablation variable, not a capability difference.
        total = next(
            (p.get("total_rows", 0) for p in profiles.values() if "total_rows" in p),
            0,
        )
        for col, p in profiles.items():
            if "error" in p:
                continue
            if p.get("null_count", 0) / max(total, 1) > 0.2:
                summary += (
                    f"\nNote: column '{col}' has a high null rate — "
                    f"results derived from it may be incomplete.\n"
                )
                break

        return summary


# ── Module-level singleton (DuckDB connection reuse) ──────────────────────────

_explorer: Optional[DataExplorer] = None


def get_explorer() -> DataExplorer:
    global _explorer
    if _explorer is None:
        _explorer = DataExplorer()
    return _explorer


def has_groupby_intent(query: str) -> bool:
    """Ported verbatim from agents/sql_agent.py::_has_groupby_intent."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in GROUP_BY_KEYWORDS)


def needs_profile(query: str, llm, model: str = "gpt-4o-mini") -> bool:
    """
    Cheap yes/no gate — same prompt and same model tier as the MAS SQL agent's
    _needs_profile (BaseAgent._llm_bool defaults to gpt-4o-mini).
    """
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": NEEDS_PROFILE_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=5,
        )
        return (resp.choices[0].message.content or "").strip().lower().startswith("yes")
    except Exception:
        return False


def explore_for_query(query: str, llm=None, force: bool = False) -> str:
    """
    Full MAS-equivalent path: gate → decide scan mode → profile.

    Returns "" when the gate says profiling is not needed, so callers can skip
    injecting anything into their context.
    """
    if not force:
        if llm is None or not needs_profile(query, llm):
            return ""
    return get_explorer().explore(query, full_categorical_scan=has_groupby_intent(query))

"""
orchestrator.py  —  HPC Multi-Agent System (P2P, SIGDIAL 2026)

P2P peer graph (v3 — full pipeline)
------------------------------------
  user
    └─▶ gateway
          ├─▶ sql_agent
          │     ├─▶ data_explorer   (SEMANTIC_AMBIGUITY: column profiling)
          │     └─▶ synthesizer
          │           ├─▶ reflector       (VALIDATE → CONFIRM/CHALLENGE)
          │           └─▶ data_explorer   (DATA_INSUFFICIENCY: after CHALLENGE)
          ├─▶ doc_agent
          │     └─▶ synthesizer
          └─▶ pa_agent
                ├─▶ sql_agent   (DATA_INSUFFICIENCY: historical stats enrichment)
                ├─▶ doc_agent   (KNOWLEDGE_GAP: policy/procedure enrichment)
                └─▶ synthesizer

Agent responsibilities:
  gateway       — classify intent (sql/doc/predict), route bare REQUEST
  sql_agent     — generate + execute SQL; delegates profiling and synthesis
  doc_agent     — hybrid BM25+dense retrieval from Fugaku manuals
  pa_agent      — ML-based job failure/runtime/energy prediction
  data_explorer — profile column distributions (no LLM)
  synthesizer   — format raw results into user-facing prose
  reflector     — review answer quality; CONFIRM or CHALLENGE
"""
from __future__ import annotations

import asyncio
import sys
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import config
from agents.data_explorer_agent import DataExplorerAgent
from agents.doc_agent import DocAgent
from agents.gateway_agent import GatewayAgent
from agents.pa_agent import PAAgent
from agents.reflector_agent import ReflectorAgent
from agents.sql_agent import SQLAgent
from agents.synthesizer_agent import SynthesizerAgent
from core.shared_log import SharedLog

console = Console()


class HpcOrchestrator:
    """
    Instantiates all agents, wires P2P peers, and drives sessions.

    Usage:
        orch = HpcOrchestrator(verbose=True)
        answer = asyncio.run(orch.run("How many failed jobs in 2023?"))
    """

    def __init__(self, verbose: bool = False) -> None:
        self.log     = SharedLog(config.LOG_DB_PATH)
        self.verbose = verbose

        # ── Instantiate agents ────────────────────────────────────────────────
        self.gateway     = GatewayAgent(self.log, verbose=verbose)
        self.sql         = SQLAgent(self.log, verbose=verbose)
        self.explorer    = DataExplorerAgent(self.log, verbose=verbose)
        self.doc         = DocAgent(self.log, verbose=verbose)
        self.pa          = PAAgent(self.log, verbose=verbose)
        self.synthesizer = SynthesizerAgent(self.log, verbose=verbose)
        self.reflector   = ReflectorAgent(self.log, verbose=verbose)

        # ── Wire P2P peers ────────────────────────────────────────────────────
        # gateway → sql  (data queries)
        self.gateway.register_peer("sql_agent", self.sql)

        # gateway → doc  (documentation / policy queries)
        self.gateway.register_peer("doc_agent", self.doc)

        # gateway → pa  (prediction queries)
        self.gateway.register_peer("pa_agent", self.pa)

        # sql → data_explorer (SEMANTIC_AMBIGUITY: optional column profiling)
        self.sql.register_peer("data_explorer", self.explorer)

        # sql → synthesizer (forwards result after execution)
        self.sql.register_peer("synthesizer", self.synthesizer)

        # sql → doc (KNOWLEDGE_GAP: when query has both SQL and doc sub-questions)
        self.sql.register_peer("doc_agent", self.doc)

        # sql → pa  (KNOWLEDGE_GAP: when query has a prediction sub-question)
        self.sql.register_peer("pa_agent", self.pa)

        # doc → synthesizer (forwards answer for formatting + reflection)
        self.doc.register_peer("synthesizer", self.synthesizer)

        # doc → sql (DATA_INSUFFICIENCY: when doc-entry query also needs historical stats)
        self.doc.register_peer("sql_agent", self.sql)

        # doc → pa (KNOWLEDGE_GAP: when doc-entry query also asks for job risk prediction)
        self.doc.register_peer("pa_agent", self.pa)

        # pa → sql (DATA_INSUFFICIENCY: historical stats enrichment)
        self.pa.register_peer("sql_agent", self.sql)

        # pa → doc (KNOWLEDGE_GAP: policy/procedure enrichment)
        self.pa.register_peer("doc_agent", self.doc)

        # pa → synthesizer (final formatting + reflector validation)
        self.pa.register_peer("synthesizer", self.synthesizer)

        # synthesizer → reflector (VALIDATE — quality review)
        self.synthesizer.register_peer("reflector", self.reflector)

        # synthesizer → data_explorer (DATA_INSUFFICIENCY — after CHALLENGE, for units)
        self.synthesizer.register_peer("data_explorer", self.explorer)

        # ── Announce wiring ───────────────────────────────────────────────────
        console.print(
            Panel(
                "[bold green]HPC Multi-Agent System online[/bold green]\n\n"
                "  Peer graph:\n"
                "    user → [cyan]gateway[/cyan]\n"
                "             ├─▶ [cyan]sql_agent[/cyan]\n"
                "             │     ├─▶ [cyan]data_explorer[/cyan]  (SEMANTIC_AMBIGUITY)\n"
                "             │     ├─▶ [cyan]doc_agent[/cyan]      (KNOWLEDGE_GAP)\n"
                "             │     └─▶ [cyan]synthesizer[/cyan]\n"
                "             │           ├─▶ [cyan]reflector[/cyan]  (VALIDATE)\n"
                "             │           └─▶ [cyan]data_explorer[/cyan]  (DATA_INSUFFICIENCY)\n"
                "             ├─▶ [cyan]doc_agent[/cyan]\n"
                "             │     └─▶ [cyan]synthesizer[/cyan]\n"
                "             └─▶ [cyan]pa_agent[/cyan]\n"
                "                   ├─▶ [cyan]sql_agent[/cyan]  (DATA_INSUFFICIENCY)\n"
                "                   ├─▶ [cyan]doc_agent[/cyan]   (KNOWLEDGE_GAP)\n"
                "                   └─▶ [cyan]synthesizer[/cyan]\n\n"
                "  Triggers: SEMANTIC_AMBIGUITY | DATA_INSUFFICIENCY | KNOWLEDGE_GAP\n"
                "  All 7 agents online.",
                title="[bold]HPC MAS v3[/bold]",
                border_style="bold blue",
                expand=False,
            )
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self, user_query: str, session_id: str | None = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())[:12]

        console.print()
        console.print(Rule(f"[bold white]Session  {session_id}[/bold white]"))
        console.print(f"[bold]Query:[/bold] {user_query}\n")

        answer = await self.gateway.run(user_query, session_id)

        console.print()
        console.print(Panel(
            answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            expand=False,
        ))

        self._print_da_summary(session_id)
        self._print_reasoning(session_id)
        return answer

    # ── DA trace display ──────────────────────────────────────────────────────

    def _print_da_summary(self, session_id: str) -> None:
        messages = self.log.get_session(session_id)
        console.print()
        console.print(Rule("[bold]Dialogue Act Trace[/bold]"))

        tbl = Table(show_header=True, header_style="bold", border_style="dim")
        tbl.add_column("Turn",      style="dim",     width=5)
        tbl.add_column("Sender",    style="cyan",    width=18)
        tbl.add_column("Recipient", style="yellow",  width=18)
        tbl.add_column("DA Type",   style="magenta", width=12)
        tbl.add_column("Trigger",   style="blue",    width=18)
        tbl.add_column("Conf",      style="green",   width=6)
        tbl.add_column("Flags",     style="red")

        for m in messages:
            conf    = f"{m.confidence:.2f}" if m.confidence is not None else "—"
            flags   = ", ".join(f.value for f in m.uncertainty_flags) or "—"
            trigger = m.delegation_trigger.value if m.delegation_trigger else "—"
            tbl.add_row(
                str(m.turn),
                m.sender_name(),
                m.recipient_name(),
                m.da_type.value,
                trigger,
                conf,
                flags,
            )

        console.print(tbl)
        console.print(f"[dim]Session: {session_id}  |  Agents: gateway/sql/doc/pa/synthesizer/reflector/data_explorer  |  Log: {config.LOG_DB_PATH}[/dim]")

    def _print_reasoning(self, session_id: str) -> None:
        traces = self.log.get_reasoning(session_id)
        if not traces:
            return
        console.print()
        console.print(Rule("[bold]Agent Reasoning Trace[/bold]"))
        for t in traces:
            console.print(
                f"  [cyan]{t['agent']}[/cyan]  [dim]{t['step']}[/dim]\n"
                f"    [white]{t['note']}[/white]"
            )


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    query   = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
              "How many jobs were submitted for each job class (pclass)?"
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    orch   = HpcOrchestrator(verbose=verbose)
    answer = await orch.run(query)
    return answer


if __name__ == "__main__":
    asyncio.run(main())

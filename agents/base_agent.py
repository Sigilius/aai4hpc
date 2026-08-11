"""
agents/base_agent.py

Foundation for every agent in the Fugaku HPC multi-agent system.

Design principles
-----------------
* P2P communication  — agents call each other directly via ask_peer();
                       the orchestrator (Gateway) is only involved at
                       entry and exit, not on every inter-agent turn.
* DA-tagged messages — every message carries a DialogueActType so the
                       full conversation trace is analyzable for SIGDIAL.
* Version stamped    — sender field is always "agent_name/version",
                       logged on startup and on every message.
* Shared log         — every sent/received message is appended to the
                       SQLite SharedLog for post-hoc DA analysis.
* No generic ReAct   — each agent manages its own LLM loop; base_agent
                       only provides infrastructure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config
from core.message_schema import (
    A2AMessage,
    DelegationTrigger,
    DialogueActType,
    UncertaintyFlag,
)


def _make_llm_client() -> OpenAI:
    """
    Build an OpenAI-compatible client pointed at the Azure AI Foundry endpoint.

    The standard openai.OpenAI client works with Azure AI Foundry project-scoped
    endpoints when base_url is set to the full /openai/v1 path.
    Authentication uses the api_key as a Bearer token (Azure accepts this).

    .env must contain:   AZURE_OPENAI_API_KEY=<your-key>
                         AZURE_OPENAI_ENDPOINT=<your-endpoint>
    """
    return OpenAI(
        base_url=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
    )
from core.shared_log import SharedLog

console = Console()


class BaseAgent(ABC):
    """
    Abstract base for every HPC agent.

    Subclasses declare class-level attributes:
        name    : str   — unique agent identifier (snake_case)
        version : str   — semantic version string, e.g. "1.0.0"

    Subclasses implement:
        handle(msg: A2AMessage) -> A2AMessage
    """

    name:    str
    version: str

    def __init__(self, log: SharedLog, verbose: bool = False) -> None:
        self.log     = log
        self.verbose = verbose
        self.llm     = _make_llm_client()   # Azure AI Foundry endpoint
        self.model   = config.MODEL
        self._peers: Dict[str, "BaseAgent"] = {}
        self._log_startup()

    # ── Identity ──────────────────────────────────────────────────────────────

    def agent_id(self) -> str:
        """Return the versioned agent identifier used in every message sender field."""
        return f"{self.name}/{self.version}"

    # ── Startup banner (version logged to console) ────────────────────────────

    def _log_startup(self) -> None:
        from shared.db import data_source_label
        console.print(Panel(
            f"[bold green]Agent online[/bold green]\n"
            f"  Name      : [cyan]{self.name}[/cyan]\n"
            f"  Version   : [yellow]{self.version}[/yellow]\n"
            f"  Model     : [blue]{self.model}[/blue]\n"
            f"  Data      : [dim]{data_source_label()}[/dim]",
            title=f"[bold]{self.name}[/bold]",
            border_style="dim green",
            expand=False,
        ))

    # ── Peer Registry (P2P wiring) ────────────────────────────────────────────

    def register_peer(self, peer_name: str, agent: "BaseAgent") -> None:
        self._peers[peer_name] = agent
        if self.verbose:
            console.print(f"  [dim][{self.name}] peer wired → {peer_name}[/dim]")

    def has_peer(self, peer_name: str) -> bool:
        return peer_name in self._peers

    def peer_names(self) -> List[str]:
        return list(self._peers.keys())

    # ── A2A Communication ──────────────────────────────────────────────────────

    async def ask_peer(
        self, peer_name: str, msg: A2AMessage
    ) -> A2AMessage:
        """
        Send a DA-typed message directly to a named peer agent.

        Steps:
          1. Validate peer is registered.
          2. Append the outgoing message to SharedLog.
          3. Print the outgoing message to console.
          4. Await peer.handle(msg).
          5. Return the peer's response (already logged inside peer.reply()).

        The orchestrator (Gateway) is NOT in the loop for these calls.
        This produces true agent-to-agent dialog turns in the shared log.
        """
        if peer_name not in self._peers:
            raise ValueError(
                f"[{self.name}] tried to contact unknown peer '{peer_name}'. "
                f"Registered peers: {list(self._peers.keys())}"
            )

        # Log + print the outgoing request
        self.log.append(msg)
        self._print_msg(msg, direction="out")

        if self.verbose:
            console.print(
                f"  [dim][{self.name}] ──{msg.da_type.value}──▶ {peer_name}: "
                f"{msg.content[:100].replace(chr(10), ' ')}[/dim]"
            )

        response = await self._peers[peer_name].handle(msg)

        if self.verbose:
            console.print(
                f"  [dim][{self.name}] ◀──{response.da_type.value}── {peer_name}: "
                f"{response.content[:100].replace(chr(10), ' ')}[/dim]"
            )

        return response

    # ── Message builders ───────────────────────────────────────────────────────

    def reply(
        self,
        incoming: A2AMessage,
        da_type: DialogueActType,
        content: str,
        *,
        confidence: Optional[float]            = None,
        flags:      Optional[List[UncertaintyFlag]] = None,
        trigger:    Optional[DelegationTrigger]     = None,
        sql_query:  Optional[str]                   = None,
        metadata:   Optional[dict]                  = None,
    ) -> A2AMessage:
        """
        Build a reply to an incoming message.

        Sets sender = self.agent_id(), recipient = incoming.sender,
        auto-increments turn, appends to SharedLog, prints to console.

        Use this as the return value from handle():
            return self.reply(msg, DialogueActType.INFORM, content="...", confidence=0.95)
        """
        out = A2AMessage(
            session_id         = incoming.session_id,
            turn               = self.log.next_turn(incoming.session_id),
            sender             = self.agent_id(),
            recipient          = incoming.sender,
            da_type            = da_type,
            delegation_trigger = trigger,
            content            = content,
            confidence         = confidence,
            uncertainty_flags  = flags or [],
            sql_query          = sql_query,
            metadata           = metadata or {},
        )
        self.log.append(out)
        self._print_msg(out, direction="out")
        return out

    def make_request(
        self,
        recipient_agent: str,
        incoming: A2AMessage,
        da_type: DialogueActType,
        content: str,
        *,
        trigger:  Optional[DelegationTrigger]     = None,
        flags:    Optional[List[UncertaintyFlag]] = None,
        metadata: Optional[dict]                  = None,
    ) -> A2AMessage:
        """
        Build an outgoing peer-request message (not a direct reply).

        Does NOT append to log — ask_peer() does that when the message is sent.

        Example:
            req = self.make_request(
                "doc_agent", incoming,
                DialogueActType.REQUEST,
                "What is the walltime limit for large jobs?",
                trigger=DelegationTrigger.KNOWLEDGE_GAP,
            )
            doc_resp = await self.ask_peer("doc_agent", req)
        """
        return A2AMessage(
            session_id         = incoming.session_id,
            turn               = self.log.next_turn(incoming.session_id),
            sender             = self.agent_id(),
            recipient          = f"{recipient_agent}/1.0.0",
            da_type            = da_type,
            delegation_trigger = trigger,
            content            = content,
            uncertainty_flags  = flags or [],
            metadata           = metadata or {},
        )

    # ── Narrative reasoning log ────────────────────────────────────────────────

    def _note(self, msg: A2AMessage, step: str, text: str) -> None:
        """
        Write a one-line reasoning note to the narrative log.

        Call this anywhere inside handle() to record WHY the agent made a
        decision — classification choices, SQL attempts, peer call rationale, etc.

        Example:
            self._note(msg, "classify",   "query asks for counts → sql")
            self._note(msg, "sql_gen",    f"generated: {sql[:80]}")
            self._note(msg, "sql_retry",  "syntax error on attempt 1, retrying")
            self._note(msg, "peer_call",  "calling data_explorer for pclass values")
        """
        self.log.log_reasoning(msg.session_id, self.agent_id(), step, text)

    # ── LLM helper ─────────────────────────────────────────────────────────────

    def _llm(
        self,
        prompt: str,
        *,
        system:          str            = "",
        response_format: Optional[dict] = None,
        max_tokens:      Optional[int]  = None,
        model:           Optional[str]  = None,
    ) -> str:
        """
        Single-turn synchronous LLM call.

        Args:
            prompt          : The user-turn content.
            system          : Optional system message.
            response_format : e.g. {"type": "json_object"} for structured output.
            max_tokens      : Optional token cap (use for cheap yes/no calls).
            model           : Override self.model (e.g. use gpt-4o-mini for cheap gates).

        Returns:
            The assistant's text response, stripped.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model":       model or self.model,
            "messages":    messages,
            "temperature": 0,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        resp = self.llm.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def _llm_bool(self, prompt: str, *, model: str = "gpt-4o-mini") -> bool:
        """
        Cheap yes/no LLM gate.
        Returns True if the model replies 'yes', False otherwise.
        Use for decisions like needs_doc_context, needs_profile, etc.
        """
        ans = self._llm(prompt, model=model, max_tokens=5)
        return ans.strip().lower().startswith("yes")

    # ── Console print ──────────────────────────────────────────────────────────

    def _print_msg(self, msg: A2AMessage, direction: str = "out") -> None:
        arrow  = "──▶" if direction == "out" else "◀──"
        flags  = (
            f"  ⚠[{', '.join(f.value for f in msg.uncertainty_flags)}]"
            if msg.uncertainty_flags else ""
        )
        conf   = f"  [{msg.confidence:.0%}]" if msg.confidence is not None else ""
        t = Text()
        t.append(f"{msg.sender_name()}", style="bold cyan")
        t.append(f" {arrow} ", style="white")
        t.append(f"{msg.recipient_name()}", style="bold yellow")
        t.append(f"  DA={msg.da_type.value}", style="bold magenta")
        t.append(conf,  style="green")
        t.append(flags, style="red")
        console.print(t)

    # ── Subclass interface ──────────────────────────────────────────────────────

    @abstractmethod
    async def handle(self, msg: A2AMessage) -> A2AMessage:
        """
        Core agent logic — implement in every subclass.

        Pattern:
            async def handle(self, msg):
                query = msg.content
                # ... do work, optionally call peers ...
                doc = await self.ask_peer("doc_agent",
                    self.make_request("doc_agent", msg,
                        DialogueActType.REQUEST, query,
                        trigger=DelegationTrigger.KNOWLEDGE_GAP))
                # ... synthesize ...
                return self.reply(msg, DialogueActType.INFORM, content=answer,
                                  confidence=0.92)
        """
        ...

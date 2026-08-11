"""
research/shared/mas_agents.py

The MAS agent implementations, exposed to the baselines as plain-text calls.

Why
---
The paper states that prompts, agent roles and tool implementations are held
constant across configurations. They were not. The MAS SQL role is
agents/sql_agent.py — 770 lines, nine LLM call sites covering sub-question
extraction, delegation gates, column profiling, generation and a sufficiency
check. The Blackboard SQL role was a 50-line function with one LLM call.
Measured directly, swapping the MAS SQL agent into Blackboard closed 100% of the
fact-recall gap on the five queries where that gap was largest.

So the reported MAS-vs-baseline difference confounded two things: the
communication architecture (the intended variable) and the agent implementation.
This module removes the second one. Every configuration now runs the same agent
code; only the substrate that carries results between agents differs.

What is deliberately NOT shared
-------------------------------
The typed A2A layer. Each agent here is constructed with an empty peer registry,
so every `self.has_peer(...)` guard in agents/*.py is False and no agent
delegates. The reply's `da_type`, `uncertainty_flags` and `delegation_trigger`
are discarded at this boundary and only `.content` is returned. A refusal
therefore arrives at the baseline as prose, exactly as before — the baseline's
own dict / log / peer channel carries it, or fails to.

That is the ablation variable, and it is now the only one.

Upstream context is appended to the request as prose, mirroring how the
baselines already passed results between roles.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Optional

from core.message_schema import A2AMessage, DialogueActType
from core.shared_log import SharedLog
from agents.sql_agent import SQLAgent
from agents.pa_agent import PAAgent
from agents.doc_agent import DocAgent
from agents.data_explorer_agent import DataExplorerAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.reflector_agent import ReflectorAgent

_agents: Optional[dict] = None
_pa_a2a = None
_turn = 0


def _get():
    """Build the MAS agents once, each with an empty peer registry."""
    global _agents
    if _agents is None:
        log = SharedLog(os.path.join(tempfile.mkdtemp(prefix="mas_adapter_"), "log.db"))
        _agents = {
            "sql":       SQLAgent(log, verbose=False),
            "pa":        PAAgent(log, verbose=False),
            "doc":       DocAgent(log, verbose=False),
            "explorer":  DataExplorerAgent(log, verbose=False),
            "synth":     SynthesizerAgent(log, verbose=False),
            "reflector": ReflectorAgent(log, verbose=False),
        }
        # No register_peer() calls: every has_peer() guard stays False, so the
        # agents never delegate and never emit typed messages to one another.
    return _agents


def _ask(agent, content: str, da=DialogueActType.REQUEST, metadata=None) -> str:
    """Run one MAS agent and return its reply text, dropping all typed fields."""
    global _turn
    _turn += 1
    msg = A2AMessage(
        sender="orchestrator", recipient=agent.name, da_type=da,
        content=content, session_id="baseline", turn=_turn,
        metadata=metadata or {},
    )
    try:
        resp = asyncio.run(agent.handle(msg))
    except Exception as exc:
        return f"{agent.name} error: {exc}"
    return resp.content or ""


def _with_context(query: str, context: str) -> str:
    if not context:
        return query
    return (f"{query}\n\n"
            f"Results already produced by other agents (plain text, use only if relevant):\n"
            f"{context}")


# ── Public plain-text interface ───────────────────────────────────────────────

def sql(query: str, context: str = "") -> str:
    return _ask(_get()["sql"], _with_context(query, context))


def pa(query: str, context: str = "") -> str:
    return _ask(_get()["pa"], _with_context(query, context))


def doc(query: str, context: str = "") -> str:
    return _ask(_get()["doc"], _with_context(query, context))


def explore(query: str, full_scan: bool = False) -> str:
    return _ask(_get()["explorer"], query,
                metadata={"full_categorical_scan": full_scan})


def synthesize(query: str, raw: str) -> str:
    """Format upstream results into a user-facing answer (no reflector peer)."""
    return _ask(_get()["synth"], raw, da=DialogueActType.INFORM,
                metadata={"original_query": query})


class _UntypedPeer:
    """
    Wraps a MAS agent so a caller receives its text but none of its typed fields.

    The Unstructured-A2A configuration keeps directed peer messaging — that is
    its defining property — but strips the typed contract. Registering this shim
    as a peer gives the shared MAS PAAgent a real delegation channel whose
    replies always arrive as INFORM with no uncertainty_flags and no
    delegation_trigger. The caller must therefore infer epistemic status from
    prose, which is exactly the difference from the Structured MAS.
    """

    def __init__(self, inner):
        self._inner = inner
        self.name = inner.name

    async def handle(self, msg: A2AMessage) -> A2AMessage:
        resp = await self._inner.handle(msg)
        return A2AMessage(
            sender=self.name, recipient=msg.sender,
            da_type=DialogueActType.INFORM,      # never REJECT/CAVEAT
            content=resp.content or "",          # flags and trigger dropped
            session_id=msg.session_id, turn=msg.turn + 1,
        )


def pa_with_untyped_peers(query: str, context: str = "") -> str:
    """
    Shared MAS PAAgent with directed — but untyped — peer access.

    Used by the Unstructured-A2A configuration only. The agent may delegate to
    sql_agent and doc_agent exactly as it does in the MAS; what it cannot do is
    read a da_type or an uncertainty_flag off the reply.
    """
    global _pa_a2a
    a = _get()
    if _pa_a2a is None:
        # A separate instance: registering peers on the shared PA singleton would
        # silently give the Blackboard and Unstructured-Blackboard configurations
        # a delegation channel they must not have.
        _pa_a2a = PAAgent(a["pa"].log, verbose=False)
        _pa_a2a.register_peer("sql_agent", _UntypedPeer(a["sql"]))
        _pa_a2a.register_peer("doc_agent", _UntypedPeer(a["doc"]))
    return _ask(_pa_a2a, _with_context(query, context))


def reflect(query: str, draft: str) -> str:
    """Single review pass; returns the reflector's verdict text."""
    return _ask(_get()["reflector"], draft, da=DialogueActType.VALIDATE,
                metadata={"original_query": query})

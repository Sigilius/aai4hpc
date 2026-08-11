from __future__ import annotations

from core.message_schema import (
    DelegationTrigger,
    DialogueActType,
    UncertaintyFlag,
)

DA_DESCRIPTIONS: dict[DialogueActType, str] = {
    DialogueActType.USER_QUERY:  "Initial query from the user",
    DialogueActType.REQUEST:     "Asking another agent for data or computation",
    DialogueActType.INFORM:      "Delivering a result or data to another agent",
    DialogueActType.CLARIFY:     "Asking for disambiguation of an ambiguous term or intent",
    DialogueActType.DELEGATE:    "Handing off a sub-task with a stated trigger reason",
    DialogueActType.CAVEAT:      "Attaching an uncertainty or warning flag to a result",
    DialogueActType.VALIDATE:    "Checking completeness and correctness of a result",
    DialogueActType.CHALLENGE:   "Flagging a specific inconsistency or error",
    DialogueActType.CONFIRM:     "Accepting a result as complete and correct",
    DialogueActType.REJECT:      "Refusing a task as out of scope",
    DialogueActType.SYNTHESIZE:  "Assembling partial results into a unified answer",
    DialogueActType.TERMINATE:   "Signalling end of session",
}

TRIGGER_DESCRIPTIONS: dict[DelegationTrigger, str] = {
    DelegationTrigger.DATA_INSUFFICIENCY: "Insufficient data to answer locally",
    DelegationTrigger.KNOWLEDGE_GAP:      "Lacks domain knowledge for this sub-task",
    DelegationTrigger.SEMANTIC_AMBIGUITY: "Query or term is ambiguous",
}

FLAG_DESCRIPTIONS: dict[UncertaintyFlag, str] = {
    UncertaintyFlag.CONFIDENCE_LOW:   "Prediction or retrieval confidence is below threshold",
    UncertaintyFlag.NOT_FOUND:        "Requested data was not found in any source",
    UncertaintyFlag.PARTIALLY_FOUND:  "Only partial data found; answer may be incomplete",
    UncertaintyFlag.SCHEMA_AMBIGUITY: "Column or table schema is ambiguous or mismatched",
    UncertaintyFlag.NULL_VALUES:      "Significant null values detected in retrieved data",
    UncertaintyFlag.STALE_DATA:       "Data may be outdated relative to the query time range",
}


def da_prompt_block(allowed: list[DialogueActType]) -> str:
    """Return a prompt snippet listing only the DA types this agent may emit."""
    lines = ["You MUST choose one of these Dialogue Act (DA) types for your response:"]
    for da in allowed:
        lines.append(f"  - {da.value}: {DA_DESCRIPTIONS[da]}")
    return "\n".join(lines)


def uncertainty_prompt_block() -> str:
    lines = ["Uncertainty flags you may attach (zero or more):"]
    for flag, desc in FLAG_DESCRIPTIONS.items():
        lines.append(f"  - {flag.value}: {desc}")
    return "\n".join(lines)

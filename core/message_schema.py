from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DialogueActType(str, Enum):
    USER_QUERY = "USER_QUERY"   # initial user input
    REQUEST    = "REQUEST"      # ask another agent for data/computation
    INFORM     = "INFORM"       # deliver a result
    CLARIFY    = "CLARIFY"      # ask for disambiguation
    DELEGATE   = "DELEGATE"     # hand off a sub-task
    CAVEAT     = "CAVEAT"       # attach uncertainty/warning to a result
    VALIDATE   = "VALIDATE"     # check completeness/correctness
    CHALLENGE  = "CHALLENGE"    # flag inconsistency or error
    CONFIRM    = "CONFIRM"      # accept result as correct
    REJECT     = "REJECT"       # refuse task as out of scope
    SYNTHESIZE = "SYNTHESIZE"   # assemble partial results into whole
    TERMINATE  = "TERMINATE"    # end of session


class DelegationTrigger(str, Enum):
    DATA_INSUFFICIENCY = "data_insufficiency"   # agent lacks enough data
    KNOWLEDGE_GAP      = "knowledge_gap"        # agent lacks domain knowledge
    SEMANTIC_AMBIGUITY = "semantic_ambiguity"   # query is ambiguous


class UncertaintyFlag(str, Enum):
    CONFIDENCE_LOW   = "confidence_low"   # model confidence below threshold
    NOT_FOUND        = "not_found"        # data not in any source
    PARTIALLY_FOUND  = "partially_found"  # only partial data available
    SCHEMA_AMBIGUITY = "schema_ambiguity" # column/table mapping unclear
    NULL_VALUES      = "null_values"      # significant nulls in retrieved data
    STALE_DATA       = "stale_data"       # data may be outdated
    LOW_SAMPLE            = "low_sample"            # prediction backed by very few historical jobs
    UNCONFIRMED_REFLECTOR = "unconfirmed_reflector"  # answer bypassed reflector after 2 challenges


class A2AMessage(BaseModel):
    id:                 str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id:         str
    turn:               int
    sender:             str                        # "agent_name/version"
    recipient:          str                        # "agent_name/version"
    da_type:            DialogueActType
    delegation_trigger: Optional[DelegationTrigger] = None
    content:            str
    confidence:         Optional[float] = None     # 0.0 – 1.0
    uncertainty_flags:  List[UncertaintyFlag] = Field(default_factory=list)
    sql_query:          Optional[str] = None       # filled by SQL agent
    raw_data:           Optional[Any] = None       # tabular result rows
    metadata:           Dict[str, Any] = Field(default_factory=dict)
    timestamp:          str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def agent_name(self) -> str:
        return self.sender.split("/")[0]

    def sender_name(self) -> str:
        return self.sender.split("/")[0]

    def recipient_name(self) -> str:
        return self.recipient.split("/")[0]

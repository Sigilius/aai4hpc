# single_agent_baseline/logger.py
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

class SingleAgentLogger:
    _instance = None

    def __init__(self, log_dir: str = "logs/single_agent/both_4_1"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_dir) / f"single_agent_{ts}.jsonl"
        self._file = open(self.log_path, "a", buffering=1)
        self.query_id: str | None = None

    @classmethod
    def get(cls) -> "SingleAgentLogger":
        if cls._instance is None:
            cls._instance = SingleAgentLogger()
        return cls._instance

    def new_query(self, query: str) -> str:
        self.query_id = uuid.uuid4().hex[:8]
        self._write({
            "event": "QUERY_START",
            "query_id": self.query_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return self.query_id

    def log_decision(self, decision: str, value, reason: str = ""):
        self._write({
            "event": "DECISION",
            "query_id": self.query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "single_agent",
            "decision": decision,
            "value": value,
            "reason": reason,
        })

    def log_llm_step(self, step: int, finish_reason: str | None = None):
        self._write({
            "event": "LLM_STEP",
            "query_id": self.query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "single_agent",
            "step": step,
            "finish_reason": finish_reason,
        })

    def log_tool_call(self, step: int, tool_call_id: str, tool_name: str, arguments: dict):
        self._write({
            "event": "TOOL_CALL",
            "query_id": self.query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "single_agent",
            "step": step,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
        })

    def log_tool_result(self, step: int, tool_call_id: str, tool_name: str, result: str):
        self._write({
            "event": "TOOL_RESULT",
            "query_id": self.query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "single_agent",
            "step": step,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result": result,
        })

    def log_final_answer(self, answer: str):
        self._write({
            "event": "FINAL_ANSWER",
            "query_id": self.query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "single_agent",
            "answer": answer,
        })

    def log_intermediate_response(self, step: int, content: str, finish_reason: str | None = None):
        self._write({
            "event": "INTERMEDIATE_RESPONSE",
            "query_id": self.query_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "single_agent",
            "step": step,
            "finish_reason": finish_reason,
            "content": content,
        })

    def _write(self, entry: dict):
        self._file.write(json.dumps(entry) + "\n")

    def close(self):
        self._file.close()
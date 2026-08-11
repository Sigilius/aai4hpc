"""
single_agent_baseline/agent.py

Single-agent GPT-4o baseline with three tools:
  - rag_search   : Fugaku documentation retrieval + rerank
  - run_sql      : DuckDB query execution on Fugaku parquet
  - predict_job  : PA model prediction

One LLM, all tools, one context window.
"""

import os
import sys
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../analytics")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../mas_system")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared")))

GPT_VERSION = os.getenv("GPT_VERSION", "gpt-4o")

_AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")


def _make_client() -> OpenAI:
    """Azure AI Foundry project-scoped endpoint via standard OpenAI client."""
    return OpenAI(base_url=_AZURE_ENDPOINT, api_key=_AZURE_API_KEY)
# print(f"======\n\nGPT_VERSION: {GPT_VERSION} in {__file__}\n\n=======")

from tools import (
    rag_search,
    run_sql,
    predict_job,
    profile_columns,
    get_schema_context,
)
from logger import SingleAgentLogger

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Retrieve relevant sections from the Fugaku supercomputer documentation. "
                "Use this for questions about job script directives, system policies, "
                "walltime limits, power control, job classes, scheduling rules, "
                "error handling, or any Fugaku-specific operational knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query for the documentation."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a DuckDB SQL query against the Fugaku job telemetry dataset. "
                "The dataset contains millions of job records with columns for user, "
                "job name, node counts, duration, power consumption, energy, exit state, "
                "timestamps, and more. Use this for quantitative questions about "
                "job counts, averages, trends, distributions, or user activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Valid DuckDB SQL query. Table name is 'jobs'."
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_job",
            "description": (
                "Run the predictive analytics model to estimate failure risk, "
                "expected runtime, and energy consumption for a job before submission. "
                "Use this when the user asks about failure probability, risk assessment, "
                "expected duration, or energy consumption for a specific job configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nnumr": {
                        "type": "integer",
                        "description": "Number of nodes requested."
                    },
                    "elpl": {
                        "type": "number",
                        "description": "Walltime limit in seconds."
                    },
                    "pclass": {
                        "type": "string",
                        "enum": ["compute-bound", "memory-bound"],
                        "description": "Job class: compute-bound or memory-bound."
                    },
                    "usr": {
                        "type": "string",
                        "description": "User ID if available."
                    },
                    "jnam": {
                        "type": "string",
                        "description": "Job name if available."
                    }
                },
                "required": ["nnumr", "elpl", "pclass"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "profile_columns",
            "description": (
                "Profile the columns of the Fugaku job telemetry table before querying "
                "them: data type, null rate, and either the full set of distinct values "
                "(for categorical columns) or the min/max/average (for numeric columns). "
                "Use this BEFORE run_sql when you are unsure what values a column "
                "actually holds, whether a column is categorical or numeric, what units "
                "it uses, or whether it has enough non-null data to support the answer. "
                "This is the reliable way to avoid inventing column values or averaging "
                "a categorical column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Text naming the columns to profile. Mention the column "
                            "names directly (e.g. 'pclass jobenv_req nnumr')."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def _dispatch(name: str, arguments: dict) -> str:
    if name == "rag_search":
        return rag_search(arguments["query"])
    if name == "run_sql":
        return run_sql(arguments["sql"])
    if name == "predict_job":
        return predict_job(arguments)
    if name == "profile_columns":
        return profile_columns(arguments["query"])
    return f"Unknown tool: {name}"


def _build_system_prompt() -> str:
    schema_ctx = get_schema_context()
    return f"""You are an expert HPC analyst assistant for the Fugaku supercomputer at RIKEN R-CCS.

You have access to four tools:
1. rag_search      — search Fugaku documentation for policies, directives, and operational knowledge
2. run_sql         — query the Fugaku job telemetry database
3. predict_job     — estimate failure risk, runtime, and energy for a job before submission
4. profile_columns — inspect a column's distinct values, data type, units, and null rate

Guidelines:
- For questions about documentation, policies, job scripts, or system behaviour: use rag_search
- For quantitative questions about job data, trends, or statistics: use run_sql
- For job failure risk, expected runtime, or energy prediction: use predict_job
- Before querying a column whose values, units, or type you are unsure of — or before
  grouping/breaking results down by a category — call profile_columns first, then write
  the SQL against what it reports. Never assume a column's distinct values
- If profile_columns shows a column is CATEGORICAL, never apply AVG, SUM, or any numeric
  operation to it. If it shows a high null rate, say so in your answer rather than
  presenting the result as complete
- For compound questions: use multiple tools and synthesize the results into one answer
- Always include units in your answer
- When using run_sql, write valid DuckDB SQL. The table is called 'jobs'
- Do not mention tool names in your final answer
- When predicting job risk, use the values from predict_job exactly — do not recalculate or scale them
- If a query can be answered better by combining documentation and prediction or documentation and SQL, do that yourself
- Keep the final answer direct and concise, but complete

{schema_ctx}"""


class SingleAgent:
    def __init__(self, openai_client: OpenAI = None, verbose: bool = False):
        self.llm = openai_client or _make_client()
        self.verbose = verbose
        self.system_prompt = _build_system_prompt()
        self.logger = SingleAgentLogger.get()

    def run(self, user_query: str) -> str:
        self.logger.new_query(user_query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        for step in range(1, 9):
            self.logger.log_llm_step(step=step)

            response = self.llm.chat.completions.create(
                model=GPT_VERSION,
                messages=messages,
                tools=TOOL_SPECS,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            self.logger.log_decision(
                decision="finish_reason",
                value=finish_reason,
            )
            self.logger.log_decision(
                decision="assistant_content_present",
                value=bool(msg.content),
            )
            self.logger.log_decision(
                decision="tool_calls_present",
                value=bool(msg.tool_calls),
            )

            if msg.content:
                self.logger.log_intermediate_response(
                    step=step,
                    content=msg.content,
                    finish_reason=finish_reason,
                )

            if not msg.tool_calls:
                final_answer = msg.content or ""
                self.logger.log_final_answer(final_answer)
                return final_answer

            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                    self.logger.log_decision(
                        decision="tool_args_parse_error",
                        value=True,
                        reason=f"tool={name}"
                    )

                self.logger.log_tool_call(
                    step=step,
                    tool_call_id=tc.id,
                    tool_name=name,
                    arguments=args,
                )

                if self.verbose:
                    print(f"[SingleAgent] Tool call: {name}({args})")

                result = _dispatch(name, args)

                self.logger.log_tool_result(
                    step=step,
                    tool_call_id=tc.id,
                    tool_name=name,
                    result=result,
                )

                if self.verbose:
                    preview = result[:160].replace("\n", " ")
                    print(f"[SingleAgent] Tool result: {preview}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        self.logger.log_decision(
            decision="max_iterations_reached",
            value=True,
            reason="Exceeded 8 tool-use steps"
        )

        messages.append({
            "role": "user",
            "content": "Please provide the final answer based on the information gathered so far."
        })

        response = self.llm.chat.completions.create(
            model=GPT_VERSION,
            messages=messages,
            temperature=0,
        )

        final_answer = response.choices[0].message.content or ""
        self.logger.log_intermediate_response(
            step=9,
            content=final_answer,
            finish_reason=response.choices[0].finish_reason,
        )
        self.logger.log_final_answer(final_answer)
        return final_answer


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    query = "How many jobs ran in January 2023?"
    agent = SingleAgent(verbose=True)
    print("\nAnswer:")
    print(agent.run(query))
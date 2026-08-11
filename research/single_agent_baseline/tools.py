"""
single_agent_baseline/tools.py

Four tools available to the single-agent baseline:
  1. rag_search(query)         -> retrieves + reranks Fugaku documentation
  2. run_sql(sql)              -> executes DuckDB query on Fugaku parquet
  3. predict_job(job)          -> runs PA model, returns structured prediction
  4. profile_columns(query)    -> column distributions, dtypes, null rates

profile_columns exposes the same profiler the Full MAS DataExplorerAgent uses,
so the single agent is not disadvantaged on capability relative to the MAS —
only on architecture.
"""

from __future__ import annotations   # defer annotation evaluation — see _load_rag_deps

import os, sys, math, json, duckdb
import numpy as np


def _load_rag_deps() -> None:
    """
    Import the Qdrant/transformers RAG stack on first use.

    These are needed only by _RAGTool. Every baseline runner calls
    tests/run_n_queries.py::_patch_rag_with_bm25, which swaps rag_search for the
    MAS BM25+numpy retriever before any agent is constructed — so _RAGTool is
    typically never instantiated. Importing torch, transformers, fastembed,
    qdrant_client and sentence_transformers at module scope made those ~3 GB of
    dependencies mandatory just to import run_sql or predict_job.
    """
    global torch, AutoTokenizer, AutoModel, SparseTextEmbedding
    global QdrantClient, qmodels, CrossEncoder

    import torch
    from transformers import AutoTokenizer, AutoModel
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient
    import qdrant_client.models as qmodels
    from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../analytics")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../mas_system")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared")))
from predict import Predictor
from schema import FUGAKU_SCHEMA_NOTES

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_PATH       = os.environ.get("QDRANT_PATH", "data/fugaku_qdrant_db")
DENSE_MODEL_NAME  = "nvidia/llama-embed-nemotron-8b"
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
RERANKER_MODEL    = "mixedbread-ai/mxbai-rerank-large-v1"
PARQUET_GLOB      = os.path.join(os.environ.get("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
COLLECTION        = "fugaku_docs"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_rag  = None
_db   = None
_pred = None


def _get_rag():
    global _rag
    if _rag is None:
        _rag = _RAGTool()
    return _rag


def _get_db():
    global _db
    if _db is None:
        _db = duckdb.connect()
        _db.execute(f"CREATE VIEW jobs AS SELECT * FROM read_parquet('{PARQUET_GLOB}')")
    return _db


def _get_predictor():
    global _pred
    if _pred is None:
        _pred = Predictor()
    return _pred


# ── RAG Tool internals ────────────────────────────────────────────────────────

class _RAGTool:
    def __init__(self):
        _load_rag_deps()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer   = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME, trust_remote_code=True)
        self.dense_model = AutoModel.from_pretrained(DENSE_MODEL_NAME, trust_remote_code=True).to(self.device)
        self.dense_model.eval()

        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        self.reranker     = CrossEncoder(RERANKER_MODEL)
        self.qdrant       = QdrantClient(path=QDRANT_PATH)
        self.collection   = COLLECTION

    @staticmethod
    def _mean_pool(output, attention_mask):
        token_embeddings     = output.last_hidden_state
        input_mask_expanded  = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return (
            torch.sum(token_embeddings * input_mask_expanded, 1)
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

    def _dense_vector(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.dense_model(**encoded)
        emb = self._mean_pool(output, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().float().numpy()[0].tolist()

    def _sparse_vector(self, text: str) -> qmodels.SparseVector:
        result = list(self.sparse_model.embed([text]))[0]
        return qmodels.SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist()
        )

    def retrieve(self, query: str, n: int = 5, fetch_k: int = 20) -> list[dict]:
        dv = self._dense_vector(query)
        sv = self._sparse_vector(query)

        res = self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=[
                qmodels.Prefetch(query=dv, using="dense",  limit=fetch_k),
                qmodels.Prefetch(query=sv, using="sparse", limit=fetch_k),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=fetch_k,
            with_payload=True
        )
        candidates = res.points
        if not candidates:
            return []

        texts  = [c.payload["text"] for c in candidates]
        scores = self.reranker.predict([(query, t) for t in texts])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        seen, chunks = set(), []
        for score, point in ranked:
            if point.id not in seen:
                chunks.append({
                    "text":       point.payload["text"],
                    "filename":   point.payload["filename"],
                    "breadcrumb": point.payload["breadcrumb"],
                    "score":      float(score)
                })
                seen.add(point.id)
            if len(chunks) == n:
                break
        return chunks


# ── Public tool functions ─────────────────────────────────────────────────────

def rag_search(query: str) -> str:
    """Retrieve and rerank relevant Fugaku documentation chunks."""
    chunks = _get_rag().retrieve(query)
    if not chunks:
        return "No relevant documentation found."
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['breadcrumb']}\n{c['text']}")
    return "\n\n".join(parts)


def run_sql(sql: str) -> str:
    """Execute a DuckDB SQL query against the Fugaku parquet dataset."""
    try:
        db   = _get_db()
        rows = db.execute(sql).fetchall()
        cols = [d[0] for d in db.execute(sql).description]
        if not rows:
            return "Query returned no results."
        header = " | ".join(cols)
        sep    = "-" * len(header)
        lines  = [header, sep]
        for r in rows[:50]:
            lines.append(" | ".join(str(v) for v in r))
        if len(rows) > 50:
            lines.append(f"... and {len(rows) - 50} more rows")
        return "\n".join(lines)
    except Exception as e:
        return f"SQL ERROR: {e}"


def predict_job(job_dict: dict) -> str:
    """
    Run the PA prediction model on a job specification.

    Required keys:
      nnumr  (int)   - number of nodes requested
      elpl   (float) - walltime limit in seconds
      pclass (str)   - 'compute-bound' or 'memory-bound'

    Optional keys:
      usr, jnam, nnuma, cnumr, mszl, msza, freq_req, pri, jobenv_req, qdt
    """
    try:
        result = _get_predictor().predict(job_dict)
        lines = [
            f"Risk level: {result['risk_level']}",
            f"Failure probability: {100 * result['p_fail']:.1f}%",
            f"Expected runtime (if successful): {result['expected_runtime']}",
            f"Expected energy: {result['expected_energy']}",
            f"If it fails: {result['fail_type_if_fails']}",
            f"Node-hours wasted if slow failure: {result['wasted_node_hrs_if_slow']}",
            "",
            "Risk factors:",
        ]
        for r in result["top_reasons"]:
            lines.append(f"  - {r}")
        return "\n".join(lines)
    except Exception as e:
        return f"PREDICTION ERROR: {e}"


def profile_columns(query: str) -> str:
    """
    Profile the Fugaku jobs columns relevant to `query`: data type, null rate,
    and either the distinct values (categorical) or min/max/avg (numeric).

    Same profiler the Full MAS DataExplorerAgent uses, so the capability is
    identical across systems.
    """
    try:
        from data_explorer import explore_for_query
        return explore_for_query(query, force=True)
    except Exception as e:
        return f"PROFILE ERROR: {e}"


def get_schema_context() -> str:
    """Return DB column list + schema notes for injection into the system prompt."""
    try:
        db   = _get_db()
        rows = db.execute("DESCRIBE jobs").fetchall()
        cols = "\n".join(f"  {r[0]} ({r[1]})" for r in rows)
        return f"Database columns:\n{cols}\n\n{FUGAKU_SCHEMA_NOTES}"
    except Exception as e:
        return f"Schema unavailable: {e}"

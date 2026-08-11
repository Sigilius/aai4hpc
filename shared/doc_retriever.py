"""
shared/doc_retriever.py

Hybrid document retriever for Fugaku documentation.

Strategy
--------
Sparse  (BM25)  — exact keyword matching, strong for command names, flags,
                   section numbers, error codes.
Dense   (embed) — semantic matching via text-embedding-3-small cosine similarity,
                   strong for paraphrased or concept-level queries.
Fusion          — Reciprocal Rank Fusion (RRF, k=60) combines both rankings
                   without requiring calibrated scores.

On first call the dense embeddings for all 922 chunks are computed in batches
and cached to disk (config.DOC_EMBED_CACHE).  Subsequent starts load in <1s.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Optional

import hashlib

import numpy as np
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi

import config


# ── Stable hash (deterministic across Python processes) ───────────────────────

def _stable_hash(text: str) -> str:
    """SHA-256 of first 500 chars — deterministic unlike Python's hash()."""
    return hashlib.sha256(text[:500].encode()).hexdigest()[:16]


# ── Tokeniser ─────────────────────────────────────────────────────────────────

_STOP = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "be", "are", "was", "were", "will", "this",
    "that", "with", "from", "by", "as", "do", "how", "what", "when",
    "which", "who", "can", "i", "you", "we", "my", "your", "if",
}

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words and short tokens."""
    tokens = re.findall(r"[a-zA-Z0-9_#-]+", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


# ── Embedding client ──────────────────────────────────────────────────────────

def _make_embed_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint = config.AZURE_EMBED_ENDPOINT,
        api_key        = config.AZURE_OPENAI_API_KEY,
        api_version    = config.EMBED_API_VERSION,
    )


def _embed_batch(client: AzureOpenAI, texts: list[str]) -> np.ndarray:
    """Embed a batch of texts; returns (N, DIM) float32 array, L2-normalised."""
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# ── DocRetriever ──────────────────────────────────────────────────────────────

class DocRetriever:
    """
    Hybrid BM25 + dense retriever over Fugaku documentation chunks.

    Usage
    -----
    retriever = DocRetriever()          # builds/loads index on first call
    results   = retriever.search("What is the walltime limit for large jobs?", top_k=8)
    for r in results:
        print(r["breadcrumb"], r["rrf_score"])
        print(r["text"][:200])
    """

    RRF_K        = 60    # RRF constant — larger = smoother rank blending
    BATCH_SIZE   = 96    # embedding batch size (stay within rate limits)
    SPARSE_POOL  = 100   # top-N from each ranker before fusion
    DENSE_POOL   = 100

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._embed_client: Optional[AzureOpenAI] = None

        # Load chunks
        with open(config.DOC_CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        if verbose:
            print(f"[doc_retriever] loaded {len(self.chunks)} chunks")

        # Build BM25 sparse index
        corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        # Build / load dense embedding matrix
        self.embeddings = self._load_or_build_embeddings()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """
        Hybrid search: BM25 + dense cosine, fused with RRF.

        Returns list of chunk dicts (sorted by descending RRF score), each with
        extra keys: rrf_score, bm25_rank, dense_rank.
        """
        tokens = _tokenize(query)

        # ── Sparse (BM25) ─────────────────────────────────────────────────────
        bm25_scores  = self.bm25.get_scores(tokens)
        bm25_ranking = np.argsort(-bm25_scores)[:self.SPARSE_POOL]

        # ── Dense (cosine) ────────────────────────────────────────────────────
        q_vec         = self._embed_query(query)
        dot_scores    = self.embeddings @ q_vec          # (N,)
        dense_ranking = np.argsort(-dot_scores)[:self.DENSE_POOL]

        # ── RRF fusion ────────────────────────────────────────────────────────
        rrf: dict[int, float] = {}
        bm25_rank_of:  dict[int, int] = {}
        dense_rank_of: dict[int, int] = {}

        for rank, idx in enumerate(bm25_ranking):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)
            bm25_rank_of[idx] = rank + 1

        for rank, idx in enumerate(dense_ranking):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)
            dense_rank_of[idx] = rank + 1

        top_indices = sorted(rrf, key=lambda i: -rrf[i])[:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["_idx"]        = idx           # unique array position (for dedup)
            chunk["rrf_score"]   = round(rrf[idx], 5)
            chunk["bm25_rank"]   = bm25_rank_of.get(idx, None)
            chunk["dense_rank"]  = dense_rank_of.get(idx, None)
            chunk["bm25_score"]  = round(float(bm25_scores[idx]), 3)
            chunk["dense_score"] = round(float(dot_scores[idx]), 4)
            results.append(chunk)

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Public: embed a single query string, L2-normalised."""
        return self._embed_query(query)

    # ── Embedding management ──────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query; returns (DIM,) float32 normalised vector."""
        client = self._get_embed_client()
        vec = _embed_batch(client, [query])[0]
        return vec

    def _get_embed_client(self) -> AzureOpenAI:
        if self._embed_client is None:
            self._embed_client = _make_embed_client()
        return self._embed_client

    def _load_or_build_embeddings(self) -> np.ndarray:
        """
        Load cached embeddings from disk, or compute them from scratch.

        Cache: config.DOC_EMBED_CACHE (.npy float32, shape [N, DIM])
        A companion JSON (config.DOC_INDEX_CACHE) records chunk order to
        detect if the chunk file was updated since the cache was built.
        """
        cache_path = Path(config.DOC_EMBED_CACHE)
        index_path = Path(config.DOC_INDEX_CACHE)

        # Check if cache is valid (same number of chunks, same first text hash)
        if cache_path.exists() and index_path.exists():
            with open(index_path) as f:
                meta = json.load(f)
            first_hash = _stable_hash(self.chunks[0]["text"])
            if (
                meta.get("n_chunks") == len(self.chunks) and
                meta.get("first_text_hash") == first_hash
            ):
                embeddings = np.load(str(cache_path))
                if self.verbose:
                    print(f"[doc_retriever] loaded embeddings cache {embeddings.shape}")
                return embeddings

        # Build from scratch
        print(f"[doc_retriever] computing embeddings for {len(self.chunks)} chunks ...")
        client = self._get_embed_client()

        all_vecs: list[np.ndarray] = []
        texts = [c["text"] for c in self.chunks]

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            vecs  = _embed_batch(client, batch)
            all_vecs.append(vecs)
            if self.verbose:
                print(f"  embedded {min(i + self.BATCH_SIZE, len(texts))}/{len(texts)}")
            time.sleep(0.2)   # gentle rate-limit pause

        embeddings = np.vstack(all_vecs)   # (N, DIM) float32, already normalised

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        with open(index_path, "w") as f:
            json.dump({
                "n_chunks":        len(self.chunks),
                "first_text_hash": _stable_hash(self.chunks[0]["text"]),
                "embed_model":     config.EMBED_MODEL,
                "embed_dim":       embeddings.shape[1],
            }, f)

        print(f"[doc_retriever] embeddings cached → {cache_path}")
        return embeddings

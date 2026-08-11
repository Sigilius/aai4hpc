# shared/db.py
# Single source of truth for database connections.
#
# Priority:
#   1. Real Fugaku parquet files at FUGAKU_GLOB (configure FUGAKU_DATA_PATH in .env)
#   2. Sample DuckDB at SAMPLE_DB_PATH (our seeded test database)
#
# To switch to real data: set FUGAKU_DATA_PATH in your .env file,
# then any call to get_connection() / get_jobs_view() will use the real data.

from __future__ import annotations

import os
from pathlib import Path

import duckdb

# ── Paths ─────────────────────────────────────────────────────────────────────

FUGAKU_GLOB    = os.path.join(os.getenv("FUGAKU_DATA_PATH", "data/fugaku"), "*.parquet")
SAMPLE_DB_PATH = str(Path(__file__).parent.parent / "data" / "fugaku.duckdb")


# ── Connection helpers ────────────────────────────────────────────────────────

def _can_read_parquet() -> bool:
    """Check whether the real Fugaku parquet files are readable."""
    import glob
    files = glob.glob(FUGAKU_GLOB)
    if not files:
        return False
    try:
        open(files[0], "rb").close()
        return True
    except PermissionError:
        return False


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection with a 'jobs' view pointing at the best
    available data source.

    Real Fugaku parquet  →  used when files are readable (prod)
    Sample DuckDB        →  fallback for development / testing
    """
    if _can_read_parquet():
        conn = duckdb.connect()
        conn.execute(f"""
            CREATE VIEW IF NOT EXISTS jobs AS
            SELECT * FROM read_parquet('{FUGAKU_GLOB}')
        """)
        return conn
    else:
        # Fallback: our seeded sample database
        conn = duckdb.connect(SAMPLE_DB_PATH, read_only=True)
        return conn


def run_query(sql: str) -> list[dict]:
    """Execute SQL and return rows as a list of dicts."""
    conn = get_connection()
    result = conn.execute(sql).fetchdf()
    return result.to_dict(orient="records")


def get_schema_str() -> str:
    """
    Return a human-readable column list for the jobs table,
    suitable for injection into LLM prompts.
    """
    conn = get_connection()
    cols = conn.execute("DESCRIBE jobs").fetchall()
    conn.close()
    lines = ["Table: jobs", "Columns:"]
    lines += [f"  {c[0]:25s} {c[1]}" for c in cols]
    return "\n".join(lines)


def data_source_label() -> str:
    """Return a string describing which data source is active — useful for logging."""
    if _can_read_parquet():
        return f"Fugaku parquet ({FUGAKU_GLOB})"
    return f"sample DuckDB ({SAMPLE_DB_PATH})"

# db.py
import os
import duckdb

_FUGAKU_DATA_PATH = os.getenv("FUGAKU_DATA_PATH", "data/fugaku")
DATA_PATH = os.path.join(_FUGAKU_DATA_PATH, "**/*.parquet")

def get_connection():
    conn = duckdb.connect()
    conn.execute(f"""
        CREATE VIEW IF NOT EXISTS jobs AS
        SELECT * FROM read_parquet('{DATA_PATH}', hive_partitioning=true)
    """)
    return conn

def run_query(sql: str) -> list[dict]:
    conn = get_connection()
    result = conn.execute(sql).fetchdf()
    return result.to_dict(orient="records")
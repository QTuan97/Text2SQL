from __future__ import annotations
from typing import Optional, List, Dict
from psycopg import connect
from psycopg.rows import tuple_row
from ..config import settings

def pg_connect():
    # Prefer POSTGRES_DSN; fall back to POSTGRES_URL for backward-compat
    dsn = getattr(settings, "POSTGRES_DSN", None) or getattr(settings, "POSTGRES_URL", None)
    if not dsn:
        raise RuntimeError("POSTGRES_DSN/POSTGRES_URL not configured")
    return connect(dsn)

def schema_snapshot(schemas: Optional[List[str]] = None) -> str:
    """
    Return a compact schema summary like:
      public.users(id:integer, email:text, created_at:timestamp)
    Only for fallback display/prompt context—RAG should be primary.
    """
    schemas = schemas or ["public"]
    with pg_connect() as conn, conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = ANY(%s)
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name, ordinal_position;
            """,
            (schemas,),
        )
        rows = cur.fetchall()

    agg: Dict[str, List[str]] = {}
    for schema, table, col, dt in rows:
        key = f"{schema}.{table}"
        agg.setdefault(key, []).append(f"{col}:{dt}")

    # Keep output small; trim if you like
    lines = [f"{tbl}({', '.join(cols)})" for tbl, cols in agg.items()]
    return "\n".join(lines[:500])  # cap to avoid huge prompts
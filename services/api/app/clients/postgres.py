from __future__ import annotations
from typing import Optional
from psycopg import connect
from ..config import settings

def pg_connect():
    if not settings.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL not configured")
    return connect(settings.POSTGRES_URL)

def schema_snapshot() -> str:
    if not settings.POSTGRES_URL:
        return "No database configured."
    with pg_connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public'
            ORDER BY table_name, ordinal_position;
        """)
        rows = cur.fetchall()
    lines: dict[str, list[str]] = {}
    for t, c, dt in rows:
        lines.setdefault(t, []).append(f"{c}:{dt}")
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in lines.items())
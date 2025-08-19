from __future__ import annotations
from typing import Any, Dict, List
from ..config import settings
try:
    from ..clients.postgres import pg_connect
except Exception:
    pg_connect = None

def execute_sql(sql: str) -> List[Dict[str, Any]]:
    if not getattr(settings, "POSTGRES_URL", None) or pg_connect is None:
        raise RuntimeError("SQL execution is disabled: POSTGRES_URL not set or pg_connect unavailable.")
    rows: List[Dict[str, Any]] = []
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            for r in cur.fetchall():
                rows.append({cols[i]: r[i] for i in range(len(cols))})
    return rows
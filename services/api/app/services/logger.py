# services/api/app/services/logger.py
from __future__ import annotations
import time, json
from decimal import Decimal
from datetime import date, datetime
from typing import Any
from psycopg import connect
from psycopg.types.json import Json
from ..config import settings

DDL = """
CREATE TABLE IF NOT EXISTS api_logs (
  id           BIGSERIAL PRIMARY KEY,
  route        TEXT NOT NULL,
  method       TEXT NOT NULL,
  status       INT  NOT NULL,
  payload_json JSONB,
  response_json JSONB,
  duration_ms  INT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);
"""

def _json_default(o: Any):
    if isinstance(o, Decimal):
        # choose float for numeric JSON; use str(o) if you prefer exact text
        return float(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # last-resort stringification for anything odd
    return str(o)

def ensure_logs_table() -> None:
    if not settings.POSTGRES_URL:
        return
    with connect(settings.POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()

def log_request(route: str, method: str, status: int, payload: Any, response: Any, duration_ms: int) -> None:
    if not settings.POSTGRES_URL:
        return
    dumps = lambda v: json.dumps(v, default=_json_default)
    try:
        with connect(settings.POSTGRES_URL) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_logs (route, method, status, payload_json, response_json, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (route, method, status, Json(payload, dumps=dumps), Json(response, dumps=dumps), duration_ms),
            )
            conn.commit()
    except Exception as e:
        # surfaces in `docker compose logs api`
        print(f"[api_logs] insert failed: {e}")
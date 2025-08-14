from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import math, time, json

from ..clients.postgres import pg_connect
from ..clients import ollama
from ..config import settings
from ..services.embeddings import embed_valid

# --- helpers ---
def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)) or 1.0
    db = math.sqrt(sum(y*y for y in b)) or 1.0
    return num / (da*db)

def _schema_sig() -> str:
    # lightweight version tag; change if you prefer a real hash of get_context()
    from ..semantic.provider import get_mdl
    mdl = get_mdl()
    ents = ",".join(sorted(e["table"] for e in mdl.get("entities", [])))
    return f"tables:{ents}"

# --- write path ---
async def record_success(question: str, sql: str, tables_used: List[str]) -> None:
    vec = await embed_valid(question)
    sig = _schema_sig()
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO nl2sql_lessons(question, sql, tables_used, vector, successes, schema_sig)
            VALUES (%s, %s, %s, %s::jsonb, 1, %s)
            ON CONFLICT DO NOTHING;
        """, (question, sql, tables_used, json.dumps(vec), sig))
        # If a near-duplicate exists, you could also implement an UPDATE by similarity; keeping it simple.

# --- read path ---
async def fetch_few_shots(question: str, k: int = 3, min_sim: float = 0.55) -> List[Dict[str, Any]]:
    """
    Returns up to K prior successful examples most similar to the incoming question.
    Computes cosine in Python; fine for hundreds/thousands of rows.
    """
    qv = await embed_valid(question)
    rows: List[Tuple[int, str, str, List[str], List[float]]] = []
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, question, sql, tables_used, vector FROM nl2sql_lessons ORDER BY last_seen DESC LIMIT 500;")
        for rid, q, s, tabs, vec_json in cur.fetchall():
            rows.append((rid, q, s, tabs, vec_json))
    scored = []
    for rid, q, s, tabs, vec in rows:
        sim = _cos(qv, vec)
        if sim >= min_sim:
            scored.append((sim, {"id": rid, "question": q, "sql": s, "tables": tabs}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in scored[:k]]
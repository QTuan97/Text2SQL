from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, time, uuid

from ..config import settings
from ..services.embeddings import embed_valid

try:
    from ..dependencies import qdrant_client as _QDRANT_CLIENT
    def _qc():
        return _QDRANT_CLIENT
except Exception:
    from qdrant_client import QdrantClient
    def _qc():
        url = os.getenv("QDRANT_URL", getattr(settings, "QDRANT_URL", "http://qdrant:6333"))
        return QdrantClient(url)

from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny

# Collection and vector settings
LESSONS_COLLECTION = getattr(settings, "LESSONS_COLLECTION", os.getenv("LESSONS_COLLECTION", "sql_lessons"))
VALID_NAME = getattr(settings, "VALID_NAME", os.getenv("VALID_NAME", "valid_vec"))
ERROR_NAME = getattr(settings, "ERROR_NAME", os.getenv("ERROR_NAME", "error_vec"))
VALID_DIM = int(getattr(settings, "VALID_DIM", os.getenv("VALID_DIM", "768")))
ERROR_DIM = int(getattr(settings, "ERROR_DIM", os.getenv("ERROR_DIM", "384")))

def _ensure_lessons_collection():
    qc = _qc()
    names = [c.name for c in qc.get_collections().collections]
    if LESSONS_COLLECTION not in names:
        qc.create_collection(
            collection_name=LESSONS_COLLECTION,
            vectors_config={
                VALID_NAME: VectorParams(size=VALID_DIM, distance=Distance.COSINE),
                ERROR_NAME: VectorParams(size=ERROR_DIM, distance=Distance.COSINE),
            }
        )

def _uuid_v5(*parts: str) -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, "wren-ai://lessons")
    return str(uuid.uuid5(ns, "::".join(parts)))

def _auto_tags(sql: str) -> List[str]:
    s = (sql or "").lower()
    tags = []
    if " join " in s:
        tags.append("join")
    if " group by " in s:
        tags.append("agg")
    if " order by " in s:
        tags.append("order")
    if " limit " in s:
        tags.append("limit")
    if " date_trunc" in s or "interval" in s or " now()" in s or " current_date" in s:
        tags.append("time")
    return tags

def _build_blob(question: str, sql: str, payload: Dict[str, Any]) -> str:
    lines = [question or "", sql or ""]
    if payload.get("topic"):
        lines.append(f"TOPIC: {payload['topic']}")
    if payload.get("tags"):
        lines.append("TAGS: " + " ".join(payload["tags"]))
    if payload.get("error"):
        lines.append(f"ERROR: {payload['error']}")
    if payload.get("rowcount") is not None:
        lines.append(f"ROWS: {payload['rowcount']}")
    return "\n".join([l for l in lines if l])

async def record_learning(question: str, sql: str, *,
                          tables_used: Optional[List[str]] = None,
                          executed: bool = False,
                          rowcount: int = 0,
                          error: Optional[str] = None,
                          topic: Optional[str] = None,
                          source: str = "text2sql") -> Dict[str, Any]:
    """
    Qdrant-only persistence for learning events (no Postgres).
    Idempotent per (question, sql) using UUIDv5.
    """
    _ensure_lessons_collection()
    qc = _qc()

    point_id = _uuid_v5(question or "", sql or "")
    payload = {
        "kind": "lesson",
        "source": source,
        "question": question,
        "sql": sql,
        "tables": tables_used or [],
        "executed": bool(executed),
        "rowcount": int(rowcount or 0),
        "error": error,
        "topic": topic or (tables_used[0] if tables_used else None),
        "tags": _auto_tags(sql),
        "created_at": int(time.time()),
    }
    text = _build_blob(question, sql, payload)

    # main vector from your existing embedding pipeline
    v_main = await embed_valid(text)
    if len(v_main) != VALID_DIM:
        raise ValueError(f"Vector size mismatch for '{VALID_NAME}': expected {VALID_DIM}, got {len(v_main)}.")

    # If you have a secondary embedding function, use it; else fill zeros to satisfy schema
    try:
        from ..services.embeddings import embed_error
        v_aux = await embed_error(text)
        if len(v_aux) != ERROR_DIM:
            raise ValueError(f"Vector size mismatch for '{ERROR_NAME}': expected {ERROR_DIM}, got {len(v_aux)}.")
    except Exception:
        v_aux = [0.0] * ERROR_DIM

    qc.upsert(
        collection_name=LESSONS_COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector={VALID_NAME: v_main, ERROR_NAME: v_aux},
            payload=payload,
        )]
    )
    return {"id": point_id, "topic": payload["topic"], "tags": payload["tags"]}

async def search_lessons(question: str, k: int = 10, min_sim: float = 0.30,
                         topic: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Drop-in replacement for the old Postgres + Python cosine search.
    Returns: [{id, question, sql, tables, score}, ...]
    """
    _ensure_lessons_collection()
    qc = _qc()
    qv = await embed_valid(question)

    # optional filter
    flt = None
    if topic or tags:
        must = []
        if topic:
            must.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
        if tags:
            must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
        flt = Filter(must=must)

    res = qc.search(
        collection_name=LESSONS_COLLECTION,
        query_vector=(VALID_NAME, qv),
        limit=k,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )
    out: List[Dict[str, Any]] = []
    for r in res:
        if r.score is not None and r.score < min_sim:
            continue
        p = r.payload or {}
        out.append({
            "id": r.id,
            "question": p.get("question"),
            "sql": p.get("sql"),
            "tables": p.get("tables") or [],
            "score": r.score,
        })
    return out
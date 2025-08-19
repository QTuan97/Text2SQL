from __future__ import annotations
from typing import List, Dict, Any, Optional
import time, uuid

from qdrant_client.models import PointStruct
from ..config import settings
from ..clients.llm_compat import embed_one
from ..semantic.repair import RepairEngine
from ..utils.qdrant_helper import _qc, _ensure_lessons_collection
from ..utils.record_helper import _qnorm, _canon_sql, _lesson_id

# ---------- API ----------
QUALITY_VALUES = {"good", "bad", "unknown"}

def record_learning(question: str, sql: str, *,
                    tables_used: Optional[List[str]] = None,
                    executed: bool = False,
                    rowcount: int = 0,
                    error: Optional[str] = None,
                    source: str = "text2sql",
                    quality: str = "unknown",
                    extra_tags: Optional[List[str]] = None) -> Dict[str, Any]:

    _ensure_lessons_collection()
    qc = _qc()

    # Try to auto-repair once if marked bad/unknown
    tags = list(dict.fromkeys(extra_tags or []))
    if quality != "good" and sql:
        try:
            eng = RepairEngine(limit=50)
            candidate, remaining, applied = eng.apply(sql)
            # accept only if parses & passes cached schema
            parse_one(candidate, read="postgres")
            ensure_known_tables_cached(candidate)
            sql = candidate
            quality = "good"
            tags += ["auto_repaired"] + applied
        except Exception:
            # keep original, mark reason
            if remaining:
                tags += [f"repair_left:{c}" for c in remaining]

    # Dedupe key & deterministic point id (overwrites instead of duplicating)
    pid = _lesson_id(question, sql)

    payload: Dict[str, Any] = {
        "kind": "lesson",
        "source": source,
        "question": question,
        "qnorm": _qnorm(question),
        "sql": sql,
        "canonical_sql": _canon_sql(sql),
        "tables": tables_used or [],
        "executed": bool(executed),
        "rowcount": int(rowcount or 0),
        "error": error,
        "topic": (tables_used or [None])[0],
        "tags": list(dict.fromkeys(tags + ([] if not error else ["error"]))),
        "quality": quality if quality in QUALITY_VALUES else "unknown",
        "created_at": int(time.time()),
    }

    text = f"{question}\n{sql}\nTOPIC:{payload['topic'] or ''}\n" + (f"ERROR:{error}" if error else "")
    v_main = embed_one(text, model=settings.VALID_EMBED_MODEL)
    v_aux  = embed_one(text, model=settings.ERROR_EMBED_MODEL)

    qc.upsert(
        collection_name=settings.LESSONS_COLLECTION,
        points=[PointStruct(
            id=pid,
            vector={settings.VALID_NAME: v_main, settings.ERROR_NAME: v_aux},
            payload=payload
        )]
    )
    return {"id": pid}

def promote_lesson(point_id: str, quality: str = "good") -> None:
    if quality not in QUALITY_VALUES:
        quality = "unknown"
    _qc().set_payload(collection_name=settings.LESSONS_COLLECTION,
                      payload={"quality": quality}, points=[point_id])

async def search_lessons(
    question: str,
    k: int = 10,
    min_sim: float = 0.30,
    topic: Optional[str] = None,
    tags: Optional[List[str]] = None,
    require_good: bool = False,
) -> List[Dict[str, Any]]:
    """
    ANN search over Qdrant. Returns [{id, question, sql, tables, score}, ...]
    Signature is async to match your callers; embeddings + Qdrant calls are sync and fast for k<=50.
    """
    _ensure_lessons_collection()
    qc = _qc()

    # embed query
    qv = embed_one(question, model=settings.VALID_EMBED_MODEL)

    # optional filters
    must = []
    if topic:
        must.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
    if tags:
        must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
    if require_good:
        must.append(FieldCondition(key="quality", match=MatchValue(value="good")))
    flt = Filter(must=must) if must else None

    results = qc.search(
        collection_name=settings.LESSONS_COLLECTION,
        query_vector=(settings.VALID_NAME, qv),
        limit=k,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )

    out: List[Dict[str, Any]] = []
    for r in results:
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

async def fetch_few_shots(question: str, k: int = 3, min_sim: float = 0.55):
    hits = await search_lessons(question, k=3, min_sim=0.55, require_good=True)
    out = []
    for h in hits:
        out.append({"question": h.get("question") or "", "sql": h.get("sql") or ""})
    return out
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
import uuid, time
from datetime import datetime

from qdrant_client.models import PointStruct
from ..config import settings
from ..clients.llm_compat import embed_one
from ..semantic.repair import RepairEngine
from ..utils.qdrant_helper import _qc, _ensure_lessons_collection
from ..utils.record_helper import _qnorm, _canon_sql, _lesson_id

# ---------- API ----------
QUALITY_VALUES = {"good", "bad", "unknown"}

def _norm_list(x: Any) -> List[str]:
    """
    Normalize input (str | list | tuple | set | None) -> list[str]
    - strips / lowercases strings
    - removes empties
    - preserves order (de-duped)
    """
    if x is None:
        return []
    if isinstance(x, str):
        raw = [x]
    elif isinstance(x, Iterable):
        raw = list(x)
    else:
        raw = [x]

    cleaned = []
    seen = set()
    for v in raw:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        s = s.lower()
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned

def _first_or_none(items: List[str]) -> Optional[str]:
    return items[0] if items else None

def _question_fp(q: str) -> str:
    import re
    s = q.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def record_learning(
    question: str,
    sql: str,
    *,
    tables_used=None,
    executed: bool,
    rowcount: int,
    error: Optional[str],
    source: str,
    quality: str,
    coverage_ok: Optional[bool] = None,
    extra_tags=None,
    ):

    qc = _qc()
    _ensure_lessons_collection()

    tables_list = _norm_list(tables_used)
    tags_list = _norm_list(extra_tags)
    pid = _lesson_id(question, sql)

    payload = {
        "kind": "lesson",
        "created_at": datetime.now() ,
        "tables": tables_list,
        "question": question,
        "sql": sql,
        "tables_used": tables_list,
        "topic": _first_or_none(tables_list),
        "executed": bool(executed),
        "rowcount": int(rowcount or 0),
        "error": (str(error) if error else None),
        "source": str(source or "text2sql"),
        "quality": str(quality or "unknown"),
        "tags": tags_list,
    }

    if coverage_ok is not None:
        payload["coverage_ok"] = bool(coverage_ok)

    text = f"{question}\n{sql}\nTOPIC:{payload['topic'] or ''}\n" + (f"ERROR:{error}" if error else "")
    v_main = embed_one(text, model=settings.VALID_EMBED_MODEL)
    v_aux  = embed_one(text, model=settings.ERROR_EMBED_MODEL)

    q_fp = _question_fp(question)
    payload["q_fp"] = q_fp

    # If this is a *good* lesson, demote any prior "good" with the same exact question (or same fingerprint)
    if str(quality or "").lower() == "good":
        try:
            filt = {"must": [
                {"key": "quality", "match": {"value": "good"}},
                {"key": "q_fp", "match": {"value": q_fp}},
            ]}
            # Scroll to collect point IDs
            ids = []
            next_off = None
            while True:
                pts, next_off = qdrant_client.scroll(
                    collection_name=settings.LESSONS_COLLECTION,
                    limit=128,
                    with_payload=False,
                    with_vectors=False,
                    offset=next_off,
                    scroll_filter=filt
                )
                if not pts: break
                ids.extend([p.id for p in pts])
                if next_off is None: break

            # Demote older goods (we'll upsert the new one right after)
            if ids:
                qdrant_client.set_payload(
                    collection_name=settings.LESSONS_COLLECTION,
                    payload={"quality": "archived"},
                    points=ids,
                )
        except Exception:
            pass

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
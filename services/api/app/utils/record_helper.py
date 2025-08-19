import re , uuid, time
from typing import Optional, List, Any
from sqlglot import parse_one
from ..config import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from ..utils.qdrant_helper import _qc

def _qnorm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def _canon_sql(sql: str) -> str:
    try:
        s = parse_one(sql, read="postgres").sql(dialect="postgres")
    except Exception:
        s = re.sub(r"\s+", " ", (sql or "").strip())
    # normalize LIMIT so only structure matters for dedupe
    s = re.sub(r"(?is)\blimit\s+\d+\b", "LIMIT N", s)
    return s.strip().lower()

def _lesson_id(question: str, sql: str) -> str:
    key = f"{_qnorm(question)}||{_canon_sql(sql)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def _normalize_scroll_result(res) -> List[Any]:
    """
    Handle qdrant-client scroll return across versions:
    """
    if isinstance(res, tuple):
        if len(res) >= 1:
            return res[0] or []
        return []
    # object style
    pts = getattr(res, "points", None)
    if pts is not None:
        return pts
    # last resort: assume iterable
    try:
        return list(res)  # may not work, but safe fallback
    except Exception:
        return []

def get_cached_good_sql(question: str, *, prefer_exact: bool = True, min_sim: float = 0.92) -> Optional[dict]:
    """
    Return {"id", "sql", "score"} if there is an existing GOOD lesson for this question.
    1) exact match by normalized question (fast), else
    2) vector fallback with a high threshold (handled elsewhere if you want)
    """
    qc = _qc()

    if prefer_exact:
        flt = Filter(must=[
            FieldCondition(key="kind", match=MatchValue(value="lesson")),
            FieldCondition(key="quality", match=MatchValue(value="good")),
            FieldCondition(key="qnorm", match=MatchValue(value=_qnorm(question))),
        ])

        res = qc.scroll(
            collection_name=settings.LESSONS_COLLECTION,
            scroll_filter=flt,
            limit=50,                 # grab a few; we’ll sort below
            with_payload=True,
            with_vectors=False,
        )
        pts = _normalize_scroll_result(res)

        if pts:
            # choose the best candidate: executed first, higher rowcount, then newest
            def rank(p):
                pl = p.payload or {}
                return (
                    1 if pl.get("executed") else 0,
                    int(pl.get("rowcount") or 0),
                    int(pl.get("last_used") or pl.get("created_at") or 0),
                )

            pts.sort(key=rank, reverse=True)
            p = pts[0]
            return {
                "id": p.id,
                "sql": (p.payload or {}).get("sql"),
                "score": 1.0,
            }

    return None

def get_same_question_negatives(question: str, limit: int = 3) -> list[str]:
    """
    Return recent SQL strings for the same normalized question that are NOT good.
    """
    qc = _qc()
    flt = Filter(must=[
        FieldCondition(key="kind", match=MatchValue(value="lesson")),
        FieldCondition(key="qnorm", match=MatchValue(value=_qnorm(question))),
        FieldCondition(key="quality", match=MatchValue(value="bad")),
    ])

    res = qc.scroll(
        collection_name=settings.LESSONS_COLLECTION,
        scroll_filter=flt,
        limit=50,                 # fetch a page; we’ll sort & slice
        with_payload=True,
        with_vectors=False,
    )
    pts = _normalize_scroll_result(res)

    # newest first
    pts.sort(key=lambda p: -int((p.payload or {}).get("created_at", 0)))

    negatives = [
        (p.payload or {}).get("sql")
        for p in pts
        if (p.payload or {}).get("sql")
    ]
    return negatives[:limit]

def touch_lesson(point_id: str) -> None:
    """Mark that we reused this lesson."""
    try:
        _qc().set_payload(collection_name=settings.LESSONS_COLLECTION,
                          payload={"last_used": int(time.time())},
                          points=[point_id])
    except Exception:
        pass
from __future__ import annotations
from ..config import settings
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Any, Dict, List, Optional
import sqlglot
from sqlglot import exp

from ..schemas.common import NL2SQLIn, NL2SQLOut
from ..nlu.classify import classify_question, build_general_answer
from ..semantic.sql_guard import validate_schema, lint_sql
from ..services.text2sql import generate_sql, repair_sql
from ..services.db_exec import execute_sql
from ..semantic.lessons import record_learning, promote_lesson
from ..utils.sql_text import _sanitize_sql
from ..utils.record_helper import get_cached_good_sql, get_same_question_negatives, touch_lesson

router = APIRouter(prefix="/text2sql", tags=["text2sql"])

def _tables_in(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
        out: List[str] = []
        for t in tree.find_all(exp.Table):
            nm = getattr(t, "name", None) or getattr(getattr(t, "this", None), "name", None)
            if nm:
                out.append(nm)
        # uniq, preserve order
        seen, keep = set(), []
        for n in out:
            if n not in seen:
                keep.append(n); seen.add(n)
        return keep
    except Exception:
        return []

@router.post("", response_model=NL2SQLOut)
async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks) -> NL2SQLOut:
    kind, reason = classify_question(body.question)

    # 1) GENERAL → plain answer, no SQL
    if kind == "general":
        ans = build_general_answer()
        return NL2SQLOut(
            sql="",
            rows=None,
            rowcount=0,
            diagnostics={"mode": "general", "reason": reason},
            answer=ans,
        )

    # 2) MISLEADING → block + learn (bad)
    if kind == "misleading":
        try:
            record_learning(
                body.question,
                "",
                tables_used=[],
                executed=False,
                rowcount=0,
                error="misleading",
                source="text2sql",
                quality="bad",
                extra_tags=["misleading"],
            )
        except Exception:
            pass
        raise HTTPException(
            400,
            (
                "This workspace handles **database questions** only. "
                "Try: 'total sales last quarter', 'top 10 users by revenue', "
                "or ask 'what is this database about?'"
            ),
        )

    # 3) TEXT2SQL → cache-first, then generate (with negatives)
    limit = body.limit if isinstance(body.limit, int) and body.limit > 0 else 50

    # ---- cache-first: reuse a GOOD lesson for the same question
    hit = None
    if getattr(settings, "CACHE_FIRST", True):
        hit = get_cached_good_sql(
            body.question,
            prefer_exact=True,
            min_sim=getattr(settings, "CACHE_MIN_SIM", 0.92),
        )

    if hit and isinstance(hit.get("sql"), str) and hit["sql"].strip():
        cached_sql = _sanitize_sql(hit["sql"], limit)
        diagnostics: Dict[str, Any] = {
            "mode": "text2sql",
            "backend": "cache",
            "cache_hit": True,
            "cache_point_id": hit["id"],
            "cache_score": hit.get("score"),
            "classifier_reason": reason,
        }
        rows: Optional[List[Dict[str, Any]]] = None
        rowcount, error = 0, None
        if body.execute:
            try:
                rows = execute_sql(cached_sql)
                rowcount = len(rows) if isinstance(rows, list) else 0
            except Exception as ex:
                error = str(ex)
                diagnostics["cache_exec_error"] = error
        try:
            touch_lesson(hit["id"])
        except Exception:
            pass
        return NL2SQLOut(sql=cached_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)

    # ---- no GOOD cache → fetch recent bad attempts as negatives
    negatives = get_same_question_negatives(body.question, limit=3)

    # Generate fresh SQL (make sure generate_sql accepts `negative_sql_examples=None` by default)
    try:
        final_sql = await generate_sql(body.question, limit=limit, negative_sql_examples=negatives)
    except ValueError as e:
        raise HTTPException(400, str(e))

    diagnostics: Dict[str, Any] = {
        "mode": "text2sql",
        "backend": "generate_sql",
        "classifier_reason": reason,
        "negatives_used": len(negatives or []),
    }
    tables = _tables_in(final_sql)

    # (optional) execute
    rows: Optional[List[Dict[str, Any]]] = None
    rowcount, error, executed = 0, None, False
    if body.execute:
        try:
            rows = execute_sql(final_sql)
            executed = True
            rowcount = len(rows) if isinstance(rows, list) else 0
        except Exception as ex:
            # optional one-shot repair
            try:
                repaired = await repair_sql(
                    body.question, limit=limit, error_message=str(ex), prev_sql=final_sql
                )
                final_sql = repaired
                rows = execute_sql(final_sql)
                executed = True
                rowcount = len(rows) if isinstance(rows, list) else 0
                diagnostics["repaired"] = True
            except Exception as ex2:
                error = str(ex2)
                diagnostics["repaired"] = False

    # static validation → derive quality/tags without hitting DB
    schema_errs = validate_schema(final_sql)
    lints = lint_sql(final_sql)
    quality = (
        "good"
        if (not schema_errs and "aggregate-without-group-by" not in lints and error is None)
        else ("bad" if schema_errs or error else "unknown")
    )
    tags = lints + (["schema-ok"] if not schema_errs else schema_errs)

    meta = record_learning(
        body.question,
        final_sql,
        tables_used=tables,
        executed=executed,
        rowcount=rowcount,
        error=error,
        source="text2sql",
        quality=quality,
        extra_tags=tags,
    )
    if executed and error is None and quality != "good":
        promote_lesson(meta["id"], "good")

    diagnostics["learn"] = {"id": meta["id"], "quality": quality, "tags": tags}
    return NL2SQLOut(sql=final_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)
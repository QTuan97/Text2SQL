from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..config import settings
from ..schemas.common import NL2SQLIn, NL2SQLOut
from ..nlu.classify import classify_question, build_general_answer
from ..services.text2sql import generate_sql, repair_sql
from ..semantic.sql_guard import (
    _tables_in,
    validate_schema,
    lint_sql,
    lint_relative_time_bad,
    SEVERE_LINTS,
    WARNING_LINTS,
)
from ..semantic.lessons import (
    record_learning,
    promote_lesson,
)
from ..utils.record_helper import (
    get_cached_good_sql,
    get_same_question_negatives,
    touch_lesson
)
from ..services.db_exec import execute_sql

router = APIRouter(prefix="/text2sql", tags=["text2sql"])

# small helper
def _norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

@router.post("", response_model=NL2SQLOut)
async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks) -> NL2SQLOut:
    # Intent gate
    kind, reason = await classify_question(body.question)

    # GENERAL
    if kind == "general":
        ans = build_general_answer()
        return NL2SQLOut(
            sql="",
            rows=None,
            rowcount=0,
            diagnostics={"mode": "general", "reason": reason},
            answer=ans,
        )

    # MISLEADING
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
            ("This workspace handles **database questions** only. "
             "Try: 'total sales last quarter', 'top 10 users by revenue', "
             "or ask 'what is this database about?'"),
        )

    # TEXT2SQL
    limit = body.limit if isinstance(body.limit, int) and body.limit > 0 else 50

    # cache-first (GOOD only)
    diagnostics: Dict[str, Any] = {"mode": "text2sql", "classifier_reason": reason}
    hit = None
    if getattr(settings, "CACHE_FIRST", True):
        hit = get_cached_good_sql(
            body.question,
            prefer_exact=True,
            min_sim=getattr(settings, "CACHE_MIN_SIM", 0.92),
        )

    if hit and isinstance(hit.get("sql"), str) and hit["sql"].strip():
        cached_sql = hit["sql"]
        try:
            cached_sql = _sanitize_sql(cached_sql, limit)
        except Exception:
            cached_sql = cached_sql.strip().strip("`")

        diagnostics.update({
            "backend": "cache",
            "cache_hit": True,
            "cache_point_id": hit.get("id"),
            "cache_score": hit.get("score"),
        })

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

    # 1b) no GOOD cache → gather negatives for this question
    negatives = get_same_question_negatives(body.question, limit=3)

    # 1c) generate fresh SQL
    try:
        final_sql = await generate_sql(
            body.question,
            limit=limit,
            negative_sql_examples=negatives,
        )
        diagnostics.update({
            "backend": "generate_sql",
            "negatives_used": len(negatives or []),
        })
    except ValueError as e:
        raise HTTPException(400, str(e))

    # execute, with a single repair pass
    rows: Optional[List[Dict[str, Any]]] = None
    rowcount, error, executed = 0, None, False

    if body.execute:
        try:
            rows = execute_sql(final_sql)
            executed = True
            rowcount = len(rows) if isinstance(rows, list) else 0
        except Exception as ex:
            diagnostics["first_exec_error"] = str(ex)
            try:
                repaired = await repair_sql(
                    body.question,
                    limit=limit,
                    error_message=str(ex),
                    prev_sql=final_sql,
                )
                # de-dup guard: if repair == original, do NOT re-exec same failure
                if _norm_sql(repaired) == _norm_sql(final_sql):
                    error = "Repair returned the same SQL; not re-executing."
                    diagnostics["repaired"] = False
                    diagnostics["repair_same_sql"] = True
                else:
                    final_sql = repaired
                    rows = execute_sql(final_sql)
                    executed = True
                    rowcount = len(rows) if isinstance(rows, list) else 0
                    diagnostics["repaired"] = True
            except Exception as ex2:
                error = str(ex2)
                diagnostics["repaired"] = False

    # static validation → classify quality with reasons
    schema_errs = validate_schema(final_sql) or []
    lints = (lint_sql(final_sql) or []) + lint_relative_time_bad(body.question, final_sql)

    severe_schema = list(schema_errs)
    severe_lints = [x for x in lints if x in SEVERE_LINTS]
    warn_lints   = [x for x in lints if x in WARNING_LINTS or x not in SEVERE_LINTS]

    if body.execute:
        if error is None and not severe_schema and not severe_lints:
            quality = "good"
        elif error is not None or severe_schema or severe_lints:
            quality = "bad"
        else:
            quality = "unknown"
    else:
        quality = "good" if (not severe_schema and not severe_lints) else ("bad" if (severe_schema or severe_lints) else "unknown")

    # learn (record & promote only when truly good)
    tables = _tables_in(final_sql)  # keep your helper; import if it lives elsewhere
    meta = record_learning(
        body.question,
        final_sql,
        tables_used=tables,
        executed=executed,
        rowcount=rowcount,
        error=error,
        source="text2sql",
        quality=quality,
        extra_tags=warn_lints + (["schema-ok"] if not severe_schema else severe_schema),
    )

    if executed and error is None and quality == "good":
        promote_lesson(meta["id"], "good")

    diagnostics["quality"] = quality
    diagnostics["quality_reasons"] = {
        "severe_schema": severe_schema,
        "severe_lints": severe_lints,
        "warn_lints": warn_lints,
        "executed": executed,
        "db_error": error,
        "classifier_reason": reason,
        "learn_id": meta["id"],
    }

    return NL2SQLOut(sql=final_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..schemas.common import NL2SQLIn, NL2SQLOut
from ..config import settings
from ..nlu.classify import classify_question, build_general_answer
from ..services.text2sql import generate_sql, repair_sql
from ..utils.sql_text import _sanitize_sql
from ..semantic.schema_cache import known_schema
from ..semantic.coverage import audit_sql_coverage
from ..semantic.sql_guard import (
    ensure_known_tables_cached,
    validate_schema,
    lint_sql,
    lint_relative_time_bad,
    SEVERE_LINTS,
    WARNING_LINTS,
    tables_in as _tables_in,
)
from ..semantic.lessons import (
    record_learning,
    promote_lesson
)
from ..utils.record_helper import (
    get_cached_good_sql,
    get_same_question_negatives,
    touch_lesson
)
from ..services.db_exec import execute_sql

router = APIRouter(prefix="/text2sql", tags=["text2sql"])

# Helpers
def _norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _suggest_tables(bad: str, k: int = 5) -> list[str]:
    names, _ = known_schema()
    pool = {n.lower() for n in names}
    pool |= {n.split(".")[-1] for n in pool}
    return get_close_matches((bad or "").lower(), list(pool), n=k, cutoff=0.45)

# Route
@router.post("", response_model=NL2SQLOut)
async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks) -> NL2SQLOut:
    # 0) Intent gate
    kind, reason = await classify_question(body.question)

    if kind == "general":
        ans = build_general_answer()
        return NL2SQLOut(sql="", rows=None, rowcount=0,
                         diagnostics={"mode": "general", "reason": reason},
                         answer=ans)

    if kind == "misleading":
        try:
            record_learning(
                body.question, "",
                tables_used=[],
                executed=False, rowcount=0, error="misleading",
                source="text2sql", quality="bad",
                extra_tags=["misleading"],
            )
        except Exception:
            pass
        raise HTTPException(
            400,
            "This workspace handles **database questions** only. "
            "Try: 'total sales last quarter', 'top 10 users by revenue', "
            "or ask 'what is this database about?'"
        )

    # 1) Text2SQL flow
    limit = body.limit if isinstance(body.limit, int) and body.limit > 0 else 50
    diagnostics: Dict[str, Any] = {"mode": "text2sql", "classifier_reason": reason}

    # 1a) Cache-first (GOOD only; exact match)
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

    # 1b) Gather negatives for this question (optional)
    negatives = get_same_question_negatives(body.question, limit=3)

    # 1c) Generate fresh SQL
    try:
        final_sql = await generate_sql(
            body.question,
            limit=limit,
            negative_sql_examples=negatives,
        )
        diagnostics.update({"backend": "generate_sql", "negatives_used": len(negatives or [])})
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 1d) Schema validation → suggest/repair on unknown tables (e.g., "order" → "orders")
    try:
        ensure_known_tables_cached(final_sql)
    except ValueError as ve:
        diagnostics["schema_validation_error"] = str(ve)
        missing = [x.strip() for x in str(ve).split(":")[-1].split(",")]
        hints = []
        for m in missing:
            sugg = _suggest_tables(m) or ["(no close match)"]
            hints.append(f"{m} → {', '.join(sugg)}")

        repaired = await repair_sql(
            body.question,
            limit=limit,
            error_message=f"Unknown tables: {', '.join(missing)}. Suggestions: {'; '.join(hints)}",
            prev_sql=final_sql,
        )

        if _norm_sql(repaired) != _norm_sql(final_sql):
            final_sql = repaired
            try:
                ensure_known_tables_cached(final_sql)
                diagnostics["schema_repaired"] = True
            except ValueError as ve2:
                raise HTTPException(400, str(ve2))
        else:
            diagnostics["schema_repaired"] = False
            raise HTTPException(400, str(ve))

    # 1e) Coverage audit (deterministic: top-K, ORDER BY, literals like 'Can Tho', bounded time)
    cov = audit_sql_coverage(body.question, final_sql)
    diagnostics["coverage"] = cov
    coverage_ok = bool(cov.get("ok", False))

    # Optional one repair pass if coverage failed
    if not coverage_ok:
        repaired = await repair_sql(
            body.question,
            limit=limit,
            error_message="; ".join(cov.get("missing") or []),
            prev_sql=final_sql,
        )
        if _norm_sql(repaired) != _norm_sql(final_sql):
            final_sql = repaired
            ensure_known_tables_cached(final_sql)
            cov = audit_sql_coverage(body.question, final_sql)
            coverage_ok = bool(cov.get("ok", False))
            diagnostics["coverage_after_repair"] = cov
        else:
            diagnostics["coverage_repair_nochange"] = True

    # 1f) Execute (with a single execution-error repair pass)
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

    # 1g) Static validation → quality
    schema_errs = validate_schema(final_sql) or []
    lints = (lint_sql(final_sql) or []) + lint_relative_time_bad(body.question, final_sql)

    # treat coverage miss as severe
    if not coverage_ok:
        lints.append("coverage-miss")
    SEVERE_LINTS.add("coverage-miss")

    if body.execute:
        if error is None and not schema_errs and "coverage-miss" not in lints and not [x for x in lints if x in SEVERE_LINTS]:
            quality = "good"
        elif error is not None or schema_errs or "coverage-miss" in lints or [x for x in lints if x in SEVERE_LINTS]:
            quality = "bad"
        else:
            quality = "unknown"
    else:
        quality = "good" if (not schema_errs and "coverage-miss" not in lints and not [x for x in lints if x in SEVERE_LINTS]) \
                  else ("bad" if (schema_errs or "coverage-miss" in lints or [x for x in lints if x in SEVERE_LINTS]) else "unknown")

    diagnostics.setdefault("quality_reasons", {})
    diagnostics["quality_reasons"].update({
        "coverage_ok": coverage_ok,
        "coverage_missing": cov.get("missing"),
        "severe_schema": schema_errs,
        "severe_lints": [x for x in lints if x in SEVERE_LINTS],
        "warn_lints": [x for x in lints if x in WARNING_LINTS or x not in SEVERE_LINTS],
    })

    # 1h) Learn + optional promote (only if truly good AND coverage_ok)
    tables_list = list(_tables_in(final_sql))
    meta = record_learning(
        body.question,
        final_sql,
        tables_used=tables_list,
        executed=executed,
        rowcount=rowcount,
        error=error,
        source="text2sql",
        quality=quality,
        extra_tags=diagnostics["quality_reasons"]["warn_lints"] + (["schema-ok"] if not schema_errs else schema_errs),
        coverage_ok=coverage_ok,  # stored in payload for cache preference
    )

    if executed and error is None and quality == "good" and coverage_ok:
        promote_lesson(meta["id"], "good")

    diagnostics["learn"] = {"id": meta["id"], "quality": quality}
    return NL2SQLOut(sql=final_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)
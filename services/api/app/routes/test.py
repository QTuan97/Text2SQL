# services/api/app/routes/text2sql.py
from ..config import settings
from ..services.lessons import get_cached_good_sql, get_same_question_negatives, touch_lesson, record_learning, promote_lesson
from ..utils.sql_text import _sanitize_sql
from ..semantic.repair import RepairEngine, detect   # engine + detector

BLOCKERS = {"parse-error","unknown-table","unknown-column","join-without-on","must-use-alias","aggregate-without-group-by","unknown-qualifier"}

def _score_sql(sql_text: str) -> tuple[int, int]:
    """(num_blockers, num_warnings). Lower is better."""
    issues = detect(sql_text)
    blockers = sum(1 for c,_ in issues if c in BLOCKERS)
    warnings = sum(1 for c,_ in issues if c not in BLOCKERS)
    return blockers, warnings

async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks) -> NL2SQLOut:
    kind, reason = classify_question(body.question)

    # 1) GENERAL
    if kind == "general":
        ans = build_general_answer()
        return NL2SQLOut(sql="", rows=None, rowcount=0,
                         diagnostics={"mode":"general","reason":reason}, answer=ans)

    # 2) MISLEADING
    if kind == "misleading":
        try:
            record_learning(body.question, "", tables_used=[], executed=False, rowcount=0,
                            error="misleading", source="text2sql", quality="bad", extra_tags=["misleading"])
        except Exception:
            pass
        raise HTTPException(400, "This workspace handles database questions only. Try: 'total sales last quarter', 'top 10 users by revenue', or ask 'what is this database about?'")

    # 3) TEXT2SQL
    limit = body.limit if isinstance(body.limit, int) and body.limit > 0 else 50

    # cache-first GOOD lesson
    if getattr(settings, "CACHE_FIRST", True):
        hit = get_cached_good_sql(body.question, prefer_exact=True, min_sim=getattr(settings, "CACHE_MIN_SIM", 0.92))
        if hit and isinstance(hit.get("sql"), str) and hit["sql"].strip():
            cached_sql = _sanitize_sql(hit["sql"], limit)
            diagnostics = {"mode":"text2sql","backend":"cache","cache_hit":True,"cache_point_id":hit["id"],"cache_score":hit.get("score"),"classifier_reason":reason}
            rows = None; rowcount = 0; error = None
            if body.execute:
                try:
                    rows = execute_sql(cached_sql); rowcount = len(rows) if isinstance(rows, list) else 0
                except Exception as ex:
                    error = str(ex); diagnostics["cache_exec_error"] = error
            try: touch_lesson(hit["id"])
            except Exception: pass
            return NL2SQLOut(sql=cached_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)

    # no good cache → build negatives
    negatives = get_same_question_negatives(body.question, limit=3)

    # Attempt A
    sql_a = await generate_sql(body.question, limit=limit, negative_sql_examples=negatives)
    # Attempt B with targeted hints (from issues in A)
    issues_a = [c for c,_ in detect(sql_a)]
    hint_lines = []
    if "must-use-alias" in issues_a: hint_lines.append("- Use the table alias everywhere (e.g., `o.amount`, not `orders.amount`).")
    if "aggregate-without-group-by" in issues_a: hint_lines.append("- Include all non-aggregated select columns in GROUP BY.")
    if "ambiguous-column" in issues_a: hint_lines.append("- Qualify ambiguous columns with the correct table alias.")
    extra_hint = "\n".join(hint_lines)
    try:
        sql_b = await generate_sql(body.question, limit=limit, negative_sql_examples=negatives, extra_hint=extra_hint)
    except Exception:
        sql_b = sql_a  # fallback to A if B fails to generate

    # Pre-exec repair engine on each, then validate & score
    eng = RepairEngine(limit=limit)
    def _repair_and_score(s: str):
        candidate, remaining, applied = eng.apply(s)
        # try to accept engine candidate
        try:
            parse_one(candidate, read="postgres")
            ensure_known_tables_cached(candidate)
            blockers, warnings = _score_sql(candidate)
            return candidate, blockers, warnings, applied, remaining
        except Exception:
            blockers, warnings = _score_sql(s)
            return s, blockers, warnings, [], issues_a

    cand_a, b_a, w_a, applied_a, rem_a = _repair_and_score(sql_a)
    cand_b, b_b, w_b, applied_b, rem_b = _repair_and_score(sql_b)

    # choose better candidate (fewest blockers, then warnings)
    final_sql, applied_rules, remaining_issues = (cand_a, applied_a, rem_a) if (b_a, w_a) <= (b_b, w_b) else (cand_b, applied_b, rem_b)

    diagnostics: Dict[str, Any] = {
        "mode": "text2sql",
        "backend": "generate_sql",
        "classifier_reason": reason,
        "negatives_used": len(negatives or []),
        "repair": {"applied": applied_rules, "remaining": remaining_issues},
    }

    tables = _tables_in(final_sql)

    # execute (optional)
    rows: Optional[List[Dict[str, Any]]] = None
    rowcount, error, executed = 0, None, False
    if body.execute:
        try:
            rows = execute_sql(final_sql)
            executed = True
            rowcount = len(rows) if isinstance(rows, list) else 0
        except Exception as ex:
            # LLM repair fallback (one shot) if execution failed
            try:
                repaired = await repair_sql(body.question, limit=limit, error_message=str(ex), prev_sql=final_sql)
                final_sql = repaired
                rows = execute_sql(final_sql)
                executed = True
                rowcount = len(rows) if isinstance(rows, list) else 0
                diagnostics["repaired"] = True
            except Exception as ex2:
                error = str(ex2)
                diagnostics["repaired"] = False

    # static validation → derive quality/tags
    schema_errs = validate_schema(final_sql)
    lints = lint_sql(final_sql)
    quality = "good" if (not schema_errs and "aggregate-without-group-by" not in lints and error is None) else ("bad" if schema_errs or error else "unknown")
    tags = lints + (["schema-ok"] if not schema_errs else schema_errs)

    meta = record_learning(
        body.question, final_sql,
        tables_used=tables, executed=executed, rowcount=rowcount,
        error=error, source="text2sql", quality=quality, extra_tags=tags
    )
    if executed and error is None and quality != "good":
        promote_lesson(meta["id"], "good")

    diagnostics["learn"] = {"id": meta["id"], "quality": quality, "tags": tags}
    return NL2SQLOut(sql=final_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)
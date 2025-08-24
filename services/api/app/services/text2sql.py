from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from sqlglot import parse_one
from qdrant_client import QdrantClient

from ..config import settings
from ..dependencies import get_qdrant
from ..clients.llm_compat import _llm_complete
from ..semantic.training import query_related_schema
from ..semantic.provider import get_context
from ..semantic.sql_guard import build_ambiguity_guard
from ..semantic.schema_cache import get_schema_text
from ..utils.sql_text import _extract_sql, _sanitize_sql, _time_guide, _normalize_ws, plan_query
from ..semantic.repair import RepairEngine

# try to keep your existing few-shot fetch; guard if missing
try:
    from ..semantic.lessons import fetch_few_shots  # uses Postgres in your tree; replace later with Qdrant impl
except Exception:
    async def fetch_few_shots(question: str, k: int = 3, min_sim: float = 0.55):
        return []

# Building prompt
def build_sql_prompt(question: str, qdrant: QdrantClient) -> str:
    schema_docs = query_related_schema(qdrant, question, top_k=8)
    context = "\n\n".join([d["text"] for d in schema_docs])

    system = (
        "You are a PostgreSQL expert. "
        "Generate **valid, executable SQL** ONLY (no explanations). "
        "Use the provided schema context; do not invent tables/columns."
    )
    return f"{system}\n\nSCHEMA CONTEXT:\n{context}\n\nQUESTION: {question}\nSQL:"

# Generate SQL
async def generate_sql(
    question: str,
    limit: int,
    negative_sql_examples: Optional[List[str]] = None,
    extra_hint: str = "",
) -> str:
    # sanitize limit
    if not isinstance(limit, int) or limit < 1:
        limit = 50

    # PLAN
    plan = await plan_query(question)

    # respect caller cap
    if isinstance(plan.get("limit"), int) and plan["limit"] > 0:
        plan["limit"] = min(plan["limit"], limit)
    else:
        plan["limit"] = limit

    # COMPOSE SQL from plan
    mdl_context = get_context()              # optional semantic rules/joins
    live_schema = get_schema_text() or ""    # authoritative names
    amb = build_ambiguity_guard()            # canonical table-name mapping

    system = (
        "You are a SQL compiler. Given a PLAN JSON, produce exactly ONE valid PostgreSQL SELECT statement.\n"
        "OUTPUT: a single fenced SQL block (```sql\\n...\\n```). No prose, JSON, or comments.\n\n"

        "# Schema (authoritative—use these names exactly)\n"
        f"{live_schema}\n\n"
        f"{amb}"

        "# Time patterns (use; never hardcode month numbers/years)\n"
        "- THIS_MONTH:  created_at >= DATE_TRUNC('month', CURRENT_DATE)\n"
        "               AND created_at <  DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'\n"
        "- TODAY:       created_at >= CURRENT_DATE AND created_at < CURRENT_DATE + INTERVAL '1 day'\n"
        "- LAST_7_DAYS: created_at >= CURRENT_DATE - INTERVAL '7 days'\n\n"

        "# Hard rules\n"
        "- Use ONLY table names in the schema. Do not invent singular/plural variants.\n"
        "- Alias DISCIPLINE: every table in FROM/JOIN must use the alias from the PLAN; qualify ALL columns with those aliases only.\n"
        "- Projection DISCIPLINE: select ONLY (a) metrics from the PLAN and (b) dimensions from the PLAN. Do NOT add extra columns.\n"
        "- If any aggregate appears, ALL non-aggregated select expressions MUST be in GROUP BY (use the PLAN dimensions only).\n"
        "- Implement PLAN.filters; map THIS_MONTH/TODAY/LAST_7_DAYS to the standard date_trunc windows.\n"
        "- ORDER BY must use select-list expressions or their aliases; ASC/DESC only in ORDER BY.\n"
        f"- If LIMIT is missing, add LIMIT {plan['limit']}.\n\n"

        "# Semantic layer (business joins/aliases/rules)\n"
        f"{mdl_context}\n"
    )

    user = (
        "PLAN JSON:\n"
        f"{json.dumps(plan, ensure_ascii=False)}\n\n"
        + (f"Extra constraints to honor: {extra_hint}\n" if extra_hint else "")
        + "Compose the final SQL now (one fenced block)."
    )

    raw = await _llm_complete(system, user)
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, plan["limit"])

    parse_one(sql, read="postgres")
    return sql

# Repair SQL
async def repair_sql(
    question: str,
    limit: int,
    error_message: str,
    prev_sql: Optional[str] = None,
) -> str:
    limit = max(1, int(limit or 50))

    # deterministic quick-fix first
    if prev_sql:
        eng = RepairEngine(limit=limit)
        candidate, remaining, applied = eng.apply(prev_sql)
        try:
            parse_one(candidate, read="postgres")  # syntax-only; route handles schema/coverage
            return _sanitize_sql(candidate, limit)
        except Exception:
            # carry forward what we tried so the LLM doesn't repeat it
            tried = f"[auto-repair tried: {', '.join(applied) or 'none'}; remaining: {', '.join(remaining) or 'none'}]"
            error_message = (error_message + "\n" + tried) if error_message else tried

    # Re-plan
    plan = await plan_query(question)
    # cap plan limit to caller limit
    plan["limit"] = min(int(plan.get("limit") or limit), limit)

    mdl_context = get_context()           # business joins/aliases
    live_schema = get_schema_text() or "" # authoritative names
    amb = build_ambiguity_guard()         # canonical table-name mapping

    system = (
        "You fix SQL by re-compiling from a PLAN JSON. Produce exactly ONE valid PostgreSQL SELECT statement.\n"
        "OUTPUT: a single fenced SQL block (```sql\\n...\\n```). No prose/JSON/comments.\n\n"

        "# Schema (authoritative—use these names exactly)\n"
        f"{live_schema}\n\n"
        f"{amb}"

        "# Time patterns (use; never hardcode month numbers/years)\n"
        "- THIS_MONTH:  created_at >= DATE_TRUNC('month', CURRENT_DATE)\n"
        "               AND created_at <  DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'\n"
        "- TODAY:       created_at >= CURRENT_DATE AND created_at < CURRENT_DATE + INTERVAL '1 day'\n"
        "- LAST_7_DAYS: created_at >= CURRENT_DATE - INTERVAL '7 days'\n\n"

        "# Hard rules\n"
        "- Use ONLY table names in the schema. Do not invent singular/plural variants.\n"
        "- Alias DISCIPLINE: every table in FROM/JOIN must use the alias from the PLAN; qualify ALL columns with those aliases only.\n"
        "- Projection DISCIPLINE: select ONLY (a) metrics from the PLAN and (b) dimensions from the PLAN. Do NOT add extra columns.\n"
        "- If any aggregate appears, ALL non-aggregated select expressions MUST be in GROUP BY (use the PLAN dimensions only).\n"
        "- Implement PLAN.filters; map THIS_MONTH/TODAY/LAST_7_DAYS to the standard date_trunc windows.\n"
        "- ORDER BY must use select-list expressions or their aliases; ASC/DESC only in ORDER BY.\n"
        f"- If LIMIT is missing, add LIMIT {plan['limit']}.\n\n"

        "# Semantic layer (business joins/aliases/rules)\n"
        f"{mdl_context}\n"
    )

    user = (
        f"Error to avoid:\n{error_message or '(none)'}\n\n"
        "PLAN JSON:\n"
        f"{json.dumps(plan, ensure_ascii=False)}\n\n"
        "Re-compile the final SQL now (one fenced block)."
    )

    raw = await _llm_complete(system, user)
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, plan["limit"])

    # De-dup guard: if unchanged, try deterministic pass once more
    def _norm_sql(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()
    if prev_sql and _norm_sql(sql) == _norm_sql(prev_sql):
        eng = RepairEngine(limit=limit)
        candidate, _, _ = eng.apply(sql)
        sql = candidate

    parse_one(sql, read="postgres")
    return sql
from __future__ import annotations
from typing import Any, Dict, List, Optional
from sqlglot import parse_one

from ..config import settings
from ..semantic.provider import get_context, get_schema_text
from ..semantic.sql_guard import ensure_known_tables_cached
from ..utils.sql_text import _extract_sql, _sanitize_sql
from ..semantic.repair import RepairEngine

# try to keep your existing few-shot fetch; guard if missing
try:
    from ..semantic.lessons import fetch_few_shots  # uses Postgres in your tree; replace later with Qdrant impl
except Exception:
    async def fetch_few_shots(question: str, k: int = 3, min_sim: float = 0.55):
        return []

# LLM client
try:
    from ..clients import ollama
except Exception:
    ollama = None  # type: ignore

async def _llm_complete(system: str, user: str) -> str:
    model = getattr(settings, "GEN_MODEL", None) or "llama3.2:3b-instruct"
    if ollama is None:
        raise RuntimeError("clients.ollama unavailable")
    # prefer chat
    if hasattr(ollama, "chat"):
        return await ollama.chat(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    if hasattr(ollama, "generate"):
        return await ollama.generate(model, f"{system}\n\n{user}")
    # fallback
    if hasattr(ollama, "complete"):
        return await ollama.complete(model, system, user)
    raise RuntimeError("No supported LLM method on clients.ollama")

async def generate_sql(
    question: str,
    limit: int,
    negative_sql_examples: Optional[List[str]] = None,
    extra_hint: str = "",
) -> str:
    if not isinstance(limit, int) or limit < 1: limit = 50

    mdl_context = get_context()
    live_schema = get_schema_text() or ""

    # GOOD-only few-shots if your impl supports it; otherwise examples may be empty
    try:
        shots = await fetch_few_shots(question, k=3, min_sim=0.55)
    except Exception:
        shots = []
    examples = ""
    if shots:
        blocks = []
        for ex in shots:
            q = (ex.get("question") if isinstance(ex, dict) else "") or ""
            s = (ex.get("sql") if isinstance(ex, dict) else "") or ""
            if s: blocks.append(f"-- Example (similar question): {q}\n{s}\n")
        if blocks:
            examples = "\n# Good examples (follow joins/structure; adapt filters):\n" + "\n".join(blocks)

    negatives = ""
    if negative_sql_examples:
        bad_blocks = [f"-- DO NOT COPY (bad example):\n{b}\n" for b in negative_sql_examples[:3] if isinstance(b, str) and b.strip()]
        if bad_blocks:
            negatives = "\n# Known-bad SQL for this question (avoid these patterns):\n" + "\n".join(bad_blocks)

    system = (
        "You generate PostgreSQL SELECT queries ONLY.\n"
        "Return a single SELECT statement for the user's question.\n"
        "No explanations or JSON; prefer fenced SQL.\n\n"
        "# Semantic Layer (authoritative rules & joins)\n"
        f"{mdl_context}\n\n"
        "# Cached schema snapshot (tables & columns)\n"
        f"{live_schema}\n"
        f"{examples}"
        f"{negatives}\n"
    )
    user = (
        f"User question: {question}\n\n"
        "Rules:\n"
        "- SELECT-only. No DDL/DML/CTE/multi-statement.\n"
        "- Use explicit JOINs via the Semantic Layer relationships.\n"
        "- For aggregates, every non-aggregated select column must be in GROUP BY.\n"
        "- For 'top ... by ...', ORDER BY the aggregate DESC.\n"
        f"- If LIMIT is missing, add LIMIT {limit}.\n"
        + (f"\nExtra constraints:\n{extra_hint}\n" if extra_hint else "")
        + "\nReturn either a single fenced SQL block, or plain SQL.\n"
    )

    raw = await _llm_complete(system, user)
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, limit)

    parse_one(sql, read="postgres")
    ensure_known_tables_cached(sql)
    return sql

async def repair_sql(question: str, limit: int, error_message: str, prev_sql: Optional[str] = None) -> str:
    limit = max(1, int(limit or 50))

    if prev_sql:
        eng = RepairEngine(limit=limit)
        candidate, remaining, applied = eng.apply(prev_sql)
        try:
            parse_one(candidate, read="postgres")
            ensure_known_tables_cached(candidate)
            return _sanitize_sql(candidate, limit)
        except Exception:
            error_message = (error_message + "\n" if error_message else "") + \
                            f"[auto-repair tried: {', '.join(applied) or 'none'}; remaining: {', '.join(remaining) or 'none'}]"

    mdl_context = get_context()
    live_schema = get_schema_text() or ""
    system = (
        "You fix invalid PostgreSQL SELECT queries.\n"
        "Return a single corrected SELECT statement only—no explanations or JSON.\n\n"
        "# Semantic Layer\n"
        f"{mdl_context}\n\n"
        "# Cached schema snapshot\n"
        f"{live_schema}\n"
        "- Use the table alias for all column references if a table is aliased.\n"
        "- Qualify ambiguous columns with the correct alias.\n"
        "- Include all non-aggregated select columns in GROUP BY when using aggregates.\n"
    )
    user = (
        f"Question: {question}\n"
        f"Previous SQL:\n{prev_sql or '(none)'}\n\n"
        f"DB/Error:\n{error_message}\n\n"
        f"Rules:\n- SELECT-only.\n- Fix names per schema.\n- If LIMIT is missing, add LIMIT {limit}.\n"
        "Return only the corrected SQL."
    )

    raw = await _llm_complete(system, user)
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, limit)
    parse_one(sql, read="postgres")
    ensure_known_tables_cached(sql)
    return sql
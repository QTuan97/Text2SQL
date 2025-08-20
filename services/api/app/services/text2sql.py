from __future__ import annotations
from typing import Any, Dict, List, Optional
from sqlglot import parse_one
from qdrant_client import QdrantClient

from ..config import settings
from ..semantic.trainning import query_related_schema
from ..semantic.provider import get_context, get_schema_text
from ..semantic.sql_guard import ensure_known_tables_cached
from ..utils.sql_text import _extract_sql, _sanitize_sql, _time_guide, _normalize_ws
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
            "- After aliasing a table, ALWAYS use the alias; never reference the base table name.\n"  # NEW
            "- Qualify ambiguous columns with the correct alias.\n"  # NEW
            "- For aggregates, every non-aggregated select column must be in GROUP BY.\n"
            "- For 'top ... by ...', ORDER BY the aggregate DESC.\n"
            f"- If LIMIT is missing, add LIMIT {limit} **only for list-like queries**.\n"  # clarify (see C)
            + (f"\nExtra constraints:\n{extra_hint}\n" if extra_hint else "")
            + "\nReturn either a single fenced SQL block, or plain SQL.\n"
    )

    raw = await _llm_complete(system, user)
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, limit)

    parse_one(sql, read="postgres")
    ensure_known_tables_cached(sql)
    return sql

# Repair SQL
async def repair_sql(
    question: str,
    limit: int,
    error_message: str,
    prev_sql: Optional[str] = None
) -> str:
    limit = max(1, int(limit or 50))

    # 0) try your deterministic fixer first (cheap)
    if prev_sql:
        eng = RepairEngine(limit=limit)
        candidate, remaining, applied = eng.apply(prev_sql)
        try:
            parse_one(candidate, read="postgres")
            ensure_known_tables_cached(candidate)
            return _sanitize_sql(candidate, limit)
        except Exception:
            # carry forward what was tried so LLM can avoid repeating it
            error_message = (
                (error_message + "\n") if error_message else ""
            ) + f"[auto-repair tried: {', '.join(applied) or 'none'}; remaining: {', '.join(remaining) or 'none'}]"

    # 1) build context (RAG schema FIRST)
    mdl_context = get_context()
    live_schema = (get_schema_text() or "")[:2000]  # fallback only
    qdrant: QdrantClient = get_qdrant()
    hits = await query_related_schema(qdrant, question=question, top_k=8)
    rag_schema = "\n\n".join([h["text"] for h in hits])[:3000]

    # 2) prompt with strict alias rules + time guide
    system = (
        "You fix invalid PostgreSQL SELECT queries.\n"
        "Return a single corrected SELECT statement only—no explanations or JSON.\n\n"
        f"{_time_guide()}\n"
        "# Retrieved schema context (most relevant)\n"
        f"{rag_schema or '(no retrieved context)'}\n\n"
        "# Semantic Layer\n"
        f"{mdl_context}\n\n"
        "# Cached schema snapshot (fallback)\n"
        f"{live_schema}\n"
        "- After aliasing a table, ALWAYS use the alias; never reference the base table name.\n"
        "- Qualify ambiguous columns with the correct alias.\n"
        "- Include all non-aggregated select columns in GROUP BY when using aggregates.\n"
    )
    user = (
        f"Question: {question}\n"
        f"Previous SQL:\n{prev_sql or '(none)'}\n\n"
        f"DB/Error:\n{error_message}\n\n"
        "Rules:\n"
        "- SELECT-only.\n"
        "- Fix names per schema.\n"
        f"- If LIMIT is missing, add LIMIT {limit} only for list-like queries (not single-row aggregates).\n"
        "Return only the corrected SQL."
    )

    raw = await _llm_complete(system, user)
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, limit)

    # 3) de-dup guard: if LLM returned same SQL, don’t “loop” it back
    if prev_sql and _normalize_ws(sql) == _normalize_ws(prev_sql):
        # last-ditch: try forcing alias normalization via RepairEngine's stricter passes
        eng = RepairEngine(limit=limit, aggressive=True)  # if you have a flag; else reuse apply()
        candidate, _, _ = eng.apply(sql)
        sql = candidate

    # 4) validate; raise to caller if still invalid (so caller can show error once)
    parse_one(sql, read="postgres")
    ensure_known_tables_cached(sql)
    return sql
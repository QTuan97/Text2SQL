from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sqlglot import parse_one

# Config + MDL context (no direct DB access)
from ..config import settings
from ..semantic.provider import get_context

# Cached schema from Qdrant (no Postgres touch)
from ..semantic.schema_cache import get_schema_text, ensure_known_tables_cached

# Async Ollama client already used in your project
from ..clients import ollama

try:
    from ..services.lessons import search_lessons, record_learning  # type: ignore

    async def fetch_few_shots(question: str, k: int = 3, min_sim: float = 0.55):
        hits = await search_lessons(question, k=k, min_sim=min_sim)
        out = []
        for h in hits:
            out.append({"question": h.get("question") or "", "sql": h.get("sql") or ""})
        return out

    async def record_success(question: str, sql: str, tables_used: List[str]) -> None:
        # Idempotent upsert keyed by (question, sql)
        await record_learning(
            question, sql,
            tables_used=tables_used,
            executed=False, rowcount=0, error=None,
            source="text2sql"
        )
except Exception:
    async def fetch_few_shots(question: str, k: int = 3, min_sim: float = 0.55):
        return []
    async def record_success(question: str, sql: str, tables_used: List[str]) -> None:
        return


# Helpers: extract / sanitize SQL
_FENCE_OPEN  = re.compile(r'^\s*```(?:json|sql)?\s*', re.I | re.M)
_FENCE_CLOSE = re.compile(r'\s*```\s*$', re.I)
_LIMIT_RE    = re.compile(r'(?is)\blimit\s+(\d+)\b')

def _extract_sql(resp: str) -> str:
    """
    Accept raw LLM output and return a clean single SELECT.
    Handles JSON {"sql": "..."} or fenced code, trims stray quotes/braces.
    Fixes cases like:  ... LIMIT 10"}  →  ... LIMIT 10
    """
    s = (resp or "").strip()

    # Strip code fences if present
    s = _FENCE_OPEN.sub("", s)
    s = _FENCE_CLOSE.sub("", s)

    # If it's JSON, prefer the 'sql' field
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("sql"), str):
            s = obj["sql"]
    except Exception:
        pass

    s = s.strip()

    # Remove surrounding literal quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Remove trailing unmatched braces from half-JSON endings
    while s.endswith("}") and s.count("{") < s.count("}"):
        s = s[:-1].rstrip()

    # Remove any trailing quotes/backticks that slipped through
    s = re.sub(r'["`\']+\s*$', "", s)
    s = s.replace("```", "").strip()

    # Keep from the first SELECT onward (in case the model added preface text)
    m = re.search(r'(?is)\bselect\b', s)
    if m:
        s = s[m.start():].strip()

    if not re.match(r'(?is)^\s*select\b', s):
        raise ValueError("Model did not return a SELECT statement.")
    return s

def _fix_mysql_limit(sql: str) -> str:
    """
    Turn MySQL 'LIMIT a,b' into Postgres 'LIMIT b OFFSET a'.
    """
    m = re.search(r'(?is)\blimit\s+(\d+)\s*,\s*(\d+)\b', sql)
    if not m:
        return sql
    a, b = int(m.group(1)), int(m.group(2))
    return re.sub(r'(?is)\blimit\s+\d+\s*,\s*\d+\b', f'LIMIT {b} OFFSET {a}', sql)

def _enforce_limit(sql: str, limit: int) -> str:
    """
    If SQL already has LIMIT n, keep the smaller of (n, limit). Otherwise append LIMIT.
    """
    m = _LIMIT_RE.search(sql)
    if m:
        try:
            existing = int(m.group(1))
            if existing <= limit:
                return sql
            # replace only the digits
            start, end = m.span(1)
            return sql[:start] + str(limit) + sql[end:]
        except Exception:
            return sql

    sql = sql.rstrip().rstrip(";")
    return f"{sql} LIMIT {limit}"

def _sanitize_sql(sql: str, limit: int) -> str:
    """
    Final sanity pass: single SELECT statement, normalized limit, no trailing junk.
    """
    sql = sql.strip().strip("`")
    sql = _fix_mysql_limit(sql)

    # Keep only the first statement (defense-in-depth)
    if ";" in sql:
        sql = sql.split(";")[0].strip()

    if not re.match(r'(?is)^\s*select\b', sql):
        raise ValueError("Only SELECT queries are allowed.")

    sql = _enforce_limit(sql, limit)
    return sql

# LLM plumbing (uses Ollama client)
async def _llm_complete(system: str, user: str) -> str:
    """
    Call your Ollama generation endpoint. Tries common call styles.
    """
    model = getattr(settings, "GEN_MODEL", None) or getattr(settings, "GENERATION_MODEL", None) \
            or getattr(settings, "LLM_MODEL", None) or "llama3.2:1b-instruct-q4_K_M"

    # Prefer a 'chat' style if available
    try:
        if hasattr(ollama, "chat"):
            return await ollama.chat(model, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
    except Exception:
        pass

    # Fallback: 'generate' style => concatenate
    try:
        if hasattr(ollama, "generate"):
            return await ollama.generate(model, f"{system}\n\n{user}")
    except Exception:
        pass

    # Last resort: 'complete' style (system & user separate)
    if hasattr(ollama, "complete"):
        return await ollama.complete(model, system, user)

    raise RuntimeError("No supported LLM method found on clients.ollama (chat/generate/complete).")

# Public API
async def generate_sql(question: str, limit: int) -> str:
    """
    Build a grounded prompt with hybrid MDL context (+ lessons), call the LLM,
    extract & sanitize SQL, and validate (syntax + cached schema).
    Raises ValueError for any non-SELECT / parse issues (route converts to 400).
    """
    if not isinstance(limit, int) or limit < 1:
        limit = 50

    mdl_context = get_context()

    # Load cached schema snapshot from Qdrant
    live_schema = get_schema_text()

    # Retrieve few-shot “lessons” to guide the model (from Qdrant)
    shots = await fetch_few_shots(question, k=3, min_sim=0.55)
    examples = ""
    if shots:
        ex_blocks = []
        for ex in shots:
            ex_blocks.append(
                f"-- Example (similar question): {ex['question']}\n"
                f"{ex['sql']}\n"
            )
        examples = "\n# Good examples (follow joins/structure; adapt filters):\n" + "\n".join(ex_blocks)

    system = (
        "You generate PostgreSQL SELECT queries ONLY.\n"
        "Return a single SELECT statement for the user's question.\n"
        "Never return explanations or JSON unless explicitly asked; prefer fenced SQL.\n\n"
        "# Semantic Layer (authoritative rules & joins)\n"
        f"{mdl_context}\n\n"
        "# Live schema snapshot (cached, authoritative)\n"
        f"{live_schema}\n"
        f"{examples}\n"
    )

    user = (
        f"User question: {question}\n\n"
        "Rules:\n"
        "- SELECT-only. No DDL/DML/CTE/multi-statement.\n"
        "- Use explicit JOINs via the relationships in the Semantic Layer.\n"
        "- For aggregates (SUM/COUNT/AVG), every non-aggregated column must be in GROUP BY.\n"
        "- For 'top ... by ...', ORDER BY the aggregate DESC.\n"
        f"- If LIMIT is missing, add LIMIT {limit}.\n\n"
        "Return either a single fenced SQL block, or plain SQL.\n"
    )

    raw = await _llm_complete(system, user)

    # Clean and validate
    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, limit)

    # Validate syntax
    try:
        parse_one(sql, read="postgres")
    except Exception as e:
        raise ValueError(f"Bad SQL from model: {e.__class__.__name__}: {e}")

    # Validate tables exist in cached schema (no DB)
    ensure_known_tables_cached(sql)

    # Record as a successful lesson (best-effort, no execution info)
    try:
        from sqlglot import exp
        tnames: List[str] = []
        tree = parse_one(sql, read="postgres")
        for t in tree.find_all(exp.Table):
            tname = None
            if hasattr(t, "name") and isinstance(t.name, str):
                tname = t.name
            elif getattr(t, "this", None) is not None:
                tname = getattr(t.this, "name", None)
            if tname:
                tnames.append(tname)
        seen = set(); tables_used: List[str] = []
        for n in tnames:
            if n not in seen:
                tables_used.append(n); seen.add(n)
        await record_success(question, sql, tables_used)
    except Exception:
        pass

    return sql


def execute_sql(sql: str):
    """
    Disabled: Postgres execution has been turned off in this deployment.
    Keeping the function for import-compatibility.
    """
    raise RuntimeError("SQL execution against Postgres is disabled in this deployment.")


async def repair_sql(question: str, limit: int, error_message: str, prev_sql: Optional[str] = None) -> str:
    """
    One-shot self-repair using a provided DB error message. Returns a new validated SQL or raises ValueError.
    Still validates against the cached schema (no DB introspection).
    """
    mdl_context = get_context()
    live_schema = get_schema_text()

    system = (
        "You fix invalid PostgreSQL SELECT queries.\n"
        "Given the user's question, the previous SQL, and the error message, "
        "return a corrected single SELECT statement only.\n\n"
        "# Semantic Layer\n"
        f"{mdl_context}\n\n"
        "# Live schema snapshot (cached)\n"
        f"{live_schema}\n"
    )

    user = (
        f"Question: {question}\n"
        f"Previous SQL:\n{prev_sql or '(none)'}\n\n"
        f"DB/Error:\n{error_message}\n\n"
        f"Rules:\n- Keep it SELECT-only.\n- Fix column/table names according to the schema above.\n"
        f"- Ensure GROUP BY correctness for aggregates.\n- If LIMIT is missing, add LIMIT {limit}.\n"
        "Return only the corrected SQL (fenced or plain).\n"
    )

    raw = await _llm_complete(system, user)

    sql = _extract_sql(raw)
    sql = _sanitize_sql(sql, limit)

    # Validate syntax
    try:
        parse_one(sql, read="postgres")
    except Exception as e:
        raise ValueError(f"Bad SQL from repair: {e.__class__.__name__}: {e}")

    # Validate tables exist against cached schema (no DB)
    ensure_known_tables_cached(sql)

    return sql
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sqlglot import parse_one

# DB + config + LLM + MDL context
from ..clients.postgres import pg_connect
from ..config import settings
from ..semantic.provider import get_context

# Optional: live schema snapshot
try:
    from ..clients.postgres import schema_snapshot  # type: ignore
except Exception:
    schema_snapshot = None  # type: ignore

# Async Ollama client already used in your project
from ..clients import ollama

# Optional: FROM/JOIN table guard (soft-required; see try/except stubs below)
try:
    from ..semantic.sql_guard import ensure_known_tables  # type: ignore
except Exception:
    def ensure_known_tables(sql: str, schema: str = None) -> None:  # fallback no-op
        return

# Optional: retrieval “lessons” (few-shot memory)
try:
    from ..semantic.lessons import fetch_few_shots, record_success  # type: ignore
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
    extract & sanitize SQL, and validate (syntax + real tables).
    Raises ValueError for any non-SELECT / parse issues (route converts to 400).
    """
    if not isinstance(limit, int) or limit < 1:
        limit = 50

    mdl_context = get_context()
    live_schema = ""
    if callable(schema_snapshot):
        try:
            live_schema = schema_snapshot()  # optional: your existing helper
        except Exception:
            live_schema = ""

    # Retrieve few-shot “lessons” to guide the model
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
        "# Live schema snapshot (for column names/types)\n"
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

    # Validate tables exist in the current DB
    ensure_known_tables(sql, schema=getattr(settings, "PG_SCHEMA", "public"))

    # Record as a successful lesson (best-effort)
    try:
        from sqlglot import exp
        tnames: List[str] = []
        tree = parse_one(sql, read="postgres")
        for t in tree.find_all(exp.Table):
            if t.this:
                tnames.append(t.this.name)
        seen = set(); tables_used: List[str] = []
        for n in tnames:
            if n not in seen:
                tables_used.append(n); seen.add(n)
        await record_success(question, sql, tables_used)
    except Exception:
        pass

    return sql


def execute_sql(sql: str) -> List[Dict[str, Any]]:
    """
    Execute validated SQL and return rows as a list of dicts.
    If your route expects (rows, columns), adapt accordingly.
    """
    rows: List[Dict[str, Any]] = []
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in (cur.description or [])]
            for r in cur.fetchall():
                rows.append({cols[i]: r[i] for i in range(len(cols))})
    return rows


async def repair_sql(question: str, limit: int, error_message: str, prev_sql: Optional[str] = None) -> str:
    """
    One-shot self-repair using the DB error. Returns a new validated SQL or raises ValueError.
    """
    mdl_context = get_context()
    live_schema = ""
    if callable(schema_snapshot):
        try:
            live_schema = schema_snapshot()
        except Exception:
            live_schema = ""

    system = (
        "You fix invalid PostgreSQL SELECT queries.\n"
        "Given the user's question, the previous SQL, and the error message, "
        "return a corrected single SELECT statement only.\n\n"
        "# Semantic Layer\n"
        f"{mdl_context}\n\n"
        "# Live schema snapshot\n"
        f"{live_schema}\n"
    )

    user = (
        f"Question: {question}\n"
        f"Previous SQL:\n{prev_sql or '(none)'}\n\n"
        f"DB Error:\n{error_message}\n\n"
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

    # Validate tables exist
    ensure_known_tables(sql, schema=getattr(settings, "PG_SCHEMA", "public"))

    return sql
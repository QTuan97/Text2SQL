from __future__ import annotations
import json, re
from typing import Dict, Any, Optional, List

from ..config import settings
from ..dependencies import get_qdrant
from ..clients.llm_compat import _llm_complete
from ..semantic.training import query_related_schema
from ..semantic.schema_cache import get_schema_text
from ..semantic.sql_guard import build_ambiguity_guard

_FENCE_OPEN  = re.compile(r'^\s*```(?:json|sql)?\s*', re.I | re.M)
_FENCE_CLOSE = re.compile(r'\s*```\s*$', re.I)
_LIMIT_RE    = re.compile(r'(?is)\blimit\s+(\d+)\b')

def _extract_sql(resp: str) -> str:
    s = (resp or "").strip()
    s = _FENCE_OPEN.sub("", s)
    s = _FENCE_CLOSE.sub("", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("sql"), str):
            s = obj["sql"]
    except Exception:
        pass
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    while s.endswith("}") and s.count("{") < s.count("}"):
        s = s[:-1].rstrip()
    s = re.sub(r'["`\']+\s*$', "", s).replace("```", "").strip()
    m = re.search(r'(?is)\bselect\b', s)
    if m: s = s[m.start():].strip()
    if not re.match(r'(?is)^\s*select\b', s):
        raise ValueError("Model did not return a SELECT statement.")
    return s

def _fix_mysql_limit(sql: str) -> str:
    m = re.search(r'(?is)\blimit\s+(\d+)\s*,\s*(\d+)\b', sql)
    if not m: return sql
    a, b = int(m.group(1)), int(m.group(2))
    return re.sub(r'(?is)\blimit\s+\d+\s*,\s*\d+\b', f'LIMIT {b} OFFSET {a}', sql)

def _enforce_limit(sql: str, limit: int) -> str:
    m = _LIMIT_RE.search(sql)
    if m:
        try:
            existing = int(m.group(1))
            if existing <= limit: return sql
            start, end = m.span(1)
            return sql[:start] + str(limit) + sql[end:]
        except Exception:
            return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {limit}"

def _normalize_ws(s: str) -> str:
    # collapse whitespace and lowercase for equality checks
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _time_guide(tz: str | None = None) -> str:
    """
    Strict instructions for relative time windows.
    - Use range predicates.
    - NEVER hardcode month/year or use EXTRACT(...)=N.
    - Replace <TS> with the correct QUALIFIED timestamp column from the fact table (e.g., o.created_at).
    """
    tz = tz or getattr(settings, "BUSINESS_TZ", "UTC")
    return (
        "# Time guidance (STRICT)\n"
        f"- Business timezone: '{tz}'. Use NOW() AT TIME ZONE '{tz}' for relative periods.\n"
        "- ALWAYS use bounded ranges on the timestamp; NEVER hardcode month/year or use EXTRACT(...)=N.\n"
        "- Replace <TS> with the correct qualified timestamp column from the relevant table (e.g., o.created_at).\n\n"

        "# Canonical windows (copy exactly, swapping <TS>):\n"
        f"- THIS_MONTH:  (<TS> AT TIME ZONE '{tz}') >= DATE_TRUNC('month', NOW() AT TIME ZONE '{tz}') "
        f"AND (<TS> AT TIME ZONE '{tz}') < DATE_TRUNC('month', NOW() AT TIME ZONE '{tz}') + INTERVAL '1 month'\n"

        f"- TODAY:       (<TS> AT TIME ZONE '{tz}') >= (NOW() AT TIME ZONE '{tz}')::date "
        f"AND (<TS> AT TIME ZONE '{tz}') < ((NOW() AT TIME ZONE '{tz}')::date + INTERVAL '1 day')\n"

        f"- YESTERDAY:   (<TS> AT TIME ZONE '{tz}') >= ((NOW() AT TIME ZONE '{tz}')::date - INTERVAL '1 day') "
        f"AND (<TS> AT TIME ZONE '{tz}') <  (NOW() AT TIME ZONE '{tz}')::date\n"

        f"- LAST_7_DAYS: (<TS> AT TIME ZONE '{tz}') >= (NOW() AT TIME ZONE '{tz}') - INTERVAL '7 days' "
        f"AND (<TS> AT TIME ZONE '{tz}') <  (NOW() AT TIME ZONE '{tz}')\n"

        f"- THIS_WEEK:   (<TS> AT TIME ZONE '{tz}') >= DATE_TRUNC('week', NOW() AT TIME ZONE '{tz}') "
        f"AND (<TS> AT TIME ZONE '{tz}') <  DATE_TRUNC('week', NOW() AT TIME ZONE '{tz}') + INTERVAL '1 week'\n"

        f"- LAST_MONTH:  (<TS> AT TIME ZONE '{tz}') >= DATE_TRUNC('month', (NOW() AT TIME ZONE '{tz}') - INTERVAL '1 month') "
        f"AND (<TS> AT TIME ZONE '{tz}') <  DATE_TRUNC('month', NOW() AT TIME ZONE '{tz}')\n"

        f"- THIS_QUARTER:(<TS> AT TIME ZONE '{tz}') >= DATE_TRUNC('quarter', NOW() AT TIME ZONE '{tz}') "
        f"AND (<TS> AT TIME ZONE '{tz}') <  DATE_TRUNC('quarter', NOW() AT TIME ZONE '{tz}') + INTERVAL '1 quarter'\n"

        f"- THIS_YEAR:   (<TS> AT TIME ZONE '{tz}') >= DATE_TRUNC('year', NOW() AT TIME ZONE '{tz}') "
        f"AND (<TS> AT TIME ZONE '{tz}') <  DATE_TRUNC('year', NOW() AT TIME ZONE '{tz}') + INTERVAL '1 year'\n"

        "\n# DO NOTS\n"
        "- Do NOT use EXTRACT(MONTH/YEAR)=N comparisons.\n"
        "- Do NOT cast to text or compare formatted strings.\n"
        "- ALWAYS qualify columns with the table alias used in the FROM/JOIN.\n"
    )

def _looks_like_single_aggregate(sql: str) -> bool:
    s = sql.strip().lower()
    # crude but effective: SELECT ... FROM ... WHERE ... without GROUP BY and with SUM/AVG/COUNT and no plain columns
    has_group = re.search(r"\bgroup\s+by\b", s) is not None
    has_agg   = re.search(r"\b(sum|avg|count|min|max)\s*\(", s) is not None
    return has_agg and not has_group

def _sanitize_sql(sql: str, limit: int) -> str:
    sql = (sql or "").strip().strip("`")
    sql = _fix_mysql_limit(sql)
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    if not re.match(r'(?is)^\s*select\b', sql):
        raise ValueError("Only SELECT queries are allowed.")
    if _looks_like_single_aggregate(sql):
        return sql
    return _enforce_limit(sql, limit)

# Plan query
def _extract_json(s: str) -> str:
    s = s.strip()
    # strip fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = s.rsplit("```", 1)[0]
    # grab first {...} block
    m = re.search(r"\{.*\}", s, re.S)
    return m.group(0) if m else s

async def plan_query(question: str) -> Dict[str, Any]:
    # RAG: pull only relevant schema to keep context small
    qc = get_qdrant()
    hits = await query_related_schema(qc, question=question, top_k=8)
    rag_schema = "\n".join(h["text"] for h in hits) if hits else ""
    fallback_schema = get_schema_text() or ""
    amb = build_ambiguity_guard()

    system = (
        "You are a SQL query PLANNER. Output STRICT JSON ONLY, no prose, no code fences.\n"
        "Your job: convert the user question into a minimal structured plan for a PostgreSQL SELECT query.\n\n"
        "# Relevant schema (authoritative names)\n"
        f"{rag_schema or fallback_schema}\n\n"
        f"{amb}\n"
        "# Time hints (use names, do not expand to SQL here)\n"
        "- THIS_MONTH, TODAY, LAST_7_DAYS\n\n"
        "# JSON schema to output (all keys required):\n"
        "{\n"
        '  "tables":     ["<table1> <alias1>", "..."],               // every table must include a short alias\n'
        '  "joins":      [{"left":"u","right":"o","on":"u.id=o.user_id"}],\n'
        '  "dimensions": ["u.name","u.city"],                        // non-aggregated select columns\n'
        '  "metrics":    [{"alias":"revenue","expr":"SUM(o.amount)"}],\n'
        '  "filters":    ["u.city = \'Can Tho\'", "THIS_MONTH?"],    // literals or time keywords only\n'
        '  "order_by":   [{"expr":"revenue","dir":"DESC"}],\n'
        '  "limit":      5\n'
        "}\n"
        "Notes: use ONLY canonical table names from the schema; attach aliases you invent (1–2 letters). "
        "If the question does not ask for a field, do not include it as a dimension."
    )
    user = f"Question: {question}\nReturn ONLY the JSON plan."
    raw = await _llm_complete(system, user)
    data = json.loads(_extract_json(raw))
    # minimal validation
    for k in ["tables","joins","dimensions","metrics","filters","order_by","limit"]:
        data.setdefault(k, [] if k!="limit" else 50)
    return data

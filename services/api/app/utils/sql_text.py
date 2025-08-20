from __future__ import annotations
import json, re

from ..config import settings

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
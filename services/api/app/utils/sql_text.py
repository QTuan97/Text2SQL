from __future__ import annotations
import json, re

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

def _sanitize_sql(sql: str, limit: int) -> str:
    sql = sql.strip().strip("`")
    sql = _fix_mysql_limit(sql)
    if ";" in sql: sql = sql.split(";")[0].strip()
    if not re.match(r'(?is)^\s*select\b', sql):
        raise ValueError("Only SELECT queries are allowed.")
    return _enforce_limit(sql, limit)
from __future__ import annotations
import re, json
from fastapi import HTTPException
from sqlglot import parse_one
from ..clients.postgres import schema_snapshot, pg_connect
from ..services.generation import llm
from ..semantic.provider import get_context

# ---------- helpers ----------
def _strip_fences(txt: str) -> str:
    # remove ```json / ```sql fences if present
    return re.sub(r"^```[\w-]*\s*|\s*```$", "", txt.strip(), flags=re.IGNORECASE | re.MULTILINE)

def _json_only(prompt: str) -> str:
    return f"""{prompt}

    Return ONLY this one-line JSON (no code fences, no extra keys):
    {{"sql": "<single valid PostgreSQL SELECT>"}}"""

def _parse_sql_from_model(txt: str) -> str:
    txt = _strip_fences(txt)
    # try JSON first
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "sql" in obj and isinstance(obj["sql"], str):
            return obj["sql"]
    except Exception:
        pass
    # fallback: extract first SELECT ... (tolerate stray text)
    m = re.search(r"select\b.*", txt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("No SQL found in model output")
    sql = m.group(0)
    # stop at first code fence or unmatched brace/extra JSON
    sql = sql.split("```")[0].split("\n\n{")[0].strip()
    return sql

def _fix_mysql_limit(sql: str) -> str:
    # Convert "LIMIT offset, count" -> "LIMIT count OFFSET offset"
    return re.sub(r"\blimit\s+(\d+)\s*,\s*(\d+)\b", r"LIMIT \2 OFFSET \1", sql, flags=re.IGNORECASE)

def _sanitize_sql(sql: str, limit: int) -> str:
    sql = sql.strip().strip("`")
    sql = _fix_mysql_limit(sql)
    if ";" in sql:
        sql = sql.split(";")[0]
    if not sql.lower().startswith("select"):
        raise HTTPException(400, detail="Only SELECT queries are allowed.")
    if not re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        sql += f" LIMIT {limit}"
    return sql

# ---------- public API ----------
async def generate_sql(question: str, limit: int) -> str:
    schema = schema_snapshot()
    mdl_ctx = get_context()
    base = f"""You generate PostgreSQL SELECT queries.
    
    # Semantic Layer (authoritative)
    {mdl_ctx}
    
    # Live schema snapshot (for column names/types)
    {schema}
    
    Instructions:
    - Follow the 'Rules' in the Semantic Layer exactly.
    - Use only tables/columns present above and relationships defined in the Semantic Layer.
    - Use explicit JOINs with short aliases (e.g., users AS u, orders AS o).
    - For aggregates (SUM/COUNT/AVG), every non-aggregated selected column MUST be in GROUP BY.
    - For "top … by total", ORDER BY the aggregate DESC.
    - If LIMIT is missing for list-like results, add LIMIT {limit}. Aggregations do not require LIMIT.
    
    Question: {question}
    Return ONLY JSON: {{"sql":"<single valid PostgreSQL SELECT>"}}"""
    txt = await llm(_json_only(base))
    try:
        sql = _parse_sql_from_model(txt)
        parse_one(sql, read="postgres")
        return _sanitize_sql(sql, limit)
    except Exception as e:
        # single self-repair with error message
        fix_prompt = _json_only(base + f"\n\nThe SQL was invalid: {e!s}\nReturn only JSON with a fixed SQL.")
        txt2 = await llm(fix_prompt)
        sql2 = _parse_sql_from_model(txt2)
        # normalize & validate again
        sql2 = _sanitize_sql(sql2, limit)
        parse_one(sql2, read="postgres")
        return sql2

def execute_sql(sql: str) -> list[dict]:
    with pg_connect() as conn:
        cur = conn.cursor()
        cur.execute("SET LOCAL statement_timeout = '5s';")
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]

async def repair_sql(question: str, limit: int, error_msg: str, previous_sql: str) -> str:
    """Ask the model to fix a SQL statement that failed at execution time."""
    base = f"""You translate questions to SQL for PostgreSQL.
    The previous SQL failed to execute with this error:
    {error_msg}
    
    Question: {question}
    
    Fix the SQL. Follow these rules:
    - Use explicit JOINs with short aliases (users AS u, orders AS o).
    - If using aggregates (SUM/COUNT/AVG), every non-aggregated selected column must be in GROUP BY.
    - For "top ... by total", ORDER BY the aggregate DESC.
    - If LIMIT is missing, add LIMIT {limit}.
    Return ONLY JSON: {{"sql":"<single valid PostgreSQL SELECT>"}}"""
    txt = await llm(_json_only(base))
    sql = _parse_sql_from_model(txt)
    sql = _sanitize_sql(sql, limit)
    # validate is Postgres-parsable
    parse_one(sql, read="postgres")
    return sql
from __future__ import annotations
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas.common import NL2SQLIn, NL2SQLOut
from ..services.text2sql import generate_sql, execute_sql, repair_sql
from ..services.logger import log_request

from ..semantic.provider import get_mdl
from ..semantic.lessons import record_learning

router = APIRouter(prefix="/text2sql", tags=["text2sql"])

def _tables_in(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
        out: List[str] = []
        for t in tree.find_all(exp.Table):
            # robust name extraction across sqlglot versions
            if hasattr(t, "name") and isinstance(t.name, str):
                out.append(t.name)
            elif getattr(t, "this", None) is not None:
                tn = getattr(t.this, "name", None)
                if isinstance(tn, str):
                    out.append(tn)
        return [x for x in out if x]
    except Exception:
        return []

@router.post("", response_model=NL2SQLOut)
async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks) -> NL2SQLOut:
    limit = body.limit if isinstance(body.limit, int) and body.limit > 0 else 50

    # 1) Generate SQL (your existing service)
    try:
        final_sql = await generate_sql(body.question, limit=limit)
    except ValueError as e:
        # model returned non-SELECT or bad SQL
        raise HTTPException(status_code=400, detail=str(e))

    diagnostics: Dict[str, Any] = {"backend": "generate_sql", "limit": limit}
    tables = _tables_in(final_sql)

    # 2) Optionally execute; attempt one-shot repair on DB error
    rows: Optional[List[Dict[str, Any]]] = None
    rowcount: int = 0
    error: Optional[str] = None
    executed: bool = False

    if body.execute:
        try:
            rows = execute_sql(final_sql)
            executed = True
            rowcount = len(rows) if isinstance(rows, list) else 0
        except Exception as ex:
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

    # 3) Record learning to Qdrant (idempotent upsert; non-blocking)
    background_tasks.add_task(
        record_learning,
        body.question,
        final_sql,
        tables_used=tables,
        executed=executed,
        rowcount=rowcount,
        error=error,
        source="text2sql",
    )

    return NL2SQLOut(sql=final_sql, rows=rows, rowcount=rowcount, error=error, diagnostics=diagnostics)
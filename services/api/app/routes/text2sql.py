from __future__ import annotations
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas.common import NL2SQLIn
from ..services.text2sql import generate_sql, execute_sql, repair_sql
from ..services.logger import log_request

from ..semantic.intent_gate_embed import gate
from ..semantic.provider import get_mdl

router = APIRouter(prefix="/text2sql", tags=["text2sql"])

@router.post("")
async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()

    # 1) Intent gate (blocks OOD like “weather”)
    allow, info = await gate(body.question, get_mdl())
    if not allow:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "OUT_OF_DOMAIN",
                "message": "This looks outside the analytics domain.",
                **info,
            },
        )

    # 2) Generate SQL (generator should raise ValueError on bad/unclean SQL)
    try:
        sql = await generate_sql(body.question, body.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "BAD_SQL", "message": str(e)})

    out = {"sql": sql}

    # 3) Execute (optional) with one-shot auto-repair
    if getattr(body, "execute", False):
        try:
            rows = execute_sql(sql)
            out["rows"] = rows
        except Exception as e:
            # Try self-repair once with the error message
            try:
                fixed_sql = await repair_sql(body.question, body.limit, str(e), sql)
                rows = execute_sql(fixed_sql)
                out.update({"sql": fixed_sql, "rows": rows})
            except Exception as e2:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "EXEC_ERROR", "message": str(e2), "previous_error": str(e)},
                )

    # 4) Log
    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(
        log_request, "/text2sql", "POST", 200, body.model_dump(), out, ms
    )
    return out
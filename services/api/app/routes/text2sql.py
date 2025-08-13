from fastapi import APIRouter, BackgroundTasks
import time
from ..schemas.common import NL2SQLIn
from ..services.text2sql import generate_sql, execute_sql
from ..services.logger import log_request
from ..config import settings

router = APIRouter(prefix="/text2sql", tags=["text2sql"])

@router.post("")
async def text2sql(body: NL2SQLIn, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    sql = await generate_sql(body.question, body.limit)
    out = {"sql": sql}
    if body.execute and settings.POSTGRES_URL:
        try:
            rows = execute_sql(sql)
        except Exception as e:
            fixed_sql = await repair_sql(body.question, body.limit, str(e), sql)
            rows = execute_sql(fixed_sql)
            out = {"sql": fixed_sql, "rows": rows}
        else:
            out["rows"] = rows

    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(
        log_request, "/text2sql", "POST", 200, body.model_dump(), out, ms
    )
    return out
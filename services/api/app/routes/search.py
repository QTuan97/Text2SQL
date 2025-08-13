from fastapi import APIRouter, HTTPException, BackgroundTasks
import time
from ..schemas.common import SearchIn, SearchByVectorIn
from ..services.embeddings import embed_valid, embed_error
from ..services.retrieval import search_named
from ..services.logger import log_request
from ..config import settings

router = APIRouter(prefix="/search", tags=["search"])

@router.post("")
async def search(body: SearchIn, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    model_func = embed_valid if body.field == "valid_vec" else embed_error
    vec = await model_func(body.query)
    using = settings.VALID_NAME if body.field == "valid_vec" else settings.ERROR_NAME
    res = search_named(using, vec, body.limit)
    out = {"hits": [{"id": p.id, "score": p.score, "payload": p.payload} for p in res]}
    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(
        log_request, "/search", "POST", 200, body.model_dump(), out, ms
    )
    return out

@router.post("/by_vector")
async def search_by_vector(body: SearchByVectorIn, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    expected = settings.VALID_DIM if body.field == "valid_vec" else settings.ERROR_DIM
    if len(body.vector) != expected:
        raise HTTPException(400, detail=f"Vector size mismatch for '{body.field}': expected {expected}, got {len(body.vector)}.")
    using = settings.VALID_NAME if body.field == "valid_vec" else settings.ERROR_NAME
    res = search_named(using, body.vector, body.limit)
    out = {"hits": [{"id": p.id, "score": p.score, "payload": p.payload} for p in res]}
    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(
        log_request, "/search/by_vector", "POST", 200, body.model_dump(), out, ms
    )
    return out
from fastapi import APIRouter, HTTPException, BackgroundTasks
import time
import uuid
from typing import Any, Dict
from ..schemas.common import IndexIn
from ..services.embeddings import embed_valid, embed_error
from ..services.retrieval import upsert_point
from ..services.logger import log_request
from ..config import settings
from ..services.chunks import make_chunks

router = APIRouter(prefix="/index", tags=["index"])

def _to_qdrant_id(val: Any | None, text: str) -> int | str:
    """Normalize to Qdrant-friendly id: unsigned int or UUID string.
    - If val is None/blank: stable UUID v5 derived from content (reindex overwrites).
    - If val is int or numeric string: return int.
    - If val is a UUID string: return as-is.
    - Else: fold arbitrary string to stable UUID v5.
    """
    if val is None or str(val).strip() == "":
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    try:
        uuid.UUID(s)  # will raise if not a valid UUID string
        return s
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

@router.post("")
async def index_doc(doc: IndexIn, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()

    # Merge behavior: honor your provided doc.id if any, but always normalize
    vid = _to_qdrant_id(getattr(doc, "id", None), doc.text)

    valid = await embed_valid(doc.text)
    error = await embed_error(doc.text)
    if len(valid) != settings.VALID_DIM or len(error) != settings.ERROR_DIM:
        raise HTTPException(500, detail="Embedding dimension mismatch from Ollama.")

    payload: Dict[str, Any] = (doc.metadata or {}).copy()
    payload["text"] = doc.text

    upsert_point(vid, valid, error, payload)

    out = {"id": vid, "ok": True}
    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(log_request, "/index", "POST", 200, doc.model_dump(), out, ms)
    return out

@router.post("/chunk")
async def index_chunked(doc: IndexIn, background_tasks: BackgroundTasks):
    """Split long text into retrievable chunks and index each with deterministic IDs.
    Safe to call with short text as well; returns 1+ points.
    """
    t0 = time.perf_counter()
    text = doc.text or ""
    chunks = make_chunks(text) or [text]
    ids: list[str] = []
    meta = (doc.metadata or {}).copy()
    meta.setdefault("source", "ui")
    total = len(chunks)

    for i, ch in enumerate(chunks):
        # Deterministic per-chunk UUID (stable across retries)
        base = f"{getattr(doc, 'id', '')}::{i}::{ch[:64]}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, base))
        valid = await embed_valid(ch)
        error = await embed_error(ch)
        if len(valid) != settings.VALID_DIM or len(error) != settings.ERROR_DIM:
            raise HTTPException(500, detail="Embedding dimension mismatch from Ollama.")
        payload: Dict[str, Any] = meta | {"text": ch, "chunk_index": i, "chunk_total": total}
        upsert_point(chunk_id, valid, error, payload)
        ids.append(chunk_id)

    out = {"ids": ids, "chunks": total, "ok": True}
    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(log_request, "/index/chunk", "POST", 200, doc.model_dump(), out, ms)
    return out
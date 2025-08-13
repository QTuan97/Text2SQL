from __future__ import annotations
import httpx
from fastapi import APIRouter
from ..config import settings
from ..clients.qdrant import qdrant_client

router = APIRouter(prefix="", tags=["health"])

@router.get("/health")
async def health():
    ok = {"qdrant": False, "ollama": False}
    try:
        qdrant_client.get_collections(); ok["qdrant"] = True
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=5) as s:
            r = await s.get(f"{settings.OLLAMA_BASE_URL}/api/tags"); r.raise_for_status(); ok["ollama"] = True
    except Exception:
        pass
    return {"ok": all(ok.values()), **ok}
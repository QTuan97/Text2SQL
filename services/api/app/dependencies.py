from __future__ import annotations
from fastapi import FastAPI
from qdrant_client.http.models import Distance, VectorParams
from .config import settings
from .clients.qdrant import qdrant_client
from .services.logger import ensure_logs_table

def ensure_qdrant_collection() -> None:
    collections = qdrant_client.get_collections().collections
    names = [c.name for c in collections]
    if settings.QDRANT_COLLECTION not in names:
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                settings.VALID_NAME: VectorParams(size=settings.VALID_DIM, distance=Distance.COSINE),
                settings.ERROR_NAME: VectorParams(size=settings.ERROR_DIM, distance=Distance.COSINE),
            }
        )

def init_app(app: FastAPI) -> None:
    @app.on_event("startup")
    def _startup() -> None:
        ensure_qdrant_collection()
        ensure_logs_table()
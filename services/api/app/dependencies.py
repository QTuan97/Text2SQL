from __future__ import annotations
from fastapi import FastAPI
from qdrant_client.http.models import Distance, VectorParams
from .config import settings
from .clients.qdrant import qdrant_client

def _ensure_named_collection(collection_name: str) -> None:
    collections = qdrant_client.get_collections().collections
    names = [c.name for c in collections]
    if collection_name not in names:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                settings.VALID_NAME: VectorParams(size=settings.VALID_DIM, distance=Distance.COSINE),
                settings.ERROR_NAME: VectorParams(size=settings.ERROR_DIM, distance=Distance.COSINE),
            },
        )

def ensure_qdrant_collections() -> None:
    _ensure_named_collection(settings.QDRANT_COLLECTION)
    _ensure_named_collection(settings.LESSONS_COLLECTION)
    _ensure_named_collection(settings.SCHEMA_COLLECTION)

def init_app(app: FastAPI) -> None:
    @app.on_event("startup")
    def _startup() -> None:
        ensure_qdrant_collections()
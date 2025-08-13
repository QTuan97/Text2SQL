from __future__ import annotations
from qdrant_client.http.models import PointStruct
from ..config import settings
from ..clients.qdrant import qdrant_client

def upsert_point(point_id: str, valid_vec: list[float], error_vec: list[float], payload: dict) -> None:
    qdrant_client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector={settings.VALID_NAME: valid_vec, settings.ERROR_NAME: error_vec},
            payload=payload
        )],
    )

def search_named(field_name: str, vector: list[float], limit: int):
    return qdrant_client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=(field_name, vector),
        limit=limit,
        with_payload=True
    )
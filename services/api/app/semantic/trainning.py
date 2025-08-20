from __future__ import annotations
from typing import List, Dict
import uuid
import asyncio , httpx
from ..config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from .schema_introspect import TableInfo, render_table_doc
from ..services.embeddings import embed_valid, embed_error
from ..clients.llm_compat import schema_embed_one

DEFAULT_COLLECTION = "semantic_schema"

def ensure_collection(client: QdrantClient, collection: str = DEFAULT_COLLECTION):
    # Named vectors: valid_vec (768), error_vec (384)
    existing = client.get_collection(collection_name=collection)
    if existing is not None:
        return
    client.recreate_collection(
        collection_name=collection,
        vectors_config={
            "valid_vec": VectorParams(size=768, distance=Distance.COSINE),
            "error_vec": VectorParams(size=384, distance=Distance.COSINE),
        }
    )

def query_related_schema(
    client: QdrantClient,
    question: str,
    top_k: int = 8,
    collection: str = DEFAULT_COLLECTION,
) -> List[Dict]:
    qvec = embed_valid(question)
    res = client.search(
        collection_name=collection,
        query_vector=("valid_vec", qvec),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [hit.payload for hit in res]

try:
    from qdrant_client.models import PointStruct
except Exception:
    from qdrant_client.http.models import PointStruct

def _make_payload(t: TableInfo, text: str) -> Dict:
    return {
        "doc_type": "schema/table",
        "table_schema": t.table_schema,
        "table_name": t.table_name,
        "pk": t.pk_columns,
        "fks": t.fks,
        "text": text,
    }

async def _embed_vectors(text: str, include_error_vec: bool) -> Dict[str, List[float]]:
    v768 = await schema_embed_one(text, model=settings.VALID_EMBED_MODEL)
    if len(v768) != settings.VALID_DIM:
        raise ValueError(f"valid_vec dim mismatch: expected {settings.VALID_DIM}, got {len(v768)}")
    vectors: Dict[str, List[float]] = {settings.VALID_NAME: v768}

    if include_error_vec:
        v384 = await schema_embed_one(text, model=settings.ERROR_EMBED_MODEL)
        if len(v384) != settings.ERROR_DIM:
            raise ValueError(f"error_vec dim mismatch: expected {settings.ERROR_DIM}, got {len(v384)}")
        vectors[settings.ERROR_NAME] = v384

    return vectors

async def upsert_schema_docs(
    client: QdrantClient,
    tables: List[TableInfo],
    collection: str = None,
    include_error_vec: bool = False,
):
    collection = collection or settings.SCHEMA_COLLECTION

    texts = [render_table_doc(t) for t in tables]
    vectors_list = await asyncio.gather(*[
        _embed_vectors(text, include_error_vec) for text in texts
    ])

    points: List[PointStruct] = []
    for t, text, vectors in zip(tables, texts, vectors_list):
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{t.table_schema}.{t.table_name}"))
        points.append(PointStruct(id=pid, vector=vectors, payload=_make_payload(t, text)))

    BATCH = 64
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=collection, points=points[i:i+BATCH])
# app/semantic/training.py
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from ..config import settings
from ..clients.llm_compat import embed_one  # ASYNC: returns list[float]


# ──────────────────────────────────────────────────────────────────────────────
# Embedding helpers (ALWAYS awaited)
# ──────────────────────────────────────────────────────────────────────────────

async def _embed_valid(text: str) -> List[float]:
    """Valid (primary) named-vector embedding."""
    model = getattr(settings, "VALID_EMBED_MODEL", settings.VALID_EMBED_MODEL)
    return await embed_one(text, model=model)

async def _embed_error(text: str) -> List[float]:
    """Secondary embedding (for error/negatives, if you store it)."""
    model = getattr(settings, "ERROR_EMBED_MODEL", getattr(settings, "VALID_EMBED_MODEL", None))
    return await embed_one(text, model=model)

async def _embed_many(texts: List[str], which: str = "valid") -> List[List[float]]:
    coros = [(_embed_valid(t) if which == "valid" else _embed_error(t)) for t in texts]
    return await asyncio.gather(*coros)


# ──────────────────────────────────────────────────────────────────────────────
# Upsert schema docs (manifest) into Qdrant
# ──────────────────────────────────────────────────────────────────────────────

async def upsert_schema_docs(
    qdrant: QdrantClient,
    tables: List[Dict[str, Any]],
    include_error_vec: bool = True,
) -> int:
    """
    Each item in `tables` should at least include:
      {
        "table_schema": "public",
        "table_name": "orders",
        "text": "Readable manifest for this table",
        # optional:
        "columns": ["id","user_id","amount","created_at"],
        "id": "uuid"
      }
    """
    if not tables:
        return 0

    texts = [str(t.get("text") or "") for t in tables]
    valid_vecs = await _embed_many(texts, which="valid")
    error_vecs: List[Optional[List[float]]] = []
    if include_error_vec and getattr(settings, "ERROR_NAME", None):
        error_vecs = await _embed_many(texts, which="error")
    else:
        error_vecs = [None] * len(texts)

    points: List[PointStruct] = []
    for i, t in enumerate(tables):
        pid = t.get("id") or str(uuid.uuid4())
        schema = (t.get("table_schema") or t.get("schema") or "public")
        name = t.get("table_name") or t.get("name")

        payload: Dict[str, Any] = {
            "doc_type": "schema/table",
            "table_schema": schema,
            "table_name": name,
            "text": texts[i],
        }
        cols = t.get("columns")
        if isinstance(cols, list) and cols:
            payload["columns"] = [str(c) for c in cols]

        vectors: Dict[str, List[float]] = {settings.VALID_NAME: valid_vecs[i]}
        if include_error_vec and getattr(settings, "ERROR_NAME", None) and error_vecs[i] is not None:
            vectors[settings.ERROR_NAME] = error_vecs[i]  # type: ignore[assignment]

        points.append(PointStruct(id=pid, vector=vectors, payload=payload))

    qdrant.upsert(collection_name=settings.SCHEMA_COLLECTION, points=points)
    return len(points)


# ──────────────────────────────────────────────────────────────────────────────
# Query related schema (RAG) for a question
# ──────────────────────────────────────────────────────────────────────────────

async def query_related_schema(
    qdrant: QdrantClient,
    question: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Embed the question (await!) and search named vector in the schema collection.
    Returns lightweight hits for prompt stuffing.
    """
    vec = await _embed_valid(question)  # ← the key fix: await the embedder
    res = qdrant.search(
        collection_name=settings.SCHEMA_COLLECTION,
        query_vector=(settings.VALID_NAME, vec),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    out: List[Dict[str, Any]] = []
    for r in res or []:
        pl = r.payload or {}
        out.append({
            "id": r.id,
            "score": float(r.score),
            "text": pl.get("text", ""),
            "table_schema": pl.get("table_schema"),
            "table_name": pl.get("table_name"),
        })
    return out
from fastapi import APIRouter, HTTPException, Depends
from psycopg import Connection
from qdrant_client import QdrantClient
from ..config import settings
from ..dependencies import get_pg_conn, get_qdrant
from ..clients.llm_compat import schema_embed_one
from ..semantic.provider import get_mdl, get_context
from ..semantic.trainning import upsert_schema_docs, query_related_schema
from ..services.embeddings import embed_valid
from ..semantic.schema_introspect import fetch_schema

import os

router = APIRouter(prefix="/semantic", tags=["semantic"])

@router.get("")
def read_semantic():
    return {"mode": os.getenv("SEMANTIC_MODE","hybrid"), "mdl": get_mdl()}

@router.get("/context")
def read_context():
    from ..semantic.provider import get_context
    return {"context": get_context()}

@router.post("/reload")
async def reload_semantic(
    include_error_vec: bool = False,
    conn: Connection = Depends(get_pg_conn),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    tables = fetch_schema(conn)
    await upsert_schema_docs(qdrant, tables, include_error_vec=include_error_vec)
    return {"ok": True, "tables_indexed": len(tables)}

@router.get("/related")
def related(question: str, k: int = 8, qdrant: QdrantClient = Depends(get_qdrant)):
    hits = query_related_schema(qdrant, question, top_k=k)
    return [
        {
            "schema": f"{h['table_schema']}.{h['table_name']}",
            "text": h["text"],
        } for h in hits
    ]

@router.get("/schema/stats")
def schema_stats(
    qdrant: QdrantClient = Depends(get_qdrant),
    conn: Connection = Depends(get_pg_conn),
):
    name = settings.SCHEMA_COLLECTION

    col = qdrant.get_collection(collection_name=name)
    total = qdrant.count(collection_name=name, exact=True).count

    # dict_row → read by key
    row = conn.execute("""
        SELECT COUNT(*)::bigint AS n
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema');
    """).fetchone()
    expected = row["n"] if row is not None else 0

    pts, _ = qdrant.scroll(
        collection_name=name,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    sample = [{
        "schema": f"{(p.payload or {}).get('table_schema')}.{(p.payload or {}).get('table_name')}",
        "chars": len(((p.payload or {}).get('text') or "")),
        "preview": (((p.payload or {}).get('text') or "")[:180]),
    } for p in pts]

    return {
        "collection": name,
        "points_total": total,
        "expected_tables": expected,
        "counts_match": (total == expected),
        "sample": sample,
    }

@router.get("/schema/search")
async def schema_search(
    question: str,
    k: int = 5,
    qdrant: QdrantClient = Depends(get_qdrant),
):
    # await the async embedder
    vec = await schema_embed_one(question, model=settings.VALID_EMBED_MODEL)

    if not isinstance(vec, list):
        raise HTTPException(status_code=500, detail="Embedding failed: not a list")
    if len(vec) != settings.VALID_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dim mismatch: expected {settings.VALID_DIM}, got {len(vec)}",
        )

    res = qdrant.search(
        collection_name=settings.SCHEMA_COLLECTION,
        query_vector=(settings.VALID_NAME, vec),  # e.g., ("valid_vec", vec)
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    return [{
        "score": float(r.score),
        "schema": f"{(r.payload or {}).get('table_schema')}.{(r.payload or {}).get('table_name')}",
        "preview": (((r.payload or {}).get('text') or "")[:220]),
    } for r in res]
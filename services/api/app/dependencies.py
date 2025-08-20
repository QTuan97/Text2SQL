from __future__ import annotations
from typing import Generator, Optional, List

from fastapi import FastAPI
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# --- Qdrant imports: support both package layouts ---
try:
    from qdrant_client.models import Distance, VectorParams
except Exception:  # pragma: no cover
    from qdrant_client.http.models import Distance, VectorParams

from .config import settings
from .clients.qdrant import qdrant_client


_PG_POOL: Optional[ConnectionPool] = None

def _get_pg_pool() -> ConnectionPool:
    global _PG_POOL
    if _PG_POOL is None:
        _PG_POOL = ConnectionPool(
            conninfo=settings.POSTGRES_URL,
            min_size=getattr(settings, "PG_POOL_MIN_SIZE", 1),
            max_size=getattr(settings, "PG_POOL_MAX_SIZE", 10),
            kwargs={"autocommit": getattr(settings, "PG_AUTOCOMMIT", True)},
        )
    return _PG_POOL

def get_pg_conn() -> Generator[Connection, None, None]:
    """
    FastAPI dependency — yields a psycopg3 Connection with dict_row factory.
    Usage: def route(conn: Connection = Depends(get_pg_conn))
    """
    pool = _get_pg_pool()
    with pool.connection() as conn:  # type: ignore[assignment]
        conn.row_factory = dict_row
        yield conn


# Qdrant dependency (returns your existing client)

def get_qdrant():
    """
    FastAPI dependency — returns your initialized qdrant_client.
    Usage: def route(qdrant = Depends(get_qdrant))
    """
    return qdrant_client


# Qdrant collection ensures (named vectors)

def _named_vectors_config():
    # Expect these in settings: VALID_NAME / VALID_DIM, ERROR_NAME / ERROR_DIM
    return {
        settings.VALID_NAME: VectorParams(size=settings.VALID_DIM, distance=Distance.COSINE),
        settings.ERROR_NAME: VectorParams(size=settings.ERROR_DIM, distance=Distance.COSINE),
    }

def _ensure_named_collection(collection_name: str) -> None:
    """
    Create collection with named vectors if missing.
    Note: Changing vector sizes/distances requires manual recreate in Qdrant.
    """
    cols = qdrant_client.get_collections().collections or []
    names: List[str] = [c.name for c in cols]
    if collection_name in names:
        # Exists → assume correct config; avoid destructive changes here.
        return

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=_named_vectors_config(),
    )

def ensure_qdrant_collections() -> None:
    """
    Ensure all collections exist with named vectors.
    """
    _ensure_named_collection(settings.QDRANT_COLLECTION)
    _ensure_named_collection(settings.LESSONS_COLLECTION)
    _ensure_named_collection(settings.SCHEMA_COLLECTION)
    _ensure_named_collection(settings.SEMANTIC_COLLECTION)


def init_app(app: FastAPI) -> None:
    @app.on_event("startup")
    def _startup() -> None:
        # Warm connections so first request isn’t slow
        _get_pg_pool()
        # Your qdrant_client is already constructed; just ensure collections
        ensure_qdrant_collections()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        global _PG_POOL
        if _PG_POOL is not None:
            _PG_POOL.close()
            _PG_POOL = None
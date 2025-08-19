from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..config import settings

# ---------- robust Qdrant client getter ----------
def _qc() -> QdrantClient:
    # support either a singleton `qdrant_client` or a factory `get_qdrant_client`
    try:
        from ..dependencies import qdrant_client as _Q
        return _Q if hasattr(_Q, "upsert") else _Q()
    except Exception:
        from ..dependencies import get_qdrant_client as _get_q
        return _get_q()

# ---------- ensure collection once (named vectors) ----------
def _ensure_lessons_collection() -> None:
    qc = _qc()
    cols = qc.get_collections().collections
    if settings.LESSONS_COLLECTION not in [c.name for c in cols]:
        qc.create_collection(
            collection_name=settings.LESSONS_COLLECTION,
            vectors_config={
                settings.VALID_NAME: VectorParams(size=settings.VALID_DIM, distance=Distance.COSINE),
                settings.ERROR_NAME: VectorParams(size=settings.ERROR_DIM, distance=Distance.COSINE),
            },
        )
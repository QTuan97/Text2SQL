from __future__ import annotations
from qdrant_client import QdrantClient
from ..config import settings

qdrant_client = QdrantClient(url=settings.QDRANT_URL)
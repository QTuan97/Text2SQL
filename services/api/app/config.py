from __future__ import annotations
import os
from typing import List

class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    POSTGRES_URL: str | None = os.getenv("POSTGRES_URL")

    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "pipeline_events")
    VALID_NAME: str = os.getenv("VALID_NAME", "valid_vec")
    ERROR_NAME: str = os.getenv("ERROR_NAME", "error_vec")
    VALID_DIM: int = int(os.getenv("VALID_DIM", "768"))
    ERROR_DIM: int = int(os.getenv("ERROR_DIM", "384"))

    GEN_MODEL: str = os.getenv("GEN_MODEL", "llama3.2:3b-instruct")
    VALID_EMBED_MODEL: str = os.getenv("VALID_EMBED_MODEL", "nomic-embed-text")
    ERROR_EMBED_MODEL: str = os.getenv("ERROR_EMBED_MODEL", "all-minilm")

    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",")

settings = Settings()
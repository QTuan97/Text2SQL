from __future__ import annotations
import os
from typing import List

def _truthy(s: str | None, default: bool) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}

class Settings:
    CACHE_FIRST: bool = os.getenv("CACHE_FIRST", "true").lower() == "true"
    CACHE_MIN_SIM: float = float(os.getenv("CACHE_MIN_SIM", "0.92"))

    # Core endpoints
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    POSTGRES_URL: str | None = os.getenv("POSTGRES_URL")

    # Named vectors
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "pipeline_events")
    VALID_NAME: str = os.getenv("VALID_NAME", "valid_vec")
    ERROR_NAME: str = os.getenv("ERROR_NAME", "error_vec")
    VALID_DIM: int = int(os.getenv("VALID_DIM", "768"))
    ERROR_DIM: int = int(os.getenv("ERROR_DIM", "384"))

    # Models
    GEN_MODEL: str = os.getenv("GEN_MODEL", "llama3.2:3b-instruct")
    VALID_EMBED_MODEL: str = os.getenv("VALID_EMBED_MODEL", "nomic-embed-text")
    ERROR_EMBED_MODEL: str = os.getenv("ERROR_EMBED_MODEL", "all-minilm")

    # Model settings
    GEN_TEMPERATURE = 0.0
    GEN_TOP_P = 1.0
    GEN_TOP_K = 0
    GEN_REPEAT_PENALTY = 1.0
    GEN_MAX_TOKENS = 512
    GEN_SEED = 42

    # Collections
    SCHEMA_COLLECTION: str = os.getenv("SCHEMA_COLLECTION", "db_schema")
    LESSONS_COLLECTION: str = os.getenv("LESSONS_COLLECTION", "sql_lessons")
    SEMANTIC_COLLECTION: str = os.getenv("SEMANTIC_COLLECTION", "semantic_schema")

    # Learning behavior
    LEARN_TO_QDRANT_ONLY: bool = _truthy(os.getenv("LEARN_TO_QDRANT_ONLY", "true"), True)
    LEARN_ASYNC: bool = _truthy(os.getenv("LEARN_ASYNC", "true"), True)

    # CORS
    CORS_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",") if o.strip()]

    # Native Ollama HTTP base for embeddings
    OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE") or OLLAMA_BASE_URL

settings = Settings()
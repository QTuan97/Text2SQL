from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class IndexIn(BaseModel):
    id: int | str | None = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchIn(BaseModel):
    query: str
    field: Literal["valid_vec", "error_vec"] = "valid_vec"
    limit: int = 5

class SearchByVectorIn(BaseModel):
    vector: List[float]
    field: Literal["valid_vec", "error_vec"]
    limit: int = 5

class AskIn(BaseModel):
    question: str
    top_k: int = 3
    min_score: float = 0.2
    rerank: bool = False
    max_context_chars: int = 3000

class NL2SQLIn(BaseModel):
    question: str
    limit: int = 50
    execute: bool = False
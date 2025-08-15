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
    limit: Optional[int] = None
    execute: bool = False

class NL2SQLOut(BaseModel):
    sql: str
    rows: Optional[List[Dict[str, Any]]] = None
    rowcount: int = 0
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

class LearnEvent(BaseModel):
    question: str
    sql: str
    executed: bool = False
    rowcount: int = 0
    error: Optional[str] = None
    tables: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source: Literal["text2sql", "manual", "import"] = "text2sql"

class LessonIn(BaseModel):
    id: int | str | None = None
    title: Optional[str] = None
    sql: str
    text: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    created_at: Optional[int] = None

class LessonsSearchIn(BaseModel):
    query: str
    k: int = 10
    min_sim: float = 0.30
    topic: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class LessonHit(BaseModel):
    id: str | int
    question: Optional[str] = None
    sql: Optional[str] = None
    tables: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    topic: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[int] = None
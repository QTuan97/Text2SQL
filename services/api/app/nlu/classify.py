from __future__ import annotations
import re
from typing import Tuple, List

from qdrant_client import QdrantClient
from ..config import settings
from ..dependencies import get_qdrant
from ..clients.llm_compat  import schema_embed_one
from ..semantic.schema_cache import known_schema

# Simple pattern bucket
_GENERAL_RE = re.compile(
    r"(?:^|\b)(hi|hello|hey|help|how (?:do|can) i|what can you do|"
    r"who are you|about (?:this|you)|docs?|documentation|examples?)\b",
    re.I,
)
_MISLEADING_PATTERNS = [
    r"\btell me a joke\b", r"\bweather\b", r"\bstory\b", r"\bpoem\b",
    r"\btranslate\b", r"\bnews\b", r"\bstock price\b", r"\b(image|draw|picture)\b",
]
# Words that usually imply a DB/analytics intent
_SQLISH_RE = re.compile(
    r"\b(count|sum|avg|min|max|total|revenue|sales|orders?|users?|"
    r"top\s+\d+|list|show|group\s+by|order\s+by|where|between|last\s+\d+\s+(days|weeks|months)|"
    r"this\s+(month|week|year)|today|yesterday)\b",
    re.I,
)

# Tunables (can live in env)
INTENT_MIN_SIM = float(getattr(settings, "INTENT_MIN_SIM", 0.12))      # accept ≥ this
BORDERLINE_SIM = float(getattr(settings, "INTENT_BORDERLINE", 0.09))   # allow if SQL-ish

# Main classifier
async def classify_question(q: str) -> Tuple[str, str]:
    """
    Returns (kind, reason) in {'general','text2sql','misleading'}.
    Policy:
      1) 'general' if smalltalk/meta.
      2) Query Qdrant on schema docs; if top score >= INTENT_MIN_SIM → text2sql.
      3) If borderline and SQL-ish → text2sql.
      4) Else misleading.
    """
    s = (q or "").strip()
    if not s:
        return "misleading", "empty"

    if _GENERAL_RE.search(s):
        return "general", "general-smalltalk"

    if any(re.search(p, s, re.I) for p in _MISLEADING_PATTERNS):
        return "misleading", "misleading-pattern"

    # Primary: schema similarity via Qdrant (RAG)
    top = 0.0
    try:
        qc: QdrantClient = get_qdrant()
        vec = await schema_embed_one(s, model=settings.VALID_EMBED_MODEL)
        res = qc.search(
            collection_name=settings.SCHEMA_COLLECTION,
            query_vector=(settings.VALID_NAME, vec),
            limit=3,
            with_payload=False,
            with_vectors=False,
        )
        top = float(res[0].score) if res else 0.0
    except Exception:
        # On retrieval/embedding failure we fall back to heuristics below
        top = 0.0

    if top >= INTENT_MIN_SIM:
        return "text2sql", f"schema-match {top:.3f}"

    if top >= BORDERLINE_SIM and _SQLISH_RE.search(s):
        return "text2sql", f"borderline+sqlish {top:.3f}"

    # Last fallback: if user explicitly mentions table names, still allow
    try:
        names, _ = known_schema()
        schema_terms = {n.lower() for n in names}
        has_schema_term = any((" " + t + " ") in (" " + s.lower() + " ") for t in schema_terms)
        if has_schema_term and _SQLISH_RE.search(s):
            return "text2sql", "schema-term+sqlish-fallback"
    except Exception:
        pass

    return "misleading", f"low-schema {top:.3f}<{INTENT_MIN_SIM:.2f}"

# Help text for 'general' responses
def build_general_answer() -> str:
    names, _ = known_schema()
    if not names:
        return (
            "This workspace converts questions → SQL over **your database**.\n"
            "Upload a schema via `PUT /schema`, then try:\n"
            "• total sales last 7 days\n• top 10 users by revenue\n• orders by status this month"
        )
    tbls = ", ".join(names)
    hints: List[str] = []
    if "orders" in names and "users" in names:
        hints = [
            "total revenue last 7 days",
            "top 10 users by total order amount",
            "orders by status this month",
        ]
    else:
        t0 = names[0]
        hints = [f"show 10 rows from {t0}", f"count rows in {t0}", f"{t0} by day, last 30 days"]
    return (
        "This workspace converts questions to SQL over your DB.\n"
        f"Available tables: {tbls}.\nTry:\n• " + "\n• ".join(hints)
    )
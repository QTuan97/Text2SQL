# app/services/classify.py
from __future__ import annotations

import re
from typing import Tuple, List

from qdrant_client import QdrantClient

from ..config import settings
from ..dependencies import get_qdrant
from ..clients.llm_compat import embed_one
from ..semantic.schema_cache import known_schema


# ───────────────────────── patterns ─────────────────────────

_GENERAL_RE = re.compile(
    r"(?:^|\b)(hi|hello|hey|help|how (?:do|can) i|what can you do|"
    r"who are you|about (?:this|you)|docs?|documentation|examples?)\b",
    re.I,
)

_MISLEADING_PATTERNS = [
    r"\btell me a joke\b",
    r"\bweather\b",
    r"\bstory\b",
    r"\bpoem\b",
    r"\btranslate\b",
    r"\bnews\b",
    r"\bstock price\b",
    r"\b(image|draw|picture)\b",
]

_SQLISH_RE = re.compile(
    r"\b(count|sum|avg|min|max|total|revenue|sales|orders?|users?|"
    r"top\s+\d+|list|show|group\s+by|order\s+by|where|between|"
    r"last\s+\d+\s+(days|weeks|months)|this\s+(month|week|year)|today|yesterday)\b",
    re.I,
)

# Tunables
INTENT_MIN_SIM = float(getattr(settings, "INTENT_MIN_SIM", 0.12))
BORDERLINE_SIM = float(getattr(settings, "INTENT_BORDERLINE", 0.09))


# ───────────────────────── classifier ─────────────────────────

async def classify_question(q: str) -> Tuple[str, str]:
    """
    Returns (kind, reason) in {'general','text2sql','misleading'}.

    Policy:
      1) 'general' for smalltalk/meta.
      2) Qdrant schema similarity (awaited embedding). If top ≥ INTENT_MIN_SIM → text2sql.
      3) If borderline and SQL-ish → text2sql.
      4) If the question mentions a known table AND is SQL-ish → text2sql.
      5) Otherwise misleading.
    """
    s = (q or "").strip()
    if not s:
        return "misleading", "empty"

    if _GENERAL_RE.search(s):
        return "general", "general-smalltalk"

    if any(re.search(p, s, re.I) for p in _MISLEADING_PATTERNS):
        return "misleading", "misleading-pattern"

    # Primary gate: schema similarity via Qdrant (RAG)
    top = 0.0
    try:
        qc: QdrantClient = get_qdrant()
        vec = await embed_one(s, model=getattr(settings, "VALID_EMBED_MODEL", None))  # ← awaited
        res = qc.search(
            collection_name=settings.SCHEMA_COLLECTION,
            query_vector=(settings.VALID_NAME, vec),
            limit=3,
            with_payload=False,
            with_vectors=False,
        )
        top = float(res[0].score) if res else 0.0
    except Exception:
        # On any retrieval/embedding failure, fall back to heuristics below.
        top = 0.0

    if top >= INTENT_MIN_SIM:
        return "text2sql", f"schema-match {top:.3f}"

    if top >= BORDERLINE_SIM and _SQLISH_RE.search(s):
        return "text2sql", f"borderline+sqlish {top:.3f}"

    # Fallback: explicit mention of a known table + SQL-ish phrasing
    try:
        names, _ = known_schema()
        terms = {n.lower() for n in names}
        # match by base table as well (schema.table → table)
        terms |= {t.split(".")[-1] for t in terms}
        hay = f" {s.lower()} "
        mentions_schema = any(f" {t} " in hay for t in terms if t)
        if mentions_schema and _SQLISH_RE.search(s):
            return "text2sql", "schema-term+sqlish-fallback"
    except Exception:
        pass

    return "misleading", f"low-schema {top:.3f}<{INTENT_MIN_SIM:.2f}"


# ───────────────────────── helper answer for 'general' ─────────────────────────

def build_general_answer() -> str:
    names, _ = known_schema()
    if not names:
        return (
            "This workspace converts questions → SQL over **your database**.\n"
            "Load a schema via /semantic/reload, then try:\n"
            "• total sales last 7 days\n"
            "• top 10 users by revenue\n"
            "• orders by status this month"
        )
    tbls = ", ".join(names)
    hints: List[str]
    lower = {n.split(".")[-1] for n in names}
    if {"orders", "users"} <= lower:
        hints = [
            "total revenue last 7 days",
            "top 10 users by total order amount",
            "orders by status this month",
        ]
    else:
        first = names[0]
        base = first.split(".")[-1]
        hints = [f"show 10 rows from {base}", f"count rows in {base}", f"{base} by day, last 30 days"]
    return (
        "This workspace converts questions to SQL over your DB.\n"
        f"Available tables: {tbls}.\nTry:\n• " + "\n• ".join(hints)
    )
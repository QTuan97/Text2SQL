from __future__ import annotations
import re
from typing import Tuple, List
from ..semantic.schema_cache import known_schema

_GENERAL_PATTERNS = [
    r"\bwhat (is|does) (this|the) (db|database|app|tool)\b",
    r"\bwhat can you do\b",
    r"\bhow (do i|to) (use|ask)\b",
    r"\bhelp\b",
    r"\b(show|list) (tables|schema)\b",
    r"\bcapabilit(y|ies)\b",
    r"\bwho are you\b",
]
_MISLEADING_PATTERNS = [
    r"\btell me a joke\b",
    r"\bweather\b",
    r"\bstory\b",
    r"\bpoem\b",
    r"\btranslate\b",
    r"\bnews\b",
    r"\bstock price\b",
    r"\bimage\b|\bdraw\b|\bpicture\b",
]
_SQL_INTENT = r"\b(list|show|count|sum|avg|min|max|top|latest|recent|first|last|where|group|order|by|rank|total|revenue|sales)\b"

def classify_question(q: str) -> Tuple[str, str]:
    """
    Returns (kind, reason) in {'general','text2sql','misleading'}.
    """
    s = (q or "").lower().strip()
    if any(re.search(p, s) for p in _GENERAL_PATTERNS):
        return "general", "general-pattern"
    if any(re.search(p, s) for p in _MISLEADING_PATTERNS):
        return "misleading", "misleading-pattern"

    names, _ = known_schema()
    schema_terms = {n.lower() for n in names}
    has_schema_term = any((" " + t + " ") in (" " + s + " ") for t in schema_terms)
    has_sql_intent = bool(re.search(_SQL_INTENT, s))
    if has_sql_intent or has_schema_term:
        return "text2sql", "sql-intent-or-schema-term"

    return "misleading", "no-schema-or-intent"

def build_general_answer() -> str:
    names, _ = known_schema()
    if not names:
        return ("This workspace converts questions → SQL over **your database**.\n"
                "Upload a schema via `PUT /schema`, then try:\n"
                "• total sales last 7 days\n• top 10 users by revenue\n• orders by status this month")
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
    return (f"This workspace converts questions to SQL over your DB.\n"
            f"Available tables: {tbls}.\nTry:\n• " + "\n• ".join(hints))

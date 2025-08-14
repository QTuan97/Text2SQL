from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import os, math, asyncio

from ..config import settings
from ..clients import ollama
from .provider import get_mdl

INTENT_MIN_SIM = float(os.getenv("INTENT_MIN_SIM", "0.60"))     # tweak if needed
POS_EXTRA = [p.strip() for p in os.getenv("INTENT_POSITIVE_EXTRA", "").split("|") if p.strip()]

def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)) or 1.0
    db = math.sqrt(sum(y*y for y in b)) or 1.0
    return num / (da * db)

async def _embed_batch(texts: List[str]) -> List[List[float]]:
    # parallelize using your async ollama client
    tasks = [ollama.embed(settings.VALID_EMBED_MODEL, t) for t in texts]
    return await asyncio.gather(*tasks)

def _build_pos_canon(mdl: Dict[str, Any]) -> List[str]:
    phrases: List[str] = []

    # 1) metrics-driven patterns from MDL
    for m in (mdl.get("metrics") or []):
        name = m.get("name")
        if not name:
            continue
        phrases += [
            f"{name} by month",
            f"{name} last 7 days",
            f"top users by {name}",
            f"{name} by city",
        ]

    # 2) examples on entities (if present)
    for e in (mdl.get("entities") or []):
        for ex in (e.get("examples") or []):
            if ex:
                phrases.append(ex)

    # 3) a few safe defaults
    phrases += [
        "revenue by month",
        "top customers by revenue last 30 days",
        "orders by city last 7 days",
        "count orders last 24 hours",
    ]

    # 4) user-supplied extras via env
    phrases += POS_EXTRA

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for p in phrases:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out

async def gate(question: str, mdl: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Embedding-based intent gate.
    Returns (allow, info_dict). info_dict includes reason/similarity/suggestions.
    """
    mdl = mdl or get_mdl()
    canon = _build_pos_canon(mdl)
    if not canon:
        # No canon => allow (or flip to False if you prefer ultra-strict)
        return True, {"reason": "no_canon", "similarity": 1.0, "suggestions": []}

    vecs = await _embed_batch([question] + canon)
    qvec, pvecs = vecs[0], vecs[1:]
    sims: List[Tuple[str, float]] = [(canon[i], _cos(qvec, pvecs[i])) for i in range(len(canon))]
    sims.sort(key=lambda x: x[1], reverse=True)

    best_sim = sims[0][1]
    suggestions = [p for p, _ in sims[:3]]

    if best_sim >= INTENT_MIN_SIM:
        return True, {"reason": "ok", "similarity": float(best_sim), "suggestions": suggestions}

    return False, {"reason": "low_similarity", "similarity": float(best_sim), "suggestions": suggestions}
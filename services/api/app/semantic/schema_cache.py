# app/services/semantic/schema_cache.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import time
import re

from qdrant_client import QdrantClient

from app.config import settings
from app.dependencies import get_qdrant

_CACHE: Optional[Tuple[List[str], Optional[Dict[str, List[str]]]]] = None
_CACHE_TTL_SEC = int(getattr(settings, "SCHEMA_CACHE_TTL", 300))
_CACHE_TS = 0.0


def bust_known_schema_cache() -> None:
    """Force the next call to reload from Qdrant."""
    global _CACHE, _CACHE_TS
    _CACHE = None
    _CACHE_TS = 0.0


# Qdrant scroll helper (API shape differs across client versions)
def _scroll_schema_points(
    qc: QdrantClient,
    collection: str,
    limit: int = 256,
):
    """
    Yield batches of points (payload only) for doc_type == 'schema/table'.
    Works across qdrant-client versions that use filter=... vs scroll_filter=...
    """
    offset = None
    flt = {"must": [{"key": "doc_type", "match": {"value": "schema/table"}}]}
    while True:
        try:
            points, offset = qc.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                scroll_filter=flt,  # newer clients
            )
        except TypeError:
            points, offset = qc.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                filter=flt,  # older clients
            )
        if not points:
            break
        yield points
        if offset is None:
            break


# Optional column extraction from free-text payload
_COL_LINE = re.compile(r"^\s*[-*]\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(|:)?", re.I)
_COL_SECTION_START = re.compile(r"^\s*columns\s*:?\s*$", re.I)

def _extract_columns_from_text(text: str) -> List[str]:
    """
    Best-effort: parse a 'Columns:' section (bullet list).
    Returns a list of column names (lowercased); empty if not found.
    """
    if not text:
        return []
    cols: List[str] = []
    in_cols = False
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not in_cols and _COL_SECTION_START.match(line):
            in_cols = True
            continue
        if in_cols:
            m = _COL_LINE.match(line)
            if m:
                cols.append(m.group(1).lower())
            elif line.strip() == "" or line.strip().startswith("#"):
                # keep scanning across blank/comment lines
                continue
            else:
                # left the columns section
                if cols:
                    break
    # de-dup preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# Loader from Qdrant
def _load_from_qdrant(qc: QdrantClient, collection: str) -> Tuple[List[str], Optional[Dict[str, List[str]]]]:
    names: List[str] = []
    cols_map: Dict[str, List[str]] = {}
    for batch in _scroll_schema_points(qc, collection, limit=256):
        for p in batch:
            pl = p.payload or {}
            schema = str(pl.get("table_schema") or "").strip()
            table = str(pl.get("table_name") or "").strip()
            if not table:
                continue
            full = f"{schema}.{table}" if schema else table
            key = full.lower()
            names.append(key)

            # Prefer explicit 'columns' if present; else try parse from 'text'
            cols = pl.get("columns")
            if isinstance(cols, list) and cols:
                cols_map[key] = [str(c).lower() for c in cols if str(c).strip()]
            else:
                parsed = _extract_columns_from_text(str(pl.get("text") or ""))
                if parsed:
                    cols_map[key] = parsed

    # de-dup names, keep stable order
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    return uniq, (cols_map or None)


# Public API
def known_schema(ttl_sec: Optional[int] = None) -> Tuple[List[str], Optional[Dict[str, List[str]]]]:
    """
    Returns (table_names, columns_map) from Qdrant:
      - table_names: lowercased, possibly schema-qualified (e.g., 'public.orders')
      - columns_map: optional dict[full_table_name] -> list[column_name]
    Uses an in-process TTL cache to avoid repeated Qdrant scans.
    """
    global _CACHE, _CACHE_TS
    ttl = int(ttl_sec if ttl_sec is not None else _CACHE_TTL_SEC)
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < ttl:
        return _CACHE

    qc = get_qdrant()
    collection = getattr(settings, "SCHEMA_COLLECTION", "semantic_schema")
    data = _load_from_qdrant(qc, collection)
    _CACHE = data
    _CACHE_TS = now
    return data


def get_schema_text(max_tables: int = 500) -> str:
    """
    Build a compact text snapshot for prompts (fallback only).
    Format: 'public.orders(id, user_id, amount, created_at)\\npublic.users(id, ...)'
    Uses columns_map when available; otherwise lists just table names.
    """
    names, cols_map = known_schema()
    lines: List[str] = []
    for full in names[:max_tables]:
        cols = cols_map.get(full) if isinstance(cols_map, dict) else None
        if cols:
            lines.append(f"{full}({', '.join(cols)})")
        else:
            lines.append(full)
    return "\n".join(lines)
# app/semantic/schema_cache.py
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sqlglot import parse_one, exp

from ..config import settings
from ..dependencies import qdrant_client as _QC
from ..clients.llm_compat import schema_embed_one  # must return a list[float]

VALID_NAME, ERROR_NAME = settings.VALID_NAME, settings.ERROR_NAME
VALID_DIM,  ERROR_DIM  = settings.VALID_DIM, settings.ERROR_DIM

SCHEMA_POINT_UUID = uuid.uuid5(uuid.NAMESPACE_URL, "wren-ai://schema/current")
SCHEMA_POINT_ID = str(SCHEMA_POINT_UUID)


# Utilities
def _flatten_tables(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Input format:
    {
      "tables":[{"name":"users","columns":[ "id","email" ]}, ...]
    }
    Columns can be strings or dicts with {"name": "..."}.
    """
    out: List[Dict[str, Any]] = []
    for t in snapshot.get("tables", []):
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        cols_raw = t.get("columns") or []
        cols: List[str] = []
        for c in cols_raw:
            if isinstance(c, str):
                cols.append(c)
            elif isinstance(c, dict) and c.get("name"):
                cols.append(str(c["name"]))
        out.append({"name": name, "columns": sorted(set(cols))})
    return out

def _schema_text(snapshot: Dict[str, Any]) -> str:
    """
    Compact, LLM-friendly description (no DB calls).
    """
    tables = _flatten_tables(snapshot)
    lines = ["# Cached Schema Snapshot"]
    for t in tables:
        cols = ", ".join(t["columns"]) if t["columns"] else "(no columns listed)"
        lines.append(f"- {t['name']}: {cols}")
    return "\n".join(lines)

def ensure_schema_collection(qc: QdrantClient) -> None:
    """
    Create the schema collection with named vectors if it doesn't exist.
    """
    try:
        existing = {c.name for c in qc.get_collections().collections}
    except Exception:
        existing = set()
    if settings.SCHEMA_COLLECTION in existing:
        return
    qc.create_collection(
        collection_name=settings.SCHEMA_COLLECTION,
        vectors_config={
            VALID_NAME: VectorParams(size=VALID_DIM, distance=Distance.COSINE),
            ERROR_NAME: VectorParams(size=ERROR_DIM, distance=Distance.COSINE),
        },
    )


# Public API
async def upsert_schema_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store the whole schema snapshot into Qdrant as a single point (deterministic id).
    """
    qc: QdrantClient = _QC
    ensure_schema_collection(qc)

    text = _schema_text(snapshot)

    # Main embedding (required)
    v_main = schema_embed_one(text, model=settings.VALID_EMBED_MODEL)
    if not isinstance(v_main, list) or len(v_main) != VALID_DIM:
        raise ValueError(f"{VALID_NAME} size mismatch: expected {VALID_DIM}, got {len(v_main) if isinstance(v_main, list) else 'N/A'}")

    # Aux embedding (best-effort)
    try:
        v_aux = schema_embed_one(text, model=settings.ERROR_EMBED_MODEL)
        if not isinstance(v_aux, list) or len(v_aux) != ERROR_DIM:
            raise ValueError("aux-embed-size-mismatch")
    except Exception:
        v_aux = [0.0] * ERROR_DIM

    payload = {
        "kind": "schema",
        "version": int(time.time()),
        "tables": _flatten_tables(snapshot),
        "text": text,
        "updated_at": int(time.time()),
    }

    qc.upsert(
        collection_name=settings.SCHEMA_COLLECTION,
        points=[PointStruct(
            id=SCHEMA_POINT_ID,
            vector={VALID_NAME: v_main, ERROR_NAME: v_aux},
            payload=payload,
        )],
    )
    return {"version": payload["version"], "tables": len(payload["tables"])}

def _load_snapshot() -> Optional[Dict[str, Any]]:
    qc: QdrantClient = _QC
    try:
        recs = qc.retrieve(
            collection_name=settings.SCHEMA_COLLECTION,
            ids=[SCHEMA_POINT_ID],
            with_payload=True,
        )
    except Exception:
        return None
    if not recs:
        return None
    return recs[0].payload or {}

def get_schema_text() -> str:
    snap = _load_snapshot() or {}
    return snap.get("text", "")

def known_schema() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns (table_names, columns_by_table) from the cached snapshot.
    """
    snap = _load_snapshot() or {}
    tables = snap.get("tables", [])
    names = [t.get("name") for t in tables if t.get("name")]
    cols = {t["name"]: list(t.get("columns") or []) for t in tables if t.get("name")}
    return names, cols

def ensure_known_tables_cached(sql: str) -> None:
    """
    Validate that all Table nodes in SQL exist in the cached schema (no Postgres calls).
    """
    names, _ = known_schema()
    known = {n.lower() for n in names}

    try:
        tree = parse_one(sql, read="postgres")
    except Exception as e:
        raise ValueError(f"SQL parse failed: {e}")

    unknown: List[str] = []
    for t in tree.find_all(exp.Table):
        # Try several shapes to get the table name robustly
        tname = None
        if getattr(t, "name", None):
            tname = t.name
        elif getattr(t, "this", None) is not None:
            # t.this could be an Identifier with .name or a dotted path
            tname = getattr(getattr(t.this, "this", None), "name", None) or getattr(t.this, "name", None)
        if tname and tname.lower() not in known:
            unknown.append(tname)

    if unknown:
        raise ValueError(f"Unknown table(s) in SQL (cached schema): {', '.join(sorted(set(unknown)))}")
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import os, time, uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlglot import parse_one, exp

from ..config import settings
from ..dependencies import qdrant_client as _QC
from ..clients.llm_compat import embed_one

VALID_NAME, ERROR_NAME = settings.VALID_NAME, settings.ERROR_NAME
VALID_DIM,  ERROR_DIM  = settings.VALID_DIM, settings.ERROR_DIM

SCHEMA_POINT_UUID = uuid.uuid5(uuid.NAMESPACE_URL, "wren-ai://schema/current")
SCHEMA_POINT_ID = str(SCHEMA_POINT_UUID)

def _uuid_ns() -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, "wren-ai://schema")

def _flatten_tables(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    # input format: {"tables":[{"name":"users","columns":[... or dicts ...]}, ...]}
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
    Make a compact, LLM-friendly description (no DB calls).
    """
    tables = _flatten_tables(snapshot)
    lines = ["# Live Schema Snapshot (cached)"]
    for t in tables:
        cols = ", ".join(t["columns"]) if t["columns"] else "(no columns listed)"
        lines.append(f"- {t['name']}: {cols}")
    return "\n".join(lines)

def upsert_schema_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store the whole schema snapshot into Qdrant (single point id).
    """
    qc: QdrantClient = _QC
    text = _schema_text(snapshot)
    v_main = embed_one(text, model=os.getenv("VALID_EMBED_MODEL", "nomic-embed-text"))
    if len(v_main) != VALID_DIM:
        raise ValueError(f"{VALID_NAME} size mismatch: expected {VALID_DIM}, got {len(v_main)}")
    try:
        v_aux = embed_one(text, model=os.getenv("ERROR_EMBED_MODEL", "all-minilm"))
        if len(v_aux) != ERROR_DIM:
            raise ValueError(f"{ERROR_NAME} size mismatch: expected {ERROR_DIM}, got {len(v_aux)}")
    except Exception:
        v_aux = [0.0] * ERROR_DIM

    payload = {
        "kind": "schema",
        "version": int(time.time()),
        "tables": _flatten_tables(snapshot),
        "text": text,
    }
    qc.upsert(
        collection_name=settings.SCHEMA_COLLECTION,
        points=[PointStruct(
            id=SCHEMA_POINT_ID,
            vector={VALID_NAME: v_main, ERROR_NAME: v_aux},
            payload=payload
        )]
    )

    return {"version": payload["version"], "tables": len(payload["tables"])}

def _load_snapshot() -> Optional[Dict[str, Any]]:
    qc: QdrantClient = _QC
    recs = qc.retrieve(
        collection_name=settings.SCHEMA_COLLECTION,
        ids=[SCHEMA_POINT_ID],
        with_payload=True
    )
    if not recs:
        return None
    return recs[0].payload or {}

def get_schema_text() -> str:
    snap = _load_snapshot() or {}
    return snap.get("text", "")

def known_schema() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns (table_names, columns_by_table)
    """
    snap = _load_snapshot() or {}
    tables = snap.get("tables", [])
    names = [t.get("name") for t in tables if t.get("name")]
    cols = {t["name"]: list(t.get("columns") or []) for t in tables if t.get("name")}
    return names, cols

def ensure_known_tables_cached(sql: str) -> None:
    """
    Validate that all table identifiers in SQL exist in cached schema (no Postgres).
    """
    names, _ = known_schema()
    known = set(n.lower() for n in names)
    try:
        tree = parse_one(sql, read="postgres")
    except Exception as e:
        raise ValueError(f"SQL parse failed: {e}")

    unknown: List[str] = []
    for t in tree.find_all(exp.Table):
        tname = None
        if hasattr(t, "name") and isinstance(t.name, str):
            tname = t.name
        elif getattr(t, "this", None) is not None:
            tname = getattr(t.this, "name", None)
        if tname and tname.lower() not in known:
            unknown.append(tname)

    if unknown:
        raise ValueError(f"Unknown table(s) in SQL (cached schema): {', '.join(sorted(set(unknown)))}")
from __future__ import annotations
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException

from ..semantic.schema_cache import known_schema, get_schema_text

router = APIRouter(prefix="/schema", tags=["schema"])

@router.get("/text")
def schema_text() -> Dict[str, str]:
    return {"text": get_schema_text()}

@router.get("/tables")
def schema_tables() -> Dict[str, Any]:
    names, _ = known_schema()
    return {"tables": names}

@router.get("/columns/{table}")
def schema_columns(table: str) -> Dict[str, Any]:
    req = (table or "").lower().strip()
    names, cols = known_schema()
    cols = cols or {}
    # accept "orders" or "public.orders"
    if req in cols:
        columns = cols[req]
    else:
        # match by base name (end of "schema.table")
        matches = [k for k in cols.keys() if k.split(".")[-1] == req]
        if not matches:
            # if we have names but no columns map, return empty list rather than 404
            if names:
                return {"table": table, "columns": []}
            raise HTTPException(404, "Schema manifest not loaded")
        columns = cols[matches[0]]
    return {"table": table, "columns": columns}
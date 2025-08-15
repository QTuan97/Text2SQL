from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..semantic.schema_cache import upsert_schema_snapshot, get_schema_text, known_schema

router = APIRouter(prefix="/schema", tags=["schema"])

class ColumnDef(BaseModel):
    name: str
    type: Optional[str] = None
    nullable: Optional[bool] = None

class TableDef(BaseModel):
    name: str
    columns: List[str] | List[ColumnDef]

class SchemaUpload(BaseModel):
    tables: List[TableDef]

@router.put("")
def upload_schema(body: SchemaUpload) -> Dict[str, Any]:
    try:
        return upsert_schema_snapshot(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/text")
def schema_text() -> Dict[str, str]:
    return {"text": get_schema_text()}

@router.get("/tables")
def schema_tables() -> Dict[str, Any]:
    names, cols = known_schema()
    return {"tables": names}

@router.get("/columns/{table}")
def schema_columns(table: str) -> Dict[str, Any]:
    _, cols = known_schema()
    return {"table": table, "columns": cols.get(table, [])}
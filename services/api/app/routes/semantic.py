from fastapi import APIRouter, HTTPException
from ..semantic.provider import get_mdl, get_context, reload_mdl
from ..clients.postgres import pg_connect
import os

router = APIRouter()

@router.get("/semantic")
def read_semantic():
    return {"mode": os.getenv("SEMANTIC_MODE","hybrid"), "mdl": get_mdl()}

@router.post("/semantic/reload")
def semantic_reload():
    reload_mdl()
    return {"ok": True}

@router.get("/semantic/context")
def read_context():
    from ..semantic.provider import get_context
    return {"context": get_context()}

@router.post("/semantic/synonyms")
def add_synonym(*args, **kwargs):
    raise HTTPException(status_code=410, detail="DB-backed synonyms are disabled.")
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
def add_synonym(phrase: str, maps_to: str, weight: float = 1.0, approved: bool = True):
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO semantic_synonyms(phrase, maps_to, weight, approved)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (phrase) DO UPDATE
                SET maps_to=EXCLUDED.maps_to, weight=EXCLUDED.weight, approved=EXCLUDED.approved;
            """, (phrase, maps_to, weight, approved))
        conn.commit()
    reload_mdl()
    return {"ok": True}
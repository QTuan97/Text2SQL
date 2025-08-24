from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from pathlib import Path
import os, time

import yaml

from ..clients.postgres import pg_connect
from .loader import build_llm_context

# Modes: static (YAML only), dynamic (DB only), hybrid (DB + YAML + DB overrides)
MODE = os.getenv("SEMANTIC_MODE", "hybrid").lower()
YAML_PATH = Path(os.getenv("SEMANTIC_MDL_PATH", "services/api/app/semantic/semantic.yaml"))
TTL_SEC = int(os.getenv("SEMANTIC_TTL_SEC", "60"))

_LAST: Dict[str, Any] = {"ts": 0.0, "yaml_mtime": 0.0, "mdl": None}

def _yaml_mtime() -> float:
    return YAML_PATH.stat().st_mtime if YAML_PATH.exists() else 0.0

def _introspect_pg(schema: str = "public") -> Dict[str, Any]:
    ent_map: Dict[str, Dict[str, Any]] = {}
    fk_rows: List[Tuple[str, str, str, str]] = []
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position;
            """, (schema,))
            for t, c, dt in cur.fetchall():
                e = ent_map.setdefault(t, {"name": t, "table": t, "primary_key": None, "dimensions": [], "synonyms": []})
                e["dimensions"].append({"name": c, "column": c, "type": dt})

            cur.execute("""
                SELECT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s;
            """, (schema,))
            fk_rows = cur.fetchall()

            # Best-effort primary keys
            cur.execute("""
                SELECT kcu.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = %s;
            """, (schema,))
            for t, c in cur.fetchall():
                if t in ent_map:
                    ent_map[t]["primary_key"] = c

            # Optional: sample low-card columns to mark enums/time
            # (kept minimal: detect timestamp-ish)
            for e in ent_map.values():
                for d in e["dimensions"]:
                    if "timestamp" in (d["type"] or "") or d["name"] in ("created_at", "order_date"):
                        d["role"] = "time"
                        d["grains"] = ["day","week","month","quarter","year"]

    entities = list(ent_map.values())
    relationships: List[Dict[str, Any]] = []
    for t, c, rt, rc in fk_rows:
        relationships.append({
            "name": f"{t}_{c}__{rt}_{rc}",
            "left": {"entity": t, "column": c},
            "right": {"entity": rt, "column": rc},
            "type": "many_to_one"
        })

    return {
        "version": 1,
        "source": {"name": "appdb", "type": "postgres", "schema": schema},
        "conventions": {"sql_dialect": "postgres", "default_time_grain": "month"},
        "entities": entities,
        "relationships": relationships,
        "metrics": [],   # filled by overrides
        "rules": [
            "SELECT-only; no INSERT/UPDATE/DELETE.",
            "Use explicit JOINs through defined relationships.",
            "Non-aggregates must be in GROUP BY when aggregates are used.",
        ],
        "synonyms": {"phrases": {}},
    }

def _load_yaml() -> Dict[str, Any]:
    if not YAML_PATH.exists():
        return {"metrics": [], "rules": [], "synonyms": {"phrases": {}}, "entities": [], "relationships": []}
    with YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_db_overrides(schema: str = "public") -> Dict[str, Any]:
    """No-op: we do not use DB-backed synonyms/metrics anymore."""
    return {"metrics": [], "synonyms": {"phrases": {}}}

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out

def _build_mdl() -> Dict[str, Any]:
    if MODE == "static":
        mdl = _load_yaml()
    elif MODE == "dynamic":
        mdl = _introspect_pg()
    else:  # hybrid
        mdl = _deep_merge(_introspect_pg(), _load_yaml())
    mdl = _deep_merge(mdl, _load_db_overrides())
    return mdl

def _stale(now: float, ts: float, ttl: int, yaml_mtime: float, last_yaml_mtime: float) -> bool:
    return (now - ts) > ttl or yaml_mtime != last_yaml_mtime

def get_mdl() -> Dict[str, Any]:
    now = time.time()
    ymt = _yaml_mtime()
    if _stale(now, _LAST["ts"], TTL_SEC, ymt, _LAST["yaml_mtime"]) or _LAST["mdl"] is None:
        mdl = _build_mdl()
        _LAST.update({"ts": now, "yaml_mtime": ymt, "mdl": mdl})
        return mdl
    return _LAST["mdl"]

def get_context() -> str:
    """LLM-friendly text block."""
    return build_llm_context(get_mdl())

def reload_mdl() -> None:
    """Manual flush."""
    _LAST.update({"ts": 0.0})

# def get_schema_text() -> str:
#     """Compact schema listing for LLM prompts, from current MDL (no DB)."""
#     mdl = get_mdl() or {}
#     parts = []
#     for e in mdl.get("entities", []):
#         cols = [d.get("name","") for d in e.get("dimensions", []) if d.get("name")]
#         parts.append(f"{e.get('name','')}({', '.join(cols)})")
#     return "\n".join(parts)
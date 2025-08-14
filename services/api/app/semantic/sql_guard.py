# Py3.9+  — validates that tables in FROM/JOIN exist in the DB
from __future__ import annotations
from typing import Dict, Set, List, Tuple
import time, os
from sqlglot import parse_one, exp
from ..clients.postgres import pg_connect

# simple TTL cache so we don't hit information_schema on every request
_CACHE = {"ts": 0.0, "ttl": float(os.getenv("DB_CATALOG_TTL", "60")), "cat": None}

def _load_catalog(schema: str = None) -> Dict[str, Set[str]]:
    schema = schema or os.getenv("PG_SCHEMA", "public")
    now = time.time()
    if _CACHE["cat"] is not None and (now - _CACHE["ts"]) < _CACHE["ttl"]:
        return _CACHE["cat"]  # type: ignore
    cat: Dict[str, Set[str]] = {}
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = %s;
                """,
                (schema,),
            )
            for t, c in cur.fetchall():
                cat.setdefault(t, set()).add(c)
    _CACHE.update({"ts": now, "cat": cat})
    return cat

def _extract_tables(sql: str) -> List[str]:
    tree = parse_one(sql, read="postgres")
    names: List[str] = []
    for t in tree.find_all(exp.Table):
        # t.this is the Identifier for the table; t.args.get("db") is the schema when qualified
        if t.this:
            names.append(t.this.name)
    # de-dup preserve order
    seen = set(); out = []
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    return out

def ensure_known_tables(sql: str, schema: str = None) -> None:
    """
    Raises ValueError if any table in FROM/JOIN is unknown to current DB schema.
    """
    catalog = _load_catalog(schema)
    refs = _extract_tables(sql)
    unknown = [t for t in refs if t not in catalog]
    if unknown:
        available = ", ".join(sorted(catalog.keys()))
        raise ValueError(
            f"Unknown table(s): {', '.join(unknown)}. Available tables: {available}."
        )
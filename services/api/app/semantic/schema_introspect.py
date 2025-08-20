from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import textwrap

@dataclass
class Column:
    table_schema: str
    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool
    column_default: Optional[str]
    column_comment: Optional[str]

@dataclass
class TableInfo:
    table_schema: str
    table_name: str
    table_comment: Optional[str]
    pk_columns: List[str]
    fks: List[Dict[str, str]]  # [{from_col, to_table, to_col}]
    columns: List[Column]

SCHEMA_SQL = """
WITH cols AS (
  SELECT
    c.table_schema,
    c.table_name,
    c.column_name,
    c.data_type,
    (c.is_nullable = 'YES') AS is_nullable,
    c.column_default,
    pgd.description AS column_comment
  FROM information_schema.columns c
  LEFT JOIN pg_catalog.pg_class pc
    ON pc.relname = c.table_name
  LEFT JOIN pg_catalog.pg_namespace pn
    ON pn.nspname = c.table_schema
  LEFT JOIN pg_catalog.pg_attribute pa
    ON pa.attrelid = pc.oid AND pa.attname = c.column_name
  LEFT JOIN pg_catalog.pg_description pgd
    ON pgd.objoid = pc.oid AND pgd.objsubid = pa.attnum
  WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
),
tbls AS (
  SELECT
    n.nspname AS table_schema,
    c.relname AS table_name,
    d.description AS table_comment
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = 0
  WHERE c.relkind = 'r'  -- base tables
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
),
pks AS (
  SELECT
    n.nspname AS table_schema,
    c.relname AS table_name,
    a.attname AS column_name
  FROM pg_index i
  JOIN pg_class c ON c.oid = i.indrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
  WHERE i.indisprimary
),
fks AS (
  SELECT
    n1.nspname AS table_schema,
    c1.relname AS table_name,
    a1.attname AS from_col,
    n2.nspname AS ref_schema,
    c2.relname AS to_table,
    a2.attname AS to_col
  FROM pg_constraint con
  JOIN pg_class c1 ON c1.oid = con.conrelid
  JOIN pg_namespace n1 ON n1.oid = c1.relnamespace
  JOIN pg_class c2 ON c2.oid = con.confrelid
  JOIN pg_namespace n2 ON n2.oid = c2.relnamespace
  JOIN UNNEST(con.conkey) WITH ORDINALITY AS ck(attnum, ord) ON TRUE
  JOIN UNNEST(con.confkey) WITH ORDINALITY AS fk(attnum, ord) ON ck.ord = fk.ord
  JOIN pg_attribute a1 ON a1.attrelid = con.conrelid AND a1.attnum = ck.attnum
  JOIN pg_attribute a2 ON a2.attrelid = con.confrelid AND a2.attnum = fk.attnum
  WHERE con.contype = 'f'
)
SELECT
  t.table_schema,
  t.table_name,
  t.table_comment,
  (SELECT array_agg(column_name ORDER BY column_name)
     FROM pks p WHERE p.table_schema = t.table_schema AND p.table_name = t.table_name) AS pk_columns,
  (SELECT json_agg(json_build_object('from_col', f.from_col,
                                     'to_table', f.to_table,
                                     'to_col', f.to_col))
     FROM fks f WHERE f.table_schema = t.table_schema AND f.table_name = t.table_name) AS fks,
  (SELECT json_agg(json_build_object(
            'table_schema', c.table_schema,
            'table_name',   c.table_name,
            'column_name',  c.column_name,
            'data_type',    c.data_type,
            'is_nullable',  c.is_nullable,
            'column_default', c.column_default,
            'column_comment', c.column_comment))
     FROM cols c WHERE c.table_schema = t.table_schema AND c.table_name = t.table_name) AS columns
FROM tbls t
ORDER BY t.table_schema, t.table_name;
"""

def fetch_schema(conn) -> List[TableInfo]:
    cur = conn.execute(SCHEMA_SQL)
    rows = cur.fetchall()
    tables: List[TableInfo] = []
    for r in rows:
        columns = [
            Column(**col) for col in (r["columns"] or [])
        ]
        tables.append(
            TableInfo(
                table_schema=r["table_schema"],
                table_name=r["table_name"],
                table_comment=r["table_comment"],
                pk_columns=r["pk_columns"] or [],
                fks=r["fks"] or [],
                columns=columns,
            )
        )
    return tables

def render_table_doc(t: TableInfo) -> str:
    col_lines = []
    for c in t.columns:
        col_lines.append(
            f"- {c.column_name} ({c.data_type})"
            + (" NOT NULL" if not c.is_nullable else "")
            + (f", default={c.column_default}" if c.column_default else "")
            + (f" — {c.column_comment}" if c.column_comment else "")
        )
    pk = ", ".join(t.pk_columns) if t.pk_columns else "—"
    if t.fks:
        fk_lines = [f"- {fk['from_col']} → {fk['to_table']}.{fk['to_col']}" for fk in t.fks]
        fk_block = "\n".join(fk_lines)
    else:
        fk_block = "—"

    return textwrap.dedent(f"""
    [SCHEMA: {t.table_schema}.{t.table_name}]
    {t.table_comment or ""}

    Primary Key: {pk}
    Foreign Keys:
    {fk_block}

    Columns:
    {chr(10).join(col_lines)}
    """).strip()
from __future__ import annotations
from typing import List, Dict, Set, Optional
import re

from sqlglot import parse_one, exp

from .schema_cache import known_schema, ensure_known_tables_cached as _ensure_cached


# Internal utilities
def _parse(sql: str) -> exp.Expression:
    return parse_one(sql, read="postgres")

def _base_to_alias(tree: exp.Expression) -> Dict[str, str]:
    """
    Map base table name -> alias (e.g., orders -> o)
    """
    m: Dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        base = getattr(getattr(t.this, "this", None), "name", None) or getattr(t.this, "name", None)
        alias = getattr(getattr(getattr(t, "alias", None), "this", None), "name", None)
        if base and alias:
            m[base.lower()] = alias
    return m

def _aliases(tree: exp.Expression) -> Set[str]:
    """
    Set of alias names used in FROM/JOIN
    """
    s: Set[str] = set()
    for t in tree.find_all(exp.Table):
        a = getattr(getattr(getattr(t, "alias", None), "this", None), "name", None)
        if a:
            s.add(a.lower())
    return s

def _joined_bases(tree: exp.Expression) -> Set[str]:
    """
    Set of base table names referenced in FROM/JOIN
    """
    s: Set[str] = set()
    for t in tree.find_all(exp.Table):
        b = getattr(getattr(t.this, "this", None), "name", None) or getattr(t.this, "name", None)
        if b:
            s.add(b.lower())
    return s


# Public helpers used by routes / services
def _tables_in(sql: str) -> List[str]:
    """
    Return unique table names referenced in the SQL (as they appear).
    """
    try:
        tree = _parse(sql)
    except Exception:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for t in tree.find_all(exp.Table):
        # Try several shapes to robustly get the table name
        nm = getattr(t, "name", None) \
             or getattr(getattr(t.this, "this", None), "name", None) \
             or getattr(t.this, "name", None)
        if nm and nm not in seen:
            out.append(nm)
            seen.add(nm)
    return out


def ensure_known_tables_cached(sql: str) -> None:
    """
    Validate that all table identifiers in SQL exist in the cached schema (Qdrant).
    Delegates to the cache module’s implementation.
    """
    _ensure_cached(sql)


def ensure_known_tables(sql: str, schema: Optional[str] = None) -> None:
    """
    Backward-compatible alias. We do NOT touch Postgres anymore.
    """
    ensure_known_tables_cached(sql)


def validate_schema(sql: str) -> List[str]:
    """
    Static schema validation using the cached schema snapshot.

    Returns a list of issue tags such as:
      - "unknown-table:<name>"
      - "unknown-column:<qualifier.column>" or "unknown-column:<column>"
      - "ambiguous-column:<column>"
      - "must-use-alias:<base.column>"
      - "unknown-qualifier:<qualifier>"
    """
    issues: List[str] = []

    # Load cached schema
    table_names, cols_by = known_schema()
    known_tables = {t.lower() for t in table_names}

    # Parse once
    try:
        tree = _parse(sql)
    except Exception as e:
        return [f"parse-error:{e}"]

    alias_of_base = _base_to_alias(tree)   # base -> alias
    alias_names   = _aliases(tree)         # alias set
    bases_in_from = _joined_bases(tree)    # bases present

    # Unknown tables in FROM/JOIN
    for t in tree.find_all(exp.Table):
        tname = getattr(t, "name", None) \
                or getattr(getattr(t.this, "this", None), "name", None) \
                or getattr(t.this, "name", None)
        if tname and tname.lower() not in known_tables:
            issues.append(f"unknown-table:{tname}")

    # Column checks
    for col in tree.find_all(exp.Column):
        tident = getattr(col, "table", None)
        cident = getattr(col, "this", None)
        tname  = getattr(tident, "name", None) if isinstance(tident, exp.Identifier) else None
        cname  = getattr(cident, "name", None) if isinstance(cident, exp.Identifier) else None
        if not cname:
            continue

        if tname:
            tl = tname.lower()
            # If the base table is aliased in FROM/JOIN, references must use alias, not base
            if tl in bases_in_from and tl in alias_of_base:
                issues.append(f"must-use-alias:{tname}.{cname}")

            # Qualifier isn't a known alias nor a known table
            if tl not in alias_names and tl not in known_tables:
                issues.append(f"unknown-qualifier:{tname}")

            # Column existence: resolve base for alias if needed
            base = tl
            if tl in alias_names:
                # find base for this alias
                for b, a in alias_of_base.items():
                    if a == tl:
                        base = b
                        break

            # If we know the base's columns, check membership
            if base in cols_by:
                if cname not in set(cols_by[base]):
                    issues.append(f"unknown-column:{tname}.{cname}")
        else:
            # Bare column: could be unknown or ambiguous
            carriers = [t for t, cols in cols_by.items() if cname in set(cols)]
            if len(carriers) == 0:
                issues.append(f"unknown-column:{cname}")
            elif len(carriers) > 1:
                issues.append(f"ambiguous-column:{cname}")

    # De-duplicate while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for it in issues:
        if it not in seen:
            uniq.append(it)
            seen.add(it)
    return uniq


def lint_sql(sql: str) -> List[str]:
    """
    Lightweight lints that don't require hitting the DB:
      - join-without-on
      - aggregate-without-group-by
      - no-limit
      - select-star
    """
    tags: List[str] = []
    try:
        tree = _parse(sql)
    except Exception as e:
        return [f"parse-error:{e}"]

    s = sql.strip().lower()

    # JOIN without ON/USING
    if " join " in s and " on " not in s and " using " not in s:
        tags.append("join-without-on")

    # Aggregation without GROUP BY
    try:
        has_agg = any(
            isinstance(f, exp.Anonymous) and f.name.lower() in {"sum", "count", "avg", "min", "max"}
            for f in tree.find_all(exp.Anonymous)
        )
        has_group = any(isinstance(g, exp.Group) for g in tree.find_all(exp.Group))
        if has_agg and not has_group:
            tags.append("aggregate-without-group-by")
    except Exception:
        pass

    # LIMIT present?
    if not re.search(r"(?is)\blimit\s+\d+\b", s):
        tags.append("no-limit")

    # SELECT *
    try:
        for sel in tree.find_all(exp.Select):
            for e in sel.expressions:
                if isinstance(e, exp.Star):
                    tags.append("select-star")
                    raise StopIteration  # one tag is enough
    except StopIteration:
        pass

    # De-duplicate
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in tags:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq
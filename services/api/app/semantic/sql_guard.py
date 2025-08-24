from __future__ import annotations
from typing import List, Set, Optional, Iterable
import re

from sqlglot import parse_one, exp

# Cached schema snapshot comes from your MDL/schema cache (no live DB)
from ..semantic.schema_cache import known_schema

# Regex + helpers
_AGG_RE      = re.compile(r"\b(sum|avg|count|min|max|array_agg|string_agg|bool_and|bool_or)\s*\(", re.I)
_GROUPBY_RE  = re.compile(r"\bgroup\s+by\b", re.I)
_LIMIT_RE    = re.compile(r"\blimit\s+\d+\b", re.I)

# For relative-time misuse (hard-coding month/year)
_RELATIVE_Q = re.compile(
    r"\b(this\s+month|today|yesterday|last\s+7\s+days|this\s+week|last\s+month|this\s+quarter|this\s+year)\b",
    re.I,
)
_HARDCODE_TIME = re.compile(r"\bextract\s*\(\s*(month|year)\s*from\b[^)]*\)\s*=\s*\d+", re.I)
_HAS_DATE_TRUNC = re.compile(r"\bdate_trunc\s*\(\s*'(month|week|quarter|year)'\s*,", re.I)

def _looks_like_single_aggregate(sql: str) -> bool:
    s = (sql or "").strip()
    return (_AGG_RE.search(s) is not None) and (_GROUPBY_RE.search(s) is None)

def _normalize_table_ref(t: exp.Table) -> Set[str]:
    """Return possible names to match against known schema (base + schema-qualified)."""
    out: Set[str] = set()
    name = (t.name or "").lower()
    if name:
        out.add(name)
    # Try schema-qualified if present
    db = t.args.get("db")
    try:
        schema = (db.name if isinstance(db, exp.Identifier) else str(db)).strip('"').lower() if db else ""
    except Exception:
        schema = ""
    if schema and name:
        out.add(f"{schema}.{name}")
    return out

def _tables_in(sql: str) -> Set[str]:
    """Collect referenced table names (lowercased)."""
    try:
        tree = parse_one(sql, read="postgres")
    except Exception:
        return set()
    refs: Set[str] = set()
    for t in tree.find_all(exp.Table):
        refs |= _normalize_table_ref(t)
    return refs

def _has_order_by(tree: exp.Expression) -> bool:
    return tree.find(exp.Order) is not None

def _has_limit(tree: exp.Expression) -> bool:
    return tree.find(exp.Limit) is not None

def _select_items(tree: exp.Expression) -> List[exp.Expression]:
    sel = tree.find(exp.Select)
    return list(sel.expressions) if sel else []

def _expr_contains_agg(e: exp.Expression) -> bool:
    return any(True for _ in e.find_all((exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max, exp.ArrayAgg)))

def _expr_is_simple_column(e: exp.Expression) -> bool:
    return isinstance(e, exp.Column)

def _variants(name: str) -> Set[str]:
    n = name.strip().lower()
    base = n.split(".")[-1]          # drop schema prefix
    out = {base}
    # naive singular/plural variants
    if base.endswith("ies"):
        out.add(base[:-3] + "y")     # categories -> category
    if base.endswith("s"):
        out.add(base[:-1])           # orders -> order
        if not base.endswith("es"):
            out.add(base + "es")     # order -> orders/es
    else:
        out.add(base + "s")
        out.add(base + "es")
    # allow underscore/space variants if you have multiword names
    out |= {base.replace("_", " "), base.replace(" ", "_")}
    return {v for v in out if v}

def build_ambiguity_guard() -> str:
    names, _ = known_schema()
    canon = sorted({t.split(".")[-1].lower() for t in names})
    lines = []
    for t in canon:
        vs = sorted(_variants(t))
        # left side = things user might say; right side = the ONLY SQL table name allowed
        lines.append(f"- {', '.join(vs)}  ->  {t}")
    return (
        "# Ambiguity guard (canonical table names)\n"
        "Use ONLY the canonical name on the right when writing SQL. "
        "Do not invent or use other variants.\n" +
        "\n".join(lines) + "\n"
    )

# Public: schema checks (cached only)
def ensure_known_tables_cached(sql: str) -> None:
    """
    Raise if the SQL references tables not present in the cached schema manifest.
    Uses base-name OR schema-qualified match.
    """
    try:
        known_names, _cols = known_schema()
    except Exception:
        return

    known_set = {str(n).lower() for n in (known_names or [])}
    if not known_set:
        return

    missing: List[str] = []
    for ref in _tables_in(sql):
        base = ref.split(".")[-1]
        match = (
            ref in known_set
            or base in known_set
            or any(k.endswith("." + base) for k in known_set)
        )
        if not match:
            missing.append(ref)

    if missing:
        raise ValueError(f"Unknown tables in SQL: {', '.join(sorted(set(missing)))}")


def validate_schema(sql: str) -> List[str]:
    """
    Return schema-related error tags (strings). Non-throwing version used by the grader.
    Currently checks table existence only; extend with column checks if your cache provides them.
    """
    tags: List[str] = []
    try:
        known_names, cols_map = known_schema()
    except Exception:
        return tags

    known_set = {str(n).lower() for n in (known_names or [])}
    if not known_set:
        return tags

    for ref in _tables_in(sql):
        base = ref.split(".")[-1]
        match = (
            ref in known_set
            or base in known_set
            or any(k.endswith("." + base) for k in known_set)
        )
        if not match:
            tags.append("schema-unknown-table")
            break

    return tags


# Public: lints (used by quality grader)
def lint_relative_time_bad(question: str, sql: str) -> List[str]:
    """
    If the user asked a relative-time question and the SQL hard-codes month/year via EXTRACT(...)=N
    without using DATE_TRUNC windows, flag it.
    """
    if _RELATIVE_Q.search(question or "") and _HARDCODE_TIME.search(sql or "") and not _HAS_DATE_TRUNC.search(sql or ""):
        return ["relative-time-hardcode"]
    return []

def lint_alias_discipline(sql: str) -> list[str]:
    try:
        tree = parse_one(sql, read="postgres")
    except Exception:
        return []
    # collect aliases for all tables
    aliases = {t.alias_or_name for t in tree.find_all(exp.Table)}
    aliases = {a.lower() for a in aliases if a}
    base_names = {getattr(t, "name", "") for t in tree.find_all(exp.Table)}
    base_names = {b.lower() for b in base_names if b}
    # if we have any alias, every column must be from an alias (not a base table name)
    if aliases:
        for c in tree.find_all(exp.Column):
            # c.table is the qualifier before the dot, if present
            q = (c.table or "").lower()
            if q and q not in aliases:
                return ["unaliased-or-base-qualified-when-aliased"]
    return []

def lint_sql(sql: str) -> List[str]:
    """
    Lightweight, dialect-agnostic lints. Returns a list of tag strings.
    """
    tags: List[str] = []
    try:
        tree = parse_one(sql, read="postgres")
    except Exception:
        return tags

    # SELECT *
    if any(isinstance(e, exp.Star) for e in tree.find_all(exp.Star)):
        tags.append("select-star")

    # LIMIT without ORDER BY (top-k stability)
    if _has_limit(tree) and not _has_order_by(tree):
        tags.append("missing-order-by-on-topk")

    # Ambiguous bare columns when multiple tables exist
    table_refs = {t.name for t in tree.find_all(exp.Table) if t.name}
    if len(table_refs) > 1:
        for col in tree.find_all(exp.Column):
            # exp.Column.table gives qualifier; None/'' means bare
            if not (col.table or "").strip():
                tags.append("ambiguous-column")
                break

    # Aggregate without GROUP BY while also selecting non-agg columns
    has_group = tree.find(exp.Group) is not None
    select_items = _select_items(tree)
    has_agg_in_select = any(_expr_contains_agg(e) for e in select_items)
    selects_non_agg = any(not (_expr_contains_agg(e) or isinstance(e, (exp.Literal, exp.Cast))) for e in select_items)

    if has_agg_in_select and not has_group and selects_non_agg:
        tags.append("aggregate-without-group-by")

    # De-duplicate
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in tags:
        if t not in seen:
            uniq.append(t); seen.add(t)

    # Don’t flag single-row aggregates as an error (e.g., SUM(...) only)
    if "aggregate-without-group-by" in uniq and _looks_like_single_aggregate(sql):
        uniq = [x for x in uniq if x != "aggregate-without-group-by"]

    return uniq

# Severity sets (used by your route’s quality grader)
SEVERE_LINTS = {
    "aggregate-without-group-by",
    "relative-time-hardcode",
    "unaliased-or-base-qualified-when-aliased",
    # add these here if you centralize schema errors as lints:
    # "schema-unknown-table", "schema-unknown-column",
}
WARNING_LINTS = {
    "missing-order-by-on-topk",
    "select-star",
    "ambiguous-column",
}

# Convenience export (some callers import this)
tables_in = _tables_in
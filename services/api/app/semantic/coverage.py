from __future__ import annotations
import re
from typing import Dict, Any, Optional
from sqlglot import parse_one, exp

_TOPK_Q        = re.compile(r"\btop\s+(\d+)\b", re.I)
_BY_METRIC_Q   = re.compile(r"\bby\s+([a-z_ ][a-z_ ]+)\b", re.I)     # e.g. "by revenue"
_RELATIVE_Q    = re.compile(r"\b(this\s+month|today|yesterday|last\s+\d+\s+days|this\s+week|last\s+month|this\s+quarter|this\s+year)\b", re.I)
_Q_LITERALS    = re.compile(r"'([^']+)'")

_FROM_CITY_Q   = re.compile(r"\b(from|in)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b")

def _topk_expected(q: str) -> Optional[int]:
    m = _TOPK_Q.search(q or "")
    return int(m.group(1)) if m else None

def _order_metric_expected(q: str) -> Optional[str]:
    m = _BY_METRIC_Q.search(q or "")
    return m.group(1).strip() if m else None

def _question_entities(q: str) -> list[str]:
    lits = _Q_LITERALS.findall(q or "")
    if lits:
        return lits
    m = _FROM_CITY_Q.search(q or "")
    return [m.group(2)] if m else []

# SQL parsers
def _int_from_expr(e: exp.Expression | None) -> Optional[int]:
    if e is None:
        return None
    try:
        return int(str(e).strip())
    except Exception:
        return None

def _limit_of(tree: exp.Expression) -> Optional[int]:
    lim = tree.find(exp.Limit)
    return _int_from_expr(lim.expression) if lim else None

def _has_order_by(tree: exp.Expression) -> bool:
    return tree.find(exp.Order) is not None

def _has_date_trunc(tree: exp.Expression, sql_text: str | None = None) -> bool:
    DT = getattr(exp, "DateTrunc", None)
    if DT and tree.find(DT): return True
    Anonymous = getattr(exp, "Anonymous", None)
    if Anonymous:
        for fn in tree.find_all(Anonymous):
            name = getattr(fn, "name", None) or str(getattr(fn, "this", "")).lower()
            if isinstance(name, str) and name.strip().lower() == "date_trunc":
                return True
    Func = getattr(exp, "Func", None)
    if Func:
        for fn in tree.find_all(Func):
            name = getattr(fn, "name", None)
            if isinstance(name, str) and name.strip().lower() == "date_trunc":
                return True
    s = sql_text or str(tree)
    return bool(re.search(r"\bdate_trunc\s*\(", s, re.I))

def _has_aggregate(tree: exp.Expression) -> bool:
    agg_nodes = [getattr(exp, n, None) for n in ("Sum","Avg","Count","Min","Max","ArrayAgg","StringAgg")]
    if any(n and tree.find(n) for n in agg_nodes): return True
    Func = getattr(exp, "Func", None)
    return any(getattr(f, "is_aggregate", False) for f in (tree.find_all(Func) if Func else []))

def _select_dims_not_in_group_by(tree: exp.Expression) -> list[str]:
    sel = tree.find(exp.Select)
    if not sel: return []
    gb = tree.find(exp.Group)
    gb_set = set()
    if gb:
        for e in (gb.expressions or []):
            gb_set.add(e.sql(dialect="postgres").strip().lower())
    missing = []
    for item in (sel.expressions or []):
        expr = item.this if isinstance(item, exp.Alias) else item
        if getattr(expr, "is_aggregate", False): continue
        s = expr.sql(dialect="postgres").strip().lower()
        if s not in gb_set:
            missing.append(s)
    return missing

def _sql_string_literals(sql: str) -> set[str]:
    return set(re.findall(r"'([^']+)'", sql or ""))

# API
def audit_sql_coverage(question: str, sql: str) -> Dict[str, Any]:
    """
    Deterministic coverage check:
      - top_k in question → LIMIT must exist and equal that K
      - "by <metric>" in question → ORDER BY must be present
      - quoted entity (or from/in City) in question → same literal must appear in SQL
      - relative-time phrase → SQL must include DATE_TRUNC(...) (bounded window)
    """
    q = (question or "").strip()
    s = (sql or "").strip()
    missing: list[str] = []

    # parse SQL once
    try:
        tree = parse_one(s, read="postgres")
    except Exception:
        return {
            "ok": False,
            "missing": ["sql-parse-failed"],
            "top_k_expected": _topk_expected(q),
            "top_k_found": None,
            "order_metric_expected": _order_metric_expected(q),
            "order_metric_found": None,
        }

    # TOP K
    k_expected = _topk_expected(q)
    k_found = _limit_of(tree)
    if k_expected is not None and k_found != k_expected:
        missing.append(f"expect LIMIT {k_expected}")

    # ORDER BY
    metric_expected = _order_metric_expected(q)
    if metric_expected and not _has_order_by(tree):
        missing.append("missing ORDER BY")

    # ENTITIES
    entities = _question_entities(q)
    if entities:
        s_lits = _sql_string_literals(s)
        for ent in entities:
            if ent not in s_lits:
                missing.append(f"missing literal '{ent}'")

    # RELATIVE TIME (require bounded window)
    if _RELATIVE_Q.search(q) and not _has_date_trunc(tree, s):
        missing.append("missing bounded time window (date_trunc)")

    # GROUP BY
    if _has_aggregate(tree):
        miss = _select_dims_not_in_group_by(tree)
        if miss:
            issues.append("non-aggregated columns not in GROUP BY: " + ", ".join(miss))

    ok = len(missing) == 0
    return {
        "ok": ok,
        "missing": missing,
        "top_k_expected": k_expected,
        "top_k_found": k_found,
        "order_metric_expected": metric_expected,
        "order_metric_found": None,
    }
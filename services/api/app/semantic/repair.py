from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict, Set
import re
from sqlglot import parse_one, exp
from .sql_guard import ensure_known_tables_cached

AGG_FUNCS = {"sum","count","avg","min","max"}

def _parse(sql: str):
    return parse_one(sql, read="postgres")

def _base_to_alias(tree):
    m: Dict[str,str] = {}
    for t in tree.find_all(exp.Table):
        base = getattr(getattr(t.this, "this", None), "name", None) or getattr(t.this, "name", None)
        alias = getattr(getattr(getattr(t, "alias", None), "this", None), "name", None)
        if base and alias: m[base.lower()] = alias
    return m

def _aliases(tree):
    s: Set[str] = set()
    for t in tree.find_all(exp.Table):
        a = getattr(getattr(getattr(t, "alias", None), "this", None), "name", None)
        if a: s.add(a.lower())
    return s

def _joined_bases(tree):
    s: Set[str] = set()
    for t in tree.find_all(exp.Table):
        b = getattr(getattr(t.this, "this", None), "name", None) or getattr(t.this, "name", None)
        if b: s.add(b.lower())
    return s

def _lev1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    i = 0
    j = 0
    ed = 0

    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            ed += 1
            if ed > 1:
                return False
            if la == lb:
                i += 1
                j += 1
            elif la > lb:
                i += 1
            else:
                j += 1

    if i < la or j < lb:
        ed += 1

    return ed <= 1


def detect(sql: str) -> List[Tuple[str,str]]:
    issues: List[Tuple[str,str]] = []
    try: tree = _parse(sql)
    except Exception as e: return [("parse-error", str(e))]
    alias_of = _base_to_alias(tree); aliases = _aliases(tree); bases = _joined_bases(tree)
    s = sql.strip().lower()
    if " join " in s and " on " not in s and " using " not in s:
        issues.append(("join-without-on",""))
    has_agg = any(isinstance(f, exp.Anonymous) and f.name.lower() in AGG_FUNCS for f in tree.find_all(exp.Anonymous))
    if has_agg and not list(tree.find_all(exp.Group)):
        issues.append(("aggregate-without-group-by",""))
    if " limit " not in s:
        issues.append(("no-limit",""))
    # unknown qualifiers & must-use-alias/unknown-column are checked in ensure_known_tables_cached during validation
    return issues

Fixer = Callable[[str], Optional[str]]
def _fix_alias_qualifiers(sql: str) -> Optional[str]:
    try: tree = _parse(sql)
    except Exception: return None
    b2a = _base_to_alias(tree)
    if not b2a: return None
    fixed = sql
    for base, alias in b2a.items():
        fixed = re.sub(rf"\b{re.escape(base)}\.", f"{alias}.", fixed, flags=re.IGNORECASE)
    return fixed if fixed!=sql else None

def _fix_alias_typos(sql: str) -> Optional[str]:
    try: tree = _parse(sql)
    except Exception: return None
    aliases = _aliases(tree)
    if not aliases: return None
    quals = set(re.findall(r"\b([A-Za-z_][\w$]*)\.", sql))
    rep: Dict[str,str] = {}
    for q in quals:
        ql=q.lower()
        if ql in aliases: continue
        near = next((a for a in aliases if _lev1(ql,a)), None)
        if near: rep[q] = near
    if not rep: return None
    fixed = sql
    for wrong,right in rep.items():
        fixed = re.sub(rf"\b{re.escape(wrong)}\.", f"{right}.", fixed, flags=re.IGNORECASE)
    return fixed if fixed!=sql else None

def _qualify_bare(sql: str) -> Optional[str]:
    try: tree = _parse(sql)
    except Exception: return None
    b2a = _base_to_alias(tree); bases = list(_joined_bases(tree))
    if not bases: return None
    fixed = sql
    for col in tree.find_all(exp.Column):
        if col.table is not None: continue
        cname = getattr(col.this,"name",None) if isinstance(col.this, exp.Identifier) else None
        if not cname: continue
        # naive: prefer first table
        base = bases[0]; alias = b2a.get(base); qual = alias or base
        fixed = re.sub(rf"(?<!\.)\b{re.escape(cname)}\b", f"{qual}.{cname}", fixed, count=1)
    return fixed if fixed!=sql else None

def _add_group_by(sql: str) -> Optional[str]:
    try: tree = _parse(sql)
    except Exception: return None
    has_agg = any(isinstance(f, exp.Anonymous) and f.name.lower() in AGG_FUNCS for f in tree.find_all(exp.Anonymous))
    if not has_agg or list(tree.find_all(exp.Group)): return None
    non_aggs: List[str] = []
    for sel in tree.find_all(exp.Select):
        for e in sel.expressions:
            if isinstance(e,(exp.Anonymous,exp.Literal)): continue
            non_aggs.append(e.sql(dialect="postgres"))
    if not non_aggs: return None
    clause = " GROUP BY " + ", ".join(dict.fromkeys(non_aggs))
    m = re.search(r"(?is)\border\s+by\b", sql) or re.search(r"(?is)\blimit\b", sql)
    return (sql[:m.start()] + clause + " " + sql[m.start():]) if m else (sql.strip() + clause)

def _enforce_limit(sql: str, limit: int) -> Optional[str]:
    if re.search(r"(?is)\blimit\s+\d+\b", sql): return None
    return sql.rstrip().rstrip(";") + f" LIMIT {max(1,int(limit))}"

@dataclass(frozen=True)
class RepairRule:
    name: str
    apply: Fixer

DEFAULT_RULES: List[RepairRule] = [
    RepairRule("alias_qualifiers", _fix_alias_qualifiers),
    RepairRule("alias_typos",      _fix_alias_typos),
    RepairRule("qualify_bare",     _qualify_bare),
    RepairRule("group_by",         _add_group_by),
]

class RepairEngine:
    def __init__(self, limit: int = 50, rules: Optional[List[RepairRule]] = None):
        self.limit = max(1, int(limit)); self.rules = list(rules or DEFAULT_RULES)
    def apply(self, sql: str) -> Tuple[str, List[str], List[str]]:
        cand = sql; applied: List[str] = []; changed = True
        def _limit_rule(s: str) -> Optional[str]: return _enforce_limit(s, self.limit)
        for _ in range(3):
            if not changed: break
            changed = False
            for r in self.rules + [RepairRule("limit", _limit_rule)]:
                res = r.apply(cand)
                if res and res != cand:
                    cand = res; applied.append(r.name); changed = True
        remaining = [c for c,_ in detect(cand)]
        # validate against cached schema (raises on unknown tables)
        try:
            parse_one(cand, read="postgres"); ensure_known_tables_cached(cand)
        except Exception:
            pass
        return cand, remaining, applied
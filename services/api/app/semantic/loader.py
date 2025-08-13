from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import yaml

DEFAULT_PATH = os.getenv("SEMANTIC_MDL_PATH", "services/api/app/semantic/semantic.yaml")

@lru_cache(maxsize=1)
def load_mdl(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MDL manifest not found at {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _fmt_dims(entity: Dict[str, Any]) -> str:
    lines: List[str] = []
    for d in entity.get("dimensions", []):
        syn = d.get("synonyms", [])
        syn_s = f" (synonyms: {', '.join(syn)})" if syn else ""
        unit = f" [{d['unit']}]" if d.get("unit") else ""
        role = f" (role: {d['role']})" if d.get("role") else ""
        lines.append(f"- {entity['table']}.{d['column']}: {d['type']}{unit}{role}{syn_s}")
    return "\n".join(lines)

def build_llm_context(mdl: Dict[str, Any]) -> str:
    parts: List[str] = []
    src = mdl.get("source", {})
    conv = mdl.get("conventions", {})
    parts.append(f"Source: {src.get('type','?')} schema={src.get('schema','public')} dialect={conv.get('sql_dialect','postgres')}")
    parts.append("\nEntities/Tables:")
    for e in mdl.get("entities", []):
        syn = f" (aka {', '.join(e.get('synonyms', []))})" if e.get("synonyms") else ""
        parts.append(f"* {e['name']} → {e['table']}{syn}")
        parts.append(_fmt_dims(e))

    if mdl.get("relationships"):
        parts.append("\nRelationships:")
        for r in mdl["relationships"]:
            parts.append(f"* {r['name']}: {r['left']['entity']}.{r['left']['column']} = {r['right']['entity']}.{r['right']['column']}")

    if mdl.get("metrics"):
        parts.append("\nMetrics:")
        for m in mdl["metrics"]:
            syn = f" (aka {', '.join(m.get('synonyms', []))})" if m.get("synonyms") else ""
            filt = f" WHERE {m['filter']}" if m.get("filter") else ""
            parts.append(f"* {m['name']}{syn}: {m['expression']}{filt}")

    if mdl.get("rules"):
        parts.append("\nRules:")
        for rule in mdl["rules"]:
            parts.append(f"- {rule}")

    if mdl.get("synonyms", {}).get("phrases"):
        parts.append("\nPhrase Mappings:")
        for k, v in mdl["synonyms"]["phrases"].items():
            parts.append(f"- '{k}' ⇒ {v}")

    return "\n".join(parts)
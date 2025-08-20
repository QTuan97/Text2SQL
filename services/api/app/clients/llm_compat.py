# services/api/app/clients/llm_compat.py
from __future__ import annotations

import os
import json , httpx
from typing import Dict, List, Optional, Any

import requests
import asyncio
from openai import OpenAI

from ..config import settings

# --- Models from config.py ---
OPENAI_MODEL = settings.GEN_MODEL
VALID_EMBED_MODEL = settings.VALID_EMBED_MODEL
ERROR_EMBED_MODEL = settings.ERROR_EMBED_MODEL

# --- Endpoints/keys (prefer settings if present; fall back to env/defaults) ---
OPENAI_BASE_URL = getattr(settings, "OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://ollama:11434/v1"))
OPENAI_API_KEY  = getattr(settings, "OPENAI_API_KEY",  os.getenv("OPENAI_API_KEY", "ollama"))
OLLAMA_API_BASE = getattr(settings, "OLLAMA_API_BASE", os.getenv("OLLAMA_API_BASE", "http://ollama:11434"))


def _client() -> OpenAI:
    """
    OpenAI-compatible client (points at Ollama/vLLM/etc. if OPENAI_BASE_URL is local).
    """
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

async def _llm_complete(system: str, user: str, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
    t = timeout if isinstance(timeout, int) and timeout > 0 else 60
    return await asyncio.to_thread(chat, user, system, model, t)

def chat(prompt: str, system: str = "You are a concise assistant.", model: Optional[str] = None) -> str:
    """
    Simple text generation via OpenAI Chat Completions API (compatible with Ollama).
    """
    m = model or OPENAI_MODEL
    resp = _client().chat.completions.create(
        model=m,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def chat_json(prompt: str, system: str = "Return ONLY minified JSON.", model: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience wrapper that asks the model for strict minified JSON and parses it.
    """
    text = chat(f"You are a strict JSON API. Return ONLY minified JSON, no prose.\n{prompt}", system=system, model=model)
    text = text.strip().strip("`").replace("\n", " ").replace(", }", "}")
    return json.loads(text)


def embed_one(text: str, model: Optional[str] = None, timeout: int = 60) -> List[float]:
    """
    Embeddings via Ollama's native /api/embeddings (fast + local).
    """
    em = model or VALID_EMBED_MODEL
    r = requests.post(
        f"{OLLAMA_API_BASE}/api/embeddings",
        json={"model": em, "prompt": text},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError(f"Unexpected embedding response: {data}")
    return vec

async def schema_embed_one(text: str, model: str | None = None) -> list[float]:
    base = settings.OLLAMA_BASE_URL.rstrip("/")
    mdl  = model or settings.VALID_EMBED_MODEL
    async with httpx.AsyncClient(timeout=60) as cx:
        r = await cx.post(f"{base}/api/embeddings", json={"model": mdl, "prompt": text})
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding") or data.get("data") or data.get("vector")
        if not isinstance(vec, list):
            raise RuntimeError("Unexpected embeddings response")
        # ensure floats
        return [float(x) for x in vec]


def embed_many(texts: List[str], model: Optional[str] = None, timeout: int = 60) -> List[List[float]]:
    """
    Batched helper (simple loop). Keeps behavior predictable with local backends.
    """
    return [embed_one(t, model=model, timeout=timeout) for t in texts]
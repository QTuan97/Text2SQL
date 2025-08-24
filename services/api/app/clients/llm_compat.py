# services/api/app/clients/llm_compat.py
from __future__ import annotations

import os , anyio
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
OLLAMA_API_BASE = getattr(settings, "OLLAMA_API_BASE", os.getenv("OLLAMA_API_BASE", "http://ollama:11434"))


def _client() -> OpenAI:
    base = getattr(settings, "OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    key  = getattr(settings, "OPENAI_API_KEY",  os.getenv("OPENAI_API_KEY"))
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(base_url=base, api_key=key)

async def _llm_complete(system: str, user: str) -> str:
    model = getattr(settings, "GEN_MODEL", "gpt-4o-mini")
    temperature = float(getattr(settings, "GEN_TEMPERATURE", 0.0))
    top_p       = float(getattr(settings, "GEN_TOP_P", 1.0))
    max_tokens  = int(getattr(settings, "GEN_MAX_TOKENS", 512))
    seed        = getattr(settings, "GEN_SEED", None)

    messages = [{"role":"system","content":system},{"role":"user","content":user}]

    # OpenAI-compatible path (preferred)
    try:
        client = _client()
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, presence_penalty=0, frequency_penalty=0,
            seed=seed,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        pass

    # Ollama fallback (if configured)
    if 'ollama' in globals() and ollama is not None:
        opts = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens, "mirostat": 0, "top_k": 0, "repeat_penalty": 1.0, "seed": seed}
        try:
            r = await ollama.chat(model=model, messages=messages, options=opts)
        except TypeError:
            r = await ollama.chat(model=model, messages=messages)
        msg = (r.get("message") or {})
        return (msg.get("content") or r.get("response") or "").strip()

    raise RuntimeError("No LLM backend available (OpenAI/Ollama)")

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

async def embed_one(text: str, model: str | None = None, timeout: int = 60) -> list[float]:
    m = model or getattr(settings, "EMBED_MODEL", "text-embedding-3-small")
    client = _client()
    def _do():
        return client.embeddings.create(model=m, input=text).data[0].embedding
    vec = await anyio.to_thread.run_sync(_do)
    return [float(x) for x in vec]

# async def schema_embed_one(text: str, model: str | None = None) -> list[float]:
#     base = (getattr(settings, "OLLAMA_API_BASE", None) or OLLAMA_API_BASE).rstrip("/")
#     mdl  = model or settings.VALID_EMBED_MODEL
#     async with httpx.AsyncClient(timeout=60) as cx:
#         r = await cx.post(f"{base}/api/embeddings", json={"model": mdl, "prompt": text})
#         r.raise_for_status()
#         data = r.json()
#         vec = data.get("embedding") or data.get("data") or data.get("vector")
#         if not isinstance(vec, list):
#             raise RuntimeError("Unexpected embeddings response")
#         # ensure floats
#         return [float(x) for x in vec]


def embed_many(texts: List[str], model: Optional[str] = None, timeout: int = 60) -> List[List[float]]:
    """
    Batched helper (simple loop). Keeps behavior predictable with local backends.
    """
    return [embed_one(t, model=model, timeout=timeout) for t in texts]
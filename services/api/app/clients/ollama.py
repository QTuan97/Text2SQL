from __future__ import annotations
import httpx
from ..config import settings

async def embed(model: str, text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as s:
        r = await s.post(f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                         json={"model": model, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]

async def generate(model: str, prompt: str) -> str:
    # lower temperature to reduce chit-chat
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1}}
    async with httpx.AsyncClient(timeout=120) as s:
        r = await s.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()["response"].strip()
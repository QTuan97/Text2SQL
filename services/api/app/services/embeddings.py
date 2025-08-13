from __future__ import annotations
from ..config import settings
from ..clients import ollama

async def embed_valid(text: str) -> list[float]:
    return await ollama.embed(settings.VALID_EMBED_MODEL, text)

async def embed_error(text: str) -> list[float]:
    return await ollama.embed(settings.ERROR_EMBED_MODEL, text)
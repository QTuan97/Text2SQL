from __future__ import annotations
from ..config import settings
from ..clients.ollama import generate

async def llm(prompt: str) -> str:
    return await generate(settings.GEN_MODEL, prompt)
from __future__ import annotations
import re

def make_chunks(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Rough, fast splitter: paragraph-aware with overlap."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paras:
        return []
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += "\n" + p
        else:
            chunks.append(buf)
            # start next with overlap tail from previous
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n" + p).strip()
    if buf:
        chunks.append(buf)
    return chunks
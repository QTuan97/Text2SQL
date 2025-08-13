from fastapi import APIRouter, BackgroundTasks
import time
from typing import List, Any
from ..schemas.common import AskIn
from ..services.embeddings import embed_valid
from ..services.retrieval import search_named
from ..services.generation import llm
from ..services.logger import log_request
from ..config import settings

router = APIRouter(prefix="/ask", tags=["ask"])

@router.post("")
async def ask(body: AskIn, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()

    # 1) Embed and search
    qvec = await embed_valid(body.question)
    candidates = search_named(settings.VALID_NAME, qvec, max(body.top_k * 3, 10))

    # 2) Min score filter (handle None)
    min_score = body.min_score if body.min_score is not None else float("-inf")
    hits = [p for p in candidates if (p.score or 0) >= min_score]

    # 3) Optional LLM rerank
    if body.rerank and hits:
        ranked: List[tuple[int, Any]] = []
        for p in hits:
            snippet = (p.payload or {}).get("text", "")[:500]
            score_txt = await llm(
                f"Question: {body.question}\n\nSnippet:\n{snippet}\n\n"
                "Give a single integer 0-100 for relevance of Snippet to Question. Return only the number."
            )
            try:
                s = int(''.join(ch for ch in score_txt if ch.isdigit()) or '0')
            except Exception:
                s = 0
            ranked.append((s, p))
        ranked.sort(reverse=True, key=lambda t: t[0])
        hits = [p for _, p in ranked]
    else:
        hits = sorted(hits, key=lambda p: (p.score or 0), reverse=True)

    # 4) Take top_k and cap context size
    selected = hits[: body.top_k]
    ctx_parts: List[str] = []
    total = 0
    for p in selected:
        t = (p.payload or {}).get("text", "")
        if not t:
            continue
        if total + len(t) > body.max_context_chars:
            t = t[: max(0, body.max_context_chars - total)]
        ctx_parts.append(t)
        total += len(t)
        if total >= body.max_context_chars:
            break

    ctx = "\n\n---\n\n".join(ctx_parts)

    # 5) Generate answer WITHOUT leaking 'prompt' outside its branch
    if ctx.strip():
        prompt = (
            'Answer only using the CONTEXT. If the answer is not in the context, say "I don\'t know" briefly.\n'
            f"CONTEXT:\n{ctx}\n\n"
            f"QUESTION: {body.question}\n"
            "Short, direct answer:"
        )
        answer = await llm(prompt)
    else:
        answer = "I don't know."

    out = {
        "answer": answer,
        "sources": [{"id": p.id, "score": p.score} for p in selected],
    }

    ms = int((time.perf_counter() - t0) * 1000)
    background_tasks.add_task(
        log_request, "/ask", "POST", 200, body.model_dump(), out, ms
    )
    return out
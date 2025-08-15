from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import init_app
from .routes import health, index, search, ask, text2sql, semantic, schema

def create_app() -> FastAPI:
    app = FastAPI(title="NL Stack API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Routers
    app.include_router(health.router)
    app.include_router(index.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(text2sql.router)
    app.include_router(semantic.router)
    app.include_router(schema.router)

    init_app(app)
    return app

app = create_app()
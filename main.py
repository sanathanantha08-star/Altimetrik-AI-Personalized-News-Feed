"""
main.py
--------
Application entry point.

Responsibilities:
  1. Wire FastAPI lifespan (DB connect/disconnect, index creation).
  2. Run the initial news ingest on startup.
  3. Schedule recurring news ingest via APScheduler.
  4. Start uvicorn.

All heavy singletons (EmbeddingService, RerankerService) are instantiated
once here and reused throughout the process lifetime.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.app import create_app
from app.repositories.news_article_repository import NewsArticleRepository
from app.services.embedding_service import EmbeddingService
from app.services.news_ingest_service import NewsIngestService
from core.config import get_settings
from core.logging import get_logger, setup_logging
from db.session import close_db, connect_db, get_client

setup_logging()
logger = get_logger(__name__)

# ── Singletons ─────────────────────────────────────────────────────────────────
_embedding_svc = EmbeddingService()
_scheduler = AsyncIOScheduler()


async def _run_ingest_job() -> None:
    """Background job: fetch news, embed, store."""
    settings = get_settings()
    try:
        db = get_client()[settings.mongodb_name]
        svc = NewsIngestService(
            repo=NewsArticleRepository(db),
            embedding_svc=_embedding_svc,
        )
        result = await svc.run()
        logger.info(
            "Scheduled ingest complete",
            fetched=result.articles_fetched,
            stored=result.articles_stored,
        )
    except Exception as exc:
        logger.error("Scheduled ingest failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("=== Application startup ===")
    await connect_db()

    # Ensure DB indexes
    db = get_client()[settings.mongodb_name]
    await NewsArticleRepository(db).ensure_indexes()

    # Initial news ingest on startup
    logger.info("Running initial news ingest …")
    await _run_ingest_job()

    # APScheduler: recurring ingest
    _scheduler.add_job(
        _run_ingest_job,
        trigger="interval",
        minutes=settings.scheduler_interval_minutes,
        id="news_ingest",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info(
        "APScheduler started",
        interval_minutes=settings.scheduler_interval_minutes,
    )

    yield  # ── Application runs ──────────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("=== Application shutdown ===")
    _scheduler.shutdown(wait=False)
    await close_db()


def build() -> FastAPI:
    app = create_app()
    app.router.lifespan_context = lifespan
    return app


app = build()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
        workers=1,  # single worker required for APScheduler shared state
    )

"""app/app.py — FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.routers import feed_router, preferences_router, scheduler_router
from core.config import get_settings
from core.exceptions import AppBaseException
from core.logging import RequestTracingMiddleware, get_logger, setup_logging

logger = get_logger(__name__)


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit_default],
    )

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        description=(
            "AI-powered personalised news feed. "
            "Uses hybrid BM25 + semantic retrieval, FlashRank reranking, "
            "query expansion, and Cohere LLM reasoning."
        ),
    )

    app.state.limiter = limiter

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)

    # ── Exception handlers ─────────────────────────────────────────────────────
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(AppBaseException)
    async def app_exception_handler(request: Request, exc: AppBaseException):
        logger.warning(
            "Application exception",
            path=request.url.path,
            detail=exc.detail,
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, **exc.extra},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected internal error occurred."},
        )

    # ── Routers ────────────────────────────────────────────────────────────────
    PREFIX = "/api/v1"
    app.include_router(preferences_router, prefix=PREFIX)
    app.include_router(scheduler_router, prefix=PREFIX)
    app.include_router(feed_router, prefix=PREFIX)

    # ── Health ─────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Liveness probe")
    async def health():
        return {"status": "ok", "version": settings.app_version}

    logger.info("Application created", env=settings.app_env)
    return app

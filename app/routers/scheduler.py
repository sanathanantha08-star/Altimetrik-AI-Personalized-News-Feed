"""
app/routers/scheduler.py
-------------------------
POST /api/v1/scheduler/run — manually triggers the news ingest job.

The scheduler also fires automatically on startup and every N minutes
via APScheduler (wired in main.py lifespan).
"""

from fastapi import APIRouter, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.repositories.news_article_repository import NewsArticleRepository
from app.schemas.user_preference import SchedulerStatusResponse
from app.services.embedding_service import EmbeddingService
from app.services.news_ingest_service import NewsIngestService
from db.session import get_db

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])


def _get_ingest_service(
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> NewsIngestService:
    return NewsIngestService(
        repo=NewsArticleRepository(db),
        embedding_svc=EmbeddingService(),
    )


@router.post(
    "/run",
    response_model=SchedulerStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Manually trigger the news ingest job",
)
async def run_scheduler(
    service: NewsIngestService = Depends(_get_ingest_service),
) -> SchedulerStatusResponse:
    return await service.run()

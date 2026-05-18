"""
app/routers/feed.py
--------------------
GET /api/v1/feed — personalized news feed using hybrid retrieval + reranking.
"""

from fastapi import APIRouter, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.repositories.news_article_repository import NewsArticleRepository
from app.repositories.user_preference_repository import UserPreferenceRepository
from app.schemas.user_preference import PersonalizedFeedResponse
from app.services.bm25_service import BM25Service
from app.services.cohere_llm_service import CohereLLMService
from app.services.embedding_service import EmbeddingService
from app.services.feed_service import FeedService
from app.services.query_expansion_service import QueryExpansionService
from app.services.reranker_service import RerankerService
from db.session import get_db

router = APIRouter(prefix="/feed", tags=["Feed"])

# Module-level singleton for the reranker (model loads once at import time)
_reranker_instance: RerankerService | None = None


def _get_reranker() -> RerankerService:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance


def _get_feed_service(
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> FeedService:
    return FeedService(
        pref_repo=UserPreferenceRepository(db),
        article_repo=NewsArticleRepository(db),
        embedding_svc=EmbeddingService(),
        bm25_svc=BM25Service(),
        reranker_svc=_get_reranker(),
        query_expansion_svc=QueryExpansionService(),
        cohere_llm_svc=CohereLLMService(),
    )


@router.get(
    "",
    response_model=PersonalizedFeedResponse,
    status_code=status.HTTP_200_OK,
    summary="Get personalised news feed (hybrid retrieval + reranking + LLM reasoning)",
)
async def get_feed(
    service: FeedService = Depends(_get_feed_service),
) -> PersonalizedFeedResponse:
    return await service.get_feed()

"""
app/repositories/news_article_repository.py
--------------------------------------------
Raw MongoDB queries for the news_articles collection.
Handles bulk upsert, full-collection fetch, and index management.
NO business logic here.
"""

from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReplaceOne

from app.models.news_article import NewsArticleDocument
from core.exceptions import DatabaseException
from core.logging import get_logger
from core.retries import db_retry

logger = get_logger(__name__)
COLLECTION = "news_articles"


class NewsArticleRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    async def ensure_indexes(self) -> None:
        """Create indexes on first use — idempotent."""
        try:
            # Standard index on published_at for sorting
            await self._col.create_index("published_at")
            logger.info("news_articles indexes ensured")
        except Exception as exc:
            logger.warning("Index creation warning", error=str(exc))

    @db_retry
    async def bulk_upsert(self, articles: List[NewsArticleDocument]) -> int:
        """Upsert a batch of articles. Returns number of documents modified/inserted."""
        if not articles:
            return 0
        ops = [
            ReplaceOne({"_id": a.id}, a.to_mongo(), upsert=True)
            for a in articles
        ]
        try:
            result = await self._col.bulk_write(ops, ordered=False)
            return result.upserted_count + result.modified_count
        except Exception as exc:
            logger.error("Bulk upsert failed", error=str(exc))
            raise DatabaseException(detail=str(exc)) from exc

    @db_retry
    async def get_all(self, limit: int = 500) -> List[NewsArticleDocument]:
        """Fetch all stored articles (capped at limit)."""
        try:
            cursor = self._col.find({}).sort("published_at", -1).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [NewsArticleDocument.from_mongo(d) for d in docs]
        except Exception as exc:
            logger.error("get_all articles failed", error=str(exc))
            raise DatabaseException(detail=str(exc)) from exc

    @db_retry
    async def count(self) -> int:
        try:
            return await self._col.count_documents({})
        except Exception as exc:
            raise DatabaseException(detail=str(exc)) from exc

    @db_retry
    async def get_by_ids(self, ids: List[str]) -> List[NewsArticleDocument]:
        try:
            cursor = self._col.find({"_id": {"$in": ids}})
            docs = await cursor.to_list(length=len(ids))
            return [NewsArticleDocument.from_mongo(d) for d in docs]
        except Exception as exc:
            raise DatabaseException(detail=str(exc)) from exc

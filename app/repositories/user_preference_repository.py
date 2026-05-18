"""app/repositories/user_preference_repository.py — Raw MongoDB queries for user prefs."""

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.user_preference import USER_PREFERENCE_DOC_ID, UserPreferenceDocument
from core.exceptions import DatabaseException
from core.logging import get_logger
from core.retries import db_retry

logger = get_logger(__name__)
COLLECTION = "user_preferences"


class UserPreferenceRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    @db_retry
    async def get(self) -> Optional[UserPreferenceDocument]:
        try:
            doc = await self._col.find_one({"_id": USER_PREFERENCE_DOC_ID})
            return UserPreferenceDocument.from_mongo(doc) if doc else None
        except Exception as exc:
            logger.error("DB get user_preference failed", error=str(exc))
            raise DatabaseException(detail=str(exc)) from exc

    @db_retry
    async def upsert(self, interests: list[str]) -> UserPreferenceDocument:
        now = datetime.now(timezone.utc)
        payload = {
            "_id": USER_PREFERENCE_DOC_ID,
            "interests": interests,
            "updated_at": now,
        }
        try:
            await self._col.replace_one(
                {"_id": USER_PREFERENCE_DOC_ID}, payload, upsert=True
            )
            return UserPreferenceDocument(**payload)
        except Exception as exc:
            logger.error("DB upsert user_preference failed", error=str(exc))
            raise DatabaseException(detail=str(exc)) from exc

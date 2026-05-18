"""app/services/user_preference_service.py — User preference business logic."""

from app.repositories.user_preference_repository import UserPreferenceRepository
from app.schemas.user_preference import UserPreferenceResponse
from core.exceptions import UserPreferenceNotFoundException, ValidationException
from core.logging import get_logger
from core.security import sanitize_interests

logger = get_logger(__name__)


class UserPreferenceService:
    def __init__(self, repo: UserPreferenceRepository) -> None:
        self._repo = repo

    async def get_preferences(self) -> UserPreferenceResponse:
        logger.info("Fetching user preferences")
        doc = await self._repo.get()
        if doc is None:
            raise UserPreferenceNotFoundException()
        logger.info("User preferences found", count=len(doc.interests))
        return UserPreferenceResponse(interests=doc.interests, updated_at=doc.updated_at)

    async def save_preferences(self, interests: list[str]) -> UserPreferenceResponse:
        logger.info("Saving user preferences", raw_count=len(interests))
        try:
            clean = sanitize_interests(interests)
        except ValueError as exc:
            raise ValidationException(detail=str(exc)) from exc

        if not clean:
            raise ValidationException(detail="At least one valid interest is required.")

        doc = await self._repo.upsert(clean)
        logger.info("User preferences saved", count=len(doc.interests))
        return UserPreferenceResponse(interests=doc.interests, updated_at=doc.updated_at)

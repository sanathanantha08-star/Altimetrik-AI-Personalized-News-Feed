"""
app/routers/preferences.py
---------------------------
GET  /api/v1/user-preference  — retrieve stored interests
POST /api/v1/user-preference  — create / replace interests

Routers call services ONLY. No business logic, no DB access.
"""

from fastapi import APIRouter, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.repositories.user_preference_repository import UserPreferenceRepository
from app.schemas.user_preference import UserPreferenceRequest, UserPreferenceResponse
from app.services.user_preference_service import UserPreferenceService
from db.session import get_db

router = APIRouter(prefix="/user-preference", tags=["User Preferences"])


# ── Dependency ────────────────────────────────────────────────────────────────

def _get_service(db: AsyncIOMotorDatabase = Depends(get_db)) -> UserPreferenceService:
    return UserPreferenceService(UserPreferenceRepository(db))


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=UserPreferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get saved user interests",
)
async def get_user_preference(
    service: UserPreferenceService = Depends(_get_service),
) -> UserPreferenceResponse:
    return await service.get_preferences()


@router.post(
    "",
    response_model=UserPreferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Save / update user interests",
)
async def post_user_preference(
    body: UserPreferenceRequest,
    service: UserPreferenceService = Depends(_get_service),
) -> UserPreferenceResponse:
    return await service.save_preferences(body.interests)

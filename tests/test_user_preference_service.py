"""tests/test_user_preference_service.py — Unit tests for preference service."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.models.user_preference import UserPreferenceDocument
from app.services.user_preference_service import UserPreferenceService
from core.exceptions import UserPreferenceNotFoundException, ValidationException


def _make_doc(interests=None):
    return UserPreferenceDocument(
        _id="singleton",
        interests=interests or ["AI", "cloud"],
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def repo():
    r = MagicMock()
    r.get = AsyncMock()
    r.upsert = AsyncMock()
    return r


@pytest.mark.asyncio
async def test_get_preferences_ok(repo):
    repo.get.return_value = _make_doc(["AI"])
    svc = UserPreferenceService(repo)
    result = await svc.get_preferences()
    assert "AI" in result.interests


@pytest.mark.asyncio
async def test_get_preferences_not_found(repo):
    repo.get.return_value = None
    svc = UserPreferenceService(repo)
    with pytest.raises(UserPreferenceNotFoundException):
        await svc.get_preferences()


@pytest.mark.asyncio
async def test_save_preferences_sanitizes(repo):
    repo.upsert.return_value = _make_doc(["AI", "cloud"])
    svc = UserPreferenceService(repo)
    result = await svc.save_preferences(["  AI ", "Cloud", "AI"])
    repo.upsert.assert_called_once()
    called_interests = repo.upsert.call_args[0][0]
    assert len(called_interests) == 2  # deduped


@pytest.mark.asyncio
async def test_save_preferences_rejects_invalid(repo):
    svc = UserPreferenceService(repo)
    with pytest.raises(ValidationException):
        await svc.save_preferences(["<script>hack</script>"])

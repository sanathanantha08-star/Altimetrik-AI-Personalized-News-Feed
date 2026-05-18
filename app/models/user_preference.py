"""app/models/user_preference.py — MongoDB document for single-user preferences."""

from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field

USER_PREFERENCE_DOC_ID = "singleton"


class UserPreferenceDocument(BaseModel):
    id: str = Field(default=USER_PREFERENCE_DOC_ID, alias="_id")
    interests: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}

    def to_mongo(self) -> dict:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict) -> "UserPreferenceDocument":
        return cls(**doc)

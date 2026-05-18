"""app/schemas/user_preference.py — HTTP request / response DTOs."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class UserPreferenceRequest(BaseModel):
    interests: List[str] = Field(
        ...,
        min_length=1,
        description="Topics the user is interested in.",
        examples=[["AI", "cloud", "startups"]],
    )

    @field_validator("interests", mode="before")
    @classmethod
    def non_empty(cls, v: List[str]) -> List[str]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Each interest must be a non-empty string.")
        return v


# ── Response ──────────────────────────────────────────────────────────────────

class UserPreferenceResponse(BaseModel):
    interests: List[str]
    updated_at: datetime


# ── News Article ──────────────────────────────────────────────────────────────

class NewsArticleOut(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    url: str
    url_to_image: Optional[str] = None
    source_name: str
    published_at: Optional[str] = None
    author: Optional[str] = None


# ── Feed ──────────────────────────────────────────────────────────────────────

class FeedItemResponse(BaseModel):
    article: NewsArticleOut
    score: float = Field(description="Combined hybrid retrieval score (0–1).")
    semantic_score: float
    bm25_score: float
    rerank_score: Optional[float] = None
    rank: int = Field(description="Final rank (1 = most relevant).")
    reason: str = Field(description="Personalisation reasoning from Cohere LLM.")
    matched_interests: List[str] = Field(default_factory=list)


class PersonalizedFeedResponse(BaseModel):
    interests: List[str]
    expanded_queries: List[str]
    total_results: int
    items: List[FeedItemResponse]


# ── Scheduler ─────────────────────────────────────────────────────────────────

class SchedulerStatusResponse(BaseModel):
    status: str
    articles_fetched: int
    articles_stored: int
    message: str

"""
app/models/news_article.py
--------------------------
MongoDB document model for the news_articles vector store collection.

Each document stores:
  - raw article data  (for frontend display)
  - dense embedding   (for semantic / cosine similarity search)
  - bm25_tokens       (pre-tokenised list for BM25 retrieval)
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class NewsArticleDocument(BaseModel):
    # MongoDB document id — we use the NewsAPI article URL as a stable dedup key.
    id: str = Field(..., alias="_id")

    # ── Raw fields (frontend display) ─────────────────────────────────────────
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    url: str
    url_to_image: Optional[str] = None
    source_name: str = "Unknown"
    published_at: Optional[str] = None
    author: Optional[str] = None

    # ── Vector store fields ───────────────────────────────────────────────────
    # Dense vector from Cohere embed-english-v3.0 (1024-dim)
    embedding: List[float] = Field(default_factory=list)

    # BM25 token list — title + description tokenised & lowercased
    bm25_tokens: List[str] = Field(default_factory=list)

    # Full text used for embedding (title + description + content snippet)
    embed_text: str = ""

    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}

    def to_mongo(self) -> dict:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict) -> "NewsArticleDocument":
        return cls(**doc)

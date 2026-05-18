"""
app/services/news_ingest_service.py
--------------------------------------
Fetches top-headline articles from NewsAPI, generates Cohere embeddings,
tokenises for BM25, and bulk-upserts everything into MongoDB.

Called:
  1. At application startup (lifespan).
  2. On a recurring APScheduler job every N minutes.
  3. Via POST /api/v1/scheduler/run (manual trigger).
"""

from typing import List
from urllib.parse import urlencode, urlunparse

import httpx

from app.models.news_article import NewsArticleDocument
from app.repositories.news_article_repository import NewsArticleRepository
from app.schemas.user_preference import SchedulerStatusResponse
from app.services.bm25_service import tokenize
from app.services.embedding_service import EmbeddingService
from core.config import get_settings
from core.exceptions import NewsAPIException
from core.logging import get_logger
from core.retries import external_api_retry

logger = get_logger(__name__)


class NewsIngestService:
    def __init__(
        self,
        repo: NewsArticleRepository,
        embedding_svc: EmbeddingService,
    ) -> None:
        self._repo = repo
        self._embed = embedding_svc
        self._settings = get_settings()

    async def run(self) -> SchedulerStatusResponse:
        logger.info("News ingest job started")

        raw_articles = await self._fetch_from_api()
        if not raw_articles:
            logger.warning("NewsAPI returned 0 articles")
            return SchedulerStatusResponse(
                status="ok",
                articles_fetched=0,
                articles_stored=0,
                message="NewsAPI returned no articles.",
            )

        docs = await self._build_documents(raw_articles)
        stored = await self._repo.bulk_upsert(docs)

        logger.info(
            "News ingest complete",
            fetched=len(raw_articles),
            stored=stored,
        )
        return SchedulerStatusResponse(
            status="ok",
            articles_fetched=len(raw_articles),
            articles_stored=stored,
            message=f"Ingested {stored} articles successfully.",
        )

    @external_api_retry
    async def _fetch_from_api(self) -> List[dict]:
        params = {
            "country": self._settings.news_fetch_country,
            "apiKey": self._settings.news_api_key,
            "pageSize": 100,
        }
        url = f"{self._settings.news_api_top_headlines_endpoint}?{urlencode(params)}"
        logger.info("Fetching top headlines", country=self._settings.news_fetch_country)

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            raise NewsAPIException(
                detail=f"NewsAPI returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        if data.get("status") != "ok":
            raise NewsAPIException(detail=data.get("message", "NewsAPI error"))

        articles = [a for a in data.get("articles", []) if a.get("url")]
        logger.info("NewsAPI raw articles received", count=len(articles))
        return articles

    async def _build_documents(
        self, raw_articles: List[dict]
    ) -> List[NewsArticleDocument]:
        # Build embed text for each article
        texts: List[str] = []
        for a in raw_articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            content_snip = (a.get("content") or "")[:300]
            embed_text = f"{title}. {desc}. {content_snip}".strip()
            texts.append(embed_text)

        # Batch embed
        embeddings = await self._embed.embed_documents(texts)

        docs: List[NewsArticleDocument] = []
        for i, a in enumerate(raw_articles):
            url = a.get("url", "")
            title = a.get("title") or "Untitled"
            desc = a.get("description") or ""
            content = a.get("content") or ""
            embed_text = texts[i]
            embedding = embeddings[i] if i < len(embeddings) else []

            doc = NewsArticleDocument(
                _id=url,
                title=title,
                description=desc,
                content=content,
                url=url,
                url_to_image=a.get("urlToImage"),
                source_name=(a.get("source") or {}).get("name", "Unknown"),
                published_at=a.get("publishedAt"),
                author=a.get("author"),
                embedding=embedding,
                bm25_tokens=tokenize(embed_text),
                embed_text=embed_text,
            )
            docs.append(doc)

        return docs

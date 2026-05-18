"""tests/test_feed_service.py — Unit tests for the feed service pipeline."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.news_article import NewsArticleDocument
from app.models.user_preference import UserPreferenceDocument
from app.services.bm25_service import BM25Service, tokenize
from app.services.feed_service import FeedService
from core.exceptions import UserPreferenceNotFoundException


# ── BM25 tests ────────────────────────────────────────────────────────────────

def test_tokenize_removes_stop_words():
    tokens = tokenize("The AI startup is raising funds for cloud computing")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "ai" in tokens
    assert "startup" in tokens


def test_bm25_search_returns_ranked_results():
    svc = BM25Service()
    corpus = [
        ("doc1", tokenize("AI startup raises funding machine learning")),
        ("doc2", tokenize("Football team wins championship final")),
        ("doc3", tokenize("Cloud security trends enterprise 2026")),
    ]
    svc.build_index(corpus)
    results = svc.search("AI startup", top_k=3)
    assert results[0][0] == "doc1"
    assert results[0][1] > results[1][1]


def test_bm25_empty_corpus():
    svc = BM25Service()
    results = svc.search("AI", top_k=5)
    assert results == []


# ── Feed service tests ────────────────────────────────────────────────────────

def _make_article(doc_id: str, title: str, desc: str = "") -> NewsArticleDocument:
    from app.services.bm25_service import tokenize as tok
    embed_text = f"{title}. {desc}"
    return NewsArticleDocument(
        _id=doc_id,
        title=title,
        description=desc,
        url=f"https://example.com/{doc_id}",
        source_name="Test",
        embedding=[0.1] * 1024,
        bm25_tokens=tok(embed_text),
        embed_text=embed_text,
    )


@pytest.fixture
def mock_pref_repo():
    repo = MagicMock()
    repo.get = AsyncMock(
        return_value=UserPreferenceDocument(
            _id="singleton",
            interests=["AI", "cloud", "startups"],
            updated_at=datetime.now(timezone.utc),
        )
    )
    return repo


@pytest.fixture
def mock_article_repo():
    repo = MagicMock()
    repo.get_all = AsyncMock(
        return_value=[
            _make_article("a1", "New AI startup raises funding", "AI and machine learning"),
            _make_article("a2", "Football team wins final", "Sports championship"),
            _make_article("a3", "Cloud security trends 2026", "Enterprise cloud security"),
        ]
    )
    return repo


@pytest.mark.asyncio
async def test_feed_raises_when_no_preferences():
    pref_repo = MagicMock()
    pref_repo.get = AsyncMock(return_value=None)

    svc = FeedService(
        pref_repo=pref_repo,
        article_repo=MagicMock(),
        embedding_svc=MagicMock(),
        bm25_svc=BM25Service(),
        reranker_svc=MagicMock(),
        query_expansion_svc=MagicMock(),
        cohere_llm_svc=MagicMock(),
    )
    with pytest.raises(UserPreferenceNotFoundException):
        await svc.get_feed()


@pytest.mark.asyncio
async def test_feed_pipeline_returns_feed(mock_pref_repo, mock_article_repo):
    from app.services.embedding_service import EmbeddingService
    from app.services.bm25_service import BM25Service
    from app.services.reranker_service import RerankerService
    from app.services.query_expansion_service import QueryExpansionService
    from app.services.cohere_llm_service import CohereLLMService

    embed_svc = MagicMock(spec=EmbeddingService)
    embed_svc.embed_queries = AsyncMock(return_value=[[0.1] * 1024])
    embed_svc.batch_cosine_similarity = MagicMock(return_value=[0.8, 0.1, 0.6])

    reranker = MagicMock(spec=RerankerService)
    reranker.rerank = MagicMock(return_value=[("a1", 0.9), ("a3", 0.7)])

    expander = MagicMock(spec=QueryExpansionService)
    expander.expand = AsyncMock(return_value=["AI", "cloud", "startups", "machine learning"])

    llm = MagicMock(spec=CohereLLMService)
    llm.generate_reasoning = AsyncMock(return_value={
        "a1": {"reason": "Matches AI and startups", "matched_interests": ["AI", "startups"], "score": 2},
        "a3": {"reason": "Matches cloud", "matched_interests": ["cloud"], "score": 1},
    })

    svc = FeedService(
        pref_repo=mock_pref_repo,
        article_repo=mock_article_repo,
        embedding_svc=embed_svc,
        bm25_svc=BM25Service(),
        reranker_svc=reranker,
        query_expansion_svc=expander,
        cohere_llm_svc=llm,
    )

    result = await svc.get_feed()

    assert result.total_results > 0
    assert result.items[0].rank == 1
    assert result.items[0].article.id == "a1"
    assert result.items[0].score >= 0
    assert "AI" in result.items[0].matched_interests
    assert result.items[1].article.id == "a3"
    assert result.items[1].rank == 2

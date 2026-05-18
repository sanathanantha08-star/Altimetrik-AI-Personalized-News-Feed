"""
app/services/feed_service.py
------------------------------
Personalized feed pipeline:

1.  Load user interests
2.  Query expansion (Cohere LLM)
3.  Hybrid retrieval:
      a. Semantic search  (cosine similarity on Cohere embeddings)
      b. BM25 lexical search
      c. Reciprocal Rank Fusion (RRF) to merge both lists
4.  Filter by similarity threshold
5.  Cross-encoder reranking (FlashRank)
6.  Cohere LLM personalisation reasoning
7.  Final sort: rerank_score desc → llm_score desc (tie-break) → return

Data structures:
  - HashMap (dict) for O(1) interest lookups
  - Heap / sorted list for ranking pipeline
"""

import heapq
from typing import Dict, List, Optional, Tuple

from app.models.news_article import NewsArticleDocument
from app.repositories.news_article_repository import NewsArticleRepository
from app.repositories.user_preference_repository import UserPreferenceRepository
from app.schemas.user_preference import (
    FeedItemResponse,
    NewsArticleOut,
    PersonalizedFeedResponse,
)
from app.services.bm25_service import BM25Service, tokenize
from app.services.cohere_llm_service import CohereLLMService
from app.services.embedding_service import EmbeddingService
from app.services.query_expansion_service import QueryExpansionService
from app.services.reranker_service import RerankerService
from core.config import get_settings
from core.exceptions import NewsArticleNotFoundException, UserPreferenceNotFoundException
from core.logging import get_logger

logger = get_logger(__name__)

# Reciprocal Rank Fusion constant — standard k=60
_RRF_K = 60


class FeedService:
    def __init__(
        self,
        pref_repo: UserPreferenceRepository,
        article_repo: NewsArticleRepository,
        embedding_svc: EmbeddingService,
        bm25_svc: BM25Service,
        reranker_svc: RerankerService,
        query_expansion_svc: QueryExpansionService,
        cohere_llm_svc: CohereLLMService,
    ) -> None:
        self._pref_repo = pref_repo
        self._article_repo = article_repo
        self._embed = embedding_svc
        self._bm25 = bm25_svc
        self._reranker = reranker_svc
        self._expander = query_expansion_svc
        self._llm = cohere_llm_svc
        self._settings = get_settings()

    async def get_feed(self) -> PersonalizedFeedResponse:
        # ── 1. Load preferences ───────────────────────────────────────────────
        pref = await self._pref_repo.get()
        if pref is None or not pref.interests:
            raise UserPreferenceNotFoundException()

        interests: List[str] = pref.interests
        # HashMap for O(1) interest membership lookups later
        interests_map: Dict[str, bool] = {i.lower(): True for i in interests}

        logger.info("Feed generation started", interests=interests)

        # ── 2. Load articles from DB ──────────────────────────────────────────
        all_articles = await self._article_repo.get_all(limit=500)
        if not all_articles:
            raise NewsArticleNotFoundException()

        logger.info("Articles loaded from DB", count=len(all_articles))

        # ── 3. Query expansion ────────────────────────────────────────────────
        expanded_queries = await self._expander.expand(interests)
        combined_query = " ".join(expanded_queries)
        logger.info("Expanded query", terms=len(expanded_queries))

        # ── 4. Semantic search ────────────────────────────────────────────────
        query_embeddings = await self._embed.embed_queries([combined_query])
        query_vec = query_embeddings[0]

        doc_embeddings = [a.embedding for a in all_articles]
        semantic_scores_raw = self._embed.batch_cosine_similarity(
            query_vec, doc_embeddings
        )

        # Map doc_id → semantic_score, filter by threshold
        threshold = self._settings.similarity_threshold
        semantic_map: Dict[str, float] = {}
        for art, score in zip(all_articles, semantic_scores_raw):
            if score >= threshold:
                semantic_map[art.id] = score

        semantic_ranked = sorted(semantic_map.items(), key=lambda x: x[1], reverse=True)
        semantic_top = semantic_ranked[: self._settings.top_k_semantic]
        logger.info("Semantic candidates", count=len(semantic_top))

        # ── 5. BM25 search ────────────────────────────────────────────────────
        corpus = [(a.id, a.bm25_tokens) for a in all_articles if a.bm25_tokens]
        self._bm25.build_index(corpus)
        bm25_results = self._bm25.search(combined_query, top_k=self._settings.top_k_bm25)
        bm25_map: Dict[str, float] = dict(bm25_results)
        logger.info("BM25 candidates", count=len(bm25_map))

        # ── 6. Reciprocal Rank Fusion (Hybrid merge) ──────────────────────────
        rrf_scores: Dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(semantic_top, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)

        bm25_sorted = sorted(bm25_map.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(bm25_sorted, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)

        # Normalise RRF scores to [0, 1]
        if rrf_scores:
            max_rrf = max(rrf_scores.values())
            rrf_scores = {k: v / max_rrf for k, v in rrf_scores.items()}

        # Use a max-heap (negate for Python's min-heap) to get top candidates
        top_candidates = heapq.nlargest(
            self._settings.top_k_rerank * 2,
            rrf_scores.items(),
            key=lambda x: x[1],
        )
        candidate_ids = [doc_id for doc_id, _ in top_candidates]
        logger.info("Hybrid RRF candidates", count=len(candidate_ids))

        # ── 7. Build candidate article lookup ─────────────────────────────────
        article_lookup: Dict[str, NewsArticleDocument] = {
            a.id: a for a in all_articles if a.id in set(candidate_ids)
        }

        # ── 8. FlashRank cross-encoder reranking ──────────────────────────────
        passages = [
            (doc_id, article_lookup[doc_id].embed_text)
            for doc_id in candidate_ids
            if doc_id in article_lookup
        ]
        reranked = self._reranker.rerank(
            query=combined_query,
            passages=passages,
            top_k=self._settings.top_k_rerank,
        )
        rerank_map: Dict[str, float] = dict(reranked)
        final_ids = [doc_id for doc_id, _ in reranked]
        logger.info("Reranked results", count=len(final_ids))

        # ── 9. Cohere LLM personalisation reasoning ───────────────────────────
        articles_for_llm = [
            {
                "id": doc_id,
                "title": article_lookup[doc_id].title,
                "description": article_lookup[doc_id].description or "",
            }
            for doc_id in final_ids
            if doc_id in article_lookup
        ]
        reasoning_map = await self._llm.generate_reasoning(
            interests=interests,
            articles=articles_for_llm,
        )

        # ── 10. Build final ranked list with tie-breaking ─────────────────────
        # Primary sort: rerank_score desc
        # Tie-break: llm interest_match_count desc
        ranked_items = []
        for doc_id in final_ids:
            art = article_lookup.get(doc_id)
            if not art:
                continue

            semantic_score = semantic_map.get(doc_id, 0.0)
            bm25_score = bm25_map.get(doc_id, 0.0)
            hybrid_score = rrf_scores.get(doc_id, 0.0)
            rerank_score = rerank_map.get(doc_id, 0.0)
            reasoning = reasoning_map.get(doc_id, {})
            llm_score = int(reasoning.get("score", 0))

            ranked_items.append(
                (
                    -rerank_score,       # primary sort key (negated for ascending sort)
                    -llm_score,          # tie-break
                    doc_id,
                    art,
                    semantic_score,
                    bm25_score,
                    hybrid_score,
                    rerank_score,
                    reasoning,
                )
            )

        ranked_items.sort(key=lambda x: (x[0], x[1]))

        # ── 11. Assemble response ─────────────────────────────────────────────
        feed_items: List[FeedItemResponse] = []
        for rank, item in enumerate(ranked_items, start=1):
            (_, _, doc_id, art, sem_sc, bm_sc, hyb_sc, rer_sc, reasoning) = item
            feed_items.append(
                FeedItemResponse(
                    article=NewsArticleOut(
                        id=art.id,
                        title=art.title,
                        description=art.description,
                        url=art.url,
                        url_to_image=art.url_to_image,
                        source_name=art.source_name,
                        published_at=art.published_at,
                        author=art.author,
                    ),
                    score=round(hyb_sc, 4),
                    semantic_score=round(sem_sc, 4),
                    bm25_score=round(bm_sc, 4),
                    rerank_score=round(rer_sc, 4),
                    rank=rank,
                    reason=reasoning.get("reason", "Matches your interests."),
                    matched_interests=reasoning.get("matched_interests", []),
                )
            )

        logger.info("Feed assembled", items=len(feed_items))
        return PersonalizedFeedResponse(
            interests=interests,
            expanded_queries=expanded_queries,
            total_results=len(feed_items),
            items=feed_items,
        )

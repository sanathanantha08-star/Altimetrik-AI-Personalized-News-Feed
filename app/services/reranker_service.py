"""
app/services/reranker_service.py
----------------------------------
Cross-encoder reranking using FlashRank (ms-marco-MiniLM-L-12-v2).

FlashRank is a lightweight open-source reranker that runs fully locally —
no API key, no network call.  It scores (query, passage) pairs and returns
a relevance score in [0, 1].
"""

from typing import List, Tuple

from flashrank import Ranker, RerankRequest

from core.logging import get_logger

logger = get_logger(__name__)

# Model choices: nano (fastest), small (balanced), large (best quality)
_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


class RerankerService:
    """
    Wraps FlashRank for cross-encoder reranking.
    The ranker is loaded once at construction time.
    """

    def __init__(self) -> None:
        logger.info("Loading FlashRank reranker", model=_FLASHRANK_MODEL)
        self._ranker = Ranker(model_name=_FLASHRANK_MODEL, cache_dir="/tmp/flashrank")
        logger.info("FlashRank reranker ready")

    def rerank(
        self,
        query: str,
        passages: List[Tuple[str, str]],  # [(doc_id, text)]
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Returns [(doc_id, rerank_score)] sorted by reranker score descending.

        passages: list of (doc_id, passage_text)
        """
        if not passages:
            return []

        rerank_input = RerankRequest(
            query=query,
            passages=[{"id": doc_id, "text": text} for doc_id, text in passages],
        )
        results = self._ranker.rerank(rerank_input)
        ranked = sorted(results, key=lambda r: r["score"], reverse=True)
        return [(r["id"], float(r["score"])) for r in ranked[:top_k]]

"""
app/services/bm25_service.py
-----------------------------
BM25 lexical retrieval over the in-memory token corpus.

Uses rank_bm25.BM25Okapi — a pure-Python, dependency-light BM25
implementation that works well for corpora up to ~100 k documents.
"""

import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from core.logging import get_logger

logger = get_logger(__name__)

# Minimal English stop-words to reduce noise
_STOP_WORDS = frozenset(
    "a an the and or but in on at to of for is are was were be been being "
    "have has had do does did will would could should may might shall can "
    "it its this that these those with from by as".split()
)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class BM25Service:
    """
    Builds a BM25 index from a list of (doc_id, token_list) pairs
    and exposes a `search` method returning (doc_id, score) tuples.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self, corpus: List[Tuple[str, List[str]]]) -> None:
        """
        corpus: list of (doc_id, token_list)
        """
        if not corpus:
            logger.warning("BM25 build_index called with empty corpus")
            return
        self._ids = [doc_id for doc_id, _ in corpus]
        token_lists = [tokens for _, tokens in corpus]
        self._bm25 = BM25Okapi(token_lists)
        logger.info("BM25 index built", docs=len(self._ids))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Returns [(doc_id, normalised_bm25_score)] sorted descending.
        Scores are normalised to [0, 1] by dividing by the max score.
        """
        if self._bm25 is None or not self._ids:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalised = [s / max_score for s in raw_scores]

        ranked = sorted(
            zip(self._ids, normalised), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

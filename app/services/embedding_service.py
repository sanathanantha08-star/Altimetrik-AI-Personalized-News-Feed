"""
app/services/embedding_service.py
-----------------------------------
Generates dense embeddings via Cohere embed-english-v3.0 and exposes
cosine-similarity utilities.

All Cohere calls are wrapped with retry+back-off.
"""

import asyncio
from typing import List

import cohere
import numpy as np

from core.config import get_settings
from core.exceptions import EmbeddingException
from core.logging import get_logger
from core.retries import cohere_retry

logger = get_logger(__name__)

# Cohere embed input types
_SEARCH_DOCUMENT = "search_document"
_SEARCH_QUERY = "search_query"


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self._model = settings.cohere_embed_model

    @cohere_retry
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts (for indexing)."""
        if not texts:
            return []
        logger.info("Embedding documents", count=len(texts), model=self._model)
        try:
            # Cohere allows max 96 texts per call — chunk if needed
            all_embeddings: List[List[float]] = []
            for chunk in _chunked(texts, 96):
                resp = await self._client.embed(
                    texts=chunk,
                    model=self._model,
                    input_type=_SEARCH_DOCUMENT,
                )
                all_embeddings.extend(resp.embeddings)
            return all_embeddings
        except Exception as exc:
            logger.error("Document embedding failed", error=str(exc))
            raise EmbeddingException(detail=str(exc)) from exc

    @cohere_retry
    async def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of query strings (for retrieval)."""
        if not texts:
            return []
        logger.info("Embedding queries", count=len(texts), model=self._model)
        try:
            all_embeddings: List[List[float]] = []
            for chunk in _chunked(texts, 96):
                resp = await self._client.embed(
                    texts=chunk,
                    model=self._model,
                    input_type=_SEARCH_QUERY,
                )
                all_embeddings.extend(resp.embeddings)
            return all_embeddings
        except Exception as exc:
            logger.error("Query embedding failed", error=str(exc))
            raise EmbeddingException(detail=str(exc)) from exc

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(
        query_vec: List[float], doc_vecs: List[List[float]]
    ) -> List[float]:
        """Vectorised cosine similarity of one query against many docs."""
        q = np.array(query_vec, dtype=np.float32)
        D = np.array(doc_vecs, dtype=np.float32)
        norms = np.linalg.norm(D, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [0.0] * len(doc_vecs)
        safe_norms = np.where(norms == 0, 1e-10, norms)
        sims = D @ q / (safe_norms * q_norm)
        return sims.tolist()


def _chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

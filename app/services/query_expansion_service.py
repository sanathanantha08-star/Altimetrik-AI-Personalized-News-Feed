"""
app/services/query_expansion_service.py
-----------------------------------------
Query Expansion: enriches the user's interest terms with semantically
related synonyms and phrases so that hybrid retrieval finds more
relevant articles.

Uses Cohere command-r-plus to generate expansions, falls back to
the original terms on any failure.
"""

from typing import List

import cohere

from core.config import get_settings
from core.logging import get_logger
from core.retries import cohere_retry

logger = get_logger(__name__)

_EXPANSION_PROMPT = """\
You are a search query expansion assistant.
Given the user interests listed below, generate 3–5 related search terms or 
short phrases for EACH interest that would help retrieve relevant news articles.
Return ONLY a flat JSON array of strings (no nesting, no explanation).

User interests: {interests}

Example output format:
["term1", "term2", "term3", ...]
"""


class QueryExpansionService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self._model = settings.cohere_model

    @cohere_retry
    async def expand(self, interests: List[str]) -> List[str]:
        """
        Returns original interests + expanded terms deduplicated.
        Falls back to originals on any error.
        """
        if not interests:
            return []

        prompt = _EXPANSION_PROMPT.format(interests=", ".join(interests))
        logger.info("Expanding queries", interests=interests)

        try:
            resp = await self._client.chat(
                model=self._model,
                message=prompt,
                temperature=0.3,
                max_tokens=400,
            )
            raw = resp.text.strip()
            expanded = _parse_json_array(raw)
            # Combine originals + expansions, deduplicated (preserve order)
            seen = set(i.lower() for i in interests)
            result = list(interests)
            for term in expanded:
                term = term.strip()
                if term and term.lower() not in seen:
                    seen.add(term.lower())
                    result.append(term)
            logger.info("Query expansion done", original=len(interests), expanded=len(result))
            return result
        except Exception as exc:
            logger.warning("Query expansion failed, using originals", error=str(exc))
            return interests


def _parse_json_array(text: str) -> List[str]:
    import json, re
    # Strip markdown code fences if present
    text = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()
    data = json.loads(text)
    if isinstance(data, list):
        return [str(x) for x in data]
    return []

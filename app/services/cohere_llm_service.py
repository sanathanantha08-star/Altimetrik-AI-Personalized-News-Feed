"""
app/services/cohere_llm_service.py
------------------------------------
Uses Cohere command-r-plus-08-2024 to generate:
  - Personalisation reasoning for each ranked article
  - Matched interests for tie-breaking
  - Final JSON output conforming to FeedItemResponse
"""

import json
import re
from typing import Any, Dict, List

import cohere

from core.config import get_settings
from core.exceptions import CohereException
from core.logging import get_logger
from core.retries import cohere_retry

logger = get_logger(__name__)

_REASONING_PROMPT = """\
You are a news personalisation engine.

User interests: {interests}

Below is a ranked list of news articles retrieved for this user.
For each article, produce a JSON object with these exact keys:
  - "article_id": the article id provided
  - "reason": one concise sentence explaining why it matches the user's interests
  - "matched_interests": a JSON array of interests from the list above that match
  - "score": integer count of matched interests (used for tie-breaking)

Return ONLY a valid JSON array of these objects, no extra text, no markdown fences.

Articles (id | title | description):
{articles_block}
"""


class CohereLLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self._model = settings.cohere_model

    @cohere_retry
    async def generate_reasoning(
        self,
        interests: List[str],
        articles: List[Dict[str, Any]],  # [{"id":..,"title":..,"description":..}]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dict keyed by article_id:
          {
            article_id: {
              "reason": str,
              "matched_interests": List[str],
              "score": int,
            }
          }
        Falls back to a default payload for each article on any failure.
        """
        if not articles:
            return {}

        articles_block = "\n".join(
            f"{a['id']} | {a['title']} | {a.get('description', '')}"
            for a in articles
        )
        prompt = _REASONING_PROMPT.format(
            interests=", ".join(interests),
            articles_block=articles_block,
        )

        logger.info(
            "Calling Cohere for personalisation reasoning",
            model=self._model,
            articles=len(articles),
        )

        try:
            resp = await self._client.chat(
                model=self._model,
                message=prompt,
                temperature=0.2,
                max_tokens=2000,
            )
            raw = resp.text.strip()
            data = _parse_json_array(raw)
            result: Dict[str, Dict[str, Any]] = {}
            for item in data:
                aid = str(item.get("article_id", ""))
                if aid:
                    result[aid] = {
                        "reason": item.get("reason", "Matches user interests."),
                        "matched_interests": item.get("matched_interests", []),
                        "score": int(item.get("score", 0)),
                    }
            return result
        except Exception as exc:
            logger.error("Cohere reasoning failed", error=str(exc))
            # Graceful degradation — return defaults for all articles
            return {
                a["id"]: {
                    "reason": "Matches your interests.",
                    "matched_interests": [],
                    "score": 0,
                }
                for a in articles
            }


def _parse_json_array(text: str) -> List[Dict]:
    text = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()
    data = json.loads(text)
    return data if isinstance(data, list) else []

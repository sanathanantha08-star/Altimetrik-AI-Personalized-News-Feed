"""core/exceptions.py — Domain exception hierarchy."""

from typing import Any, Dict, Optional


class AppBaseException(Exception):
    status_code: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(
        self,
        detail: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.detail = detail or self.__class__.detail
        self.extra = extra or {}
        super().__init__(self.detail)


class ValidationException(AppBaseException):
    status_code = 422
    detail = "Validation failed."


class NotFoundException(AppBaseException):
    status_code = 404
    detail = "Resource not found."


class UserPreferenceNotFoundException(NotFoundException):
    detail = "User preferences not found. POST /api/v1/user-preference first."


class NewsArticleNotFoundException(NotFoundException):
    detail = "No news articles found in the store."


class AlreadyExistsException(AppBaseException):
    status_code = 409
    detail = "Resource already exists."


class RateLimitException(AppBaseException):
    status_code = 429
    detail = "Too many requests. Please slow down."


class ExternalServiceException(AppBaseException):
    status_code = 502
    detail = "External service error."


class NewsAPIException(ExternalServiceException):
    detail = "Failed to fetch news from NewsAPI."


class CohereException(ExternalServiceException):
    detail = "Cohere API call failed."


class DatabaseException(AppBaseException):
    status_code = 503
    detail = "Database operation failed."


class EmbeddingException(AppBaseException):
    status_code = 502
    detail = "Failed to generate embeddings."

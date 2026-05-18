"""core/retries.py — Tenacity-based retry decorators with exponential back-off."""

import logging
from functools import wraps
from typing import Callable, Tuple, Type

from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_logger = logging.getLogger(__name__)


def with_retry(
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    reraise_as: Type[Exception],
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @retry(
            reraise=False,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_on),
            before_sleep=before_sleep_log(_logger, logging.WARNING),
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except RetryError as exc:
                raise reraise_as(
                    f"Retry exhausted after {max_attempts} attempts: {exc}"
                ) from exc

        return wrapper

    return decorator


def external_api_retry(func: Callable) -> Callable:
    """3-attempt exponential back-off for external HTTP calls."""
    from core.exceptions import NewsAPIException

    return with_retry(
        max_attempts=3,
        min_wait=1,
        max_wait=8,
        reraise_as=NewsAPIException,
    )(func)


def cohere_retry(func: Callable) -> Callable:
    """3-attempt exponential back-off for Cohere API calls."""
    from core.exceptions import CohereException

    return with_retry(
        max_attempts=3,
        min_wait=1,
        max_wait=8,
        reraise_as=CohereException,
    )(func)


def db_retry(func: Callable) -> Callable:
    """2-attempt retry for transient MongoDB errors."""
    from core.exceptions import DatabaseException

    return with_retry(
        max_attempts=2,
        min_wait=0.5,
        max_wait=3,
        reraise_as=DatabaseException,
    )(func)

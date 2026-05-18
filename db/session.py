"""db/session.py — Async MongoDB client lifecycle and get_db FastAPI dependency."""

from typing import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

_client: AsyncIOMotorClient | None = None


async def connect_db() -> None:
    global _client
    settings = get_settings()
    host_hint = settings.mongodb_uri.split("@")[-1]
    logger.info("Connecting to MongoDB", host=host_hint)
    _client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5_000,
        connectTimeoutMS=5_000,
        socketTimeoutMS=15_000,
    )
    await _client.admin.command("ping")
    logger.info("MongoDB connected", db=settings.mongodb_name)


async def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")


def get_client() -> AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("DB client not initialised. Call connect_db() first.")
    return _client


async def get_db() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """FastAPI dependency — yields the application database."""
    settings = get_settings()
    yield get_client()[settings.mongodb_name]

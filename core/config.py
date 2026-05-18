from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── News API ──────────────────────────────────────────────────────────────
    news_api_key: str
    news_api_url: str = "https://newsapi.org/v2"
    news_api_top_headlines_endpoint: str = "https://newsapi.org/v2/top-headlines"
    news_fetch_country: str = "us"

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongodb_uri: str
    mongodb_name: str

    # ── Cohere ────────────────────────────────────────────────────────────────
    cohere_api_key: str
    cohere_model: str = "command-r-plus-08-2024"
    cohere_embed_model: str = "embed-english-v3.0"

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    app_title: str = "AI News Feed Personalizer"
    app_version: str = "1.0.0"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_default: str = "60/minute"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: List[str] = ["http://localhost:3000"]

    # ── Feed / Retrieval Tuning ───────────────────────────────────────────────
    similarity_threshold: float = 0.30
    top_k_semantic: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 10
    scheduler_interval_minutes: int = 30

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

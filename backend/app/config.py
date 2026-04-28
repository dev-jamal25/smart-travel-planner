from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", env_file=".env")

    # Required
    database_url: str
    supabase_jwt_jwks_url: str
    supabase_jwt_audience: str = "authenticated"
    supabase_jwt_issuer: str
    anthropic_api_key: str
    openweather_api_key: str
    webhook_url: str

    # Optional with defaults
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"
    embedding_model: str = "all-MiniLM-L6-v2"
    weather_cache_ttl_seconds: int = 600
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

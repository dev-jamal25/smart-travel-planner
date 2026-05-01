from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
    )

    # Required
    database_url: str
    supabase_jwt_jwks_url: str
    supabase_jwt_audience: str = "authenticated"
    supabase_jwt_issuer: str
    supabase_anon_key: str
    supabase_service_role_key: str
    anthropic_api_key: str
    webhook_url: str

    # Optional with defaults
    classifier_model_path: str = "models/final_travel_style_pipeline.joblib"
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"
    embedding_model: str = "all-MiniLM-L6-v2"
    weather_cache_ttl_seconds: int = 600
    log_level: str = "INFO"

    # CORS — list of allowed frontend origins
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Agent runtime and tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "smart-travel-planner"
    max_agent_steps: int = 8
    tool_arg_repair_attempts: int = 1
    anthropic_timeout_seconds: int = 30

    @field_validator("database_url")
    @classmethod
    def ensure_async_scheme(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

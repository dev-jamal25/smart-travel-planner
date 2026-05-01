import os
from uuid import UUID

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SUPABASE_JWT_JWKS_URL", "https://test.supabase.co/auth/v1/.well-known/jwks.json")
os.environ.setdefault("SUPABASE_JWT_ISSUER", "https://test.supabase.co/auth/v1")
os.environ.setdefault("SUPABASE_JWT_AUDIENCE", "authenticated")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")
os.environ.setdefault("WEBHOOK_URL", "https://test.example.com/webhook")

# Force LangSmith tracing off in tests — no network calls to LangSmith.
# Use os.environ directly (not setdefault) so this overrides any shell value.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langsmith-key")
os.environ.setdefault("LANGCHAIN_PROJECT", "smart-travel-planner-test")

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.dependencies import get_current_user, get_session  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.auth import CurrentUser  # noqa: E402


@pytest.fixture
def client():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    fake_user = CurrentUser(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        email="test@test.com",
    )

    async def _session():
        async with factory() as session:
            yield session

    async def _current_user() -> CurrentUser:
        return fake_user

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_current_user] = _current_user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()

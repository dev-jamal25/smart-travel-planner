from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_session
from app.main import app


def test_healthz_returns_200(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "connected"}


def test_healthz_db_unreachable_returns_503() -> None:
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.execute.side_effect = Exception("DB unreachable")

    async def broken_session():
        yield mock_session

    app.dependency_overrides[get_session] = broken_session
    try:
        with TestClient(app) as c:
            response = c.get("/healthz")
        assert response.status_code == 503
        assert response.json() == {"status": "degraded", "db": "unreachable", }
    finally:
        app.dependency_overrides.pop(get_session, None)

from datetime import UTC, datetime, timedelta
from typing import Annotated
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_current_user
from app.exceptions import AuthError
from app.schemas.auth import CurrentUser

_app = FastAPI()


@_app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": str(exc)})


@_app.get("/me")
async def _me(user: Annotated[CurrentUser, Depends(get_current_user)]) -> dict[str, str]:
    return {"user_id": str(user.user_id), "email": user.email}


@pytest.fixture
def auth_client() -> TestClient:
    return TestClient(_app, raise_server_exceptions=False)


def test_missing_auth_returns_401(auth_client: TestClient) -> None:
    response = auth_client.get("/me")
    assert response.status_code == 401


def test_invalid_jwt_returns_401(auth_client: TestClient) -> None:
    with patch("app.dependencies._get_jwks_client") as mock_jwks_client:
        mock_client = MagicMock()
        mock_jwks_client.return_value = mock_client
        mock_client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError("Invalid token")
        
        response = auth_client.get("/me", headers={"Authorization": "Bearer bad.token.here"})
        assert response.status_code == 401


def test_valid_jwt_returns_current_user(auth_client: TestClient) -> None:
    """Test ES256 token verification via JWKS."""
    settings = get_settings()
    
    # Create a test payload
    test_payload = {
        "sub": "00000000-0000-0000-0000-000000000001",
        "email": "test@test.com",
        "aud": settings.supabase_jwt_audience,
        "iss": settings.supabase_jwt_issuer,
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    
    # For testing, we create a fake token (any string works for mocking)
    fake_token = "fake.jwt.token"
    
    with patch("app.dependencies._get_jwks_client") as mock_get_jwks:
        with patch("jwt.decode") as mock_decode:
            mock_client = MagicMock()
            mock_get_jwks.return_value = mock_client
            
            # Mock the signing key
            mock_key = MagicMock()
            mock_key.key = "mock-public-key"
            mock_client.get_signing_key_from_jwt.return_value = mock_key
            
            # Mock jwt.decode to return our test payload
            mock_decode.return_value = test_payload
            
            # Clear lru_cache to force fresh call to patched _get_jwks_client
            from app.dependencies import _get_jwks_client
            _get_jwks_client.cache_clear()
            
            response = auth_client.get("/me", headers={"Authorization": f"Bearer {fake_token}"})
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "00000000-0000-0000-0000-000000000001"
            assert data["email"] == "test@test.com"

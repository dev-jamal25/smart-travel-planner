from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.exceptions import AuthError
from app.schemas.auth import CurrentUser
from app.services.classifier_service import ClassifierService
from app.services.rag_service import RagService

# Security scheme for Swagger auth display
bearer_scheme = HTTPBearer(auto_error=False)


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.session_factory() as session:
        yield session


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> CurrentUser:
    """Verify Supabase JWT token and return current user.

    Uses FastAPI HTTPBearer security scheme for Swagger auth display.
    """
    if credentials is None:
        raise AuthError("Missing authorization header")

    token = credentials.credentials
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=settings.supabase_jwt_audience,
        )
    except (ExpiredSignatureError, InvalidTokenError) as err:
        raise AuthError("Invalid or expired token") from err
    return CurrentUser(user_id=UUID(payload["sub"]), email=payload.get("email", ""))


def get_classifier_service(request: Request) -> ClassifierService:  # noqa: ARG001
    """Dependency to inject classifier service.

    Uses request app.state.classifier loaded in FastAPI lifespan.
    """
    model = getattr(request.app.state, "classifier", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Classifier model not available")
    return ClassifierService(model=model)


def get_rag_service(request: Request) -> RagService:
    return RagService(embedder=request.app.state.embedder)

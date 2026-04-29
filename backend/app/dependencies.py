from collections.abc import AsyncGenerator
from uuid import UUID

import jwt
from fastapi import Request
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.exceptions import AuthError
from app.schemas.auth import CurrentUser


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.session_factory() as session:
        yield session


async def get_current_user(request: Request) -> CurrentUser:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise AuthError("Missing authorization header")
    token = auth_header.removeprefix("Bearer ")
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


def get_classifier() -> None:
    raise NotImplementedError

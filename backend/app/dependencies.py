from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    raise NotImplementedError


async def get_current_user() -> None:
    raise NotImplementedError


def get_classifier() -> None:
    raise NotImplementedError

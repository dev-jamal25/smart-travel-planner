import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import create_engine, create_session_factory
from app.dependencies import get_session
from app.exceptions import AuthError
from app.logging_setup import configure_logging
from app.routers import auth, chat, history

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    # TODO: load classifier here
    # TODO: load embedder here
    engine = create_engine()
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    yield
    await app.state.engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(history.router)


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError) -> Response:
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=401, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz(
    response: Response,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ok", "db": "connected"}
    
    except Exception as exc:
        logger.exception("Health check database connection failed")

        response.status_code = 503
        return {
            "status": "degraded",
            "db": "unreachable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

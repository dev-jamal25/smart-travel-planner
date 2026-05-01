import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_session
from app.exceptions import AuthError
from app.lifespan import lifespan
from app.logging_setup import configure_logging
from app.routers import auth, chat, classifier, history, rag, traces, weather, webhook

logger = logging.getLogger(__name__)

configure_logging()
app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(classifier.router)
app.include_router(history.router)
app.include_router(rag.router)
app.include_router(traces.router)
app.include_router(weather.router)
app.include_router(webhook.router)


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

    except Exception:
        logger.exception("Health check database connection failed")

        response.status_code = 503
        return {
            "status": "degraded",
            "db": "unreachable",
        }

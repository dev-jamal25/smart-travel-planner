import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
from fastapi import Depends, FastAPI, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BACKEND_DIR, get_settings
from app.db.session import create_engine, create_session_factory
from app.dependencies import get_session
from app.exceptions import AuthError
from app.logging_setup import configure_logging
from app.routers import auth, chat, classifier, history, rag, weather, webhook
from app.services.weather_service import WeatherService
from app.services.webhook_service import WebhookService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from sentence_transformers import (
        SentenceTransformer,  # lazy: keeps PyTorch out of module-level imports
    )

    configure_logging()
    settings = get_settings()

    app.state.embedder = await asyncio.to_thread(SentenceTransformer, settings.embedding_model)

    model_path = BACKEND_DIR / settings.classifier_model_path
    try:
        app.state.classifier = await asyncio.to_thread(joblib.load, model_path)
        logger.info("Classifier loaded from %s", model_path)
    except Exception:
        logger.warning(
            "Classifier model not found at %s — /classifier/predict will return 503", model_path
        )
        app.state.classifier = None

    app.state.weather_service = WeatherService(settings=settings)
    app.state.webhook_service = WebhookService(settings=settings)

    engine = create_engine()
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    yield
    await app.state.weather_service.close()
    await app.state.webhook_service.close()
    await app.state.engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(classifier.router)
app.include_router(history.router)
app.include_router(rag.router)
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

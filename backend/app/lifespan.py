"""FastAPI lifespan context manager for resource initialization and cleanup."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.config import BACKEND_DIR, get_settings
from app.db.session import create_engine, create_session_factory
from app.logging_setup import configure_logging
from app.services.weather_service import WeatherService
from app.services.webhook_service import WebhookService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared resources on startup; clean up on shutdown.

    Creates singletons for:
    - Embedding model (lazy-loaded SentenceTransformer)
    - ML classifier (joblib pipeline)
    - Database engine and session factory
    - Weather service with async HTTP client
    - Webhook service with async HTTP client

    All resources are attached to app.state and available via dependency injection.
    """
    # Logging must be configured early
    configure_logging()
    settings = get_settings()

    # --- EMBEDDER (lazy-loaded to keep PyTorch out of module-level imports) ---
    app.state.embedder = await asyncio.to_thread(
        SentenceTransformer, settings.embedding_model
    )
    logger.info("Embedding model loaded: %s", settings.embedding_model)

    # --- CLASSIFIER (joblib-saved sklearn pipeline) ---
    model_path = BACKEND_DIR / settings.classifier_model_path
    try:
        app.state.classifier = await asyncio.to_thread(joblib.load, model_path)
        logger.info("Classifier loaded from %s", model_path)
    except Exception:
        logger.warning(
            "Classifier model not found at %s — /classifier/predict will return 503",
            model_path,
        )
        app.state.classifier = None

    # --- WEATHER SERVICE ---
    app.state.weather_service = WeatherService(settings=settings)
    logger.info("Weather service initialized")

    # --- WEBHOOK SERVICE ---
    app.state.webhook_service = WebhookService(settings=settings)
    logger.info("Webhook service initialized")

    # --- DATABASE ---
    engine = create_engine()
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    logger.info("Database engine and session factory created")

    # Yield control back to FastAPI; the app is now ready to serve requests
    yield

    # --- CLEANUP (on shutdown) ---
    logger.info("Shutting down services and closing connections")
    await app.state.weather_service.close()
    await app.state.webhook_service.close()
    await app.state.engine.dispose()
    logger.info("Shutdown complete")

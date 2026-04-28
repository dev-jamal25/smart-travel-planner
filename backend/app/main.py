from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.logging_setup import configure_logging
from app.routers import auth, chat, history


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    # TODO: load classifier here
    # TODO: load embedder here
    # TODO: create db engine here
    yield
    # TODO: cleanup resources here


app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(history.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

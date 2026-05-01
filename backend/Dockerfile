FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY rag_data/ ./rag_data/
COPY models/ ./models/
COPY alembic.ini ./
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

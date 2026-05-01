# Smart Travel Planner

## Overview

## Architecture

## Setup

## Running

### Backend

```bash
cd backend
uv run uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The frontend expects the backend running on `http://localhost:8000`.

Copy `frontend/.env.example` to `frontend/.env` and fill in your Supabase project URL and anon key.

## Docker

1. Copy `backend/.env.example` to `backend/.env` and fill in the backend secrets and provider settings.
2. Copy `frontend/.env.example` to `frontend/.env` and fill in the frontend variables.
3. Run `docker compose up --build` from the repository root.
4. Open the backend docs at [http://localhost:8000/docs](http://localhost:8000/docs).
5. Open the frontend at [http://localhost:5173](http://localhost:5173).

If migrations do not apply automatically, run `docker compose exec backend uv run alembic upgrade head`.

If the local RAG tables are empty, run `docker compose exec backend uv run python scripts/ingest_rag_documents.py`.

## Tests

## Env Vars

## Project Structure

## Deployment

## Limitations

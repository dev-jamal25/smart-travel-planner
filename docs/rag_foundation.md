# RAG Foundation

## Why Postgres + pgvector

The project already uses Postgres for all persistence (users, runs, tool calls). pgvector adds a native vector column type and cosine/L2/dot-product operators directly in SQL. Using the same database means:

- No extra service to run or Dockerize.
- Embeddings and metadata live in the same transaction boundary.
- Joins between chunks and other tables (e.g. filtering by destination) are plain SQL.
- Backup, migrations, and access control are unified.

## Embedding Model

**Model:** `all-MiniLM-L6-v2` (sentence-transformers)
**Dimension:** 384
**Why:** Fast CPU inference (~10 ms per sentence), strong semantic coverage for travel prose, under 100 MB. The model name comes from `Settings.embedding_model` so it can be swapped without code changes.

## Chunking Strategy

**Chunk size:** 500 characters
**Overlap:** 75 characters (15%)

Rationale: a typical travel paragraph is 300–600 characters. 500 fits one substantial paragraph while staying well within the model's 256-token input limit. A 75-character overlap ensures that sentences split across a chunk boundary appear in both chunks, preserving retrieval quality at seam points.

Constants are defined in `app/services/rag_service.py` as `CHUNK_SIZE` and `CHUNK_OVERLAP`.

## Retrieval Strategy

Similarity is measured by **cosine distance** (pgvector `<=>` operator). Results are ordered ascending (lower = more similar) and limited to `top_k=5` by default. An optional `destination_filter` narrows search to a single destination by exact name match.

The `retrieve_top_k` method in `RagService` joins `DestinationChunk` with `DestinationDocument` to return `source_title` alongside each chunk, giving the agent provenance information.

## Raw Corpus Collection

The raw corpus for RAG is collected exclusively from **Wikivoyage** through the MediaWiki Action API.

**Approach:**
- 10 destination articles fetched from Wikivoyage (real sources only; no synthetic fallback content)
- Each destination split into 2 documents (_overview.txt + _details.txt) for balanced chunk retrieval
- HTML cleaned, citations removed, formatted as plain text with headers (destination, source_title, source_url)
- Stored in `backend/rag_data/raw/` before ingestion

**Source Attribution:**
Every raw text file includes full source attribution in its header:
- `source_title`: Human-readable title from Wikivoyage
- `source_url`: Direct link to the Wikivoyage article, e.g., `https://en.wikivoyage.org/wiki/Interlaken`

**Real Data Commitment:**
The collection script (`backend/scripts/fetch_wikivoyage_raw.py`) is production-safe:
- Fetches exclusively from Wikivoyage MediaWiki API with proper User-Agent header
- Falls back to direct HTML page fetch if API is rate-limited
- Skips destinations if neither source returns valid content
- Never generates synthetic or sample content

## Implemented Now

- `DestinationDocument` and `DestinationChunk` ORM models (`app/models/db.py`)
- Alembic migration `a4f8c2d1e9b3` enabling the `vector` extension and creating both tables
- `RagService` with `store_document_with_chunks` and `retrieve_top_k` (`app/services/rag_service.py`)
- `chunk_text` pure function (unit-testable without DB)
- `get_rag_service` FastAPI dependency (`app/dependencies.py`)
- `SentenceTransformer` singleton loaded in FastAPI lifespan (`app/main.py`)
- Ingestion script `scripts/ingest_rag_documents.py` reading from `rag_data/raw/*.txt`
- Tests covering chunking logic and retrieval service (no Postgres required)

## Deferred

- **HNSW vector index** — must be added after the first data load (pgvector requires rows to build the index). Run manually:
  ```sql
  CREATE INDEX ON destination_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  ```
  Requires pgvector >= 0.5.0.
- **Real destination documents** — 20–30 `.txt` files should be placed in `rag_data/raw/` following the header format (`destination:`, `source_title:`, `source_url:`, then `---`, then body).
- **Hybrid BM25 re-ranking** — combining keyword and vector scores for higher precision.
- **Haiku query rewriting** — using Claude Haiku to reformulate the user's question into an optimal embedding query before retrieval.
- **Agent wiring** — connecting `RagService.retrieve_top_k` to the LangGraph agent as the destination knowledge retrieval tool.

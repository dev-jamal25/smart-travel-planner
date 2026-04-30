# RAG Retrieval Evaluation

## Overview

Before wiring the RAG service into the LangGraph agent, retrieval quality is tested via two HTTP endpoints:
1. **POST /rag/search** — Ad-hoc query endpoint for manual Swagger testing
2. **GET /rag/eval** — Fixed evaluation set with 7 reference queries

Both endpoints return the same retrieval response structure and use the existing `RagService.retrieve_top_k()` method.

## Endpoint: POST /rag/search

**Purpose:** Manual retrieval testing via Swagger or curl.

**Request:**
```json
{
  "query": "I want beaches and relaxation",
  "top_k": 5,
  "destination_filter": null
}
```

**Parameters:**
- `query` (str, required): The search query
- `top_k` (int, default 5): Number of results (1–20)
- `destination_filter` (str | null, optional): Filter by destination name

**Response:**
```json
{
  "query": "I want beaches and relaxation",
  "top_k": 5,
  "results": [
    {
      "destination_name": "Bali",
      "source_title": "Wikivoyage — Bali",
      "source_url": "https://en.wikivoyage.org/wiki/Bali",
      "distance": 0.18,
      "chunk_preview": "Bali is a tropical island...",
      "chunk_index": 1
    },
    ...
  ]
}
```

**Fields:**
- `destination_name`: Destination article title
- `source_title`: Full source title from Wikivoyage
- `source_url`: Wikivoyage URL (reconstructed from destination name)
- `distance`: Cosine distance (lower = more similar; 0.0 = exact match)
- `chunk_preview`: First 200 characters of the chunk
- `chunk_index`: Which part of the document (0 = overview, 1+ = details)

## Endpoint: GET /rag/eval

**Purpose:** Automated evaluation of retrieval quality across 7 reference queries.

**Request:**
```
GET /rag/eval
```

No parameters required.

**Response Structure:**
```json
{
  "total_cases": 7,
  "top_3_passes": 5,
  "top_5_passes": 6,
  "cases": [
    {
      "name": "adventure_mountains",
      "query": "I want hiking, mountains, lakes, and outdoor adventure.",
      "expected_destinations": ["Interlaken", "Banff"],
      "top_3_pass": true,
      "top_5_pass": true,
      "search": {
        "query": "I want hiking, mountains, lakes, and outdoor adventure.",
        "top_k": 5,
        "results": [...]
      }
    },
    ...
  ]
}
```

**Scoring:**
- `top_3_pass`: True if any `expected_destinations` appears in the first 3 results
- `top_5_pass`: True if any `expected_destinations` appears in the first 5 results
- `top_3_passes`: Count of cases where `top_3_pass == true`
- `top_5_passes`: Count of cases where `top_5_pass == true`

## Evaluation Cases Storage

The 7 evaluation cases are stored in a JSON file independent of the router module:
- **Location:** `backend/rag_data/eval/retrieval_eval_cases.json`
- **Format:** Array of objects with `name`, `query`, and `expected_destinations`
- **Loading:** The GET /rag/eval endpoint calls `load_rag_eval_cases()` from `app.evaluation.rag_eval_cases`
- **Caching:** Loaded once and cached for the lifetime of the process (uses `@lru_cache`)

This separation ensures the router does not own the evaluation dataset and makes it easy to add/modify test cases without changing code.

## The 7 Evaluation Cases

### 1. adventure_mountains
- **Query:** "I want hiking, mountains, lakes, and outdoor adventure."
- **Expected:** Interlaken, Banff
- **Rationale:** Iconic mountain destinations with outdoor activity content

### 2. culture_history
- **Query:** "I want temples, historic districts, old architecture, and cultural sightseeing."
- **Expected:** Kyoto, Istanbul
- **Rationale:** Strong cultural/historical content in Wikivoyage corpus

### 3. beaches_relaxation
- **Query:** "I want beaches, warm weather, and a relaxing island trip."
- **Expected:** Bali, Santorini
- **Rationale:** Island destinations with beach/relaxation focus

### 4. luxury_modern_city
- **Query:** "I want luxury shopping, modern attractions, and premium hotels."
- **Expected:** Dubai, Singapore
- **Rationale:** Modern urban luxury destinations

### 5. budget_history
- **Query:** "I want a budget-friendly city with history, walkable areas, and affordable food."
- **Expected:** Kraków, Tbilisi
- **Rationale:** Affordable destinations with cultural content

### 6. family_practical_city
- **Query:** "I am travelling with family and want safe transport, easy attractions, and a city that is simple to navigate."
- **Expected:** Singapore
- **Rationale:** Safe, well-organized city infrastructure

### 7. logistics_safety
- **Query:** "I care about getting around, safety, and practical travel logistics more than sightseeing."
- **Expected:** Singapore, Dubai, Kyoto, Istanbul
- **Rationale:** Destinations with strong logistics/safety/navigation coverage

## Interpreting Results

### Full Passes (top_3_pass = true)
If an expected destination appears in the top-3 results:
- Embedding/RAG correctly understands the query intent
- Destination content strongly matches the semantic query
- Query likely will work well when integrated into the agent

### Top-5 Only Passes (top_3_pass = false, top_5_pass = true)
If an expected destination appears in positions 4–5:
- Embedding understands the query but is not maximally confident
- May work in agent context if agent re-ranks by classifier or other signals
- Consider query rewriting or expanding corpus if many cases fall here

### Fails (top_5_pass = false)
If no expected destination appears in top-5:
- Embedding may not understand the query intent well
- Destination content may not be well-represented in corpus
- Consider adding more destination content or adjusting query syntax

## Integration with Agent

Once these tests pass (ideally ≥5/7 top-5 passes), the RAG retrieval tool is wired into the LangGraph agent as:
- **Tool name:** destination_knowledge_retrieval
- **Tool input:** RagSearchRequest (query, top_k=5, destination_filter)
- **Tool output:** RagSearchResponse results

The agent uses these results to ground its travel plan synthesis in real destination knowledge.

## Testing the Endpoints

### Via Swagger
1. Start the backend: `uv run uvicorn app.main:app --reload`
2. Open http://127.0.0.1:8000/docs
3. Expand the `/rag` section
4. Try **POST /rag/search** with your own queries
5. Try **GET /rag/eval** to see the fixed evaluation set

### Via curl

**Search:**
```bash
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "mountains and hiking", "top_k": 5}'
```

**Eval:**
```bash
curl http://127.0.0.1:8000/rag/eval
```

### Via Python
```python
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000")

# Ad-hoc search
response = client.post("/rag/search", json={"query": "beaches", "top_k": 5})
print(response.json())

# Evaluation
response = client.get("/rag/eval")
results = response.json()
print(f"Passed top-3: {results['top_3_passes']}/7")
print(f"Passed top-5: {results['top_5_passes']}/7")
```

## Future Improvements

- **Haiku query rewriting:** Use Claude Haiku to reformulate user queries before embedding (e.g., "I want chill vibes" → "beaches, relaxation, resort")
- **Hybrid BM25 re-ranking:** Combine vector score with keyword matching
- **Classifier re-ranking:** Use the travel style classifier to rank results by relevance to the user's travel style
- **Expand corpus:** Add more destinations or more detailed content per destination
- **Query feedback:** Log which queries fail to improve corpus or rewriting strategy

# Agent Orchestration Foundation

## Overview

This document describes the agent foundation slice for the Smart Travel Planner.

The foundation establishes the structure and patterns for the LangGraph-based travel planning agent, but **does not yet implement the full agent orchestration**.

## What Is Implemented

### 1. Lifespan Management (`app/lifespan.py`)

The FastAPI lifespan has been extracted into a dedicated module for clarity and testability.

**Initialized on startup:**
- Embedding model (SentenceTransformer, lazy-loaded)
- ML classifier (scikit-learn pipeline, via joblib)
- Database engine and session factory
- Weather service with async HTTP client
- Webhook service with async HTTP client

All resources are stored in `app.state` and exposed via FastAPI dependency injection.

**Why separate from main.py?**
- Cleaner module organization
- Easier to test and modify
- Preparation for more complex startup logic (e.g., Anthropic client initialization)

### 2. Retry Foundation (`app/retries.py`)

Shared retry configuration using Tenacity, available for future external calls (LLM APIs, model router, etc).

```python
from app.retries import DEFAULT_WAIT, DEFAULT_STOP

# Future use:
@retry(stop=DEFAULT_STOP, wait=DEFAULT_WAIT)
async def call_anthropic_api(...):
    ...
```

Currently used by WeatherService and WebhookService with their own retry logic. This module centralizes retry patterns for consistency.

### 3. System Prompts (`app/prompts/`)

Separated system prompts for different agent stages:

- **`safety.py`**: Universal safety guardrails (stay in scope, protect secrets, use only allowed tools)
- **`agent.py`**: Haiku mechanical task prompts (intent extraction, query rewriting, argument repair)
- **`synthesis.py`**: Sonnet synthesis prompt (genuine plan creation, tension resolution, no internal details)

These will be injected into the respective model calls when the agent is built.

### 4. Agent Schemas (`app/schemas/agent.py`)

Core request/response types:

- **`PlanTripRequest`**: User message
- **`PlanTripResponse`**: Agent run ID, final answer, destination, webhook status
- **`TripIntent`**: Extracted structured intent (budget, duration, style, activities, etc.)
- **`AgentToolError`**: Structured error from tool failure

### 5. Tool Input Schemas (`app/schemas/agent_tools.py`)

Validated input schemas for the three allowed tools:

- **`DestinationKnowledgeInput`**: Query, top_k (1–10), optional destination filter
- **`ClassifyDestinationInput`**: Destination name
- **`LiveWeatherInput`**: Destination, forecast_days (1–7)

All schemas include:
- Non-empty string validation
- Range validation (top_k, forecast_days)
- Normalization (e.g., destination_filter "none" → None)

### 6. Destination Profiles (`app/services/destination_profiles.py`)

Hardcoded profiles for the 10 supported destinations:

- Interlaken, Banff, Bali, Santorini, Kyoto, Istanbul, Tbilisi, Kraków, Dubai, Singapore

Each profile contains fields required by the ML classifier:
- Country, continent, destination type
- Cost, season, rating, visitors, UNESCO status

**Functions:**
- `get_destination_profile(destination_name)` — Lookup with case-insensitive, whitespace-tolerant matching
- `list_supported_profile_destinations()` — Return supported destinations

**Why hardcoded?**
- Small fixed set of supported destinations
- No need for DB migration until agent integration
- Deterministic and easy to test

### 7. Tool Registry (`app/tools/registry.py`)

Explicit allowlist of tools the agent may use:

```python
ALLOWED_AGENT_TOOLS = {
    "destination_knowledge_retrieval",
    "classify_destination_style",
    "fetch_live_weather",
}
```

The allowlist prevents the agent from inventing or calling tools outside this set.

**Functions:**
- `is_allowed_tool(name)` — Check if tool is allowed
- `get_allowed_tools()` — Return the set of allowed tools

### 8. Tool Result Base Type (`app/tools/base.py`)

`ToolResult` represents the outcome of a tool execution:

```python
class ToolResult:
    status: Literal["ok", "error"]
    output: dict[str, Any]  # if status=="ok"
    error: str              # if status=="error"
    retryable: bool         # whether error is transient
```

**Helper methods:**
- `ToolResult.ok(tool_name, output)` — Create success result
- `ToolResult.fail(tool_name, error, retryable=False)` — Create error result

Tool wrappers always return `ToolResult`, never raise exceptions.

### 9. Tool Wrappers

Three async functions that call existing services directly (not via HTTP):

#### `destination_knowledge_retrieval()`
- Accepts `DestinationKnowledgeInput`
- Calls `RagService.retrieve_top_k()` directly
- Returns `ToolResult` with chunks formatted for agent

#### `classify_destination_style()`
- Accepts `ClassifyDestinationInput`
- Looks up destination profile from `destination_profiles.py`
- Calls `ClassifierService.predict()` with profile
- Returns `ToolResult` with travel style and confidence
- **Error if destination not in supported list** (non-retryable)

#### `fetch_live_weather()`
- Accepts `LiveWeatherInput`
- Calls `WeatherService.get_forecast()` directly
- Converts `WeatherServiceError` to `ToolResult.fail(retryable=True)`
- Returns `ToolResult` with forecast data

**Key design:**
- All tools are async
- No HTTP calls to own backend routes (direct service calls)
- Comprehensive error handling → structured `ToolResult`
- No unhandled exceptions escape to caller

### 10. Agent Persistence (`app/models/db.py`, `app/db/repositories/agent_runs.py`)

Four new ORM models for comprehensive agent run tracking:

#### **AgentRun** — Top-level agent execution record
- `id` (UUID, PK)
- `user_id` (UUID, indexed) — Always scoped to user
- `user_query` (Text) — Original user request
- `final_answer` (Text, nullable) — Synthesized response
- `recommended_destination` (String, nullable)
- `status` (String) — "running", "completed", or "failed"
- `total_cost_usd` (Float, nullable) — Accumulated LLM + webhook cost
- `webhook_delivered` (Boolean, nullable) — Discord webhook success
- `webhook_status_code` (Integer, nullable)
- `error_message` (Text, nullable) — If status="failed"
- `created_at`, `completed_at` (DateTime, timezone-aware)

#### **ToolCallLog** — Individual tool invocation
- `id` (UUID, PK)
- `run_id` (UUID, FK to agent_runs, indexed)
- `tool_name` (String, indexed) — "destination_knowledge_retrieval", "classify_destination_style", "fetch_live_weather"
- `input_json` (JSONB, nullable) — Tool input serialized
- `output_summary` (Text, nullable) — Human-readable summary of output
- `status` (String) — "ok" or "error"
- `latency_ms` (Integer, nullable)
- `created_at` (DateTime)

#### **LLMUsageLog** — API call metrics for cost tracking
- `id` (UUID, PK)
- `run_id` (UUID, FK, indexed)
- `step_name` (String, indexed) — "intent_extraction", "synthesis", etc.
- `model` (String) — Model name used (e.g., "claude-3-haiku-20240307")
- `input_tokens`, `output_tokens` (Integer, nullable)
- `cost_usd` (Float, nullable) — Calculated cost
- `latency_ms` (Integer, nullable)
- `created_at` (DateTime)

#### **AgentTraceEvent** — Observability and debugging
- `id` (UUID, PK)
- `run_id` (UUID, FK, indexed)
- `event_type` (String, indexed) — "tool_call", "llm_call", "decision", "error", etc.
- `event_name` (String) — Human-readable name
- `detail_json` (JSONB, nullable) — Event-specific context
- `latency_ms` (Integer, nullable)
- `created_at` (DateTime)

**Repository functions** (`app/db/repositories/agent_runs.py`):
- `create_agent_run(session, user_id, user_query)` → AgentRun
- `mark_agent_run_completed(session, run_id, final_answer, ...) → AgentRun
- `mark_agent_run_failed(session, run_id, error_message) → AgentRun
- `log_tool_call(session, run_id, tool_name, input_json, output_summary, status, latency_ms) → ToolCallLog
- `log_llm_usage(session, run_id, step_name, model, input_tokens, output_tokens, cost_usd, latency_ms) → LLMUsageLog
- `log_trace_event(session, run_id, event_type, event_name, detail_json, latency_ms) → AgentTraceEvent
- `list_agent_runs_for_user(session, user_id, limit) → list[AgentRun]
- `get_agent_run_for_user(session, run_id, user_id) → AgentRun | None`

**Key design:**
- All DB functions are async
- User scoping enforced: runs never exposed across users
- Flush-based persistence: no session.commit() inside helpers (caller decides)
- JSONB for flexible tool inputs/outputs and trace details
- Indexes on foreign keys + frequently-filtered columns (user_id, status, tool_name, step_name, event_type)

## What Is NOT Implemented

### ✗ LangGraph Graph
The agent orchestration graph is not built yet. This will be added in the next slice when:
- Anthropic model setup and cost tracking is ready
- Chat route integration is planned

### ✗ Anthropic Integration
No LLM calls yet. The agent will call Anthropic in the next slice using:
- Haiku for mechanical tasks (intent extraction, query rewriting, etc.)
- Sonnet for final synthesis

### ✗ Chat Route Integration
The `/chat` endpoint is not wired to the agent yet. That will happen when the agent graph is built.

### ✗ Frontend
No changes to React frontend.

## Testing

Components tested with:

- **`test_agent_schemas.py`** — 20+ tests for request/response types and validation
- **`test_agent_tools.py`** — 10+ tests for tool wrappers with mocked services
- **`test_destination_profiles.py`** — 10+ tests for profile lookup and registry
- **`test_agent_persistence.py`** — 15+ tests for repositories with in-memory SQLite

Run tests:
```bash
uv run pytest tests/test_agent_schemas.py tests/test_agent_tools.py tests/test_destination_profiles.py tests/test_agent_persistence.py -v
```

Or all tests:
```bash
uv run pytest -q
```

## Design Rationale

### Why tools call services directly, not HTTP endpoints?

**Problem:** Tools in an agent orchestration layer should call services, not make HTTP requests to the same backend.

**Solution:** Tool wrappers call services from `app/services/` directly (RagService, ClassifierService, WeatherService), bypassing HTTP and Pydantic request/response conversion. This:
- Reduces latency (no HTTP overhead)
- Simplifies testing (mock services, not mocked HTTP)
- Prevents circular dependencies (agent → /classifier/predict HTTP → classifier_service)
- Enables better error recovery in the agent (structured ToolResult, not HTTPException)

The existing HTTP endpoints (`/rag/search`, `/classifier/predict`, `/weather/forecast`) remain for manual/Swagger testing and future API clients (frontend, external apps).

### Why prompts are separated?

**Problem:** Mixing system prompts with code makes them hard to iterate on and version-control separately.

**Solution:** Prompts live in `app/prompts/`, not hardcoded in agent logic. This:
- Makes prompts visible and reviewable
- Enables A/B testing of different prompt versions
- Keeps code clean and focused on orchestration
- Prepares for dynamic prompt management (database-backed prompts in future)

### Why destination profiles are hardcoded now?

**Problem:** Destinations are currently fixed to the 10 RAG sources and trained classifier labels.

**Solution:** Use hardcoded profiles in `destination_profiles.py` now, with clear upgrade path later:
- No DB migration needed yet
- Easy to test (no DB dependency)
- Will migrate to DB when more destinations are added

## Next Steps

### Slice 2: Agent Graph & LLM Integration
- Add Anthropic client to lifespan
- Create LangGraph graph with state, nodes, and edges
- Implement model router (Haiku/Sonnet selection)
- Add DB models for AgentRun, ToolCall, LLMUsage
- Create Alembic migrations
- Wire `/chat` endpoint to the agent

### Slice 3: Chat Route Integration
- Connect the agent graph to the FastAPI `/chat` endpoint
- Stream plan synthesis results (if beneficial)
- Persist run data to database
- Integrate with webhook service for Discord notifications

### Slice 4: Observability & Dashboard
- LangSmith tracing integration
- Admin dashboard for run history, costs, and error rates
- Tool performance metrics

## References

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [Pydantic Validation](https://docs.pydantic.dev/latest/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Anthropic API](https://docs.anthropic.com/)

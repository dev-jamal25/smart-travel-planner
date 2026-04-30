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

### 10. Model Router (`app/agents/model_router.py`)

Routes LLM calls to Haiku (mechanical work) or Sonnet (synthesis) with automatic token/cost tracking.

#### Haiku Methods (Mechanical Tasks)
- **`extract_trip_intent(message)`** — Extracts structured TripIntent from user message
- **`rewrite_rag_query(message, intent)`** — Rewrites user query for better RAG matching
- **`select_candidate_destination(message, retrieved_destinations)`** — Selects best destination from list
- **`repair_tool_arguments(tool_name, invalid_payload, validation_error)`** — Fixes invalid tool arguments before execution

#### Sonnet Method (Synthesis)
- **`synthesize_final_answer(...)`** — Creates final user-facing travel plan, synthesizing RAG + classifier + weather results

#### Token & Cost Tracking
Every LLM call extracts:
- `input_tokens`, `output_tokens` from Anthropic response
- `cost_usd` calculated using static pricing table keyed by model name
- `latency_ms` measured from call start to end

#### Optional Persistence
All ModelRouter methods accept optional `session: AsyncSession | None` and `run_id: UUID | None`:
- If both provided, automatically calls `log_llm_usage()` to persist metrics to `LLMUsageLog` table
- If either is None, still returns result normally (no persistence)
- Does not commit inside ModelRouter (caller controls transaction)

#### Settings Integration
- Model names sourced from `Settings.haiku_model` and `Settings.sonnet_model` (never hardcoded)
- Timeout from `Settings.anthropic_timeout_seconds`
- Retry config: 3 attempts with exponential backoff (0.5s → 8s max)

#### Dependency Injection
- ModelRouter instantiated once in `app/lifespan.py` and stored on `app.state.model_router`
- Exposed via `get_model_router()` dependency in `app/dependencies.py`

### 11. Agent Persistence (`app/models/db.py`, `app/db/repositories/agent_runs.py`)

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
- `input_json` (JSON, nullable) — Tool input serialized
- `output_summary` (Text, nullable) — Human-readable summary of output
- `status` (String) — "ok" or "error"
- `latency_ms` (Integer, nullable)
- `created_at` (DateTime)

#### **LLMUsageLog** — API call metrics for cost tracking
- `id` (UUID, PK)
- `run_id` (UUID, FK, indexed)
- `step_name` (String, indexed) — "intent_extraction", "synthesis", etc.
- `model` (String) — Model name used (e.g., "claude-haiku-4-5-20251001")
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
- JSON for flexible tool inputs/outputs and trace details
- Indexes on foreign keys + frequently-filtered columns (user_id, status, tool_name, step_name, event_type)

## What Is NOT Implemented

### ✗ LangGraph Graph
The agent orchestration graph is not built yet. This will be added in the next slice after model router is complete.
Graph will orchestrate:
1. Intent extraction (Haiku via ModelRouter)
2. Query rewriting (Haiku)
3. RAG retrieval (via destination_knowledge_retrieval tool)
4. Destination selection (Haiku)
5. Classification (via classify_destination_style tool)
6. Weather fetch (via fetch_live_weather tool)
7. Final synthesis (Sonnet via ModelRouter)
8. Persistence to database

### ✗ Chat Route Integration
The `/chat` endpoint is not wired to the agent yet. That will happen when the LangGraph graph is built.

### ✗ Frontend
No changes to React frontend.

## Testing

Components tested with:

- **`test_agent_schemas.py`** — 20+ tests for request/response types and validation
- **`test_agent_tools.py`** — 10+ tests for tool wrappers with mocked services
- **`test_destination_profiles.py`** — 10+ tests for profile lookup and registry
- **`test_model_router.py`** — 25+ tests for Haiku/Sonnet model calls, cost estimation, token tracking
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

### Slice 1: LangGraph Agent Graph
- Create a controlled LangGraph state graph.
- Wire nodes for safety, intent extraction, query rewriting, RAG retrieval, destination selection, classification, weather, synthesis, and webhook delivery.
- Use ModelRouter for Haiku/Sonnet calls.
- Use existing tool wrappers for RAG/classifier/weather.
- Persist tool calls, LLM usage, and trace events using the repository helpers.

### Slice 2: Chat Route Integration
- Wire the existing `/chat` route to the agent service.
- Return `PlanTripResponse`.
- Keep internal cost/tool logs out of the user-facing response.

### Slice 3: Trace Routes and README Evidence
- Add DB-backed trace inspection routes if time allows.
- Enable LangSmith tracing.
- Run one full multi-tool query.
- Add the LangSmith screenshot and one full-query cost breakdown to README.

## References

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [Pydantic Validation](https://docs.pydantic.dev/latest/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Anthropic API](https://docs.anthropic.com/)

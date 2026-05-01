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

## LangGraph Controlled Graph

### Node Order

```
START
  → extract_intent          (Haiku: TripIntent from user message)
  → rewrite_rag_query       (Haiku: optimise query for vector search)
  → retrieve_knowledge      (RAG tool: destination_knowledge_retrieval)
  → select_destination      (Haiku: pick best candidate from RAG results)
  → classify_destination    (ML tool: classify_destination_style)
  → fetch_weather           (Weather tool: fetch_live_weather)
  → synthesize_answer       (Sonnet: final travel plan)
  → deliver_webhook         (Discord: send plan to configured webhook)
  → END
```

### Why This Is Not ReAct

A ReAct agent lets the LLM decide at runtime which tools to call and in what
order.  This graph does the opposite: every node is explicit, every edge is
hardcoded, and the LLM can only produce text — it never names a tool.

The controlled graph guarantees:
- The three tools always fire in the same order (RAG → classifier → weather).
- The LLM cannot invent a tool name not in `app/tools/registry.py`.
- Tool failure never crashes the graph — nodes return `ToolResult.fail()` and
  the graph continues to synthesis, which explains missing evidence.
- Token/cost behaviour is deterministic and observable.

### How the Graph Satisfies the Requirements

| Requirement | How it is met |
|---|---|
| 3 tools | Explicit nodes call `destination_knowledge_retrieval`, `classify_destination_style`, `fetch_live_weather` |
| Pydantic input validation | Each tool node builds a Pydantic schema (`DestinationKnowledgeInput`, `ClassifyDestinationInput`, `LiveWeatherInput`) before calling the tool wrapper |
| Explicit tool allowlist | Tools are called directly by name, never via an LLM tool-call API; `app/tools/registry.py` defines `ALLOWED_AGENT_TOOLS` |
| Two-model routing | Haiku for `extract_intent`, `rewrite_rag_query`, `select_destination`; Sonnet for `synthesize_answer` |
| Synthesis across RAG/classifier/weather | `synthesize_answer` node combines all three tool outputs and passes them to `ModelRouter.synthesize_final_answer()` (Sonnet), which explains tensions between sources |
| Persistence | `log_tool_call` after every tool, `log_trace_event` at key nodes, `create_agent_run` / `mark_agent_run_completed` / `mark_agent_run_failed` in `AgentService` |
| Webhook delivery | `deliver_webhook` node calls `WebhookService.send_trip_plan()`; failure is isolated and does not raise |

### Key Files

| File | Responsibility |
|---|---|
| `app/agents/state.py` | `AgentGraphState` TypedDict flowing through the graph |
| `app/agents/graph.py` | `build_travel_graph()` — assembles and compiles the StateGraph |
| `app/services/agent_service.py` | `AgentService.plan_trip()` — creates `AgentRun`, invokes graph, marks completed/failed, commits |
| `app/dependencies.py` | `get_agent_service()` — wires per-request services into `AgentService` |

### Session and Transaction Boundary

- Repository helpers (`log_tool_call`, `log_trace_event`, etc.) call `flush()` only — they never commit.
- `AgentService.plan_trip()` calls `session.commit()` exactly once at the service boundary (success or failure).
- Graph nodes share the same `AsyncSession` via `state["session"]`.

### Tool Failure Strategy

- If a tool node raises an unhandled exception, the `ToolResult` from the
  tool wrapper catches it and returns `ToolResult.fail(...)`.
- The graph continues to the next node; the failing tool's error text is
  appended to `state["errors"]`.
- `synthesize_answer` checks each tool result and substitutes a plain-language
  "unavailable" string when a tool failed, so Sonnet can still produce a
  coherent (if partial) plan.
- Webhook failure is fully isolated: `WebhookService.send_trip_plan()` never
  raises; the graph always reaches `END`.

## Chat Route — POST /chat/plan-trip

### Overview

`POST /chat/plan-trip` is the single user-facing entrypoint for the travel-planning agent.
It is implemented in `app/routers/chat.py` and registered in `app/main.py`.

### Request / Response

| Field | Type | Notes |
|---|---|---|
| `message` | `string` | User's trip request (non-empty) |

| Field | Type | Notes |
|---|---|---|
| `run_id` | `UUID` | Agent run ID for audit/history lookup |
| `answer` | `string` | Synthesized travel plan |
| `recommended_destination` | `string \| null` | Best destination selected by the agent |
| `webhook_delivered` | `bool \| null` | Whether Discord delivery succeeded |

Internal fields (cost, tokens, tool logs, trace events) are **never** included in the response.
They are persisted to the database only.

### Dependencies

```
POST /chat/plan-trip
  ├── get_current_user  → CurrentUser (Supabase JWT, 401 on missing/invalid)
  ├── get_session       → AsyncSession  (per-request DB session)
  └── get_agent_service → AgentService  (per-request; compiles graph once in __init__)
```

### Error Handling

| Condition | HTTP Status |
|---|---|
| Missing or invalid JWT | 401 |
| Empty or whitespace message | 422 |
| Agent graph raises unhandled exception | 500 (detail: "Travel planning failed. Please try again.") |

Webhook failure is **not** a 500 — it is isolated inside `deliver_webhook` node and reflected only in `webhook_delivered: false` in the response.

### Structured Logs

| Event | When |
|---|---|
| `chat.plan_trip.start` | Immediately on request receipt |
| `chat.plan_trip.success` | After graph completes and session commits |
| `chat.plan_trip.failed` | If the agent graph raises an unhandled exception |

### Key Files

| File | Responsibility |
|---|---|
| `app/routers/chat.py` | Route handler — logs, calls AgentService, raises HTTPException on failure |
| `app/services/agent_service.py` | `plan_trip()` — creates AgentRun, invokes graph, commits session |
| `app/dependencies.py` | `get_agent_service()` — wires per-request services into AgentService |

## LangSmith Tracing

### Overview

The agent is instrumented with LangSmith custom tracing using `langsmith.traceable`.
Tracing is activated when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set.
When disabled, every decorated function behaves exactly as if the decorator were not present.

### How It Is Enabled

`app/lifespan.py` bridges pydantic-settings → `os.environ` at startup:

```python
if settings.langchain_tracing_v2:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
```

`setdefault` is used so Docker / shell environment values are always respected.

### Instrumented Functions

| Function | Run Type | Where |
|---|---|---|
| `AgentService.plan_trip` | chain | `app/services/agent_service.py` — top-level run |
| `ModelRouter.extract_trip_intent` | chain | `app/agents/model_router.py` |
| `ModelRouter.rewrite_rag_query` | chain | `app/agents/model_router.py` |
| `ModelRouter.select_candidate_destination` | chain | `app/agents/model_router.py` |
| `ModelRouter.synthesize_final_answer` | chain | `app/agents/model_router.py` |
| `destination_knowledge_retrieval` | tool | `app/tools/rag_tool.py` |
| `classify_destination_style` | tool | `app/tools/classifier_tool.py` |
| `fetch_live_weather` | tool | `app/tools/weather_tool.py` |

### What Is NOT Sent to LangSmith

`process_inputs` filters strip the following before any data leaves the process:

- `self` (contains Anthropic client + settings with API key)
- `session` (SQLAlchemy AsyncSession — internal DB handle)
- `run_id` (internal DB FK)
- Service objects (`rag_service`, `classifier_service`, `weather_service`)
- System prompts (never passed as arguments to public methods)
- Auth tokens and webhook URLs (never in traced function arguments)

### Settings Fields

| Field | Default | Purpose |
|---|---|---|
| `langchain_tracing_v2` | `False` | Enable/disable LangSmith tracing |
| `langchain_api_key` | `None` | LangSmith API key |
| `langchain_project` | `"smart-travel-planner"` | LangSmith project name |

### LangGraph Native Tracing

LangGraph also emits native LangSmith traces for each graph node when
`LANGCHAIN_TRACING_V2=true`. This happens automatically alongside the custom
`@traceable` instrumentation. The custom decorators add LLM call and tool-level
spans that LangGraph's built-in integration does not cover (raw Anthropic SDK calls).

### Note on README Screenshot

A cost breakdown screenshot should be captured manually after running one real
end-to-end query with `LANGCHAIN_TRACING_V2=true` and added to the README.

## Trace Inspection Routes

### Overview

Two protected read-only endpoints let a logged-in user inspect their own agent run history.
Internal fields (error messages, webhook URLs, stack traces) are never included in any response.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/traces` | Recent runs for the current user (default 20, max 50) |
| `GET` | `/traces/{run_id}` | Full trace detail: run + tool calls + LLM usage + trace events |

### Security

- Both endpoints require `Depends(get_current_user)` (Supabase JWT, 401 on missing/invalid).
- `get_agent_run_for_user` enforces user scoping: runs belonging to a different user return `None`, which becomes a 404.
- The following fields are **never** included in any response: `error_message`, `webhook_status_code`, API keys, system prompts, webhook URLs, raw stack traces.

### Response Schemas (`app/schemas/traces.py`)

| Schema | Used in |
|---|---|
| `AgentRunSummary` | `GET /traces` list items |
| `AgentRunTraceDetail` | `GET /traces/{run_id}` root object |
| `ToolCallLogResponse` | Nested in detail: tool invocations |
| `LLMUsageLogResponse` | Nested in detail: model calls and costs |
| `AgentTraceEventResponse` | Nested in detail: observability events |

### Repository helpers (`app/db/repositories/agent_runs.py`)

Three new query helpers (all existing user-scoping helpers were already in place):

- `list_tool_calls_for_run(session, run_id)` — chronological tool call logs
- `list_llm_usage_for_run(session, run_id)` — chronological LLM usage records
- `list_trace_events_for_run(session, run_id)` — chronological trace events

### Key Files

| File | Responsibility |
|---|---|
| `app/routers/traces.py` | Route handlers with structured logging |
| `app/schemas/traces.py` | Response schemas (5 Pydantic models) |
| `app/db/repositories/agent_runs.py` | Added 3 read query helpers |
| `tests/test_traces.py` | 15 tests covering auth, scoping, fields, and forbidden keys |

## Next Steps

## References

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [Pydantic Validation](https://docs.pydantic.dev/latest/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Anthropic API](https://docs.anthropic.com/)

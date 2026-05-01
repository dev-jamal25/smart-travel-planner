# AI Worker Instructions — Smart Travel Planner

## Project Identity

This repository is for the project: **Smart Travel Planner**.

The app is an AI travel planning system that:
1. Accepts a user trip request from a React UI.
2. Uses an agent to understand the request.
3. Retrieves destination knowledge through RAG.
4. Classifies destinations by travel style using a trained ML classifier.
5. Fetches live conditions such as weather, flights, or FX.
6. Synthesizes a real travel plan.
7. Stores users, agent runs, tool calls, and embeddings in Postgres + pgvector.
8. Sends the final plan to a webhook destination.
9. Runs as a full Dockerized stack.

The project is not just a chatbot. It is a multi-tool AI + ML system with production-style engineering standards.

---

## Non-Negotiable Engineering Rules

These rules apply to every backend change.

### Async

- All FastAPI routes must be `async def`.
- All I/O must be awaited.
- Use `httpx.AsyncClient`, never `requests`.
- Use SQLAlchemy 2.x async sessions.
- No `time.sleep()` in request paths.
- External LLM/API/webhook calls must be async.
- Scikit-learn prediction is CPU-bound; it can run inline only if lightweight. If it becomes heavy, wrap it with `asyncio.to_thread()`.

### Dependency Injection

- Use FastAPI `Depends()` for:
  - database sessions
  - current user
  - LLM clients
  - ML classifier service
  - RAG/vector store service
  - agent executor
  - settings/config
- Do not instantiate clients inside routes.
- Do not use module-level globals for shared resources.

### Lifespan Singletons

Create expensive shared resources once in FastAPI lifespan and expose them through dependencies:

- database engine
- loaded ML model
- embedding model/client
- LLM clients
- HTTP client pool if used
- agent executor
- vector store connection

Do not load `joblib` models, embedding models, or API clients per request.

### Pydantic Boundaries

Every external boundary must use Pydantic:

- HTTP request bodies
- HTTP response bodies
- auth payloads
- agent tool inputs
- tool outputs
- LLM structured outputs
- webhook payloads
- config/settings

Validate at the edge. Do not scatter defensive validation inside business logic.

### Configuration

- Use one `Settings` class with `pydantic-settings`.
- Use `extra="forbid"`.
- No scattered `os.getenv`.
- No hardcoded API URLs, model names, secrets, JWT settings, database URLs, or webhook URLs.
- `.env.example` must include every required variable with fake placeholder values only.
- `.env` must never be committed.

### Error Handling

- Every external call must have:
  - timeout
  - retry with bounded exponential backoff
  - structured logging on failure
- Catch specific exceptions, not bare `except`.
- Do not return `200 OK` with an error body for real API errors.
- Use proper `HTTPException` status codes in routes.
- Tool failures must return structured tool errors to the agent instead of crashing the whole run.
- Webhook failure must be logged but must not break the user-facing response.

### Logging

- Use structured logging.
- No `print()` in backend code.
- Logs should include useful context:
  - request ID if available
  - user ID
  - agent run ID
  - tool name
  - latency
  - error type
- Never log secrets, tokens, passwords, full auth headers, or sensitive request bodies.

### Testing

Every critical feature must include tests.

Minimum required tests:
- Pydantic schema tests with valid and invalid inputs.
- One unit test per agent tool with external services mocked.
- One end-to-end agent flow test with mocked LLM/API calls.
- Auth tests for signup/login/protected route access.
- RAG retrieval tests for a few hand-written queries.
- ML training/evaluation code must prevent leakage and use reproducible seeds.

Tests must be runnable through a single command.

### Code Hygiene

- Keep backend and frontend separate.
- Do not put everything in `main.py`.
- Use routers from day one.
- Split by responsibility:
  - routes
  - schemas
  - services
  - tools
  - agents
  - db/models
  - config
  - dependencies
  - tests
- Avoid vague files like `utils.py`, `helpers.py`, `stage1.py`, or `misc.py`.
- Use descriptive names.

### Type Hints

- Every function must have parameter and return type hints.
- Avoid `Any` unless there is a real reason.
- Prefer explicit domain models over loose dictionaries.

---

## Required Project Features

The final project must include the following.

### 1. ML Classifier

Build a classifier for destination travel style:

- Adventure
- Relaxation
- Culture
- Budget
- Luxury
- Family

Rules:
- Labels must be justified with clear labeling rules.
- Features must be justified.
- Use a scikit-learn `Pipeline`.
- Preprocessing must live inside the pipeline.
- Compare at least 3 classifiers.
- Use k-fold cross-validation.
- Report accuracy and macro F1 with mean and standard deviation.
- Tune at least one model.
- Address class imbalance honestly.
- Report per-class metrics.
- Save all experiments to `results.csv`.
- Save the winning model with `joblib`.
- Fix random seeds.

Never preprocess the full dataset before train/test split if that causes leakage.

### 2. RAG Tool

Build retrieval over real destination content.

Rules:
- Use 10–15 destinations.
- Use 20–30 real documents from sources such as Wikivoyage, travel blogs, or tourism boards.
- Store embeddings in Postgres using pgvector.
- Use the same database as the rest of the app.
- Justify chunk size, overlap, and retrieval strategy in the README.
- Test retrieval with hand-written queries before connecting it to the agent.

### 3. Agent With Three Tools

Use LangGraph
Required tools:
1. Destination knowledge retrieval tool.
2. Destination style classifier tool.
3. Live conditions tool: weather

Rules:
- Every tool input must have a Pydantic schema.
- Maintain an explicit tool allowlist.
- The LLM must not be allowed to invent tools.
- Invalid tool inputs should cause retry/repair, not a crash.
- Tool errors should be returned as structured errors.
- The final answer must synthesize across tool outputs, not concatenate them.
- If RAG says one thing and live data says another, the final answer must explain the tension.

## Live Conditions Tool MVP

The live conditions tool is weather-only for MVP.

Use Open-Meteo through async HTTP calls.

Rules:
- Use `httpx.AsyncClient`, never `requests`.
- Add timeout.
- Add retry with bounded exponential backoff.
- Cache weather responses with a TTL cache where sensible.
- Weather cache TTL must come from Settings.
- Weather failure must return a structured tool error.
- Weather failure must not crash the agent run.
- The final answer should clearly say when fresh weather could not be fetched.

Do not implement flights or FX unless explicitly requested.

### 4. Model Routing Strategy

This project uses Anthropic model routing.

Use Haiku for cheap mechanical work:
- extracting structured travel preferences from user text
- validating or repairing tool arguments
- rewriting RAG search queries
- simple routing decisions
- summarising retrieved chunks before final synthesis if needed

Use Sonnet for high-value reasoning:
- final travel plan synthesis
- resolving conflicts between RAG and weather data
- deciding what trade-offs to explain to the user
- producing the final user-facing answer

Every LLM call must log:
- model name
- purpose of the call
- input token count if available
- output token count if available
- estimated cost if available
- latency
- agent run ID

Model names must come from Settings, not hardcoded strings.

### 5. Persistence

Use Postgres + pgvector + SQLAlchemy.

Persist at minimum:
- users
- agent runs
- original user query
- final answer
- tool calls
- tool inputs
- tool outputs or summaries
- timestamps
- embeddings/chunks
- webhook delivery status

Every agent run must be scoped to the logged-in user.

### 6. Auth Strategy

This project uses Supabase JWT verification with **JWKS public-key validation**.

Rules:
- Verify Bearer tokens using PyJWT.
- Use FastAPI `HTTPBearer` so protected endpoints show the Swagger auth lock.
- The current Supabase project issues `ES256` access tokens.
- Because `ES256` is asymmetric, verify tokens using Supabase JWKS, not the old HS256 shared secret.
- Use these Settings fields:
  - `supabase_jwt_jwks_url`
  - `supabase_jwt_audience`
  - `supabase_jwt_issuer`
  - `supabase_anon_key`
  - `supabase_service_role_key`
- Do not hardcode the Supabase project ref or JWKS URL outside Settings.
- Cache JWKS/public-key lookup where appropriate so the app does not fetch JWKS on every request.
- Validate:
  - token signature
  - `aud`
  - `iss`
  - `exp`
- Return a `CurrentUser` object with:
  - `user_id: UUID`
  - `email: str`
- Missing token must return 401.
- Invalid or expired token must return 401.
- Protected routes must use `Depends(get_current_user)`.
- Every agent run must be scoped to the current user.

Legacy note:
- Do not use HS256 verification for this project unless Supabase is explicitly reconfigured to issue HS256 tokens.
- `supabase_jwt_secret` is not used for the current ES256/JWKS auth flow.

### 7. React Frontend

Use Vite + React.

Required UI:
- signup/login flow
- chat-style trip planning interface
- display final trip plan
- show what tools fired
- show useful tool results/summaries
- show run history if feasible

Streaming is optional, but do not break core functionality to chase streaming.

### 8. Webhook Delivery

The webhook target is Discord.

Rules:
- Send the final trip plan to the configured Discord webhook URL.
- The webhook URL must come from Settings.
- Use async HTTP through `httpx.AsyncClient`.
- Add timeout.
- Add at least one retry with backoff.
- Log success and failure with structured logs.
- Persist webhook delivery status if the relevant table exists.
- Webhook failure must not break the API response shown to the user.
- Never log the full Discord webhook URL.

### 9. Docker

The whole stack must run with:

## No Duplicate Architecture Rule

Before creating a new file, search for an existing file with the same responsibility.

Do not create:
- a second config system
- a second database session system
- a second auth system
- a second logger setup
- a second agent runner
- a second classifier wrapper
- a second webhook client
- a second RAG service

Extend the existing system unless it clearly violates the standards.
If replacing something, explain why in `docs/ai_change_log.txt`.

```bash
docker compose up
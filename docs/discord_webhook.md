# Discord Webhook Delivery

## Why Discord

Discord webhooks are the simplest zero-auth delivery mechanism for structured messages.
A single POST with a JSON body sends a formatted message to any Discord channel.
No bot token, no OAuth — just a URL stored in Settings.

## Payload

The service sends a Discord `content` message (plain text, up to 2,000 characters) containing only **user-facing information**.

Fields included:

| Field | Source |
|---|---|
| Title | Hardcoded "Smart Travel Planner — New Trip Plan" |
| User query | `WebhookTripPlanRequest.user_query` |
| Destination | `WebhookTripPlanRequest.destination` (omitted if None) |
| Trip plan | `WebhookTripPlanRequest.trip_plan` (truncated at 1,400 chars if needed) |

The full `content` field is clamped to 2,000 characters before sending.

### Internal Data Not Sent to Discord

The following are **not** included in the Discord message:

- **Tools used**: Which agent tools fired (RAG retrieval, classifier, weather)
- **Estimated cost**: LLM API usage costs

These internal details are valuable for debugging and cost tracking, and will be stored in:
- Database tables: `AgentRun`, `ToolCall`, LLM usage logs
- Structured logs: request ID, user ID, tool names, token counts, latencies
- Admin dashboards: for monitoring and cost analysis

## Timeout and Retry

- `httpx.Timeout(10.0)` on every request.
- Tenacity `AsyncRetrying`: 3 attempts, exponential backoff 0.5 s → 8 s max.
- Retries only on `httpx.RequestError` (network failures) and HTTP 5xx (treated as `RequestError`).
- HTTP 4xx responses from Discord are **not retried** — they indicate a bad payload or revoked webhook URL and should be fixed, not retried.

## Failure Isolation

`WebhookService.send_trip_plan()` never raises to its caller.
On any failure (network error, 4xx, 5xx after retries) it returns:

```json
{"delivered": false, "provider": "discord", "status_code": 400, "error": "..."}
```

The `POST /webhook/test-discord` endpoint always returns HTTP 200.
A failed delivery is communicated through `delivered: false` in the response body.
This matches the CLAUDE.md rule: webhook failure must not break the user-facing response.

The full Discord webhook URL is **never logged**.

## How the Agent Will Call It

After the LangGraph agent produces the final synthesized trip plan, it will:

1. Construct a `WebhookTripPlanRequest` from:
   - the original user query stored on the agent run
   - the recommended destination from the classifier/RAG tools
   - the final plan text from the Sonnet synthesis step
   - (NOT: tools_used — stored in DB/logs instead)
   - (NOT: cost — stored in LLM usage logs instead)

2. Call `WebhookService.send_trip_plan(req)` via the injected dependency.

3. Persist the returned `delivered` flag and `status_code` on the `AgentRun` row
   (or a `WebhookDelivery` table if one is added later).

4. Return the final plan to the user regardless of webhook outcome.

## Manual Swagger Test

1. Start the server: `uv run uvicorn app.main:app --reload`
2. Open `http://localhost:8000/docs`
3. Authenticate with a valid Supabase Bearer token (click the lock icon).
4. Expand `POST /webhook/test-discord`.
5. Use this body:

```json
{
  "user_query": "I want two weeks in July, warm weather, hiking, and around $1500.",
  "destination": "Interlaken",
  "trip_plan": "Interlaken is a strong adventure match with mountain access, lakes, and hiking routes. Weather should be checked before final booking."
}
```

6. A successful delivery returns `{"delivered": true, "provider": "discord", "status_code": 204}`.
7. If the webhook URL in `.env` is not a real Discord URL, the response will be
   `{"delivered": false, ...}` — this is correct failure isolation behaviour.

# Weather Tool — Design Notes

## Why weather-only for MVP

The CLAUDE.md specification names three live conditions: weather, flights, and FX.
For MVP, weather is the only one implemented because:

- It is the most universally useful signal for any trip type (adventure, beach, culture).
- Open-Meteo provides a free, no-key, globally-available forecast API.
- Adding flights or FX requires third-party API keys and quota agreements.
- The agent can still produce a useful travel plan with weather + RAG + classifier.

Flights and FX can be added later as separate tools inside the same `WeatherService`-style pattern.

## Why fixed coordinates instead of geocoding

The 10 supported RAG destinations each have a single fixed (latitude, longitude) pair
stored in `app/services/destination_coordinates.py`.

Reasons:
- Geocoding requires an external API call on every request (added latency, added failure mode).
- All 10 destinations are large, well-known cities or regions — their coordinates are stable.
- Keeping the coordinate map in code makes it auditable, testable, and dependency-free.
- The agent only needs weather for destinations that are in the RAG corpus.
  If the corpus grows, the coordinate map grows with it (one line per destination).

Kraków is indexed under both `"krakow"` (ASCII) and `"kraków"` (Unicode) so that
casual user input or Haiku-extracted strings both resolve correctly.

## Open-Meteo as the weather source

URL: `https://api.open-meteo.com/v1/forecast`

Parameters used:
- `current=temperature_2m,weather_code` — live temperature and WMO weather condition code
- `daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max` — daily forecast
- `timezone=auto` — Open-Meteo infers the timezone from coordinates
- `forecast_days=<1–7>` — controlled by the caller

No API key is required. Rate limits are permissive for personal/research use.
If this app scales to production traffic, a commercial Open-Meteo plan should be considered.

## Timeout, retry, and cache

### Timeout

`httpx.AsyncClient` is configured with a 10-second timeout (all phases: connect + read).
Open-Meteo responses are small JSON payloads; 10 seconds is generous.

### Retry

`tenacity.AsyncRetrying` is used with:
- `stop_after_attempt(3)` — at most 3 total attempts
- `wait_exponential(multiplier=0.5, max=8)` — waits ~0.5 s, ~1 s, ~4 s between retries
- `retry_if_exception_type(httpx.RequestError)` — retries only on network-level errors
  (DNS failures, connection resets, timeouts). HTTP status errors (4xx/5xx from the server)
  are not retried because they indicate a permanent parameter problem or a server outage
  that will not resolve in seconds.

### TTL cache

`cachetools.TTLCache(maxsize=128, ttl=settings.weather_cache_ttl_seconds)` is held on the
`WeatherService` instance. The `WeatherService` is a lifespan singleton stored on
`app.state.weather_service`, so the cache survives across requests within one server process.

Default TTL is 600 seconds (10 minutes), configurable via `WEATHER_CACHE_TTL_SECONDS` in `.env`.

When a cache hit occurs, the response is returned with `cached: true` in the JSON body.
This lets the agent (and Swagger testers) see whether the data came from the network or
from the in-process cache.

## How this becomes the live_conditions agent tool

The planned LangGraph tool will look like:

```python
class LiveConditionsInput(BaseModel):
    destination: str
    forecast_days: int = 3

async def live_conditions_tool(
    input: LiveConditionsInput,
    weather_service: WeatherService,  # injected by agent dependency
) -> WeatherForecastResponse | WeatherToolError:
    try:
        return await weather_service.get_forecast(
            destination=input.destination,
            forecast_days=input.forecast_days,
        )
    except WeatherServiceError as exc:
        return WeatherToolError(error=str(exc), retryable=exc.retryable)
```

Key points:
- The same `WeatherService` instance is reused (no new client per tool call).
- `WeatherToolError` (already in `app/schemas/weather.py`) is the structured error returned
  to the agent instead of crashing the run.
- If the agent gets a `WeatherToolError`, it should note in the final answer that live
  weather data was unavailable for the destination and fall back to the RAG context.
- Haiku can be used to validate or repair `LiveConditionsInput` before calling the tool.

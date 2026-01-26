# Architecture & Conventions

This document describes the architectural patterns, coding standards, and design decisions for the "Shall We Swim Today?" project.

**Architecture pattern**: Data aggregator (not a proxy). The app fetches data from multiple upstream APIs in background tasks, caches it in memory, and serves processed results. User requests never trigger external API calls.

## 1. Architectural Patterns

### Project Structure

```
shallweswim/
├── main.py          # App entry point, web UI routes, templates
├── api.py           # JSON API routes (delegates to data.py)
├── data.py          # LocationDataManager - coordinates feeds per location
├── feeds.py         # Feed classes with caching/expiration
├── config.py        # Location configs, station IDs, feed settings
├── plotting.py      # Chart generation (runs in process pool)
├── util.py          # Shared utilities
└── clients/         # External API clients
    ├── base.py      # BaseApiClient with retry logic, error hierarchy
    ├── coops.py     # NOAA CO-OPS (tides, currents, coastal temps)
    ├── ndbc.py      # NOAA NDBC (buoy temperatures)
    └── nwis.py      # USGS NWIS (river temps, discharge)

tests/               # Unit and integration tests
templates/           # Jinja2 HTML templates
static/              # CSS, JS, images
```

### Modular Design

- **API (`api.py`)**: Contains _only_ route handlers and request validation. Delegates business logic to `data.py` or `feeds.py`.
- **Data Management (`data.py`)**: `LocationDataManager` is the facade for all data operations per location. It coordinates multiple `Feed`s.
- **Feeds (`feeds.py`)**: Encapsulates data fetching, caching, and validation. Each data source type (e.g., `NoaaTidesFeed`) is a separate class inheriting from `Feed`.
- **Clients (`clients/`)**: Pure API clients. They handle HTTP requests, retries, and raw data parsing, but know nothing about the application's business logic (caching, expiration).

### Data Flow

**User requests** serve cached data (no external calls):

```
API Handler → LocationDataManager → Feed (cached data) → Response
```

**Background refresh** updates the cache on intervals:

```
Background Task → Feed → ApiClient → External Service → Update Cache
```

### Configuration

- All static configuration is in `config.py`.
- Use Pydantic models for configuration schemas (e.g., `LocationConfig`, `BaseFeedConfig`).
- Avoid hardcoding constants in logic files; move them to `config.py` or module-level constants if they are "magic numbers".

## 2. Coding Standards

### Typing

- **Strict Typing**: All functions and methods must have type hints.
- **Pydantic**: Use Pydantic v2 models for all data structures passed between layers (especially API responses).
  - Use `model_config = ConfigDict(...)` for model configuration.
- **Pandas**: When passing DataFrames, use `pandera` models or clear docstrings to describe the expected columns.

### Asynchronous Programming

- **FastAPI**: Route handlers must be `async`.
- **I/O Bound**: All network I/O must be asynchronous (`aiohttp`).
- **CPU Bound**: Heavy computation (like plotting with Matplotlib) must be offloaded to a process pool (`app.state.process_pool`) to avoid blocking the event loop.

### Error Handling

- **API Layer**: Catch internal exceptions and convert them to `HTTPException` with appropriate status codes (404, 503, etc.).
- **Data Layer**: Fail fast. Use `AssertionError` for "impossible" states (e.g., missing app state). Raise `ValueError` or specific custom exceptions for operational errors (e.g., parsing failure).
- **Feeds**: Feeds call API clients (which handle retries internally) and expose a "healthy" status property based on data freshness.

### Logging

- Use the standard `logging` module.
- Format: `logging.info(f"[{context}] Message")` where context is often the location code or component name.

## 3. Testing

- **Unit Tests**: Must run fast. Mock all external network calls using `unittest.mock` or `pytest-mock`.
- **Integration Tests**: Marked with `@pytest.mark.integration`. These hit real external APIs and are run separately.
  - Locations with `test_required=True` in config (e.g., NYC) must pass
  - Other locations skip gracefully on data unavailability
  - Run with: `uv run pytest -m integration --run-integration`
- **Fixtures**: Use `conftest.py` for shared fixtures.

## 4. Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes.
  - **Args**: List arguments and their types.
  - **Returns**: Describe the return value.
  - **Raises**: List exceptions raised.

## 5. Station Outage Handling

External data sources (NOAA CO-OPS, NOAA NDBC, USGS NWIS) may have temporary outages. The application handles these gracefully through principled error handling.

### Two Code Paths

**Background feed refresh** (populates data):

- `StationUnavailableError` → WARNING log (no alert)
- Other `BaseClientError` → ERROR log (alert)
- Other exceptions → ERROR log (alert)
- Feed stays stale, retries on next interval

**API request handlers** (serves users):

- Feed not ready → 503 + WARNING log
- Other exceptions → 500 + ERROR log (bug)

### Exception Classes (`clients/base.py`)

- **`StationUnavailableError`**: Use ONLY for confirmed "no data" conditions

  - NDBC returns empty dict `{}`
  - COOPS returns "No data was found"
  - Empty DataFrame for time range

- **`*DataError`**: Unexpected data format, parsing failures

  - May indicate API changed - needs investigation

- **`RetryableClientError`**: Transient network issues (timeouts, connection errors)
  - Automatically retried by `BaseApiClient.request_with_retry()`

### Logging Guidelines

- **WARNING**: Expected operational issues (station outages)

  - Does NOT trigger GCP alerts (query: `severity=ERROR`)
  - Visible in logs for debugging

- **ERROR**: Unexpected issues requiring attention
  - Triggers GCP alerts
  - Indicates potential bug or API change

### Health Check (`/api/healthy`)

- Returns **200** if at least 1 location has data (fresh or stale)
- Returns **503** only if NO location can serve any data
- Used by Cloud Run for routing decisions
- Single station outages do NOT trigger 503

### Monitoring (`/api/status`)

- Returns detailed status for all locations/feeds
- Shows `is_healthy`, `is_expired`, `age_seconds` per feed
- Use external monitoring (GCP Cloud Monitoring) to alert on stale data
- Recommended: Alert if `is_healthy: false` persists > 30 minutes for critical feeds

## 6. Feed Lifecycle

Each feed has an **expiration interval** that determines how often it refreshes:

| Feed Type              | Refresh Interval | Rationale              |
| ---------------------- | ---------------- | ---------------------- |
| Tides                  | 24 hours         | Predictions are stable |
| Currents               | 24 hours         | Predictions are stable |
| Live Temperature       | 10 minutes       | Real-time observations |
| Historical Temperature | 3 hours          | Slower-changing data   |

**Health status** uses a 15-minute buffer to prevent flapping:

- `is_expired`: Data older than refresh interval
- `is_healthy`: Data within interval + 15-minute buffer

Background tasks continuously refresh feeds. Failed fetches leave the feed stale (serving old data) until the next successful refresh.

# Architecture & Conventions

This document describes the architectural patterns, coding standards, and design decisions for the "Shall We Swim Today?" project.

**Architecture pattern**: Data aggregator (not a proxy). The app fetches data from multiple upstream APIs in background tasks, caches it in memory, and serves processed results. User requests never trigger external API calls.

## 1. Architectural Patterns

### Project Structure

```text
shallweswim/
├── main.py              # App entry point, web UI routes, templates
├── api/                 # API layer
│   ├── __init__.py      # Re-exports from routes
│   └── routes.py        # JSON API routes (delegates to core/)
├── config/              # Configuration layer
│   ├── __init__.py      # Re-exports from locations
│   └── locations.py     # Location configs, station IDs, feed settings
├── core/                # Business logic layer
│   ├── __init__.py
│   ├── manager.py       # LocationDataManager - coordinates feeds
│   ├── queries.py       # Query functions (temp, tide, current info)
│   ├── updater.py       # Background update helpers
│   └── feeds.py         # Feed classes with caching/expiration
├── clients/             # External API clients
│   ├── base.py          # BaseApiClient with retry logic, error hierarchy
│   ├── coops.py         # NOAA CO-OPS (tides, currents, coastal temps)
│   ├── ndbc.py          # NOAA NDBC (buoy temperatures)
│   └── nwis.py          # USGS NWIS (river temps, discharge)
├── plotting.py          # Chart generation (runs in process pool)
└── util.py              # Shared utilities

tests/                   # Unit and integration tests
templates/               # Jinja2 HTML templates
static/                  # CSS, JS, images
```

**Backwards compatibility**: Top-level shim files (`api.py`, `config.py`, `data.py`, `feeds.py`) re-export from the new locations for import compatibility.

### Modular Design

- **API (`api/routes.py`)**: Contains _only_ route handlers and request validation. Delegates business logic to `core/`.
- **Core Manager (`core/manager.py`)**: `LocationDataManager` is the facade for all data operations per location. It coordinates multiple `Feed`s and manages the background update loop.
- **Core Queries (`core/queries.py`)**: Standalone functions for querying feed data (temperature, tide info, current predictions).
- **Core Updater (`core/updater.py`)**: Helper functions for the background update loop (expiration checks, dataset updates, exception handling).
- **Feeds (`core/feeds.py`)**: Encapsulates data fetching, caching, and validation. Each data source type (e.g., `CoopsTidesFeed`) is a separate class inheriting from `Feed`.
- **Config (`config/locations.py`)**: Location configurations, station IDs, and feed settings using Pydantic models.
- **Clients (`clients/`)**: Pure API clients. They handle HTTP requests, retries, and raw data parsing, but know nothing about the application's business logic (caching, expiration).

### Data Flow

**User requests** serve cached data (no external calls):

```text
API Handler → LocationDataManager → Feed (cached data) → Response
```

**Background refresh** updates the cache on intervals:

```text
Background Task → Feed → ApiClient → External Service → Update Cache
```

### Configuration

- All static configuration is in `config/locations.py`.
- Use Pydantic models for configuration schemas (e.g., `LocationConfig`, `BaseFeedConfig`).
- Avoid hardcoding constants in logic files; move them to `config/` or module-level constants if they are "magic numbers".

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

Two error types for data availability, at different layers:

| Error | Layer | When | Result |
|-------|-------|------|--------|
| `StationUnavailableError` | Backend (`clients/`) | NOAA/USGS returns no data | Schedules retry, swallowed |
| `DataUnavailableError` | Core (`core/queries.py`) | Request for unavailable data | HTTP 503 |

- **API Layer**: Catch `DataUnavailableError` and convert to `HTTPException(503)`. Other exceptions become 500.
- **Core Layer**: `get_feed_data()` raises `DataUnavailableError` when feed data is unavailable. This centralizes the check.
- **Data Layer**: Fail fast. Use `AssertionError` for "impossible" states (e.g., missing app state).
- **Feeds**: Feeds call API clients (which handle retries internally) and expose a "healthy" status property based on data freshness.

### Logging

- Use the standard `logging` module.
- Format: `logging.info(f"[{context}] Message")` where context is often the location code or component name.

## 3. Testing

- **Unit Tests**: Must run fast. Mock all external network calls using `unittest.mock` or `pytest-mock`.
- **Integration Tests**: Marked with `@pytest.mark.integration`. These hit real external APIs and are run separately.
  - Each feed (temperature, tides, currents) is validated independently per location
  - Locations with `test_required=True` (e.g., NYC): any missing or stale feed fails the test
  - Other locations: unavailable feeds are collected as skip reasons; test skips if any feed is unavailable but never blocks validation of other feeds
  - Run with: `uv run pytest -m integration --run-integration`
- **Fixtures**: Use `conftest.py` for shared fixtures.
- **Warnings as Errors**: Pytest treats all warnings as errors (`filterwarnings = ["error"]`).
  - New warnings fail tests immediately, forcing explicit decisions
  - Known third-party warnings are filtered with comments explaining why
  - Filtered warnings should be revisited periodically (check for upstream fixes)

## 4. Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes.
  - **Args**: List arguments and their types.
  - **Returns**: Describe the return value.
  - **Raises**: List exceptions raised.

## 5. Station Outage Handling

External data sources (NOAA CO-OPS, NOAA NDBC, USGS NWIS) may have temporary outages. The application handles these gracefully through principled error handling.

### Feed Scheduling

Feeds track two timestamps:

- `_fetch_timestamp`: When data was successfully fetched (for `age` monitoring)
- `_next_fetch_after`: When to attempt next fetch (for scheduling)

Both success and failure update `_next_fetch_after`, preventing runaway retries:

- **Success**: Set both `_fetch_timestamp` and `_next_fetch_after`
- **StationUnavailableError**: Only set `_next_fetch_after` (data stays stale but won't retry immediately)
- **Other errors**: Set `_next_fetch_after`, then propagate error for logging

### Two Code Paths

**Background feed refresh** (populates data):

- `StationUnavailableError` → WARNING log (no alert), schedules retry at normal interval
- Other `BaseClientError` → ERROR log (alert), schedules retry at normal interval
- Other exceptions → ERROR log (alert), schedules retry at normal interval

**API request handlers** (serves users):

- Requested location/feed/capability is not configured → 404
- No data available → 503 + WARNING log
- Partial location data available → 200 with unavailable fields set to `null`
- Stale data → Serve it (better than 503 for users)
- Missing/empty app manager state for configured locations → 500 + ERROR log
- Other exceptions → 500 + ERROR log (bug)

### Exception Classes

**Backend (`clients/base.py`)**:

- **`StationUnavailableError`**: Use ONLY for confirmed "no data" conditions

  - NDBC returns empty dict `{}`
  - COOPS returns "No data was found"
  - Empty DataFrame for time range

- **`*DataError`**: Unexpected data format, parsing failures

  - May indicate API changed - needs investigation

- **`RetryableClientError`**: Transient network issues (timeouts, connection errors)
  - Automatically retried by `BaseApiClient.request_with_retry()`

**Core (`core/queries.py`)**:

- **`DataUnavailableError`**: Feed data requested but not currently available
  - Raised by `get_feed_data()` when `feed is None or feed._data is None`
  - Expected operational condition (station outage or startup race)
  - API routes catch this and return HTTP 503

### Client Timeouts

All API clients enforce a 30-second timeout on individual requests (`REQUEST_TIMEOUT` in `clients/base.py`):

- **COOPS**: Uses aiohttp per-request timeout
- **NWIS/NDBC**: Uses `asyncio.wait_for()` around synchronous library calls

Timeouts raise `RetryableClientError` and are automatically retried by `request_with_retry()`.

### Plot Generation

Plot generation runs in a `ProcessPoolExecutor` (bounded to `os.cpu_count()` workers) using a fire-and-forget pattern:

- **Submit**: Each update loop iteration submits plot tasks to the pool via `loop.run_in_executor()` and returns immediately (no awaiting)
- **Collect**: On the next iteration, completed futures are harvested and results stored. The async loop is never blocked by plot generation
- **Guard**: `_pending_plot_futures` tracks in-flight work per feed — prevents duplicate submissions while a worker is still running
- **Hard timeout** (`PLOT_HARD_TIMEOUT`, 300s): If a worker hasn't finished, tracking is dropped (orphaned worker cannot be killed — Python limitation) and the location can retry
- **Key constraint**: `ProcessPoolExecutor` futures cannot be cancelled. Never use `asyncio.wait_for` or `asyncio.gather` on them — cancelled futures appear "done" but the worker keeps running, causing task stacking

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
- Missing/empty location manager state returns **503** here because the
  health endpoint answers "can this instance serve traffic?"

### Monitoring (`/api/status`)

- Returns detailed status for all locations/feeds
- Shows `is_healthy`, `is_expired`, `age_seconds` per feed
- Use external monitoring (GCP Cloud Monitoring) to alert on stale data
- Recommended: Alert if `is_healthy: false` persists > 30 minutes for critical feeds
- Missing/empty location manager state returns **500** because `/api/status`
  is diagnostic; no managers after startup indicates an internal
  initialization/configuration bug, not an upstream station outage.

## 6. Feed Lifecycle

Each feed has an **expiration interval** that determines how often it refreshes:

| Feed Type              | Refresh Interval | Rationale              |
| ---------------------- | ---------------- | ---------------------- |
| Tides                  | 24 hours         | Predictions are stable |
| Currents               | 24 hours         | Predictions are stable |
| Live Temperature       | 10 minutes       | Real-time observations |
| Historical Temperature | 3 hours          | Slower-changing data   |

**Feed status properties**:

- `has_data`: Any data exists (used by API endpoints - serve stale over 503)
- `is_expired`: Data older than refresh interval (triggers background refresh)
- `is_healthy`: Data within interval + 15-minute buffer (for monitoring display)

Background tasks continuously refresh feeds. Failed fetches leave the feed stale (serving old data) until the next successful refresh.

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
├── plot.py              # Chart generation (runs in process pool)
└── util.py              # Shared utilities

tests/                   # Unit and integration tests
frontend/                # React/Vite app source and frontend build tooling
templates/               # Jinja2 HTML templates
static/                  # CSS, JS, images
```

**Backwards compatibility**: Top-level shim files (`api.py`, `config.py`, `data.py`, `feeds.py`) re-export from the new locations for import compatibility.

### Modular Design

- **API (`api/routes.py`)**: Contains _only_ route handlers and request validation. Delegates business logic to `core/`.
- **Frontend app (`frontend/`)**: React/Vite/TypeScript app mounted at root
  location routes.
  It consumes the FastAPI JSON API and generated OpenAPI types. It must not own
  NOAA/USGS fetching, feed orchestration, caching, plotting logic, or station
  configuration.
- **Core Manager (`core/manager.py`)**: `LocationDataManager` is the facade for all data operations per location. It coordinates multiple `Feed`s and manages the background update loop.
- **Core Queries (`core/queries.py`)**: Standalone functions for querying feed data (temperature, tide info, current predictions).
- **Core Updater (`core/updater.py`)**: Helper functions for the background update loop (expiration checks, dataset updates, exception handling).
- **Feeds (`core/feeds.py`)**: Encapsulates data fetching, caching, and validation. Each data source type (e.g., `CoopsTidesFeed`) is a separate class inheriting from `Feed`.
- **Config (`config/locations.py`)**: Location configurations, station IDs, and feed settings using Pydantic models.
- **Clients (`clients/`)**: Pure API clients. They handle HTTP requests, retries, and raw data parsing, but know nothing about the application's business logic (caching, expiration).

### Data Flow

**User requests** serve cached data or derived in-memory views (no external calls):

```text
API Handler → LocationDataManager → Feed / Derived Cache → Response
```

**Background refresh** updates raw feeds and derived caches on intervals:

```text
Background Task → Feed → ApiClient → External Service → Update Feed Cache
Background Task → Derived Data Precompute → Update Derived Cache
```

Keep user-facing condition endpoints on the fast path. Expensive, repeatable
work that depends only on cached feed data should run during background updates
or immediately after a feed changes, not on every request. Request handlers may
perform cheap point-in-time lookups and response serialization, but should not
loop across full feed DataFrames for high-traffic or above-the-fold endpoints
such as `/api/{location}/conditions` and `/api/{location}/currents`.

Known tech debt: `/api/{location}/plots/current_tide` still renders a Matplotlib
SVG per request in the process pool. It should eventually cache or precompute
common hourly `at` / `shift` values, while preserving on-demand fallback
behavior for less common planning times.

### Frontend App Serving And Durable HTML

The React app is built to static files in `frontend/dist` with root-relative
Vite asset paths. FastAPI serves:

- `/`, `/locations`, and configured `/{location}` routes as a thin,
  FastAPI-rendered HTML shell with `Cache-Control: no-cache, must-revalidate`
- `/assets/...` from Vite hashed assets with immutable one-year caching
- `/manifest.json` from the existing static manifest route, with `start_url`
  set to `/?source=pwa-react` for installed-app log visibility and `scope` set
  to `/`
- `/legacy/...` for the temporary Jinja-rendered experience while it remains
  available

FastAPI enables gzip compression for compressible responses larger than 1 KiB,
including HTML shells, JSON APIs, XML/text responses, and Vite CSS/JavaScript
assets. Hashed Vite assets still keep immutable one-year caching; compression is
negotiated per request with `Accept-Encoding`.

The app shell reuses `frontend/dist/index.html` so Vite-managed script and
stylesheet tags remain the source of truth. FastAPI adds route-specific
`title`, description, canonical, Open Graph, JSON alternate links, compact
`noscript` fallback content, and conservative JSON-LD before returning the
shell. This is the project's "good web citizen" layer: canonical app routes are
useful to crawlers, sharing previews, no-JavaScript clients, agents, and
archives before React loads. It is not full React SSR and does not introduce a
Node production runtime.

The durable HTML layer reads only static `LocationConfig` metadata and canonical
URL helpers. It does not make loopback HTTP calls and does not include live
condition summaries, ratings, safety guidance, or forecast claims. React remains
the primary interactive UI once JavaScript loads.

Local development may leave `frontend/dist` absent; app routes then return a
clear not-built response. Production/container startup passes
`--require-frontend-dist` and fails loudly if the built shell is missing.

The frontend contract is generated from FastAPI OpenAPI:

```bash
uv run python -m shallweswim.scripts.export_openapi > frontend/openapi.json
corepack pnpm@10.18.3 --dir frontend generate-api
```

API/config ownership:

- `/manifest.json` is the canonical browser install/PWA manifest. Do not mirror
  manifest fields through app bootstrap payloads.
- `/api/locations` is the public discovery endpoint for configured swim
  locations. It should remain general-purpose location metadata plus availability
  summary, not React render configuration. Canonical HTML routes advertise this
  endpoint with `rel="alternate"` links and no-JavaScript fallback links where
  relevant.
- `/api/app/bootstrap` is an app-internal React startup payload. It is still
  exported in OpenAPI so the bundled frontend can use generated types, but it is
  not a stable external-consumer API. It may intentionally duplicate selected
  static location metadata also exposed by `/api/locations` and compose it with
  app presentation configuration to avoid extra startup requests. It may include
  display labels, feature flags, water-movement planner/detail capabilities,
  trusted citation HTML, and external embed configuration, but should not include
  dynamic condition data, manifest metadata, station IDs, or feed internals
  unless there is an explicit frontend need and approval.

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
  - Teardown intentionally uses bounded waits for blocking external-API worker threads. Earlier unbounded cleanup caused GitHub Actions integration jobs to stall when live NOAA/USGS/NDBC calls hung. Local full-suite integration runs may finish all tests and still report unclosed socket `ResourceWarning`s during pytest teardown; preserve bounded cleanup unless CI behavior is revalidated.
- **Performance Tests**: Marked with `@pytest.mark.performance` and kept in `tests/performance/`.
  - Use deterministic in-memory feed data, not live APIs
  - Guard important user-facing request paths against accidental feed-scale DataFrame work
  - Run separately with: `uv run pytest tests/performance -v --run-performance`
  - GitHub Actions runs them on every push and pull request, plus a weekly
    schedule and manual dispatch
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

Feed scheduling is a small state machine:

| State | `_fetch_timestamp` | `_next_fetch_after` | Behavior |
| --- | --- | --- | --- |
| Never attempted | `None` | `None` | Fetch immediately |
| Success, refreshable feed | Updated to success time | Success time + `expiration_interval` | Refresh on normal cadence |
| Success, never-expiring feed | Updated to success time | `None` | Do not refresh automatically |
| Failure before first success | `None` | Failure time + retry delay | Retry after backoff |
| Failure after prior success | Unchanged | Failure time + retry delay | Serve stale data and retry after backoff |

Failure retries use a shared feed-level sequence: **1 minute, 2 minutes,
5 minutes, 10 minutes, 20 minutes, 30 minutes**. If a feed has an
`expiration_interval`, the retry delay is capped at that interval so retries
never become less frequent than the feed's ordinary refresh cadence. Feeds with
`expiration_interval=None` still retry failed attempts using the same sequence;
after a successful fetch, they do not refresh automatically.

### Two Code Paths

**Background feed refresh** (populates data):

- `StationUnavailableError` → WARNING log (no alert), schedules feed-level retry
- Other `BaseClientError` → ERROR log (alert), schedules feed-level retry
- Other exceptions → ERROR log (alert), schedules feed-level retry

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

  - NDBC text files are unavailable or empty for the requested range
  - COOPS returns "No data was found"
  - USGS NWIS continuous-values response has no observations for the requested
    site, parameter, and time range
  - Empty DataFrame for time range

- **`*DataError`**: Unexpected data format, parsing failures

  - May indicate API changed - needs investigation

- **`RetryableClientError`**: Transient network/service issues

  - Timeouts and connection errors
  - Broken protocol responses such as chunked transfer or content decoding errors
  - Retryable HTTP statuses: `429`, `500`, `502`, `503`, `504`
  - Automatically retried by `BaseApiClient.request_with_retry()`

- **`*ApiError`**: Unexpected client/library/API failures that are not known
  transient conditions and are not confirmed no-data responses

### Client Request Wrapping

`BaseApiClient` owns shared retry and logging behavior through
`request_with_retry()`. Each concrete client owns its request helper signature
because CO-OPS, NDBC, and NWIS all require different request parameters.
Do not add `_execute_request` back to the base class contract just to share a
name; pass the concrete helper into `request_with_retry()` instead.

Client request retries and feed scheduling are separate layers:

- `BaseApiClient.request_with_retry()` uses tenacity to retry transient
  HTTP/network failures inside one fetch attempt over seconds.
- `Feed.update()` schedules the next whole-feed attempt over minutes after the
  fetch attempt finishes. This state is visible in `/api/status`.

**Core (`core/queries.py`)**:

- **`DataUnavailableError`**: Feed data requested but not currently available
  - Raised by `get_feed_data()` when `feed is None or feed._data is None`
  - Expected operational condition (station outage or startup race)
  - API routes catch this and return HTTP 503

### Current Phase Semantics

Tidal current API responses include structured fields for displays and API
consumers:

- `phase: "flood"` or `"ebb"` when the current is meaningfully moving
- `phase: "slack_before_flood"` or `"slack_before_ebb"` when absolute current
  magnitude is below `0.2` knots and the next non-slack prediction indicates
  the upcoming direction
- `phase: "slack"` when magnitude is below `0.2` knots and the next direction
  cannot be inferred
- `strength: "light"`, `"moderate"`, or `"strong"` for non-slack tidal
  predictions, based on thirds of `magnitude_pct`
- `trend: "building"`, `"easing"`, or `"steady"` for non-slack tidal
  predictions, based on the directional slope of the prediction curve
- `state_description`, a display-ready phrase such as
  `"strong ebb and building"` or `"slack before flood"`

The `direction` field remains `flooding`/`ebbing` for backwards compatibility
and directional context. Consumers that need compact display text should prefer
`phase`; consumers that need user-facing prose can use `state_description`.

`magnitude` is absolute speed in knots. `magnitude_pct` is cycle-relative: it is
normalized against the peak within the current continuous flood or ebb segment,
not against a fixed theoretical maximum. A neap and spring tide can both report
`magnitude_pct` near `1.0` while having different absolute `magnitude` values.

`range` is optional slack-to-peak context for prediction-backed current displays.
When present, `range.slack` is the relevant slack boundary for the current
segment and trend, and `range.peak` is the peak of the current continuous flood
or ebb segment. `range` is `null` for observation sources, non-tidal currents,
slack-only data, slack phases that are not associated with a non-slack segment,
or incomplete segment context.

For prediction-based current feeds, `LocationDataManager` precomputes the
derived prediction frame when the raw current feed changes. That frame contains
the segment peak-relative magnitude, direction, slope, segment peak metadata,
and adjacent slack-boundary metadata needed for current state displays. User
requests then do an `asof` lookup for the requested time instead of
recalculating those columns across the full current prediction DataFrame.
Preserve this split: derived data belongs in the background/feed-update path;
request handlers should remain cheap lookups.

Tide state follows the same derived-cache pattern. The raw tide feed currently
stores NOAA high/low prediction events (`interval=hilo`). When that feed changes,
`LocationDataManager` precomputes a minute-resolution tide-height frame from
those events for future point-in-time tide state lookups. This derived frame is
optional: if a location has no tide source, no loaded tide data, or too few
events to interpolate, the manager leaves the derived tide frame unavailable
without blocking the existing high/low tide event responses. Request handlers
must not recompute the tide curve across the full feed DataFrame.

`/api/{location}/conditions` exposes this as `tides.state` when available.
`tides.past` and `tides.next` remain the high/low event lists. `tides.state` is
the point-in-time estimated tide state and may be `null` even when high/low
events are available. It includes `timestamp`, `estimated_height`, `units`,
`trend`, and `height_pct`. The timestamp uses a real Pydantic `datetime` field
so OpenAPI exposes it as `format: date-time`; existing timestamp fields still
use string models until a future coordinated cleanup.

### Client Timeouts

All API clients enforce a 30-second timeout on individual requests (`REQUEST_TIMEOUT` in `clients/base.py`):

- **COOPS**: Uses aiohttp per-request timeout
- **NDBC**: Uses direct aiohttp text-file requests with a process-local
  concurrency gate. The NDBC base URL, path fragments, historical cutoff, and
  request concurrency limit are named constants in `shallweswim/clients/ndbc.py`;
  do not duplicate endpoint paths elsewhere.
- **NWIS**: Uses direct aiohttp requests against the modern USGS Water Data
  continuous-values endpoint. The base URL, path, page limit, and instantaneous
  statistic id are named constants in `shallweswim/clients/nwis.py`; do not
  duplicate endpoint paths elsewhere. The client follows USGS pagination links
  and maps empty FeatureCollections to `StationUnavailableError`.
  `shallweswim.scripts.debug_nwis_fetch` is the operational validation tool for
  configured NWIS request counts, response statuses, retry behavior, and
  rate-limit headers. Current configured sources did not produce live pagination
  during migration validation, so pagination remains unit-tested rather than
  live-proven against production station configs.

Timeouts raise `RetryableClientError` and are automatically retried by `request_with_retry()`.

### Plot Generation

Plot generation runs in a `ProcessPoolExecutor` (bounded to `os.cpu_count()` workers) using a fire-and-forget pattern:

- **Submit**: Each update loop iteration submits plot tasks to the pool via `loop.run_in_executor()` and returns immediately (no awaiting)
- **Collect**: On the next iteration, completed futures are harvested and results stored. The async loop is never blocked by plot generation
- **Guard**: `_pending_plot_futures` tracks in-flight work per feed — prevents duplicate submissions while a worker is still running
- **Hard timeout** (`PLOT_HARD_TIMEOUT`, 300s): If a worker hasn't finished, tracking is dropped (orphaned worker cannot be killed — Python limitation) and the location can retry
- **Key constraint**: `ProcessPoolExecutor` futures cannot be cancelled. Never use `asyncio.wait_for` or `asyncio.gather` on them — cancelled futures appear "done" but the worker keeps running, causing task stacking

#### Feed Processing vs Plot Processing

Keep source-of-truth data cleanup in the feed layer and visualization-specific
cleanup in the plot layer:

- **Feed layer belongs to data integrity**: API client normalization,
  dataframe schema validation, unit conversion, timestamp normalization,
  configured known-bad source `outliers`, and source-specific failures. Data
  returned by feeds is the canonical data used by API responses, condition
  summaries, and future derived products.
- **Plot layer belongs to visual representation**: interpolation for readable
  lines, rolling means, gap rendering, chart-only visual artifact suppression,
  axis formatting, labels, and styling. Plot transforms must not mutate cached
  feed data or silently redefine what the API considers the measured value.
- **Promote logic out of plotting only when it becomes product data**: if a
  conditioned historical temperature series is needed outside SVG generation,
  move it into a named core/query derivation with its own contract, tests, and
  API semantics instead of reusing a private plotting helper.

This boundary is especially important for heuristic visual artifact
suppression. A configured feed outlier says "this source timestamp is known
bad." A plot artifact mask says "this point or segment would make this chart
misleading or unreadable." Those are different claims and should stay in
different layers.

#### Historical Temperature Plot Processing

Historical temperature plots are backend-rendered SVGs generated from the
`historic_temps` feed. The feed fetches each configured year from the same
temperature source used for the location, normalizes and validates each year
independently, and only publishes a new combined dataset when every required
year succeeds. Successful years are cached in memory for the process lifetime:
past years do not expire once fetched, while the current year refreshes on the
historical feed interval. Per-year normalization uses the same hourly resampling
path as the final combined feed, so source quirks such as duplicate local
timestamps around daylight-saving transitions are resolved before schema
validation. Incomplete attempts record the successful and failed years for
diagnostics but leave the previously published complete dataset and plots
untouched. After a complete fetch, the feed combines years, sorts by timestamp,
and resamples to hourly rows. Plot generation then pivots the data with
`util.pivot_year()`, which moves the year into columns and normalizes every
timestamp onto leap-year calendar year 2020 so all years can be compared on one
month/day axis.

`plot._historic_temperature_plot_frame()` owns the visible yearly and monthly
trend-line preparation. Treat this output as **visual artifact suppression for
backend-rendered charts**, not a general-purpose cleaned temperature record.
The stages are intentionally layered because NOAA/NDBC and USGS historical
feeds can contain both missing spans and implausible isolated or short-lived
artifacts:

1. **Raw isolated spike artifact mask**: per year, suppress points whose
   residual from a centered 7-day rolling median exceeds
   `MAX_HISTORIC_TEMP_PLOT_SPIKE_RESIDUAL_F`.
2. **Short-gap interpolation**: interpolate missing spans up to
   `MAX_HISTORIC_TEMP_PLOT_GAP`; longer missing spans remain `NaN` so
   Matplotlib breaks the plotted line instead of drawing a false diagonal.
3. **24-hour smoothing**: apply the rolling mean used for the visible trend
   lines.
4. **Cross-year seasonal artifact mask**: compare each smoothed year to the
   same day/hour median across years and suppress points whose residual exceeds
   `MAX_HISTORIC_TEMP_PLOT_CROSS_YEAR_RESIDUAL_F`.
5. **Volatility artifact mask**: suppress smoothed windows whose 48-hour range
   exceeds `MAX_HISTORIC_TEMP_PLOT_SMOOTHED_RANGE_F`; this catches short runs
   that are not single-point spikes but still produce jagged visual artifacts.
6. **Short-segment cleanup**: after the other masks are applied, remove
   remaining visible segments shorter than `MIN_HISTORIC_TEMP_PLOT_SEGMENT`.

These masks affect only the rendered historical temperature plots. They do not
mutate cached feed data, live temperature readings, condition summaries, API
responses, or future modeling datasets. `generate_historic_temp_plots()` logs
per-location counts by visual artifact stage and year so threshold changes can
be tuned from production or local logs.

### Logging Guidelines

- **WARNING**: Expected operational issues (station outages)

  - Does NOT trigger GCP alerts (query: `severity=ERROR`)
  - Visible in logs for debugging

- **ERROR**: Unexpected issues requiring attention
  - Triggers GCP alerts
  - Indicates potential bug or API change

### Health Check (`/api/healthy`, `/api/health`)

- Returns **200** if at least 1 location has data (fresh or stale)
- Returns **503** only if NO location can serve any data
- Used by Cloud Run for routing decisions
- Single station outages do NOT trigger 503
- Missing/empty location manager state returns **503** here because the
  health endpoint answers "can this instance serve traffic?"

### Canonical Routes

- Canonical production host is `https://shallweswim.today`
- `www.shallweswim.today` redirects to the apex host with a permanent redirect
- `/`, `/locations`, and enabled `/{location}` routes serve the React app shell
- `/robots.txt` advertises `/sitemap.xml`
- `/sitemap.xml` lists `/locations` and enabled location pages

### Monitoring (`/api/status`)

- Returns detailed status for all locations/feeds
- Shows `is_healthy`, `is_expired`, `age_seconds`,
  `consecutive_failures`, `next_fetch_after`, and
  `seconds_until_next_fetch` per feed
- Shows year-level `historic_temps` diagnostics, including required, cached,
  available, missing, fetched, and failed years
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
- `is_expired`: Feed is due for a scheduled refresh or retry attempt
- `is_healthy`: Data within interval + 15-minute buffer (for monitoring display)
- `next_fetch_after`: Next scheduled refresh or retry attempt
- `seconds_until_next_fetch`: Countdown to the next scheduled attempt
- `consecutive_failures`: Failed attempts since the last successful update
- `historical_temp_status`: Optional diagnostics for the `historic_temps` feed,
  exposing year-cache progress without changing the public swimming data API.

Background tasks continuously check feeds. Successful fetches schedule the next
normal refresh; failed fetches leave the feed stale (serving old data) and
schedule a bounded retry.

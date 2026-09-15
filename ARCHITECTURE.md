# Architecture & Conventions

This document describes the architectural patterns, coding standards, and design decisions for the "Shall We Swim Today?" project.

**Architecture pattern**: Data aggregator (not a proxy). A scheduled job fetches data from multiple upstream APIs, processes it, and publishes one immutable generation to a shared store; the web service loads a generation into memory and serves from it. User requests never trigger external API calls, and the web service holds no provider client at all.

## 1. Architectural Patterns

### Project Structure

```text
shallweswim/
├── web.py               # Web service entry point, web UI routes, templates
├── update.py            # One-shot bounded observation capture job entry point
├── local.py             # Local entry point: job cycle plus web app, one process
├── archive/             # Observation schemas, conditional stores, and merge writer
├── snapshot/            # Serving snapshot model, Parquet/SVG objects, manifests, publisher, generation collector, incremental loader, read-only manager, web serving state
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
│   ├── serving.py       # FeedData and LocationServing Protocols the routes and queries read through
│   ├── updater.py       # Background update helpers
│   └── feeds.py         # Feed classes with caching/expiration
├── clients/             # External API clients
│   ├── __init__.py      # create_api_clients() provider client-set factory
│   ├── base.py          # BaseApiClient with retry logic, error hierarchy
│   ├── coops.py         # NOAA CO-OPS (tides, currents, coastal temps)
│   ├── cspf.py          # CSPF Sandettie historical temperatures
│   ├── irish_lights.py  # Irish Lights MetOcean buoy temperatures
│   ├── marine_institute.py # Marine Institute Ireland ERDDAP
│   ├── ndbc.py          # NOAA NDBC (buoy temperatures)
│   └── nwis.py          # USGS NWIS (river temps, discharge)
├── plot.py              # Chart generation (runs in process pool)
└── util.py              # Shared utilities

tests/                   # Unit and integration tests
frontend/                # React/Vite app source and frontend build tooling
templates/               # Jinja2 HTML templates
static/                  # CSS, JS, images
```

**Backwards compatibility**: `api/__init__.py` and `config/__init__.py` re-export from `routes.py` and `locations.py` for import compatibility.

### Entry Points And Store Locators

Three entry points run this code:

- `shallweswim.web` is the web service: it loads published generations and
  serves from them, and fetches nothing.
  `SHALLWESWIM_SNAPSHOT_READ_BUCKET` is required, because a web process with no
  store has nothing to serve; an unset variable fails startup with one message
  naming it and pointing at `shallweswim.local`. Production runs it.
- `shallweswim.update` is the bounded job: one capture cycle, and with
  `SHALLWESWIM_SNAPSHOT_PUBLISH=1` one published generation. It is the only
  process that contacts a provider. Production runs it on a schedule.
- `shallweswim.local` is the clone-and-run local command: it runs the job's
  publishing cycle (`update.publish_locations`) on a timer inside the web app's
  process, against a local store, and serves that app. It composes rather than
  branches: it wraps the app's lifespan, and inside it runs the first cycle and
  loads the generation that cycle published before the server accepts a
  request, then starts the cadence task and cancels it at shutdown, so
  `web.py` holds no local mode. The web app opens no HTTP session, so this
  module opens the one the cycles fetch over; the pool they plot in is the
  app's. It runs the application object under uvicorn, not the factory string,
  because the store lives in the process; `--reload` is therefore unsupported.

Every store the application builds comes from `archive/store.py`'s
`object_store(locator)`: a bare name is a GCS bucket (with the per-bucket client
cache), a locator containing `/` is a `FilesystemObjectStore` rooted there, and
the literal `memory` is one process-wide `MemoryObjectStore`. The `ObjectStore`
protocol is four operations - `read`, `compare_and_swap`, `list(prefix)`
returning each key with a timezone-aware creation time, and `delete(key)`,
which an absent key satisfies - and all three implementations answer them
identically: memory records write times, the filesystem reports modification
times and skips its dot-named locks and temporaries, and GCS reports
`time_created` and tolerates a `NotFound` delete. The four
environment-driven sites - archive capture, historical hydration, the snapshot
publisher, and the web loader - call that helper and differ in nothing else, so
a locator kind is never a code path. `shallweswim.local` sets
`SHALLWESWIM_ARCHIVE_BUCKET`, `SHALLWESWIM_ARCHIVE_READ_BUCKET`, and
`SHALLWESWIM_SNAPSHOT_READ_BUCKET` in its own process to one locator,
unconditionally, so a local run can never reach the operator's bucket.

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

**User requests** serve the loaded generation or derived in-memory views (no external calls):

```text
API Handler → app.state.snapshot.managers → SnapshotLocationManager → Loaded Frames / Plots → Response
```

Routes resolve the location's manager from `app.state.snapshot` on every
request rather than from a mapping captured at startup, so the generation an
elected request loads is served by the next one. A location the generation does
not carry, or one whose manager holds no data, answers 503.

**The publishing job** updates raw feeds and derived caches on intervals:

```text
Scheduled Job → Feed → ApiClient → External Service → Update Feed Cache
Scheduled Job → Derived Data Precompute → Published Generation
```

Every client returns frames indexed by timezone-aware UTC instants, as does the
local harmonic tide feed. Feeds derive the naive location-local serving index in
one step before validation, converting to the location timezone and keeping the
first reading of a repeated daylight-saving fall-back wall time. Archive capture
receives the UTC client frame itself, unconverted and unfiltered by configured
outliers, so both folds of that hour are archived as the distinct instants they
are. Composite feeds combine frames their member feeds already converted, so
that step passes a naive frame through unchanged.

When `SHALLWESWIM_ARCHIVE_BUCKET` is set, successful temperature updates and
successful observational currents updates also merge observations into the
private GCS archive. Production sets that variable only for the bounded capture
job (`shallweswim/update.py`), never for the web service, so the web runtime
never writes to the archive. The job builds feeds through the same
`core.manager.build_feeds()` builder the web manager uses, updates each
archivable feed once, and exits. Live feeds publish and schedule before capture; historical
feeds capture only freshly fetched years, including successful years in a
partial fetch, at the provider's native cadence, from the per-year frame before
the hourly serving resample. Cached years retain their retrieval times. Merging is
value-aware per observation instant: an unchanged reading keeps the stored row
and its original retrieval time, a changed reading is replaced by the more
recently retrieved claim, and a fetch with nothing new or revised leaves the
partition byte-identical. Each merge event reports those row counts and the
job's run summary reports their totals for the run.
When the job definition also sets `SHALLWESWIM_SNAPSHOT_PUBLISH=1`, the job
instead runs each location's full serving cycle through `LocationDataManager`
(`update_once()`, then `wait_for_plots()` for the process-pool plots) and, after
the cycle, publishes every location's served frames, plots, and feed metadata as
one immutable content-addressed generation under `published/` in the same
bucket (`shallweswim/snapshot/`); publication failure is isolated from the run's
outcome, and the web service serves those generations, below.
`snapshot/gc.py` then sweeps once per run, isolated the same way: it keeps the
generation the current pointer names whatever its age and every generation
published inside `RETAINED_GENERATION_AGE`, deletes the other manifests, and
deletes only objects that no retained manifest references and that are older
than `OBJECT_SAFETY_AGE`, because content addressing means a new generation may
reference an old object. It deletes nothing it did not list in the same run,
keeps every object when a retained manifest will not parse, never touches
`archive/`, and logs one `snapshot.gc` event instead of raising.
The builder reports every
configured feed, as a served frame or as a failure, and manifest assembly
resolves each failure against the generation the publisher observed at start:
an entry published for the same `citation_key` is carried forward unchanged
except for its accumulated failure count, this run's error, and this run's
retry time, as is any plot the run did not produce whose feed the assembled
manifest still publishes. Carried objects are
referenced by key, never reread or rewritten, so a repeatedly failing feed
publishes a manifest and no object. After assembly, and whether or not the
generation is promoted, the job logs one `snapshot.freshness` event per
location and feed with the served frame's `age_seconds` and a bounded
`success`, `carried`, or `absent` outcome.
A loaded generation can already serve: `snapshot/manager.py` builds a
read-only `SnapshotLocationManager` per location from the manifest entry,
restored frames, and plot bytes, computes the derived tide and current
prediction frames once at construction, and derives each feed's status (age,
expiry, health) from the manifest timestamps with the feed rules. It and
`LocationDataManager` both satisfy `core/serving.py`'s `LocationServing`
Protocol, which is what the API routes are typed against, and the query
functions read any `FeedData` (a `has_data` flag and a `values` frame), which a
fetched feed and a loaded snapshot feed both provide.
`SHALLWESWIM_SNAPSHOT_READ_BUCKET`, which the web service and local runs set
and the job never does, names the store the web service serves from. Its
lifespan builds the store on a worker thread, runs one bounded (20-second)
startup load, and starts the instance whether or not that load succeeded; a
failed startup load is logged at ERROR, because the instance then has nothing
to serve, while a failed refresh over a loaded generation stays WARNING. The
lifespan constructs no `LocationDataManager`, opens no client session, and
starts no update loop; the process pool it does create serves only the
on-demand detail plots. An HTTP middleware elects one arriving request per
60-second interval to check `published/current.json` before its handler runs,
skipping when a check is already in flight. `snapshot/load.py` loads
incrementally:
objects are content-addressed, so a key the process already holds is reused and
only new keys are read, eight at a time, each checked against the size and key
the manifest recorded and validated through its feed model.
`snapshot/refresh.py` holds that generation and the `SnapshotLocationManager`s
built from it, publishing them with one assignment of an immutable mapping so a
request in flight during a swap finishes on one whole generation, and logs one
`snapshot.load` event per load that does work. That state is what every route
reads; `app.state.data_managers` no longer exists.
When `SHALLWESWIM_ARCHIVE_READ_BUCKET` is set, which local development and the
publishing job do and the web service never does, the historical temperature
feed first hydrates each
required past year that is not already cached. Archive partitions are UTC years
while a historical year frame is a station-local year, so hydration reads that
year's partition and the next one, keeps the rows inside the local year, and
runs them through the same serving index, hourly
resample, and validation as a provider fetch; the current year, years the
archive lacks, and years whose read fails still fetch from the provider, and
hydrated years are never captured.
Archive failures emit failed merge events without changing serving or retry
state. Normalization, Parquet work, and synchronous GCS operations run in worker
threads. Tide feeds and prediction currents feeds do not enter this archive.

Keep user-facing condition endpoints on the fast path. Expensive, repeatable
work that depends only on cached feed data should run during background updates
or immediately after a feed changes, not on every request. Request handlers may
perform cheap point-in-time lookups and response serialization, but should not
loop across full feed DataFrames for high-traffic or above-the-fold endpoints
such as `/api/{location}/conditions` and `/api/{location}/currents`.

Known tech debt: water-movement detail plots
(`/api/{location}/plots/current_tide` and `/api/{location}/plots/tide`) still
render Matplotlib SVGs per request in the process pool. They should eventually
cache or precompute common hourly `at` / `shift` values, while preserving
on-demand fallback behavior for less common planning times.

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
  trusted citation HTML, source caveats used for presentation, external resource
  links such as official water-quality information, and external embed
  configuration, but should not include dynamic condition data, manifest
  metadata, station IDs, or feed internals unless there is an explicit frontend
  need and approval.

Temperature API contract:

- Internal temperature feed data is currently normalized to Fahrenheit in the
  `water_temp` dataframe column.
- `/api/{location}/conditions` exposes explicit `water_temp_f` and
  `water_temp_c` fields for public API consumers.
- Default display unit is static location metadata exposed through
  `/api/app/bootstrap`, not dynamic condition data.
- Live and historical temperature acquisition sources are configured
  independently as `live_temp_source` and `historic_temp_source`. They may use
  different upstream services, but should represent the same physical
  measurement location unless a source-location mismatch is explicitly reviewed
  and approved.
- Temperature source citations are de-duplicated by stable source identity. If
  live and historical temperature sources differ, `/api/app/bootstrap` exposes
  separate live and historical citation fields for the frontend.

Tide source contract:

- `CoopsTidesFeed` is the normal online tide prediction source for NOAA CO-OPS
  locations. It fetches high/low events and publishes the app's standard tide
  dataframe shape. Requests use NOAA's `MLLW` datum, which keeps swimmer-facing
  tide heights in a familiar local low-water frame.
- `MarineInstituteTidesFeed` supports Irish tide prediction locations. It
  fetches Marine Institute ERDDAP high/low summary rows and publishes the same
  app-native event shape as NOAA CO-OPS. Marine Institute high/low heights are
  relative to Ordnance Datum Malin, so location configs may declare a
  `height_offset_m` to convert them into a swimmer-facing local tide-height
  frame before the app's standard feet conversion. Cork/Sandycove uses the
  Kinsale high/low station with a 2.01 m offset to align with Marine
  Institute's local LAT-style dense prediction heights.
- `LocalHarmonicTidesFeed` supports locations where a suitable external tide
  prediction API is unavailable or not appropriate for the app's lightweight
  runtime. It loads static harmonic coefficients generated by offline tooling
  and locally produces the same short high/low prediction window as CO-OPS. The
  model computes in UTC throughout, so the feed emits events indexed by
  timezone-aware UTC instants like a client frame.
  Model JSON files must declare their source datum; Dover's packaged model uses
  Environment Agency E71624 data treated as already chart-datum-style, per the
  location config comment.
- The tide dataframe's `prediction` column is in feet. NOAA CO-OPS already
  provides feet; Marine Institute and local harmonic models may be meter-native
  and must convert to feet before publishing data to the rest of the app.
- Local harmonic model derivation belongs in `shallweswim/scripts/`, not in app
  startup. The running service must not fetch large tide-gauge archives or fit
  harmonic coefficients.

### Configuration

- All static configuration is in `config/locations.py`.
- Use Pydantic models for configuration schemas (e.g., `LocationConfig`, `BaseFeedConfig`).
- Avoid hardcoding constants in logic files; move them to `config/` or module-level constants if they are "magic numbers".
- Every `LocationConfig` must explicitly set `default_temperature_unit`; do not infer it from geography or rely on hidden defaults.
- New location work should start with `NEW_LOCATION.md`. If the location needs
  a new upstream source client or feed type, use `NEW_DATA_FEED.md` first.

## 2. Coding Standards

### Generalization

- Generalize a shared mechanism through parameters only when a second use exists
  in a committed plan and the change adds no new indirection; keep domain
  bindings as named constants.

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
- Local runs default to human-readable logs on stderr. Hosted deployments set
  `SHALLWESWIM_LOG_FORMAT=json` to emit one structured JSON object per line on
  stdout; the container platform captures and routes that stream.
- The Uvicorn application factory configures logging in the server process so
  reload child processes use the same application format. Uvicorn access logs
  retain their own human-readable development format in console mode. JSON
  deployments disable Uvicorn access logs because the hosting platform emits
  richer native request logs; this avoids duplicate production request events.
- Application code must not use a provider logging SDK. Only approved,
  bounded-cardinality fields from `logging_utils.py` may be supplied through
  `extra`; arbitrary fields could leak secrets or create unbounded log labels.
- Feed updates and background plot generation emit one structured completion
  event per attempt. Their stable fields and bounded outcomes support managed
  log-based metrics; request starts and provider-specific details remain DEBUG
  diagnostics rather than routine INFO events.
- Successful route-entry and response events rely on platform-native request
  logs rather than duplicate application INFO messages. Application logs record
  domain work, state changes, degraded availability, and failures.

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

External data sources (NOAA CO-OPS, NOAA NDBC, USGS NWIS, CSPF, Marine Institute Ireland, Irish Lights) may have temporary outages. The application handles these gracefully through principled error handling.

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
  - CO-OPS `200` responses that carry a NOAA error message instead of data, as the whole body or as a prose row under a CSV header,
    other than the stable "no data" answer, which stays
    `StationUnavailableError`
  - Automatically retried by `BaseApiClient.request_with_retry()`

- **`*ApiError`**: Unexpected client/library/API failures that are not known
  transient conditions and are not confirmed no-data responses

### Client Request Wrapping

`BaseApiClient` owns shared retry and logging behavior through
`request_with_retry()`, plus small HTTP mechanics such as standard timeout
objects, retryable status checks, and retryable network/timeout error messages.
Each concrete client owns its request helper signature because CO-OPS, NDBC,
NWIS, CSPF, and Marine Institute all require different request parameters.
Do not add `_execute_request` back to the base class contract just to share a
name; pass the concrete helper into `request_with_retry()` instead.

Client request retries and feed scheduling are separate layers:

- `BaseApiClient.request_with_retry()` uses tenacity to retry transient
  HTTP/network failures inside one fetch attempt over seconds. CO-OPS also
  reports transient rejections under HTTP `200` with an error body in place of
  CSV, so its client classifies the body before parsing.
- `Feed.update()` schedules the next whole-feed attempt over minutes after the
  fetch attempt finishes. This state is visible in `/api/status`.

External request pressure is controlled at the actual HTTP boundary, not by
blocking async fanout at higher layers:

- Composite feeds may intentionally create many async tasks, such as one
  historical temperature task per year.
- Each concrete client wraps `aiohttp` calls in a provider request slot before
  opening the upstream request.
- Provider request slots are process-local semaphores keyed by upstream provider.
  They bound active HTTP requests within one app instance, while queued
  coroutines remain suspended by the event loop.
- Cloud Run instances multiply these limits. For example, two instances with an
  NWIS cap of `2` can make up to four active NWIS HTTP requests at once.
- These gates are not rate-limit accounting or cross-instance distributed locks;
  they are local backpressure so startup, refresh, and retry bursts do not
  overwhelm upstream services or the app process.
- The CO-OPS client requests `time_zone=gmt` and returns frames indexed by
  timezone-aware UTC instants, so both folds of a daylight saving fall-back hour
  and the skipped spring-forward hour are exact. Request windows stay
  station-local days: callers pass naive local dates plus the station
  `timezone`, and the client sends each window edge as the UTC instant that
  local day boundary names.
- The NDBC client returns frames indexed by timezone-aware UTC instants,
  because NDBC text products report UTC. It de-duplicates overlapping realtime,
  monthly, and yearly components on the absolute instant, keeping the first
  component that supplied one. Its request window is a span of UTC days, so the
  client takes no station `timezone`.
- The NWIS client returns frames indexed by timezone-aware UTC instants. USGS
  stamps each observation with an explicit offset, which the client normalizes
  to UTC so both folds of a fall-back hour stay distinct. Request windows stay
  station-local days: callers pass naive local dates plus the station
  `timezone`, and the client converts each edge to a UTC RFC3339 instant.
- The CSPF client is intentionally narrow: it fetches Dover/Sandettie
  historical temperature fallback data from CSPF Sandettie pages. It parses the
  embedded sea-temperature JavaScript series, normalizes Celsius to the internal
  Fahrenheit `water_temp` column, uses monthly pages as the primary source
  because they are denser than annual summaries, and falls back to annual pages
  only when monthly pages have no data. Its points carry epoch milliseconds, so
  it returns frames indexed by timezone-aware UTC instants and de-duplicates on
  the instant, keeping the value the last page supplied. Request windows stay
  station-local: callers pass naive local edges plus the station `timezone`,
  which name the local calendar year whose pages are read and trim the result.
- The Irish Lights client fetches MetOcean buoy observations for configured
  Irish temperature sources. It uses the public MetOcean endpoint constants in
  `shallweswim/clients/irish_lights.py`, converts Celsius source values to the
  internal Fahrenheit `water_temp` column, and applies source-configured
  plausible Celsius bounds before publishing rows. MetOcean reports UTC ISO
  instants, so the client returns frames indexed by timezone-aware UTC instants
  and de-duplicates on the instant, keeping the last row reported for it. Its
  request window is UTC too, so the client takes no station `timezone`.
- The Marine Institute client is currently tide-only. It keeps ERDDAP tabledap
  endpoint constants in `shallweswim/clients/marine_institute.py`, fetches the
  `IMI_TidePrediction_HighLow` summary series for configured stations such as
  `Kinsale`, and converts that source-specific shape before feeds publish
  app-native tide data. ERDDAP reports UTC ISO instants, so the client returns
  frames indexed by timezone-aware UTC instants and de-duplicates events on the
  instant. Its request window is a span of UTC days, so the client takes no
  station `timezone`.

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

### Client Timeouts And Provider Gates

All API clients enforce a 30-second timeout on individual requests (`REQUEST_TIMEOUT` in `clients/base.py`):

- **COOPS**: Uses aiohttp per-request timeout and a process-local request gate
  capped by `COOPS_MAX_CONCURRENT_REQUESTS` in `shallweswim/clients/coops.py`.
- **NDBC**: Uses direct aiohttp text-file requests with a process-local
  request gate capped by `NDBC_MAX_CONCURRENT_REQUESTS`. The NDBC base URL,
  path fragments, historical cutoff, and request concurrency limit are named
  constants in `shallweswim/clients/ndbc.py`; do not duplicate endpoint paths
  elsewhere.
- **NWIS**: Uses direct aiohttp requests against the modern USGS Water Data
  continuous-values endpoint with a process-local request gate capped by
  `NWIS_MAX_CONCURRENT_REQUESTS`. The base URL, path, page limit, and
  instantaneous statistic id are named constants in `shallweswim/clients/nwis.py`;
  do not duplicate endpoint paths elsewhere. The client follows USGS pagination
  links and maps empty FeatureCollections to `StationUnavailableError`. If
  `USGS_WATERDATA_API_KEY` is set, the client sends it as an `X-Api-Key` header
  on every page request; otherwise requests remain unauthenticated. Local runs
  read the variable from `.env`; Cloud Run injects it from the
  `waterdata_usgs_gov_api_key` Secret Manager secret, with runtime access scoped
  to that secret.
  `shallweswim.scripts.debug_nwis_fetch` is the operational validation tool for
  configured NWIS request counts, response statuses, retry behavior, and
  rate-limit headers. Current configured sources did not produce live pagination
  during migration validation, so pagination remains unit-tested rather than
  live-proven against production station configs.
- **Marine Institute**: Uses direct aiohttp ERDDAP requests with a process-local
  request gate capped by `MARINE_INSTITUTE_MAX_CONCURRENT_REQUESTS`. The tide
  high/low summary endpoint is a constant in
  `shallweswim/clients/marine_institute.py`.
- **Irish Lights**: Uses direct aiohttp MetOcean JSON requests with a
  process-local request gate capped by `IRISH_LIGHTS_MAX_CONCURRENT_REQUESTS`.
  The MetOcean endpoint URL, public access token, default plausible Celsius
  bounds, and concurrency limit are named constants in
  `shallweswim/clients/irish_lights.py`.

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
`historic_temps` feed. The feed fetches each configured year from the
location's `historic_temp_source`, normalizes and validates each year
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
   lines. Dense sources use the global 24-row minimum. Sparse sources may lower
   `smoothing_min_periods` through source-level plot policy overrides.
4. **Cross-year seasonal artifact mask**: compare each smoothed year to the
   same day/hour median across years and suppress points whose residual exceeds
   `MAX_HISTORIC_TEMP_PLOT_CROSS_YEAR_RESIDUAL_F`.
5. **Volatility artifact mask**: suppress smoothed windows whose 48-hour range
   exceeds `MAX_HISTORIC_TEMP_PLOT_SMOOTHED_RANGE_F`; this catches short runs
   that are not single-point spikes but still produce jagged visual artifacts.
6. **Short-segment cleanup**: after the other masks are applied, remove
   remaining visible segments shorter than `MIN_HISTORIC_TEMP_PLOT_SEGMENT`.

Historical temperature source configs may provide `historic_plot_policy`
overrides for presentation-only tuning. Defaults preserve the global dense-source
behavior. CSPF Sandettie uses a sparse-source override (`smoothing_min_periods=3`
and `min_segment=6 hours`) so recent valid monthly-page observations remain
visible without changing the global NOAA/NDBC/NWIS plot policy.

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

- Returns **200** if a generation is loaded and at least 1 of its locations has
  data (fresh or stale)
- Returns **503** only if no generation is loaded, or NO location in it can
  serve any data
- Used by Cloud Run for routing decisions
- Single station outages do NOT trigger 503
- An instance whose startup load failed answers **503** until an elected request
  loads a generation, because the health endpoint answers "can this instance
  serve traffic?"
- Successful health and aggregate-status probes rely on platform-native request
  logs and do not emit duplicate application INFO events. Unhealthy states and
  internal status failures remain application WARNING or ERROR events.

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
- Each location also reports the generation it was served from:
  `generation_id`, `published_at`, and `loaded_at`
- The response is an empty object while no generation is loaded, which is what
  an instance that has not loaded one has to report; `/api/healthy` is the
  endpoint that calls that unhealthy.

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

### React condition embeds

Public `/{location}/embed` routes use the same durable React HTML shell and
location validation as the dashboard, with location-specific metadata and an
embed canonical URL. The React route sits outside `AppShell` to omit navigation
and installation UI. `EmbedPage` uses bootstrap presentation metadata and
`useLocationConditions` (including its periodic refresh), the shared temperature
summary (with an optional container class for embed styling) and Windy component,
and shared formatters. The embed arranges swimming data in responsive cards
and keeps the Windy map/forecast together below them. It never fetches upstream
station data. Links to the dashboard/detail view open safely in a new tab.
The original Jinja embed remains under `/legacy/{location}/embed` for comparison.
Cross-origin framing is allowed by the app; no API CORS changes are required.
Embed presentation is scoped under `.swim-embed` in `frontend/src/styles/embed.css`,
including local palette overrides for shared components. Poppins Latin subsets
are bundled as Vite assets with their SIL Open Font License; the main dashboard
keeps its existing typography and palette. A root `:has(.swim-embed)` selector
colors the full iframe canvas only while the embed is mounted. The host supplies
the visible page heading; the embed retains a location-specific accessible main
label and document title.

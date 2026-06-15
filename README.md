# Shall We Swim Today?

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-Managed-blueviolet)](https://github.com/astral-sh/uv)

**A web application that helps open water swimmers make informed decisions about swim conditions.**

[shallweswim.today](https://shallweswim.today) aggregates tide, current, and temperature data from public coastal and inland-water sources for popular open water swimming locations:

- **New York** - Coney Island / Brighton Beach
- **San Diego** - La Jolla Cove
- **Chicago** - Ohio Street Beach
- **San Francisco** - Aquatic Park
- **Louisville** - Community Boathouse (Ohio River)
- **Austin** - Barton Springs
- **Boston** - L Street Beach
- **Seattle** - Alki Beach
- **Dover** - Swimmer’s Beach
- **Cork** - Sandycove

## Features

- **Current conditions and historical trends** from NOAA CO-OPS, NOAA NDBC,
  USGS NWIS, Marine Institute Ireland, Irish Lights, and CSPF sources
- **Tide predictions** with high/low tide times, heights, and estimated tide state
- **Current velocity** data with flood/ebb/slack phase, strength, trend, and absolute speed
- **Water temperature trends** (48-hour, 2-month, and multi-year)
- **Transit information** for NYC locations (subway status and alerts)
- **Mobile-friendly interface** for on-the-go swimmers
- **JSON API** for programmatic access to swim conditions
- **Durable canonical HTML** with route-specific metadata, API discovery links,
  compact no-JavaScript fallback content, and conservative JSON-LD

## Architecture

Shall We Swim is a FastAPI application with a modular architecture. FastAPI
serves a thin route-aware React/Vite app shell as the primary web experience at
root location URLs, with the older Jinja-rendered pages temporarily available
under `/legacy`.

### Runtime Model

The application is **fully stateless** with no database or persistent storage. On startup, each instance fetches historical data (~8 days) directly from external APIs and holds it in memory. This design was chosen for Cloud Run deployment simplicity - instances can scale to zero and spin back up without managing storage.

**Implications:**

- Cold starts require fetching all data before serving (gated by `/api/healthy`)
- Instance shutdown loses all data (automatically re-fetched on next startup)
- Multiple instances don't share state (each fetches independently)

### Request Flow

**User requests** always serve cached or precomputed data (fast, no external calls):

```text
HTTP Request → API Handler → LocationDataManager → Feed / Derived Cache → Response
```

**Background tasks** refresh feeds and derived data on intervals (10 min to 24 hours):

```text
Background Task → Feed → ApiClient → External API → Update Cache
Background Task → Derived Data Precompute → Update Derived Cache
```

Failed fetches leave data stale until the next successful refresh.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation, coding standards, and error handling patterns.

## Getting Started

### Prerequisites

- Python 3.13
- Node 24 LTS for frontend development
- [uv](https://github.com/astral-sh/uv) for dependency management
- Docker (optional, for containerized deployment)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/jeremywhelchel/shallweswim.git
cd shallweswim

# Install dependencies, including the default dev group
uv sync

# Run the development server
uv run python -m shallweswim.main --port=12345
```

Then visit <http://localhost:12345> in your browser.

### Frontend App Development

The React app lives in `frontend/` and consumes the same-origin FastAPI JSON API.
Use the pinned `pnpm` version from `frontend/package.json` through Corepack.

Preferred package-manager path:

```bash
corepack enable
corepack pnpm@10.18.3 --dir frontend install
```

Fallback when Corepack is unavailable:

```bash
npx --yes pnpm@10.18.3 --dir frontend install
```

Avoid global package-manager installs in repo docs and project setup:

```bash
npm install -g pnpm
npm install -g corepack
```

Corepack keeps the package-manager version tied to the project. The `npx`
fallback is explicit and does not persist a global `pnpm` install. Global
installs conflict with the project's no-global-state rule and can interfere with
existing Yarn, pnpm, or Corepack shims.

```bash
# Export the backend OpenAPI contract and generate TypeScript API types
uv run python -m shallweswim.scripts.export_openapi > frontend/openapi.json
corepack pnpm@10.18.3 --dir frontend install
corepack pnpm@10.18.3 --dir frontend generate-api

# Run the Vite dev server
corepack pnpm@10.18.3 --dir frontend dev

# Build the static app shell served by FastAPI at root routes
corepack pnpm@10.18.3 --dir frontend build
```

If your Node installation does not provide Corepack, use an ephemeral pinned
`pnpm` invocation instead of installing package managers globally:

```bash
npx --yes pnpm@10.18.3 --dir frontend install
npx --yes pnpm@10.18.3 --dir frontend generate-api
npx --yes pnpm@10.18.3 --dir frontend dev
npx --yes pnpm@10.18.3 --dir frontend build
```

The production Docker image builds `frontend/dist`. FastAPI reuses the built
Vite shell at `/`, `/locations`, and configured location routes such as `/nyc`,
then injects route-specific title/meta/canonical data, JSON API discovery links,
compact no-JavaScript fallback content, and conservative JSON-LD. Local FastAPI
app route requests return a clear not-built response until the frontend has
been built.

### Run with Docker

```bash
# Build the Docker image
docker buildx build -t shallweswim .

# Run the container
docker run -e PORT=80 -p 12345:80 shallweswim
```

Then visit <http://localhost:12345> in your browser.

## Deployment

The application is hosted on Google Cloud Run:

```bash
# Deploy to Google Cloud Run
./build_and_deploy.sh
```

### Canonical URLs

The canonical production host is `https://shallweswim.today`. The app redirects
`www.shallweswim.today` to the apex host, exposes canonical tags on app and
legacy HTML pages, and serves `/robots.txt` plus `/sitemap.xml` for crawler
discovery. The React app owns `/`, `/locations`, and canonical location paths
such as `/nyc`; legacy Jinja pages live under `/legacy` while they remain
available.

## Development

**Documentation**

- [Architecture & Conventions](ARCHITECTURE.md) - **Read this first** before making changes.

### Setup

```bash
# Install uv (recommended method)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies, including dev/test/tooling dependencies
uv sync --dev

# Set up pre-commit hooks
uv run pre-commit install
```

### Testing and Code Quality

The project uses pytest for tests and several tools to maintain code quality.
Pre-commit runs the fast local subset: file hygiene, formatting/linting, Markdown
linting, Ruff, and Pyrefly. Run the broader test, browser, integration, and
performance checks explicitly or through CI.

```bash
# Run unit tests (excluding integration tests)
uv run pytest -v -k "not integration"

# Run integration tests (connects to external APIs)
uv run pytest -v -m integration --run-integration

# Run optional Python browser tests (requires Playwright Chromium)
uv run playwright install chromium
uv run pytest tests/test_react_stack_browser.py -v --run-browser

# Run frontend checks
corepack pnpm@10.18.3 --dir frontend typecheck
corepack pnpm@10.18.3 --dir frontend test
corepack pnpm@10.18.3 --dir frontend build
corepack pnpm@10.18.3 --dir frontend test:e2e:install
corepack pnpm@10.18.3 --dir frontend test:e2e:smoke
corepack pnpm@10.18.3 --dir frontend test:e2e

# Same checks without Corepack, using ephemeral pinned pnpm
npx --yes pnpm@10.18.3 --dir frontend typecheck
npx --yes pnpm@10.18.3 --dir frontend test
npx --yes pnpm@10.18.3 --dir frontend build
npx --yes pnpm@10.18.3 --dir frontend test:e2e:install
npx --yes pnpm@10.18.3 --dir frontend test:e2e:smoke
npx --yes pnpm@10.18.3 --dir frontend test:e2e

# Run optional performance guardrails
uv run pytest tests/performance -v --run-performance

# Run type checking
uv run pyrefly check .

# Run linting
uv run ruff check .

# Format code
uv run ruff format .

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Run with code coverage
uv run pytest --cov=shallweswim

# Run with code coverage and generate HTML report
uv run pytest --cov=shallweswim --cov-report=html
```

Note: Integration tests connect to live external APIs (NOAA CO-OPS, NOAA NDBC, USGS NWIS, CSPF, Marine Institute Ireland, Irish Lights) and may occasionally fail if services are experiencing issues or data is temporarily unavailable. Browser tests are also opt-in; they use Playwright to run a real Chromium browser and are skipped unless `--run-browser` is passed.

#### Optional Browser Tests

Browser tests exercise the frontend JavaScript in a real Chromium browser.
They are not part of the default test run. Python Playwright and frontend
Playwright are pinned to the same version and use the default shared Playwright
browser cache, so either install command below prepares Chromium for both.
Frontend Playwright tests run against the production Vite build via
`vite preview`, so run `corepack pnpm@10.18.3 --dir frontend build` before
`test:e2e:smoke` or `test:e2e`.

```bash
# Install the Playwright Chromium browser binary via Python Playwright
uv run playwright install chromium

# Or install the same browser via frontend Playwright
corepack pnpm@10.18.3 --dir frontend test:e2e:install

# If Playwright reports missing Linux system libraries in a disposable dev
# container, VM, or CI image, install them too
uv run playwright install-deps chromium

# Run the optional Jinja browser tests
uv run pytest tests/test_frontend_browser.py -v --run-browser

# Run the optional React/FastAPI browser stack test
corepack pnpm@10.18.3 --dir frontend install --frozen-lockfile
corepack pnpm@10.18.3 --dir frontend build
uv run pytest tests/test_react_stack_browser.py -v --run-browser
```

`playwright install-deps chromium` modifies system packages with `apt` on Linux.
Do not run it as routine setup on a host machine; use it only after Playwright
reports missing system libraries, and only in a disposable development
container/VM or CI image where that mutation is expected. If you prefer explicit
package installation in a Dockerfile or GitHub Actions step, Playwright's
Chromium dependency warning lists the needed packages for the current image.

Integration test teardown intentionally uses bounded waits for blocking live-API
worker threads. This avoids GitHub Actions stalls that happened when teardown
waited indefinitely on stuck external HTTP calls. A local full-suite integration
run can pass all tests and still exit nonzero from unclosed socket
`ResourceWarning`s during pytest teardown; treat that as a known tradeoff unless
the scheduled GitHub Actions integration job starts failing.

#### Frontend Debug Mode

Location pages include a lightweight browser debug tool for diagnosing frontend
loading issues. Add `?debug=1` to a location URL to enable the visible debug UI:

```text
http://localhost:12345/nyc?debug=1
```

When enabled, a small debug button appears in the lower-right corner. Click it to
open a panel with browser details, selected DOM state, recent API calls, and
recent frontend errors. The panel can copy the captured debug data for sharing.

The debug script is loaded on all pages so it can passively capture fetch and
error state in `window.SWS_DEBUG_STATE`, but it stays visually hidden and avoids
debug console logging unless `?debug=1` is present.

#### Frontend Loading Behavior

The app intentionally serves HTML before every feed and generated plot is ready.
This keeps local startup and production cold starts non-blocking. Location pages
load the primary condition text first with `/api/<location>/conditions`; visible
placeholders use quiet `...` text until data arrives.

Temperature trend plots are deferred until after the first conditions request.
The browser then loads each plot independently and retries transient `503`
responses with backoff while plots are still being generated. If a plot remains
unavailable after retries, only that plot shows `Plot unavailable`; the rest of
the page remains usable.

NYC transit status loads independently from swim conditions and plots. If the
third-party transit feed is unavailable on first load, each train card shows an
unavailable state instead of leaving placeholder text on screen.

#### Inspecting Historical Temperature Plot Visual Artifacts

Historical temperature plots apply visual artifact suppression during plot
generation to avoid rendering bad station artifacts as misleading trend lines.
This is plot-only cleanup, not a replacement for the source feed data. Some
sources, such as CSPF Sandettie, provide source-specific plot policy overrides
for sparse historical observations. The full pipeline is documented in
[ARCHITECTURE.md](ARCHITECTURE.md#historical-temperature-plot-processing).

To inspect what the plot artifact masks suppress for a real location, run:

```bash
uv run python -m shallweswim.scripts.inspect_historic_temp_plot_artifacts bos \
  --output-dir tmp/historic-temp-plot-artifacts/bos
```

The command fetches live historical data from the configured historical
temperature source, applies the same configured plot policy used by runtime
chart generation, prints counts by visual artifact stage and year, and writes:

- `plot_suppressed_points.csv`: one row per plot-suppressed point with stage, year,
  pivoted calendar timestamp, original timestamp, source temperature, value
  suppressed from the plot, seasonal median, and residual
- `final_plot_frame.csv`: the post-suppression frame that feeds the rendered
  historical plot

Use `--start-year` and `--end-year` to narrow a tuning run. Because this command
hits configured external temperature sources directly, results can change as
upstream station data changes.

#### Debugging NDBC Temperature Fetches

Use the NDBC client debug script to exercise station fetches without starting the
full service:

```bash
uv run python -m shallweswim.scripts.debug_ndbc_fetch --location bos \
  --start-year 2011 --end-year 2026 --yearly --concurrency 2
```

The command uses the same first-party NDBC client as the app and reports per-year
row counts, missing temperature counts, date bounds, and elapsed time. The
script's `--concurrency` flag controls diagnostic workload fanout; runtime
upstream HTTP concurrency is bounded by provider gates in the clients.

#### Debugging NWIS Temperature and Current Fetches

Use the NWIS client debug script to exercise configured USGS fetches without
starting the full service:

```bash
uv run python -m shallweswim.scripts.debug_nwis_fetch --location aus --feed live-temp
uv run python -m shallweswim.scripts.debug_nwis_fetch --location sdf --feed currents
uv run python -m shallweswim.scripts.debug_nwis_fetch --all --startup-workload
```

The command uses the same first-party NWIS client as the app and reports row
counts, date bounds, HTTP request counts, response statuses, rate-limit headers,
and an estimated multi-instance request count. The script's `--concurrency`
flag controls diagnostic yearly fanout; runtime upstream HTTP concurrency is
bounded by provider gates in the clients.

Use this script before deploying NWIS client changes. During the modern USGS
Water Data API migration, configured NWIS cold-start work measured about 19-21
HTTP attempts per instance: Austin historical temperature years completed in
one request per year with no live pagination observed, while two year requests
timed out once and succeeded on retry. Louisville live temperature and currents
each completed in one request. The modern API supports authenticated requests
with an optional key, which increases quota and exposes rate-limit headers.

For local authenticated NWIS testing, create `.env` from `.env.example` and set
`USGS_WATERDATA_API_KEY`. The client sends it as an `X-Api-Key` header when
present and falls back to unauthenticated requests when it is omitted. Do not
commit API keys.

#### Debugging CSPF Sandettie Historical Temperatures

Use the CSPF debug script to exercise Dover's Sandettie historical temperature
fallback without starting the full service:

```bash
uv run python -m shallweswim.scripts.debug_cspf_fetch --location dov \
  --start-year 2011 --end-year 2026
```

The command fetches the same CSPF Sandettie pages as the runtime client and
reports per-year row counts, date bounds, failures, and elapsed time. The client
uses monthly CSPF pages first because they are denser than annual summaries, and
falls back to an annual page only when monthly pages have no data.

#### Debugging Irish Lights Temperature Fetches

Use the Irish Lights debug script to exercise Cork/Sandycove buoy temperature
fetches without starting the full service:

```bash
uv run python -m shallweswim.scripts.debug_irish_lights_fetch --location cor
uv run python -m shallweswim.scripts.debug_irish_lights_fetch --location cor \
  --start-year 2024 --end-year 2026
```

The command fetches the same Irish Lights MetOcean endpoint as the runtime
client and reports row counts, date bounds, Fahrenheit min/max values, failures,
and elapsed time. Cork uses the Irish Lights Cork Buoy as a shared live and
historical temperature source, with source-specific filtering for implausible
water-temperature outliers.

#### Deriving Local Harmonic Tide Models

For locations where a suitable tide prediction API is unavailable, local
harmonic tide models can be derived offline from observed gauge history:

```bash
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --fetch \
  --cache /tmp/dover_ea_local.csv \
  --archive-start 2025-06-25 \
  --archive-end 2026-06-01

uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --fit --eval --backtest \
  --cache /tmp/dover_ea_local.csv \
  --output /tmp/dov_harmonics.json
```

The generated JSON contains compact coefficients for `LocalHarmonicTidesFeed`,
which generates the normal short high/low tide prediction window locally at
runtime. Model derivation is intentionally offline; the app should not fetch
large tide-gauge archives or fit harmonics during startup. If a source model is
meter-native, the runtime feed converts generated tide heights to feet before
the values enter the shared API and plotting path.

### Testing Philosophy

The test suite uses a tiered strategy:

| Tier | Files | External APIs | Config | Run By Default |
|------|-------|---------------|--------|----------------|
| **Unit** | `test_*.py` (most) | Mocked | Fake test configs | Yes |
| **E2E Stack** | `test_mocked_stack.py` | Mocked | Fake test configs | Yes |
| **Integration** | `test_*_integration.py` | Real external APIs | Real configs | No (`--run-integration`) |
| **Browser Jinja** | `test_frontend_browser.py` | Mocked | Real templates/static assets | No (`--run-browser`) |
| **React Browser Stack** | `test_react_stack_browser.py` | Mocked | Real FastAPI routes + built React app | No (`--run-browser`) |

**Key principles:**

- **Unit/E2E tests are deterministic** - No external dependencies, fake configs defined in `tests/conftest.py`
- **Integration tests validate real-world compatibility** - May fail due to external factors (station outages, API changes). NWIS integration tests must fail, not skip, if the live USGS API returns retryable failures such as rate limiting.
- **Fake configs are explicit** - Each test controls exactly what scenario it tests, independent of production config

## Monitoring & Station Outages

External data sources (NOAA CO-OPS, NOAA NDBC, USGS NWIS, CSPF, Marine Institute Ireland, Irish Lights) occasionally experience outages. The application handles these gracefully:

- **Health check (`/api/healthy`, alias `/api/health`)**: Returns 200 if at least one location can serve data. Single station outages don't mark the entire service unhealthy.
- **Status endpoint (`/api/status`)**: Returns detailed per-feed status including `is_healthy`, `is_expired`, `age_seconds`, `consecutive_failures`, and the next scheduled fetch time. Historical temperature feeds also include year-level diagnostics for required, cached, missing, fetched, and failed years. Use this for granular monitoring and alerting.

For production deployments, set up external monitoring on `/api/status` to alert when critical feeds become unhealthy. See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed station outage handling strategy.

### HTTP Error Codes

- **404 Not Found**: Requested resource doesn't exist for this location
- **503 Service Unavailable**: External station has no data (expected, retry later)
- **500 Internal Server Error**: Bug in our code or app initialization state (needs immediate attention)

## API Documentation

When running locally, API documentation is available at:

- Swagger UI: <http://localhost:12345/docs>
- ReDoc: <http://localhost:12345/redoc>

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [NOAA CO-OPS API](https://tidesandcurrents.noaa.gov/api/) (Center for Operational Oceanographic Products and Services) for tide, current, and temperature data
- [NOAA NDBC API](https://www.ndbc.noaa.gov/) (National Data Buoy Center) for buoy-based water temperature data
- [USGS Water Data APIs](https://api.waterdata.usgs.gov/) (National Water Information System) for water temperature and river current data
- [Channel Swimming and Piloting Federation](https://cspf.co.uk/sandettie-data) for Sandettie historical water temperature data sourced from the Met Office
- [Marine Institute Ireland ERDDAP](https://erddap.marine.ie/erddap/index.html) for Irish tide prediction data
- [Irish Lights MetOcean](https://www.irishlights.ie/technology-data-services/metocean-charts.aspx) for Irish buoy water temperature data
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Matplotlib](https://matplotlib.org/) for data visualization
- [Feather Icons](https://feathericons.com/) for UI icons
- [GoodService.io](https://goodservice.io/) for NYC subway information

## Continuous Integration

GitHub Actions workflows automatically verify the following on every push:

- **Unit Tests**: All unit tests pass
- **Type Checking**: No type errors found by pyrefly
- **Code Quality**: Ruff linting and formatting checks pass
- **Frontend**: React linting, type checking, unit tests, build, generated API
  freshness, and frontend Playwright browser tests pass
- **Browser Tests**: Python Playwright browser tests pass against the Jinja
  frontend and the built React/FastAPI stack

Additionally, a separate integration test workflow runs daily to ensure compatibility with external APIs.

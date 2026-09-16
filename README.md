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

Want to see another swim spot here? Contributions are welcome. Start with
[Adding a New Location](NEW_LOCATION.md), which explains what local knowledge
and data sources make a location useful.

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

## Frequently Asked Questions (FAQ)

### Why Doesn't the App Show Water Quality?

Available recreational-water samples are generally intermittent and delayed.
Traditional bacterial culture results take about a day after collection, and
public datasets may be updated later still. By then, rain, discharges, tides,
currents, and mixing may have changed the water at the swim location. A dated
sample is useful evidence about longer-term water quality, but it is not a
reliable measurement of what a swimmer will encounter now.

Official beach statuses do not solve that mismatch consistently. For example,
New York City's status combines water-quality information with environmental,
public-health, safety, and operational reasons for advisories or closures. The
app therefore does not currently turn laboratory samples or a municipal
open/closed status into a swimmer-facing water-quality rating. This decision
should be revisited if a location has a validated, timely, swimmer-relevant
source such as same-day rapid testing or a transparent local nowcast.
Where available, the app may link to official local water-quality information
without interpreting it as a current condition or swimming recommendation.

## Architecture

Shall We Swim is a FastAPI application with a modular architecture. FastAPI
serves a thin route-aware React/Vite app shell as the primary web experience at
root location URLs, with the older Jinja-rendered pages temporarily available
under `/legacy`.

### Runtime Model

The application has **two processes and one shared store**. A scheduled job
fetches every configured feed from the external APIs, draws the plots, and
publishes one immutable generation into object storage. The web service loads
the current generation into memory and serves every request from it; it
contacts no external API. There is no database: the store holds plain
content-addressed objects, and the web service reads them.

**Implications:**

- A cold start loads one published generation instead of fetching, and
  `/api/healthy` gates readiness on having loaded one
- Instance shutdown loses nothing: the generation is in the store
- Every instance serves the same generation, and picks up a new one within a
  check interval
- Upstream availability and latency never reach a user request

### Request Flow

**User requests** always serve the loaded generation (fast, no external calls):

```text
HTTP Request → API Handler → Snapshot Manager → Loaded Frames / Plots → Response
```

**The publishing job** refreshes each feed on its own interval (10 min to 24
hours) and publishes what changed:

```text
Scheduled Job → Feed → ApiClient → External API → Archive + Published Generation
Web instance → elected request → load new generation → serve it
```

A failed fetch leaves the last published frame in the generation, so the web
serves stale data rather than none.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation, coding standards, and error handling patterns.

The job, the bundle, and the web servers, with the archive and the rules that
bind them, are documented in [DATA_PIPELINE.md](DATA_PIPELINE.md).

The metrics, alert policies, dashboard, uptime check, and production log
queries are documented in [infra/MONITORING.md](infra/MONITORING.md).

To add a new swim spot, start with [NEW_LOCATION.md](NEW_LOCATION.md). If the
spot needs an unsupported upstream API or parser, use
[NEW_DATA_FEED.md](NEW_DATA_FEED.md) before wiring it into production config.

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

# Run the publishing job and the web app in one process
uv run python -m shallweswim.local --port=12345
```

Then visit <http://localhost:12345> in your browser.

`shallweswim.local` is the recommended local command. It needs no bucket and no
credentials: it points the three store variables
(`SHALLWESWIM_ARCHIVE_BUCKET`, `SHALLWESWIM_ARCHIVE_READ_BUCKET`,
`SHALLWESWIM_SNAPSHOT_READ_BUCKET`) at one local store, overriding whatever the
shell or `.env` holds, runs the publishing job's cycle against that store, and
serves the web app from the same process, which loads each published generation
the way the deployed service loads the job's.

- `--store-dir PATH` keeps the archive and the published generations in a
  directory. Served history comes from that archive, so what one run captured
  the next run still serves, and a fresh store holds only what its own cycles
  have captured: the current year, from the first cycle on. To see a
  location's full history locally, run the backfill once against the same
  directory (`SHALLWESWIM_ARCHIVE_BUCKET=PATH ... --backfill-from --location
  CODE`, see [Backfilling Deep History](#backfilling-deep-history)). Without
  `--store-dir` the store is in process memory and starts empty every run.
- `--cadence MINUTES` sets how often the cycle runs, ten minutes by default,
  which is the cadence the production job targets. The first cycle runs to
  completion during startup and its generation is loaded before the server
  accepts a request, so the first page already has data; later cycles fetch
  only the feeds that are due, so the process fetches each feed once per
  interval.
- `--historic-years N` limits the historical temperature range to the last N
  years, counting the current one: the default 10 in 2026 serves 2017 through
  2026. It exists because the configured ranges now reach back decades and a
  fresh local store has an empty archive, so every year before the floor would
  only be read and reported as a gap. Raise it, with `--store-dir`, when the
  archive behind it holds those years. The job and the deployed web service are
  unaffected.
- `--reload` is not supported here, because the store lives in this process.

```bash
# Persist the archive and published generations between runs
uv run python -m shallweswim.local --port=12345 --store-dir=.local-store
```

The web half alone serves published generations and fetches nothing, so it
needs a store to read and `SHALLWESWIM_SNAPSHOT_READ_BUCKET` is required:

```bash
SHALLWESWIM_SNAPSHOT_READ_BUCKET="$PWD/.local-store" \
  uv run python -m shallweswim.web --port=12345 --reload
```

Without that variable it fails at startup with a message naming it. A locator
containing `/` is a directory, `memory` is this process's memory, and a bare
name is a bucket, so pass a path when you mean the `--store-dir` above. Use
this to serve a store something else is publishing into; to publish and serve
in one process, use `shallweswim.local`.

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

The application is portable: it needs an object store, a scheduled job, and
web servers, and nothing outside `infra/` names a provider
(ARCHITECTURE.md "Documentation"). [infra/README.md](infra/README.md) is the
reference deployment on Google Cloud: the identities and grants, the bucket,
the build and deploy, the service, the job and its schedule, manual runs,
and pausing. Its deploy command is `./infra/build_and_deploy.sh`, run from
the repository root.

## Canonical URLs

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
They are not part of the default test run. Python Playwright, frontend
`@playwright/test`, and the official Playwright image in the frontend CI
workflow are pinned to the same version. Update all three together. The local
Python and frontend packages use the default shared Playwright browser cache,
so either install command below prepares Chromium for both.
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
row counts, missing temperature counts, UTC instant bounds, and elapsed time. The
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
timed out once and succeeded on retry. Louisville currents completed in one
request. Louisville temperature polling is disabled because USGS confirmed
that the site's diagnostic temperature sensor was exposed publicly by mistake
and will remain internal-only. The modern API supports authenticated requests
with an optional key, which increases quota and exposes rate-limit headers.

For local authenticated NWIS testing, create `.env` from `.env.example` and set
`USGS_WATERDATA_API_KEY`. The client sends it as an `X-Api-Key` header when
present and falls back to unauthenticated requests when it is omitted. Do not
commit API keys.

Production injects the same environment variable into Cloud Run from the
`waterdata_usgs_gov_api_key` Secret Manager secret. The runtime service account
has accessor permission scoped to that secret.

#### Optional Observation Archive

Set `SHALLWESWIM_ARCHIVE_BUCKET` to a private GCS bucket name to preserve
temperature and observational currents measurements after successful fetches.
Leave it empty to disable capture. What is archived, how partitions are laid
out, and how merges behave is in [DATA_PIPELINE.md](DATA_PIPELINE.md#the-archive).

Each of the three store variables holds a store *locator*, not only a bucket
name: a bare name is a GCS bucket, a value containing `/` is a directory, and
`memory` is a store in the running process. `shallweswim.local` uses that to
run without any bucket; nothing else changes with the locator kind.

Production capture runs from a scheduled one-shot job rather than the web
service. The job fetches every archivable feed once and exits:

```bash
# Scheduled run: current historical year only
SHALLWESWIM_ARCHIVE_BUCKET=my-archive-bucket \
  uv run python -m shallweswim.update

# One-time backfill of every configured historical year
SHALLWESWIM_ARCHIVE_BUCKET=my-archive-bucket \
  uv run python -m shallweswim.update --full-history
```

In this capture-only mode the job fetches only live temperatures, historical
temperatures, and observational currents; it never fetches tide or current
predictions, generates plots, or starts the web app. A missing bucket variable
fails the run before any upstream request. One failing feed leaves the run
`partial` and still exits zero; a run that publishes nothing exits non-zero
([run summary](DATA_PIPELINE.md#run-summary)).

##### Backfilling Deep History

`--backfill-from` archives everything a provider still holds, rather than the
configured historical range:

```bash
# Walk every year each location's historical temperature source still holds
SHALLWESWIM_ARCHIVE_BUCKET=my-archive-bucket \
  uv run python -m shallweswim.update --backfill-from

# Walk one location, no further back than 2009
SHALLWESWIM_ARCHIVE_BUCKET=my-archive-bucket \
  uv run python -m shallweswim.update --backfill-from 2009 --location pbi
```

Each selected location's historical temperature source is walked one calendar
year at a time, from this year down to the floor year, newest first; without a
year the floor is 1900, so the walk ends by running out of data rather than by
reaching the floor. CO-OPS stations are fetched as both products per year — one
hourly request for the whole year and twelve six-minute requests, one per month
— because neither product covers the other and a month the station lacks would
otherwise abort the year. Every other source is one request per year. A year is
empty only when every one of its requests returned no data; after five
consecutive empty years the source's walk stops, and a year with data resets
that count. One INFO line reports each year as it completes. Locations run one
after another, and requests inside a source run one at a time, a second apart.
CO-OPS answers a rate block with HTTP 403 rather than 429, so the walk reads a
403 as a temporary block: it waits five minutes and asks for the same window
again, up to three times, before that source ends in an error. Walking every
source takes roughly ninety minutes.

The run captures only: it publishes no snapshot, reads nothing back from the
archive, and generates no plots, so it needs `SHALLWESWIM_ARCHIVE_BUCKET` and
nothing else. It is rejected together with `--full-history` or
`SHALLWESWIM_SNAPSHOT_PUBLISH=1`, and an unknown location code fails before any
request. Run it from your own machine against the archive bucket, under a
temporary write grant; the scheduled job keeps capturing meanwhile, and
re-archiving a year already held changes nothing. A source whose fetch fails
unexpectedly ends there, is logged at ERROR, and leaves the run `partial`,
which still exits zero; only a failure of the run itself exits non-zero. The
summary event uses `operation=backfill`, so the scheduled capture metrics and
alerts never count it, and reports per source the years archived, the years
empty, and the earliest year with data — that earliest year is what the
source's `start_year` in `shallweswim/config/locations.py` is then lowered to,
in a reviewed change, so the served range and the plots follow.

##### Published Snapshots

The deployed job also publishes a serving snapshot after its capture cycle, and
runs every ten minutes. `SHALLWESWIM_SNAPSHOT_PUBLISH=1` switches a run to the
full serving cycle of every location, all four feeds including tide and current
predictions, derived frames, and plots in a process pool, and then writes one
immutable generation under `published/` in the archive bucket. Publishing also
requires `SHALLWESWIM_ARCHIVE_READ_BUCKET`, set to the same bucket, because the
historical feed serves every year from the archive. The web service sets
neither variable.

```bash
SHALLWESWIM_SNAPSHOT_PUBLISH=1 \
SHALLWESWIM_ARCHIVE_BUCKET=my-archive-bucket \
SHALLWESWIM_ARCHIVE_READ_BUCKET=my-archive-bucket \
  uv run python -m shallweswim.update
```

The current generation is also the job's feed schedule: a feed not due before
the next run is held and its published entry carried forward, so each feed
keeps its own interval whatever the job cadence is, and `--full-history`
fetches every feed regardless. The generation layout, schedule restoration,
carry-forward, the freshness and publish events, and the sweep of superseded
generations are in [DATA_PIPELINE.md](DATA_PIPELINE.md#the-jobs-cycle).

##### Serving From Published Snapshots

`SHALLWESWIM_SNAPSHOT_READ_BUCKET` names the bucket whose `published/` prefix
the app reads, and it is required: the web service serves every request from
the generation it has loaded and contacts no provider. It is read-only, so the
credential needs no more than `roles/storage.objectViewer` on the bucket, and
the deployed service sets only this variable, substituted from the same Cloud
Build value as the job's bucket.

```bash
SHALLWESWIM_SNAPSHOT_READ_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET" \
  uv run python -m shallweswim.web --port=12345
```

The instance loads the current generation at startup and starts even if that
load fails, answering 503 from `/api/healthy` until one is loaded. One request
a minute is elected to check for a new generation and load it; `/api/status`
reports the served `generation_id`, `published_at`, and `loaded_at`. The load
rules, the swap, and the load event are in
[DATA_PIPELINE.md](DATA_PIPELINE.md#the-web-servers). To roll back a deploy,
route traffic to the previous revision; the store needs no change, and the job
keeps publishing throughout.

##### Comparing The Published Bundle With A Local Fetch

Whether serving from the bundle is equivalent to serving from in-process
fetches is answered by a separate local command, because one local process can
hold both sides in memory at once exactly as a production instance would. The
command never runs in production:

```bash
SHALLWESWIM_SNAPSHOT_READ_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET" \
  uv run python -m shallweswim.scripts.compare_snapshot

# One location, at a chosen local instant
SHALLWESWIM_SNAPSHOT_READ_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET" \
  uv run python -m shallweswim.scripts.compare_snapshot \
  --location nyc --at 2026-06-01T04:00:00
```

It fetches every enabled location from the providers exactly as the job does,
loads the current generation from the read bucket, and asks both sides the same
questions at one location-local instant (now, or `--at`, applied as each
location's own local time). It reads and writes no archive:
`SHALLWESWIM_ARCHIVE_BUCKET` and `SHALLWESWIM_ARCHIVE_READ_BUCKET` are removed
from the run's environment even when `.env` sets them, so the fetched side is a
pure provider fetch that writes nothing. Served history comes from the archive
alone, so with none to read that side holds no historical temperatures and the
feed compares as `extra`; the live temperature, tide, and current comparisons
are what the command is for. Only the viewer credential is needed:
`roles/storage.objectViewer` on the bucket, the same grant the web service
uses. A missing `SHALLWESWIM_SNAPSHOT_READ_BUCKET` fails as a usage error
before any upstream request.

The report is one table per location, then the differing rows. Each configured
feed gets one outcome: `missing` (only the legacy side has data, expected while
the job's own fetch of that feed fails), `extra` (only the bundle has data,
expected when the local fetch failed), `absent` (neither side has data),
`disjoint` (both have data but share no timestamp, so the bundle is too stale to
compare), `mismatch`, or `match`. Alongside it: how many timestamps the two
sides share, how many of those differ, the largest absolute difference, each
side's first and last timestamp, and the lag between the two latest timestamps,
which is the bundle's lag for that feed. Float columns agree within a relative
tolerance of 1e-6 and every other column must agree exactly; a mismatch also
lists the first 20 differing rows with both values. A feed mismatches when a
derived answer differs even if its rows agree: `get_tide_info_at_time` and
`predict_tide_at_time` for tides, `predict_flow_at_time` for prediction
currents, and `get_current_temperature` for live temperature when both frames
end at the same timestamp. Each location also reports which plots exist on each
side; plot bytes are not compared, because the two sides draw different fetch
windows.

The command exits 0 only when every feed is `match`, `extra`, or `absent`, so it
can run in a loop for days and its exit status is the verdict.

##### Serving Historical Temperatures From The Archive

`SHALLWESWIM_ARCHIVE_READ_BUCKET` names the archive the historical temperature
feed serves every year from, so it is required wherever that feed runs. Only
fetching processes read it: the capture job, whose manifest sets it for a
publishing run and which defaults it to `SHALLWESWIM_ARCHIVE_BUCKET` on a
capture-only run, since both name the same store; and a local run, which points
it at the same local store as `--store-dir`. The deployed web service fetches
nothing, so it never sets it.

`SHALLWESWIM_ARCHIVE_READ_BUCKET` is independent of
`SHALLWESWIM_ARCHIVE_BUCKET`, which remains the only variable that enables
writes. Hydration calls only the store's read operation, so the local
credential needs no more than `roles/storage.objectViewer` on the bucket. The
feed's own fetch is one top-up of the current year, which is captured and then
read back like every other year. Archived partitions are UTC years, so reading
a station-local year reads that year's partition and the next one and keeps the
rows inside the local year; its final local hours are served exactly as a
provider fetch would return them. A year the archive lacks, or one whose read
or validation fails, logs a warning and is a gap in the frame and the plots,
never a failed feed.

See [infra/README.md](infra/README.md) for the one-time bucket commands and
the job identity, deployment, scheduling, and validation steps. The operations dashboard includes
archive merges by outcome, capture runs and snapshot publishes per hour by
outcome, maximum published feed age by feed, and new and revised observations
per hour by source; bucket setup and job deployment are separate from applying
monitoring Terraform.

#### Debugging CSPF Sandettie Historical Temperatures

Use the CSPF debug script to exercise Dover's Sandettie historical temperature
fallback without starting the full service:

```bash
uv run python -m shallweswim.scripts.debug_cspf_fetch --location dov \
  --start-year 2011 --end-year 2026
```

The command fetches the same CSPF Sandettie pages as the runtime client and
reports per-year row counts, UTC instant bounds, failures, and elapsed time. The
client uses monthly CSPF pages first because they are denser than annual
summaries, and falls back to an annual page only when monthly pages have no
data.

#### Debugging Irish Lights Temperature Fetches

Use the Irish Lights debug script to exercise Cork/Sandycove buoy temperature
fetches without starting the full service:

```bash
uv run python -m shallweswim.scripts.debug_irish_lights_fetch --location cor
uv run python -m shallweswim.scripts.debug_irish_lights_fetch --location cor \
  --start-year 2024 --end-year 2026
```

The command fetches the same Irish Lights MetOcean endpoint as the runtime
client and reports row counts, UTC instant bounds, Fahrenheit min/max values,
failures, and elapsed time. Cork uses the Irish Lights Cork Buoy as a shared live and
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

[infra/MONITORING.md](infra/MONITORING.md) owns the metrics, alert policies, dashboard,
and log queries built on the application's events; they are applied with
Terraform from [`infra/monitoring`](infra/monitoring/README.md).
[ARCHITECTURE.md](ARCHITECTURE.md) covers station-outage handling.

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

### Embedding swimming conditions

`/{location}/embed` (for example `/nyc/embed`) serves a compact React panel
without app navigation or charts. `/embed` redirects to `/nyc/embed`.
The former Jinja panel remains at `/legacy/{location}/embed` for comparison;
`/legacy/embed` redirects to `/legacy/nyc/embed`.

The panel uses CIBBOWS yellow (`#f3b03d`), charcoal text, blue controls,
and rounded white cards. Yellow fills the entire iframe, including unused space.
The panel omits a visible page heading so the host site can provide its own;
the location remains in the document title and accessible main label. Its isolated `frontend/src/styles/embed.css` uses
self-hosted Poppins Latin fonts (SIL Open Font License), with system-font
fallback for other characters. Temperature and
water-movement cards sit side by side on wider screens and stack on mobile;
the three tide events use compact tiles. Windy stays together as a full-width
map/forecast below the swimming data. There is no separate air-temperature
card or custom weather forecast table.

The panel shows configured live water temperature and station information,
the last and next two tides, current state and magnitude in knots, a current
details link where supported, availability messages, the Windy map/forecast,
and a link to the full location page. Full-app links open in a new tab.
It shares the app's cached conditions API, refresh hook, temperature UI and
formatters, location presentation metadata, and responsive Windy component.

Paste this into a WordPress Custom HTML block:

```html
<iframe
  src="https://shallweswim.today/nyc/embed"
  title="Brighton Beach swimming conditions"
  width="100%"
  height="1200"
  style="border: 0; display: block;"
  loading="lazy"
></iframe>
```

Build with `corepack pnpm@10.18.3 --dir frontend build`, then run
`uv run python -m shallweswim.web --port=12345` and open
`http://localhost:12345/nyc/embed`. Other configured location codes work too.

This baseline uses a fixed iframe height; leave scrolling enabled for larger
text, longer location content, and small screens. It requires JavaScript and
network access to Shall We Swim and Windy. Windy is a third-party interactive
forecast with its own mobile layout and availability. Host WordPress policies
must allow this iframe; no WordPress API calls, proxy, or CORS changes are needed.
After previewing, evaluate panel height, readability, the usefulness of Windy
on mobile, and whether the existing content meets the site's needs before
agreeing on styling or additional products.

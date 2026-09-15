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

A proposed future separation of scheduled data materialization, durable
observation storage, and request-serving is documented in
[Persistent Data Pipeline Design](PERSISTENT_DATA_PIPELINE_DESIGN.md). It is a
design proposal, not the current production architecture.

The proposed provider-neutral telemetry contracts and managed GCP monitoring
approach are documented in [Observability Design](OBSERVABILITY_DESIGN.md).

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
  directory. The next start hydrates historical years from that archive instead
  of refetching every configured year, so a restart is fast. Without it the
  store is in process memory and starts empty every run.
- `--cadence MINUTES` sets how often the cycle runs, ten minutes by default,
  which is the cadence the production job targets. The first cycle runs to
  completion during startup and its generation is loaded before the server
  accepts a request, so the first page already has data; later cycles fetch
  only the feeds that are due, so the process fetches each feed once per
  interval.
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

The application is hosted on Google Cloud Run:

```bash
# Deploy to Google Cloud Run
./build_and_deploy.sh
```

The observation capture job is a separate bounded entry point
(`python -m shallweswim.update`) deployed from the same image as a Cloud Run
Job and triggered by Cloud Scheduler instead of running inside the web
service. See the [capture job runbook](infra/capture-job/README.md).

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
Leave it empty to disable capture. The service continues serving its in-memory
data; archive failures are logged and do not change feed success or retry
scheduling. Repeated fetches of unchanged readings leave the stored partition
byte-identical, so only new observations and upstream revisions are written. Only fresh historical years are captured, so cached years keep their
original retrieval times. Historical years are archived in UTC at the provider's
native cadence, before the hourly serving resample, so both folds of a
daylight-saving fall-back hour reach the archive. Prediction feeds, including
tide and NOAA CO-OPS currents predictions, are excluded.

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
fails the run before any upstream request, because fetching without capturing
has no purpose. Locations run concurrently and each location's feeds run in
sequence. One failing feed leaves the run `partial` and still exits zero; a run
that publishes nothing exits non-zero.

##### Published Snapshots

The deployed job also publishes a serving snapshot after its capture cycle, and
runs every ten minutes.
`SHALLWESWIM_SNAPSHOT_PUBLISH=1` switches a run to the full serving cycle of
every location, exactly as the web service runs it: all four feeds including
tide and current predictions, derived frames, and plots in a process pool. It
then writes one immutable generation under `published/` in the archive bucket:
content-addressed Parquet objects (one per served feed frame) and SVG objects
(one per plot) under `published/objects/`, a manifest under
`published/manifests/`, and finally the `published/current.json` pointer,
replaced conditionally. A generation identical to the current one is not
written. The web service loads these generations and serves only the loaded
one, described below.

The current generation is also the job's feed schedule. Before running a
location's cycle, the job restores each feed's next fetch time from the current
manifest, for every entry that still names the feed's configured source. A feed
that would not come due before the next run starts is not fetched — it is
*held* — and the new manifest keeps its entry and its plots exactly as
published; a feed due before the next run fetches on this one. Each feed thus
keeps its own interval whatever the job cadence is: live temperature every ten
minutes, historical temperature every three hours, tide and current predictions
daily. A feed that is due, that the manifest does not describe, or whose source
identity changed fetches as a fresh feed does. `--full-history` skips
restoration entirely, so every feed fetches. A run in which no feed was due
publishes nothing new and reports `outcome=unchanged`.

Every configured feed of every enabled location is reported. A feed that was
due and fetched nothing keeps the entry the current generation published for
the same source: the object, its fetch timestamp, and its record count stay as
published, while the failure count accumulates and the last error and next
retry become this run's, so a transient provider failure never drops
last-known-good data from a generation. A plot this run did not produce is
copied the same way, but only while the feed it was drawn from is still
published. A feed the current generation never published, one whose source
identity changed, and a feed or location that is no longer configured are
simply absent. A run in which nothing can be referenced at all publishes
nothing. After assembly the job logs one `snapshot.freshness` event per
location and feed with `outcome` `success`, `held`, `carried`, or `absent` and
the served frame's `age_seconds`, at INFO when the feed was fetched or held and
WARNING when it was carried or absent. A held feed counts as published in the
run summary, because the generation keeps serving its frame. Publishing requires
`SHALLWESWIM_ARCHIVE_READ_BUCKET`, set to the same bucket, so the historical
feed restores past years from the archive instead of refetching them; a
publishing run always uses the full historical range. A failed publish is
logged, does not change the run's outcome or exit code, and is named in the run
summary. The web service sets neither variable.

After publishing, each run sweeps the generations that publication superseded.
The generation the current pointer names is kept whatever its age, as is every
generation published in the last 24 hours — the rollback window, and the window
a slow instance could still be loading from. The manifests of older generations
are deleted; `published/current.json` never is. Objects are content-addressed
and shared between generations, so the sweep deletes an object only when no
retained manifest references it and it was created more than an hour ago; that
safety window protects a publisher that has written a generation's objects but
has not promoted it yet. The sweep deletes nothing it did not list in that same
run, never touches the `archive/` prefix, and logs one `snapshot.gc` event with
`outcome` `success` or `failed` and the objects deleted as `record_count`. Like
a failed publish, a failed sweep changes neither the run's outcome nor its exit
code. The local entry point's cycle sweeps too, so a store directory does not
grow without bound.

##### Serving From Published Snapshots

`SHALLWESWIM_SNAPSHOT_READ_BUCKET` names the bucket whose `published/` prefix
the app reads, and it is required: the web service serves every request from
the generation it has loaded and contacts no provider, so a process with no
store has nothing to serve and fails at startup with a message naming the
variable. It is read-only: loading calls only the store's read operation, so
the credential needs no more than `roles/storage.objectViewer` on the bucket.
It is deliberately distinct from `SHALLWESWIM_ARCHIVE_BUCKET`, which enables
writes, and from `SHALLWESWIM_ARCHIVE_READ_BUCKET`, which hydrates historical
years; the deployed service sets only the snapshot read bucket, substituted
from the same Cloud Build value as the job's bucket.

```bash
SHALLWESWIM_SNAPSHOT_READ_BUCKET=shallweswim-archive \
  uv run python -m shallweswim.web --port=12345
```

Every response, health check, and status field comes from the loaded
generation. The instance loads the current one at startup, bounded to 20
seconds, and keeps it current; the only work it does per request that is not a
lookup is the on-demand tide and current detail plot, drawn in its process pool
from the loaded frames. A startup failure or timeout is logged at ERROR and the
instance starts anyway: `/api/healthy` answers 503 until an elected request
loads a generation, the startup probe elects itself every check interval, and
the platform restarts an instance that never becomes ready.

`/api/status` reports the served generation alongside the per-feed status:
`generation_id`, `published_at`, and `loaded_at` on each location, and an empty
object while no generation is loaded. `/api/locations` reports `has_data` for
the locations the generation carries, and a request for any other location
answers 503.

Refresh is request-piggybacked. An HTTP middleware runs on every request,
health checks included: when 60 seconds have elapsed since the last check and
no check is in flight, that request is elected and awaits the check before its
handler runs, and concurrent requests proceed immediately. A check reads
`published/current.json`. If it names the loaded generation, nothing else
happens. If it names a new one, the instance reads that manifest and only the
objects whose content-addressed keys it does not already hold, eight at a time,
validates every frame through its feed model, builds the read-only per-location
managers, and swaps them in with one assignment, so a request running during
the swap finishes on the generation it started with. A location the generation
does not carry simply has no manager. A failed check schedules the next one a
full interval later, never sooner. The app never lists `published/manifests/`
and never deletes anything.

Each load that does work logs one structured event with
`component=snapshot operation=load`, `outcome` `success` or `failed`, the
`generation_id`, `duration_ms`, `record_count` as the objects read, and
`age_seconds` as the lag between the job publishing the generation and this
instance picking it up. An unchanged check logs nothing above DEBUG. There is
no new route: the load events carry the platform's instance identity, which
answers "is every instance loading the bundle" better than an endpoint can.

To roll back, route traffic to the previous revision, which fetches for itself.
No store change is needed, and the job keeps publishing throughout.

##### Comparing The Published Bundle With A Local Fetch

Whether serving from the bundle is equivalent to serving from in-process
fetches is answered by a separate local command, because one local process can
hold both sides in memory at once exactly as a production instance would. The
command never runs in production:

```bash
SHALLWESWIM_SNAPSHOT_READ_BUCKET=shallweswim-archive \
  uv run python -m shallweswim.scripts.compare_snapshot

# One location, at a chosen local instant
SHALLWESWIM_SNAPSHOT_READ_BUCKET=shallweswim-archive \
  uv run python -m shallweswim.scripts.compare_snapshot \
  --location nyc --at 2026-06-01T04:00:00
```

It fetches every enabled location from the providers exactly as the job does,
loads the current generation from the read bucket, and asks both sides the same
questions at one location-local instant (now, or `--at`, applied as each
location's own local time). It hydrates nothing: `SHALLWESWIM_ARCHIVE_BUCKET`
and `SHALLWESWIM_ARCHIVE_READ_BUCKET` are removed from the run's environment
even when `.env` sets them, so the legacy side is a pure provider fetch that
writes nothing. Only the viewer credential is needed:
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
can run in a loop for days and its exit status is the verdict. Historical
temperature frames should agree on overlap except where a provider revised a
reading between the two fetches; investigate every such mismatch rather than
loosening the rule.

##### Hydrating Historical Temperatures From The Archive

`SHALLWESWIM_ARCHIVE_READ_BUCKET` makes a fetching process read historical
temperature years from the archive instead of refetching them from the
provider. Only fetching processes hydrate: the capture job, which sets it in
its manifest, and a local run, which points it at the same local store as
`--store-dir`, so a second start is fast. The deployed web service fetches
nothing, so it never hydrates and never sets it.

`SHALLWESWIM_ARCHIVE_READ_BUCKET` is independent of
`SHALLWESWIM_ARCHIVE_BUCKET`, which remains the only variable that enables
writes. Hydration calls only the store's read operation, so the local
credential needs no more than `roles/storage.objectViewer` on the bucket. The
current year and any year the archive does not hold still fetch from the
provider, and hydrated years are never captured back. Archived partitions are
UTC years, so hydrating a station-local year reads that year's partition and the
next one and keeps the rows inside the local year; its final local hours are
served exactly as a provider fetch would return them. A failed read or
validation for one year logs a warning and leaves that year to the provider, so
hydration never fails startup. The web service manifest never sets this
variable; the capture job sets it because publishing a snapshot hydrates the
full historical range.

See [archive setup](infra/monitoring/README.md#observation-archive-setup) for
the one-time bucket commands and the
[capture job runbook](infra/capture-job/README.md) for the job identity,
deployment, scheduling, and validation steps. The operations dashboard includes
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

The reference GCP deployment's log-based metrics and operations dashboard are
managed by Terraform under [`infra/monitoring`](infra/monitoring/README.md).
See [OBSERVABILITY_DESIGN.md](OBSERVABILITY_DESIGN.md) for the broader monitoring
and alert migration and [ARCHITECTURE.md](ARCHITECTURE.md) for station-outage
handling.

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

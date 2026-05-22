# Frontend Application Design Notes

This document captures the current working plan for a proper frontend application
for Shall We Swim. It is intentionally a work in progress. The frontend approach
is still being discussed and iterated, so decisions here should be treated as
directional until implementation validates them.

## Goals

- Keep the existing Python/FastAPI backend as the source of truth for data,
  domain logic, API routes, caching, background updates, and plot generation.
- Build a frontend application that consumes the existing JSON API instead of
  replacing backend responsibilities.
- Allow the current Jinja-rendered site to stay live while the new frontend is
  developed and deployed incrementally.
- Preserve a simple single-service release model where frontend and backend ship
  together.

## Non-Goals

- Do not replace FastAPI.
- Do not introduce a second backend runtime for application logic.
- Do not move NOAA, USGS, plotting, caching, or feed orchestration logic into the
  browser.
- Do not require the frontend and backend to be independently versioned at this
  stage.

## Proposed Repository Layout

Use `frontend/` for frontend source code and build tooling.

```text
repo/
  shallweswim/              Python backend package
  tests/                    Backend and integration tests
  frontend/                 Frontend app source and tooling
    DESIGN.md
    package.json
    vite.config.ts
    tsconfig.json
    openapi.json
    src/
      api/
      components/
      pages/
      routes/
      styles/
      main.tsx
    dist/                   Build output, likely gitignored
```

Use `/app` as the public route for the new frontend application.

```text
filesystem: frontend/
public route: /app
API routes:  /api/...
```

This avoids overloading `app` as both a source directory name and a public URL
concept.

## `/app` Routing Model

The new frontend should be a standard client-routed SPA mounted under `/app`.
Everything after `/app` belongs to the React router unless it is a built static
asset.

Initial URL shape:

```text
/app                    app landing/default location, initially NYC
/app/nyc                location dashboard (canonical per-location URL)
/app/locations          all locations/status page
/app/assets/...         Vite-built static assets
/app/manifest.webmanifest
```

There is no `/app/{loc}/currents` route in the React app. Currents planning
lives directly inside the location dashboard's Water Movement card: the compact
planner control opens in-card, and deeper local-current context opens as a
detail panel in the same card.

Query parameters carry app state. `planner=open` controls the in-card time
scrubber, `detail=open` controls the current/tide detail panel, and `at=...`
carries the shifted planning time as a location-local ISO-8601 timestamp:

```text
/app/nyc?planner=open
/app/nyc?at=2026-05-18T15:30:00
/app/nyc?planner=open&at=2026-05-18T15:30:00
/app/nyc?detail=open&at=2026-05-18T15:30:00
```

See Milestone 3 below for the full route, URL state, and capability
contract.

`/app` should render the default location view, initially NYC, while preserving
the `/app` URL. Do not redirect `/app` to `/app/nyc` in the first
implementation. Canonicalization can be revisited after the app has real usage.

Implementation requirements:

- configure Vite with `base: "/app/"`
- configure React Router with `basename="/app"`
- register FastAPI `/app` routes before the dynamic `/{location}` route
- serve `/app/assets/...` as static files from the built frontend output
- serve the frontend `index.html` for `/app`, `/app/`, and app-owned nested
  paths such as `/app/nyc`
- return real 404s for missing files under `/app/assets/...`
- keep `/api/...`, existing Jinja routes, and existing `/static/...` behavior
  unchanged

The app subdomain can redirect to `/app` later. Do not make a second deployment
or second backend service for the initial implementation.

## App Shell Caching And Versioning

The frontend must not accidentally trap home-screen users on stale application
code. Because offline behavior is out of scope, favor correctness and fast update
visibility over aggressive app-shell caching.

Initial caching policy:

- serve `/app`, `/app/`, nested app routes, and `/app/manifest.webmanifest` with
  `Cache-Control: no-cache, must-revalidate`
- serve Vite-built hashed assets under `/app/assets/...` with long-lived immutable
  caching, for example `Cache-Control: public, max-age=31536000, immutable`
- do not add a service worker in the initial implementation
- do not cache API responses in HTTP caches; use TanStack Query for in-memory
  client-side freshness behavior

This means a saved-to-home-screen user may keep the old JavaScript bundle while
the app process remains open, but a fresh app load should revalidate the app
shell and discover the latest asset references.

Backend/frontend compatibility policy:

- frontend and backend deploy together in one Docker image
- generated frontend API types must match the backend OpenAPI schema at build
  time
- backend API changes should remain backward-compatible within a release window
  whenever practical, because already-open browser tabs can keep old JavaScript
  in memory while calling the newly deployed backend
- avoid removing or renaming API fields without a compatibility period
- additive API changes are preferred

If we later add a service worker, it must include an explicit update strategy and
clear stale-data messaging before it is enabled.

## Deployment Shape

The current preference is one release artifact and one FastAPI service:

```text
FastAPI serves:
  /api/...       JSON API
  /app/...       New frontend app
  /static/...    Existing static assets and/or built frontend assets
```

The existing template-rendered pages can remain available while `/app` is built
out. A future `app.shallweswim.today` or `app.shallweswim.com` host can redirect
to `/app` without changing the frontend source layout.

Serving the frontend under the same origin keeps API requests simple:

```text
/api/nyc/conditions
/api/nyc/currents
```

This avoids CORS configuration during the initial migration.

## Durable Web And App-Only Future

The long-term direction may be to make the React app the only user-facing
interface and eventually remove the `/app` prefix from canonical location URLs.
That should not mean shipping a blank JavaScript shell as the public web
surface. If React replaces the Jinja pages, the replacement should preserve the
old-web virtues that matter for this project:

- location URLs should remain useful to a human, bot, archive, or agent that
  fetches HTML without executing JavaScript
- structured JSON APIs should stay first-class and discoverable from the page
- app HTML should include meaningful title/meta/canonical data, API links, and
  enough fallback content to identify the location and latest conditions
- search engines and agents should be able to understand the page through
  ordinary HTML plus machine-readable structured data such as JSON-LD or an
  embedded bootstrap JSON payload
- curl/no-JavaScript checks should be part of the migration acceptance criteria
  before replacing the existing template-rendered location URLs

Possible implementation paths:

- FastAPI can serve a location-aware app HTML template instead of one generic
  Vite `index.html`, embedding a small server-rendered fallback summary and
  links such as `/api/{loc}/conditions`.
- React can continue to hydrate/enhance that page in the browser after load.
- If static fallback HTML becomes too limiting, evaluate server-rendered or
  pre-rendered React output, but avoid adding a second production backend runtime
  unless the benefit is clear.

Open question: when the React app reaches parity, should `/nyc` and other
location URLs serve the progressively enhanced React app directly, with `/app`
retained only as a transition path or removed entirely?

## Build And Deploy Integration

Production deploys currently build one Docker image through Cloud Build and
deploy it to Cloud Run. The frontend should integrate into that existing Docker
build rather than introduce a separate deploy pipeline.

Expected Docker changes:

- add a frontend build stage using an exact Node major LTS image
- enable/pin `pnpm` through Corepack or an equivalent non-global invocation
- pin the frontend package manager in `frontend/package.json` with the
  `packageManager` field
- copy `frontend/package.json`, `frontend/pnpm-lock.yaml`, and frontend config
  files before installing dependencies to preserve Docker layer caching
- run `pnpm install --frozen-lockfile`
- copy the full `frontend/` source
- run `pnpm build`
- copy `frontend/dist` into the final Python image, outside `shallweswim/static`
  unless the asset manifest script is intentionally extended to understand it
- keep the existing `uv` Python dependency install and Cloud Run startup command

Expected repository changes:

- commit `frontend/pnpm-lock.yaml`
- add `frontend/node_modules/` and `frontend/dist/` to `.gitignore`
- add relevant frontend paths to `.dockerignore` only when they should not be
  sent as Docker build context
- do not commit `frontend/dist`

`cloudbuild.yaml` can probably remain structurally unchanged because it already
delegates the build to `docker buildx build`. The Dockerfile should own the
frontend build steps.

Initial choice: pin a current Node LTS major image rather than `node:lts-slim`.
Use `node:24-slim` while Node 24 is the active LTS line, and update
deliberately when changing Node major versions.

## Initial Decisions

These decisions are sufficient to start the first implementation. Some are
provisional and should be reassessed after the first useful `/app` route is
running in production.

| Area | Initial decision | Reassess later? |
| --- | --- | --- |
| Backend | Keep FastAPI as the only application backend | No |
| Source directory | `frontend/` | No |
| Public route | `/app` | Yes, before replacing the root experience |
| Framework | React | Unlikely |
| Language | TypeScript | No |
| Build tool | Vite | Yes, if Bun/Rspack materially reduce risk or complexity |
| Package manager | `pnpm` | Yes, after a focused Bun spike |
| Styling | Tailwind CSS with project-specific tokens/components | Yes |
| Data fetching | TanStack Query | Unlikely |
| API client | `openapi-typescript` + `openapi-fetch` | Yes |
| Routing | React Router | Yes |
| Formatting | Biome, matching the existing repo formatter config | Yes |
| Unit/component tests | Vitest | Yes |
| Browser tests | Playwright | No |
| Plots | Existing backend-generated image endpoints | Yes |
| External embeds | Local React wrappers around iframe/API integrations | Yes |
| Offline behavior | Explicitly out of scope | Yes, much later |

## Candidate Stack

The current leading option is:

```text
Vite
React
TypeScript
Tailwind CSS
TanStack Query
openapi-typescript + openapi-fetch
Vitest
Playwright
```

React is the current working choice for the frontend component model. Tailwind
CSS is also the current working styling choice, but it remains easier to adjust
later than the component framework decision.

React does not mean running a Node application in production. Node is expected to
be a development and build-time dependency for the frontend toolchain. The built
frontend output should be static HTML, JavaScript, CSS, and assets served by the
existing FastAPI service.

### Vite

Frontend build tool and development server. Vite would compile TypeScript/JSX,
provide local hot reload, and produce static production assets that FastAPI can
serve.

### React

UI component model for the browser application. React would replace the current
combination of Jinja markup plus ad hoc frontend JavaScript for the new app.

### TypeScript

Adds compile-time checking for frontend code and generated API types.

### Tailwind CSS

Utility-first styling layer for building responsive layouts and component styles.
Tailwind should be used with project-specific design tokens and components, not
as a substitute for intentional visual design.

The current expectation is that Tailwind helps implement the app layout and
component styling quickly while still leaving room to change colors, typography,
spacing, and visual treatment as the product design evolves.

### TanStack Query

Manages frontend server-state concerns:

- API fetch lifecycle
- loading and error states
- caching
- stale data
- periodic refetching
- retry behavior
- retaining prior data during refresh failures

TanStack Query does not generate types from OpenAPI by itself. It should wrap a
typed API client generated from the backend OpenAPI contract.

### OpenAPI Type Generation

FastAPI already exposes an OpenAPI schema. The intended contract flow is:

```text
FastAPI/Pydantic code
  -> exported OpenAPI schema
  -> generated TypeScript API types/client
  -> TanStack Query hooks
  -> React components
```

Likely tooling:

```text
openapi-typescript
openapi-fetch
```

This keeps the frontend aligned with the backend API contract without importing
Python implementation details.

## Frontend Toolchain

The frontend toolchain is expected to be Node-based for development and build
steps only.

Expected local tooling:

```text
Node 24 LTS
pnpm
Vite
TypeScript
Tailwind CSS
Biome or Prettier-compatible formatting
Vitest
Playwright
```

The working package-manager decision is `pnpm`. Avoid plain `npm` for this
frontend. `pnpm` is a conservative upgrade that offers stricter dependency
isolation while keeping broad compatibility with the Vite/React/Tailwind
ecosystem.

`bun` is still worth evaluating later because it provides a newer, faster, more
integrated JavaScript/TypeScript toolchain. Do not use Bun as the first frontend
toolchain bet unless a focused spike shows that Vite, Tailwind, OpenAPI
generation, type checking, Playwright, and CI all work cleanly with it.

The project already uses newer tooling on the Python side, including `uv`,
`ruff`, `pyrefly`, and `rumdl`. Choosing modern frontend tooling is consistent
with that approach as long as the decision is deliberate, bounded, and backed by
lockfiles and CI checks.

Package-manager decision:

```text
pnpm  initial frontend package manager
bun   future evaluation candidate
npm   explicitly not preferred
```

Expected scripts:

```bash
pnpm dev           # Vite development server
pnpm build         # TypeScript check plus Vite production build
pnpm typecheck     # tsc --noEmit
pnpm generate-api  # Generate TS API types/client from frontend/openapi.json
pnpm test          # Frontend unit/component tests
pnpm test:e2e:install  # Install Playwright Chromium into frontend/node_modules
pnpm test:e2e      # Playwright browser tests
```

Production output:

```text
frontend/dist/
  index.html
  assets/*.js
  assets/*.css
  assets/*
```

FastAPI should serve the production output under `/app` as static assets. No
Node server should be required in production.

Local missing-dist behavior:

- if `frontend/dist` is missing in local development, `/app` should return 404
  or a clear development-only not-built response
- local developers should run `pnpm build` before testing production-style
  FastAPI serving of `/app`

Production missing-dist behavior:

- the Docker image build should fail if `frontend/dist` is not produced
- a production container configured to serve `/app` should fail loudly at startup
  or during route setup if the built frontend shell is missing
- do not silently serve a partial or empty `/app` in production

## Dependency And Supply-Chain Rules

The frontend should keep the JavaScript dependency graph intentionally small.

Initial rules:

- use `pnpm`, not plain `npm`
- commit `pnpm-lock.yaml`
- use frozen lockfile installs in CI and Docker builds
- review lockfile changes as part of code review
- prefer first-party or widely adopted packages
- avoid third-party React wrappers for simple iframe integrations
- avoid packages that require install/postinstall scripts unless clearly
  justified
- add dependency update automation later, likely Dependabot or Renovate
- run frontend build/type/test checks in CI before deploy
- keep Node/pnpm as build-time dependencies only; no Node runtime in production

## GitHub Actions

Milestones 0 and 1 should include a dedicated frontend GitHub Actions workflow.
Do not punt frontend CI until feature parity; the scaffold and `/app` shell need
continuous validation as soon as they are introduced.

Initial frontend workflow:

```text
.github/workflows/frontend.yml
```

Triggers:

```text
push
pull_request
workflow_dispatch
```

Required first-milestone jobs:

- check out the repository
- set up Python 3.13 and `uv`
- install backend dev dependencies with `uv sync --dev`
- set up Node 24
- enable Corepack or use pinned `pnpm@10.18.3` without a global install
- install frontend dependencies with `pnpm --dir frontend install --frozen-lockfile`
- run `pnpm --dir frontend lint`
- run `pnpm --dir frontend typecheck`
- run `pnpm --dir frontend test`
- run `pnpm --dir frontend build`
- run `pnpm --dir frontend check:api`
- install Playwright Chromium for the frontend test environment
- run `pnpm --dir frontend test:e2e`

The existing Python workflows should remain in place:

- unit tests
- Ruff
- Pyrefly
- existing Python Playwright browser tests for the Jinja frontend
- integration and performance workflows

Do not run full frontend builds or Playwright from pre-commit hooks. Keep
pre-commit fast: formatting, linting, and file hygiene only. Full frontend checks
belong in scripts and GitHub Actions.

Future/deferred workflow work:

- Docker image build validation can remain in Cloud Build for now.
- Browser test consolidation between Python Playwright and JavaScript Playwright
  is deferred until the React app has real feature coverage.
- Dependency update automation can be added later with Dependabot or Renovate.

## API Contract Workflow

Because the frontend and backend are expected to release together, the OpenAPI
schema can be treated as a generated contract artifact.

Likely files:

```text
frontend/openapi.json
frontend/src/api/generated.ts
```

Commit both files initially. This makes frontend builds reproducible and makes
API contract changes visible in code review. CI should regenerate both artifacts
and fail if they are stale.

Expected checks:

```bash
uv run python -m shallweswim.scripts.export_openapi > frontend/openapi.json
pnpm generate-api
git diff --exit-code frontend/openapi.json frontend/src/api/generated.ts
```

The desired invariant:

```text
Every deployed frontend bundle was built against the exact backend API contract
shipped with it.
```

`openapi-typescript` and `openapi-fetch` are the standard tools selected for the
initial implementation. The missing backend work is not a generator; it is a
small deterministic script that exports the current FastAPI schema to
`frontend/openapi.json` without requiring a running development server.

## Backend Support Work

The frontend can and should drive small backend additions when the current JSON
API does not expose data that Jinja currently receives through server-side
template context.

Required backend changes for the first implementation:

- add `shallweswim.scripts.export_openapi`
- add a freshness check for `frontend/openapi.json` and generated TypeScript API
  files
- add FastAPI routes to serve the built frontend under `/app`
- add tests that `/app` and `/app/nyc` serve the frontend shell while
  `/api/...` and existing Jinja routes still work
- add a frontend bootstrap API for non-secret presentation config

Initial bootstrap endpoint:

```text
GET /api/app/bootstrap
```

Initial response content:

```text
app name and short name
default location code, initially `nyc`
location list and nav labels
per-location display metadata:
  code
  name
  swim_location
  swim_location_link
  description
  latitude
  longitude
  timezone
  feature flags for temperature, tides, currents, webcam, transit, Windy
  source citation HTML or structured citation data
NYC-specific external integration config:
  YouTube live channel/embed settings
  transit train list and GoodService route IDs
```

Location metadata should come from `/api/app/bootstrap`, not from hardcoded
frontend tables. The frontend may contain generic presentation code and default
fallback copy, but adding a location, changing a swim-location link,
enabling/disabling a camera, changing transit route IDs, or updating citations
should be a backend configuration/API contract change, not a frontend source-code
change.

Do not expose station IDs or backend feed internals through this endpoint unless
there is a clear frontend need and explicit approval. The endpoint is for
presentation metadata, not operational configuration.

Citation data can initially be returned as trusted HTML from the existing config,
because the content is controlled by this repository. If this becomes
user-editable or externally sourced later, switch to structured citation data
instead of rendering HTML.

The bootstrap endpoint is part of the first implementation chunk. It can return a
minimal typed response at first, but it should exist before frontend routes start
hardcoding app or location metadata.

### App Config Versus Locations API

Keep `/api/locations` focused on simple location-summary data:

```text
code
name
swim_location
latitude
longitude
has_data
```

That endpoint answers: "What swim locations exist, where are they, and do they
currently have data?"

Keep `/api/app/bootstrap` focused on frontend bootstrap and presentation data:

```text
app name and default location
location display order
per-location feature flags
source citations
external embed settings
transit route settings
installable-app presentation metadata
```

That endpoint answers: "How should the React app render itself?"

Some fields will intentionally overlap between the two endpoints, such as code,
name, swim location, and coordinates. That is acceptable because the endpoint
purposes are different. Avoid turning `/api/locations` into a kitchen-sink
frontend bootstrap endpoint.

## Migration Plan

1. Keep the existing Jinja site live.
2. Add frontend build tooling under `frontend/`.
3. Export the FastAPI OpenAPI schema and generate TypeScript API types.
4. Build the first React route for a location conditions page.
5. Serve the built frontend under `/app`.
6. Add pages incrementally:
   - location conditions
   - currents
   - all locations/status
   - widgets or embeds if still needed
7. Link to `/app` from the existing site once useful.
8. Decide later whether the new app replaces the root Jinja experience.

## Incremental Milestones

### Milestone 0: Scaffolding And Contract

- create Vite/React/TypeScript/Tailwind app under `frontend/`
- configure pnpm, Biome, Vitest, Playwright, and basic scripts
- add OpenAPI export script
- add a minimal typed `/api/app/bootstrap` endpoint
- generate and commit `frontend/openapi.json`
- generate and commit the typed API client
- update pre-commit to run fast frontend formatting/lint checks, including
  `frontend/`
- add script-level checks for frontend typecheck, build, tests, and API contract
  freshness

### Milestone 1: Serve A Minimal App Shell

- build frontend in Docker
- serve built output under `/app`
- configure Vite base and React Router basename
- add app manifest and icons
- prove `/app` and `/app/nyc` reach the React shell
- prove production builds fail loudly if the frontend build output is missing
- keep current Jinja pages unchanged

### Milestone 2: NYC Location Page Vertical Slice

- implement `/app/nyc`
- consume `/api/app/bootstrap`
- consume `/api/nyc/conditions`
- render core summary, tides, current estimate, Windy iframe, YouTube webcam,
  temperature plot images, NYC transit cards, citations, and location nav
- match current loading/unavailable/stale-data behavior
- add mobile and desktop Playwright browser coverage

#### Milestone 2 Implementation Spec

Milestone 2 should make the React app useful for the default NYC location while
staying close to the current Jinja/HTML experience. Bias toward parity with the
existing site unless React gives a clearly simpler implementation with the same
user-facing behavior.

Route scope:

- `/app` and `/app/nyc` should both render the real NYC location page. `/app`
  should preserve the URL rather than redirecting.
- No `/app/{loc}/currents` route exists in the React app. The currents
  planning surface lives in the Water Movement card on the dashboard
  (Milestone 3); deep links use `?planner=open&at=...` for the scrubber
  and `?detail=open&at=...` for the supporting detail panel.
- `/app/locations` remains a placeholder in this milestone.
- Unsupported location codes can continue using placeholder or not-found
  behavior until Milestone 3 generalizes location pages.

Data dependencies:

- Use `/api/app/bootstrap` for app name, default location, location navigation,
  NYC display metadata, feature flags, citations, YouTube config, and transit
  route config.
- Use `/api/nyc/conditions` for water temperature, tide timing, and current
  estimate.
- Use existing plot image endpoints directly:
  - `/api/nyc/plots/live_temps`
  - `/api/nyc/plots/historic_temps?period=2mo`
  - `/api/nyc/plots/historic_temps?period=12mo`
- Use direct GoodService fetches for NYC transit routes from bootstrap:
  `https://goodservice.io/api/routes/{goodservice_route_id}`.
- Do not add raw chart-data endpoints, a backend transit proxy, or new backend
  feed logic for this milestone unless direct browser transit access proves
  impossible.

Page content and ordering should match the current NYC page:

1. Header with `shall we swim today?` and the configured swim-location link.
2. Conditions summary:
   - water temperature: `The water is currently {water_temp}{degree symbol}{units}`
   - station note: `at {station_name} as of {formatted timestamp}.`
   - last tide, next tide, and following tide rows
   - current estimate: `{state_description or direction} at {magnitude} knots`
   - show planner/detail controls (Milestone 3) only when current prediction
     data is available
3. Forecast section with the Windy iframe.
4. Live Webcam section with the YouTube live embed and the existing EarthCam
   alternative link.
5. Temperature Trends section with the three existing plot images when enabled.
6. Transit Status section with B and Q train cards.
7. Sources section with configured temperature/tide/current citations plus NYC
   webcam, GoodService, GitHub, and location navigation entries.

Formatting rules:

- Format tide dates as weekday, month, and day, matching the current frontend.
- Format tide times and station timestamps with 12-hour US English display.
- Format current magnitude to one decimal place when numeric.
- Use `Unavailable`, `N/A`, and `unavailable` text consistently with the current
  JavaScript behavior for first-load failures.
- Citation HTML is trusted repository-controlled content from bootstrap and may
  be rendered as HTML in this milestone. Keep that rendering isolated in a small
  component so it can be replaced with structured citation rendering later.

Runtime behavior:

- Fetch conditions immediately on page load and refetch every 60 seconds.
- Avoid overlapping condition requests for the same location; TanStack Query
  should be configured so it does not create concurrent duplicate fetches.
- On first condition failure, render unavailable states for temperature, tide,
  and current areas and show:
  `Unable to load latest conditions. Please try again later.`
- On refresh failure after a successful load, keep prior data visible and show:
  `Could not refresh latest conditions. Showing last loaded data.`
- Start deferred plot image loading only after the first conditions request has
  settled, whether it succeeds or fails.
- Probe plot image loads before displaying `src`; retry failures after 1s, 3s,
  and 7s; then show `Plot unavailable` for that plot without breaking the rest
  of the page.
- Fetch transit independently from swim conditions and refetch every 60 seconds.
- On first transit failure, show `Unavailable`, destination `unavailable`, and no
  alert detail rows for that train. On later failures, keep the previously loaded
  transit data visible.

External integration parity:

- Windy should use the same iframe URL shape as the current `windy_iframe`
  template macro, with location latitude/longitude, `overlay=waves`,
  `product=ecmwfWaves`, `message=true`, `marker=true`, `calendar=now`,
  `detail=true`, and Fahrenheit temperature units.
- YouTube should use the bootstrap embed URL and preserve the current player
  parameters: `enablejsapi=1`, `controls=0`, `playsinline=1`,
  `iv_load_policy=3`, and `rel=0`. Load the official iframe API only if needed
  to mute and attempt autoplay.
- Transit status should preserve the current GoodService southbound behavior:
  prefer `direction_statuses.south`, treat route-level `Not Scheduled`
  specially, show `destinations.south[0]`, and render south/both delay, service
  change, and service irregularity summaries when present.
- Preserve the current status color mapping semantically, though the React/Tailwind
  class names do not need to match the old CSS names.

Suggested component boundaries:

- `LocationPage`
- `ConditionsSummary`
- `TideSummary`
- `CurrentSummary`
- `WindyEmbed`
- `YouTubeLiveEmbed`
- `DeferredPlotImage`
- `TemperaturePlots`
- `TransitStatusSection`
- `TransitRouteCard`
- `SourcesList`
- `LocationNav`

Suggested hooks/clients:

- `useAppBootstrap`
- `useLocationConditions(locationCode)`
- `useDeferredImage(src, enabled)`
- `useTransitRoute(routeConfig)`

Acceptance checks:

- Existing backend unit tests still pass.
- `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`,
  `pnpm check:api`, and `pnpm test:e2e` pass.
- Add unit/component tests for formatting helpers and at least the conditions
  summary unavailable/success states.
- Add Playwright browser tests for `/app` and `/app/nyc` on mobile and desktop
  that verify the real NYC page renders the header, conditions summary, Windy
  section, webcam section, temperature trends, transit section, and sources.
- Add Playwright coverage that `/app` does not redirect away from `/app`.
- Mock external GoodService and bootstrap/conditions responses in browser tests
  so CI does not depend on third-party services.

Non-goals:

- Do not redesign the product or replace backend-generated plots with client
  charting.
- Do not implement full `/app/:locationCode` generalization beyond what is
  naturally needed for NYC.
- Do not implement URL-backed planner/detail controls — those land in Milestone 3.
- Do not add service workers or offline data caching.
- Do not add the existing-site adoption banner yet.

### Milestone 2.5: NYC Layout And Information Hierarchy Pass

Milestone 2 intentionally biases toward current Jinja/HTML parity. After that
slice is committed, take a dedicated layout pass before spreading the same
structure to every location.

Goal:

- keep the Milestone 2 data, routes, embed behavior, refresh behavior, and tests
  intact
- improve the NYC page's information hierarchy, scanability, density, and mobile
  ergonomics
- make the page feel like a focused conditions app rather than a literal port of
  the old document layout

Primary design question:

```text
What does a swimmer need to know first before deciding whether to swim today?
```

Expected direction:

- prioritize the current swim decision signals near the top: water temperature,
  tide timing, current estimate, and forecast context
- make supporting material secondary: plots, webcam, transit, sources, and
  location navigation
- reduce excessive vertical scanning without hiding important details
- use layout density carefully: compact where information is routine, spacious
  where comparison or interpretation matters
- preserve direct access to the full supporting evidence, especially plots and
  citations
- keep the design restrained, operational, and mobile-first

Temperature plot treatment:

- Prefer a single temperature plot module instead of showing all temperature
  plots as separate stacked images.
- The module should provide a compact selector for:
  - live
  - 2-month historical
  - 12-month historical
- Only the selected plot should be visually primary. This should reduce vertical
  scanning while preserving access to the same plot endpoints.
- Keep the selector state local to the component for now. It does not need to be
  encoded in the URL unless a later sharing/bookmarking use case appears.
- Continue to use the existing backend image endpoints in this phase.

Compact tide/current visual summaries:

- Explore compact bar-style instruments for tides and currents in the top
  decision area. These should summarize state, not replace the fuller supporting
  plots elsewhere on the page.
- The bar direction matters. It should communicate whether the value is filling
  toward a peak or draining away from one, not only the current percentage.
- For currents, use backend-derived `current.magnitude_pct`,
  `current.trend`, `current.phase`, `current.magnitude`,
  `current.state_description`, and `current.range` when available.
- Current bar labeling should make the basis explicit, for example:
  - current direction/phase: flooding, ebbing, slack before flood, etc.
  - whether the current is building, easing, or steady
  - current value in knots
  - relevant slack value/time and segment peak value/time when
    `current.range` is available
- For tides, do not approximate the normalized percentage in the frontend. Add a
  backend-derived tide summary before building a tide bar so the app can display
  the current predicted tide height, normalized height/progress, rising/falling
  direction, and the relevant high/low event time and height.
- Tide bar labeling should make the basis explicit, for example:
  - rising toward high or falling toward low
  - current predicted tide height in feet
  - next high/low height and time
  - percentage toward the relevant high/low range or event, as defined by the
    backend field
- Use animation sparingly. A subtle leading-edge pulse/shimmer can indicate
  filling/building/rising; a subtle trailing-edge pulse or inward motion can
  indicate draining/easing/falling. Avoid distracting blinking and respect
  `prefers-reduced-motion`.
- Treat current and tide bars as independent instruments. Their percentages have
  different physical meanings, so the UI must not imply they are directly
  comparable.

Provisional visual direction:

- Treat the visual style as exploratory in Milestone 2.5. The goal is to see a
  concrete design in the app, review it on desktop and mobile, and iterate.
- Do not lock in a final design system yet. Any typography, color, spacing, card,
  or bar-chart choices made in this milestone are provisional until reviewed with
  real content.
- A full terminal-style interface is probably too strong for the whole app. It is
  distinctive, but it can hurt readability and make the app feel more like a
  novelty dashboard than a practical swim-conditions tool.
- Use terminal/console aesthetics selectively as an accent language for
  instrument-like elements where precision and compactness matter.
- Good candidates for selective monospace or console treatment:
  - tide/current bar instruments
  - numeric measurements
  - timestamps
  - compact status labels such as `LIVE`, `RISING`, `FALLING`, `BUILDING`, or
    `EASING`
  - station/source codes when they are shown as data labels
- Keep body copy, explanatory text, section headings, links, and most navigation
  in a more natural readable sans-serif treatment unless a prototype proves the
  monospace style works better.
- The ASCII-style bar idea is explicitly worth prototyping. It may give the app a
  distinctive identity while remaining contained to the tide/current instruments.
- Prefer a restrained operational UI overall: useful, readable, compact, and
  mobile-safe. Avoid letting the aesthetic overpower the swim-condition
  information.
- A possible hybrid pattern:

```text
CURRENT  FLOODING / BUILDING
1.3 kt now   peak 1.6 kt 5:10 PM
[############----] 72%  filling
```

This is only an example of tone and density, not a required final component.

Implementation guidance:

- this is a React/Tailwind layout refactor, not a backend/API milestone
- prefer rearranging and refining existing Milestone 2 components over rewriting
  data hooks or behavior
- add small presentational components only when they make the layout easier to
  reason about
- keep component and browser tests focused on behavior and key visible content,
  not pixel-level layout
- do not generalize all locations yet; use NYC as the design proving ground

Acceptance checks:

- `/app` and `/app/nyc` still render the NYC page without redirecting `/app`
- conditions, Windy, YouTube, deferred plots, transit, sources, and location nav
  still render with mocked Playwright data
- first-load and refresh failure behavior remains unchanged
- no horizontal overflow at the existing mobile viewport test size
- frontend lint, typecheck, unit tests, build, API freshness check, and e2e tests
  pass

### Milestone 3: Location Dashboard with Planner Mode

This milestone supersedes the earlier sketches of both a dedicated
`/app/{loc}/currents` detail route and a standalone "Generalize
Location Pages" milestone. The dashboard shape is the single
per-location page, so generalizing to non-NYC locations is just step
`3.D` (per-location rendering plus the router/404 plumbing). Earlier
wireframes in `shallweswim/static/claude_app_structure.html` explored a
full-screen/side-drawer planner overlay, but the implemented dashboard direction
keeps the planning controls and detail context embedded in the Water Movement
card. That turned out to be simpler, more legible, and better aligned with the
dashboard's compact decision workflow.

#### Chosen direction

The location page at `/app/{loc}` is a single dashboard with no nested route.
Three orthogonal URL query params drive its variable behavior:

- `planner=open` controls the **in-card planner controls**: a compact time
  scrubber in the sticky Water Movement card header. With no `at`, it opens at
  "now".
- `detail=open` controls the **in-card current/tide detail panel**: the
  projection plot plus NYC local-current context such as map, direction
  guidance, timing notes, historic harbor chart, and estimate methodology.
- `at=...` carries the **shifted planning time** as an ISO-8601 timestamp. When
  set, every prediction-respecting Water Movement component — tide state,
  current state, meters, and detail plot URL — shows the selected time. When not
  set, the dashboard remains in its right-now state.

The params are independent: `?planner=open` opens only the scrubber,
`?detail=open` opens only the supporting evidence/detail panel, and `?at=...`
shifts the dashboard without forcing either panel open. In conversational
shorthand, the page is in "planner mode" whenever `at` is set; the URL contract
is the authoritative model.

Deep currents detail stays close to the Water Movement instrument rather than
moving into a separate overlay. This keeps the map, projection plot, current
direction guidance, and legacy chart connected to the selected time while
avoiding a second page-like interaction model inside the dashboard.

NYC's domain depth — direction commentary, physics notes, custom map selection,
legacy chart copy — is app code, not `/api/app/bootstrap` data. The bootstrap
endpoint stays focused on operational config (capability flags, webcam URLs,
transit route IDs, citation links). A later organization pass may move NYC-only
content into `frontend/src/locations/nyc/`, but that is a code-structure cleanup,
not a product requirement for this milestone.

#### Route contract

- Canonical URL: `/app/{loc}` with query params. There is no separate
  `/currents` route in the React app — the React app isn't bound to
  the Jinja URL shape, so we adopt the dashboard convention directly.
- Primary planning URL: `/app/{loc}?planner=open&at=…` — dashboard with the
  in-card planner scrubber open at the chosen time.
- Primary detail URL: `/app/{loc}?detail=open&at=…` — dashboard with the
  current/tide detail panel open at the chosen time.
- Known location without prediction data: the dashboard loads
  normally; planner/detail controls are quietly suppressed; deep-linked
  `?planner=open` or `?detail=open` URLs are no-ops rather than errors.
- Unknown location code: hard HTTP 404. The 404 is reserved for codes
  the app doesn't know about, not for known locations that simply
  lack a planner.

#### URL state contract

- `?at=` is a location-local ISO-8601 timestamp without a timezone
  offset, e.g. `?at=2026-05-18T15:30:00`. The location code supplies the
  timezone; backend endpoints reject `Z`, explicit offsets, DST-ambiguous
  local instants, and times outside the ±24h prediction window.
- `?planner=open` controls in-card scrubber visibility independently of `?at=`.
- `?detail=open` controls in-card detail-panel visibility independently of
  `?planner=open` and `?at=`.
- Shift-aware backend endpoints accept `?at=` directly:
  `/api/{loc}/currents?at=…` and
  `/api/{loc}/plots/current_tide?at=…`. `?shift=` remains available for
  simple relative controls and legacy callers; `at` wins if both are present.

History semantics:

| Action                                             | URL change                                | History  |
| -------------------------------------------------- | ----------------------------------------- | -------- |
| Open planner controls from no-shift state          | adds `?planner=open`                      | push     |
| Open detail panel from no-shift state              | adds `?detail=open`                       | push     |
| Scrub time in planner controls                     | updates `?at=…`                           | replace  |
| Close planner controls                             | drops `?planner=open`, keeps `?at`        | replace  |
| Close detail panel                                 | drops `?detail=open`, keeps `?at`         | replace  |
| Reset to now                                       | drops `?at`                               | replace  |

Direct-entry edge case: if a user lands on
`/app/{loc}?planner=open&at=…` via a shared link, browser Back follows
normal browser history (leaves the app entirely). The in-app close control still
replaces to `/app/{loc}?at=…` so dismissal behaves the same as in a fresh
session. The same rule applies to `?detail=open&at=…`.

#### Capability / fallback rules

| Location type                              | Planner controls   | Detail panel     | Detail content                                                           |
| ------------------------------------------ | ------------------ | ---------------- | ------------------------------------------------------------------------ |
| Tidal · current prediction · NYC           | enabled            | enabled          | full — projection plot + map + commentary + legacy charts                |
| Tidal · current prediction · other         | enabled            | enabled          | generic — projection plot                                                |
| Tidal · tide prediction only (SAN/SFO/BOS/SEA) | optional · module-driven | optional · module-driven | tide projection only                                             |
| River · current observation (SDF)          | disabled           | disabled         | n/a — "Recent flow (observed)" card on home instead                      |
| Lake (CHI)                                 | disabled           | disabled         | n/a — no Water Movement card                                             |
| Spring · temperature-only (AUS)            | disabled           | disabled         | n/a — minimal home layout                                                |

Default for any new location: the planner trigger is **disabled** unless
the location either has `currents_source.type === 'prediction'` in its
bootstrap config (auto-enables the trigger with generic detail content)
or registers custom planner/detail content (NYC-style).

Planner and detail interaction:

- The Water Movement control strip is sticky within the card so the selected
  time, reset affordance, and planner/detail buttons remain close to the meters.
- Opening planner or detail pushes one history entry. Closing either panel uses
  replace semantics and preserves `at`.
- The `Now` reset drops `at` and returns the card to live conditions without
  changing unrelated query params.
- The planner scrubber updates predictions in real time through
  `/api/{loc}/conditions?at=...`; debouncing can be added if API cadence becomes
  noisy.
- Touch targets should be widened as the controls mature; the current compact
  implementation is acceptable for the first shippable planner pass but needs a
  mobile polish pass.

#### Shipping sequence

Each step is independently shippable and adds a layer to the dashboard. Step 3.A
is the foundation; 3.B polishes the control surface; 3.C and 3.D add smarter
time jumps and broader location support.

- **3.A · In-card planner and detail foundation:** Water Movement gets
  URL-backed `planner=open`, `detail=open`, and `at=...` state. The compact
  scrubber shifts `/api/{loc}/conditions?at=...`; the detail panel uses
  `/api/{loc}/plots/current_tide?at=...` plus NYC local-current map selection,
  direction guidance, timing note, legacy harbor chart, and methodology copy.
  Result: feature parity with the useful parts of the existing Jinja currents
  page while keeping the interaction in the dashboard.
- **3.B · Planner control polish:** refine the sticky Water Movement control
  strip, selected-time readout, reset affordance, slider labels, mobile spacing,
  and loading state when shifted data or shifted plot images update. Users should
  be able to keep scrubbing without opening a separate surface. This replaces
  the older full-page sticky time-bar idea.
- **3.C · In-card Now/Soon toggle + smart presets:** Water Movement
  card's `Now` / `Soon` chips for the 2-hour outlook. Adds
  `next slack` / `peak flood` / `peak ebb` preset chips to the planner controls.
  *Depends on the backend follow-up below.*
- **3.D · Generalize to other locations:** ship `locations/sdf/`,
  `locations/chi/`, `locations/aus/`, and `locations/generic/`. Each
  module configures which generic components render and provides
  per-location copy. This step also handles the router-level work
  formerly tracked as a standalone "Generalize Location Pages"
  milestone: `/app/:locationCode` routing for any registered location,
  per-location feature-flag rendering, graceful absence of features
  (no temp / tide / current / webcam / transit on a given location
  collapses without leaving holes), and preservation of 404 /
  unavailable states. The dashboard shape is the single per-location
  page; per-location modules just configure it.

`/app/locations` (all-locations grid, matching the Jinja `/all` page)
and the embed widget (`/{loc}/widget`) decisions stay in this milestone
but are independent of the dashboard rework.

#### Backend follow-up track

Optional, additive, none of these block 3.A or 3.B:

- Named inflection-point timestamps in `/api/{loc}/currents`
  (`next_slack`, `next_peak_flood`, `next_peak_ebb`). Unblocks 3.C's
  smart presets without client-side arithmetic.
- Replace `/api/{loc}/plots/current_tide` image endpoint with a JSON
  time-series so the projection plot renders client-side (interactive
  tooltips, sharper labels, dynamic time-window bound to the planner
  slider). The image endpoint stays for the Jinja site until that's
  retired.
- Future: a temperature prediction feed. The dashboard's planner mode
  is already wired to accept it — adding a new prediction-respecting
  component is the per-component renderer plus the feed.

### Milestone 4: Adoption Banner

- add the existing-site "try the new app" banner
- include dismiss-forever behavior
- link users from Jinja pages to `/app`

## First Implementation Milestone

The first implementation should cover Milestones 0 and 1. It should produce a
deployable but clearly provisional app shell under `/app`; it does not need to
replace or duplicate the current site yet.

Milestone scope:

- scaffold `frontend/` with Vite, React, TypeScript, Tailwind, pnpm, Biome,
  Vitest, and Playwright
- add OpenAPI export, minimal `/api/app/bootstrap`, and TypeScript API generation
- serve the built frontend under `/app`
- add placeholder routes for `/app` and `/app/nyc`
- include installable-app manifest metadata for `/app`
- keep the existing Jinja pages unchanged except for wiring needed to serve `/app`

Milestone acceptance checks:

- `pnpm build`
- `pnpm typecheck`
- `pnpm test`
- `pnpm test:e2e` with JavaScript Playwright browser tests under `frontend/`
- existing backend unit tests
- OpenAPI/generated-client freshness check

The first pass should create JavaScript Playwright tests under `frontend/` for
the React app shell. Existing Python Playwright tests can remain where they are
for current Jinja behavior. Moving or consolidating browser tests is a later
cleanup decision.

This milestone should prefer parity and correctness over redesign. Visual design
can improve during implementation, but avoid large product changes until the
React app has matched current behavior.

Feature parity starts in Milestone 2, not in the first scaffolding pass.

### First Milestone Follow-Up Notes

The initial app shell is intentionally provisional. Before or during Milestone 2,
address these known rough edges:

- Replace developer-facing placeholder copy with real condition content as the
  NYC vertical slice is implemented.
- Improve the location nav for narrow mobile viewports; the current full-button
  nav is acceptable for shell validation but will need a compact overflow or menu
  treatment as app content grows.
- Revisit the placeholder card styling once real condition, tide, current, plot,
  webcam, transit, and citation content is present. The final layout should be
  denser and closer to the current site's useful information hierarchy.
- Local development can use Node 24+ with pinned `pnpm@10.18.3`; production
  Docker remains pinned to the active Node LTS major. Corepack is preferred when
  available; otherwise use an ephemeral pinned invocation such as
  `npx --yes pnpm@10.18.3 ...` rather than installing package managers globally.
  If local Node Current releases cause toolchain issues, reproduce on Node 24
  before treating the issue as a project bug.
- JavaScript Playwright tests use `PLAYWRIGHT_BROWSERS_PATH=0`, so browser
  binaries live under `frontend/node_modules`. After deleting or recreating
  `frontend/node_modules`, run `pnpm --dir frontend test:e2e:install` before
  `pnpm --dir frontend test:e2e`.

## Feature Parity Plan

The React app should reach feature parity with the current template-based
experience before replacing the existing site or introducing major product
changes. Feature parity starts in Milestone 2 after the app shell and build
pipeline exist.

Current frontend features to preserve:

- location-specific condition summary
- tide timing
- current estimate and link to current details
- Windy forecast embed
- NYC webcam embed
- temperature trend plots
- NYC transit status cards
- source citations
- location navigation
- current prediction page
- all-locations/status views
- widget/embed views if they remain useful

## Responsive Design Requirements

The new frontend must work well on mobile and desktop from the beginning. Mobile
should be treated as a primary use case, not as a later adaptation of a desktop
layout.

Responsive requirements:

- design mobile-first and progressively enhance for wider screens
- avoid horizontal scrolling for normal page content
- keep condition summaries readable at phone widths
- use touch-friendly tap targets for navigation and controls
- constrain plots, embeds, cards, and tables to the viewport
- preserve a coherent information hierarchy across mobile and desktop
- let desktop layouts use available width without spreading related content too
  far apart
- ensure Windy and YouTube embeds use responsive wrappers
- ensure transit cards stack cleanly on narrow screens
- test representative mobile and desktop viewport sizes with Playwright

Implementation should prefer shared layout tokens and reusable responsive
containers over one-off width rules.

Initial viewport test matrix:

```text
mobile: 390x844
desktop: 1440x900
```

Additional widths can be added when a layout risk is identified.

## Launch And Adoption Path

The existing template-rendered site should eventually promote the new frontend
without forcing an immediate migration.

Add a lightweight top-of-page banner on the existing site once the new app is
useful:

```text
Try our new app
```

Expected behavior:

- appears on existing Jinja-rendered pages
- links to `/app` or the app host if one is configured
- includes a clear dismiss action
- stores dismissal permanently in `localStorage` or a long-lived cookie
- is accessible by keyboard and screen readers
- does not block the existing page content
- can be removed after the new app becomes the primary experience

## Installable App Support

The new frontend should work well when saved to a phone home screen. This means
supporting the installable/app-like parts of Progressive Web App behavior without
committing to offline behavior.

Initial requirements:

- web app manifest
- app name: `shallweswim`
- short name: `shallweswim`
- iOS home screen label should display as `shallweswim`: one word, all
  lowercase
- `start_url`: `/app`
- `scope`: `/app/`
- `display`: `standalone`
- `theme_color`: `#000099`, matching the current site theme color
- `background_color`: `#fcffff`, matching the current site background
- required icon sizes
- Apple touch icon support
- mobile viewport and safe-area polish

Initial icon approach:

- reuse the existing static icons where dimensions are suitable
- add app-specific icons under the frontend public assets if needed
- avoid generated placeholder icons in production

Do not prioritize service workers or offline behavior in the initial frontend
build. Swim conditions are time-sensitive, and stale data should not be served as
if it were current. If service workers are considered later, they should be
limited to app-shell/static-asset caching with explicit stale/unavailable data
messaging.

### Persisted App State

The installable app should remember the user's last viewed location so reopening
from a phone home screen returns to the relevant swim location.

Persisted app state is client-only preference state. Store it in
`localStorage`, not cookies, because the backend does not need these values and
they should not be sent with requests. Treat the store as best-effort: it can be
missing, unavailable, invalid, or cleared by the user at any time. Do not store
secrets, live condition payloads, precise user location, or third-party API
responses.

Current behavior:

- Explicit routes remain authoritative. Visiting `/app/nyc` should render NYC
  regardless of previously persisted state.
- When a user views a concrete location route, persist that location code in
  `localStorage`.
- Opening `/app` should use the persisted location code if it is valid in
  `/api/app/bootstrap`; otherwise fall back to the backend-provided default
  location.
- Prefer preserving the `/app` URL rather than redirecting solely to restore
  state, unless routing or analytics needs make canonicalization useful later.
- If `localStorage` is unavailable, private, or cleared, silently fall back to
  the default location.
- Record a local `organicVisitCount` for the future install-app prompt. This is
  not analytics: it stays on the device, is not sent to the backend, and exists
  only so the UI can avoid showing a prompt before the user has demonstrated
  repeat interest.

Potential future persisted preferences:

- selected temperature plot mode
- dismissed existing-site adoption banner
- dismissed or snoozed install-app prompt
- preferred units if the app ever supports more than Fahrenheit

## External Embeds And Third-Party Data

Several current features are intentionally not served by our backend. The new
frontend should preserve those integrations as isolated components with clear
fallback states.

### Windy Forecast

Use a React component that renders the Windy iframe URL for a location. This
keeps the integration close to the current implementation and avoids bringing a
Windy API key or map SDK into the client.

The component should accept location configuration as props:

```text
lat
lon
preferred layer/options
```

Using the iframe also keeps Windy as an external visual embed rather than making
Windy forecast data part of the Shall We Swim data model.

### YouTube Webcam

Use a dedicated React component for the NYC YouTube live stream. Prefer a direct
iframe plus the official YouTube iframe API only for behavior we actually need,
such as muting and autoplay attempts.

Do not depend on a third-party React wrapper unless it removes meaningful code.
The YouTube iframe API is already the stable integration boundary, and a local
component keeps player parameters explicit.

### NYC Transit Status

Keep the initial frontend implementation close to the current direct
GoodService request pattern:

```text
https://goodservice.io/api/routes/{train}
```

Wrap it in a small typed client module and TanStack Query hook so loading,
refresh, retry, and unavailable states are consistent with the rest of the app.

If direct browser access becomes unreliable because of CORS, availability, or
rate limiting, consider adding a backend proxy endpoint later. That would be an
operational reliability change, not a frontend requirement for feature parity.

## Runtime Data Behavior

Initial TanStack Query behavior should match the operational intent of the
current frontend rather than use library defaults blindly.

Conditions:

- fetch immediately on page load
- refetch every 60 seconds
- do not issue overlapping requests for the same location
- on first-load failure, show an unavailable state for each affected display area
- on refresh failure after successful load, keep last loaded data visible and show
  a quiet stale/refresh warning
- treat HTTP 503 as data temporarily unavailable

Plots:

- do not start loading deferred plot images until the first conditions request has
  completed, whether it succeeds or fails
- use the existing image endpoints for phase 1
- probe image loads before showing them
- retry transient image load failures with delays equivalent to the current
  frontend: 1s, 3s, and 7s
- after retries are exhausted, show a per-plot unavailable message without
  breaking the rest of the page

Planner currents:

- fetch immediately when shifted `at=...` state needs current detail
- use `planner=open` to reveal the compact time scrubber and `detail=open` to
  reveal the current/tide plot plus supporting local-current guidance
- use `at` as the preferred query parameter for shift-aware backend endpoints;
  `shift` remains available for simple relative controls and legacy callers
- preserve the current behavior of keeping prior loaded data visible on refresh
  failure when refresh behavior is added

Transit:

- fetch independently from swim conditions
- keep prior transit data visible if a later refresh fails
- show unavailable state on first-load failure
- initial refetch interval: 60 seconds

External embeds:

- render Windy and YouTube independently from API data loading
- show layout-stable fallback text if an embed cannot be loaded

## Frontend Error Instrumentation

The React app should eventually surface client-side failures to developers. This
is not a replacement for user-visible unavailable states; it is for detecting
bugs and integration failures after deployment.

Initial goals:

- catch React render failures with an app-level error boundary
- catch uncaught `window.error` and `unhandledrejection` events
- capture API request failures at the client/query layer with endpoint, status,
  and coarse error category
- capture external integration failures where practical, especially YouTube,
  Windy, plot image loading, and GoodService transit failures
- include enough context to debug: app version/build identifier, route,
  location code, viewport class, user agent, timestamp, and whether the app is
  running in standalone display mode
- avoid collecting precise location, personally identifying information, full
  URLs with arbitrary query strings, or raw third-party response bodies

Possible implementation paths:

- Lightweight first-party endpoint: add a backend endpoint such as
  `/api/app/client-error` that accepts sanitized client error events and writes
  them to server logs for Cloud Run/Cloud Logging.
- Third-party monitoring service: consider Sentry or similar if first-party logs
  are not enough, but do not introduce a monitoring vendor without a separate
  decision.
- Local-only diagnostics: keep a small in-memory or `sessionStorage` ring buffer
  of recent frontend errors to support manual bug reports, similar in spirit to
  the existing Jinja debug tooling.

Initial recommendation:

- Do not block Milestone 2.5 on full observability.
- Before the adoption banner sends real users to `/app`, add at least an error
  boundary plus sanitized first-party client-error logging or an explicit
  decision to defer it.
- Frontend error logging must be best-effort and must never break the app if the
  logging endpoint fails.

## Plot Strategy

For feature parity, keep using the existing FastAPI plot image endpoints.

```text
/api/{location}/plots/live_temps
/api/{location}/plots/historic_temps?period=2mo
/api/{location}/plots/historic_temps?period=12mo
/api/{location}/plots/current_tide?at=2026-05-18T15:30:00
```

Reasons to keep plots as backend images for the first frontend milestone:

- Existing plots already encode domain-specific processing and presentation.
- The backend already owns plot generation, caching, and data availability
  behavior.
- Image endpoints are easy to embed in React with the same deferred loading and
  retry behavior used today.
- This avoids adding raw chart-data endpoints before the new frontend has reached
  parity.
- It reduces the number of moving parts during the initial migration.

Client-rendered charts may still be desirable later. If we move in that
direction, the backend should expose processed chart series through typed JSON
endpoints. The frontend should not reimplement tide/current/temperature business
logic that already belongs to the backend.

Potential later charting goals:

- better mobile responsiveness
- interactive tooltips
- accessible chart summaries
- consistent styling with the rest of the app
- user-selectable time windows without regenerating images

This should be treated as a second phase:

```text
Phase 1: backend-generated plot images for parity
Phase 2: processed chart-data APIs plus client-rendered charts where useful
```

## Open Questions

- Which plot views are worth rebuilding as interactive client-rendered charts
  after image-based parity exists?
- How much of the existing browser tests should move into frontend Playwright
  tests versus remain in Python?
- Should the existing-site new-app banner store dismissal in `localStorage` or a
  cookie?
- Should Bun replace pnpm/Vite for any part of the toolchain after a focused
  compatibility spike?
- Should the new app eventually live at `/`, `/app`, or a dedicated app hostname?

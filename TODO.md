# TODO

## Product And UX

### Location Page Experience

- Review whether the remaining legacy Jinja pages should stay, move behind a
  narrower compatibility surface, or be removed. Keep `/legacy/...` and legacy
  embed routes as long as they serve a real compatibility need, but make that an
  explicit decision rather than permanent drift.
- Add an explicit location-code alias mechanism if friendly alternate URLs are
  useful. Model aliases in typed `LocationConfig`, keep bootstrap/location order
  canonical, and redirect alias routes to the canonical location URL rather than
  rendering duplicate canonical pages.

### External Embed Consumers

- Audit consumers that still use the historical legacy embed URLs (`/embed` and
  `/{location}/embed`). Either migrate them to a new supported React/embed
  surface or keep the legacy embed route intentionally.

### Water Movement Experience

- Rework the meter layout into a clearer tide/current instrument. Candidate
  improvements: combine the tide bar with last/next/following tide context,
  include relevant event times on or near the meter, make endpoint labels less
  cluttered on mobile, clarify that tide endpoints are low/high range labels
  rather than chronological order, and preserve the current compact data-dense
  mono treatment.
- Extend current `range` context to slack rows when useful. The current API only
  exposes slack-to-peak range for non-slack rows with complete segment context;
  slack phases can later be associated with the upcoming flood/ebb segment so
  the current meter can show where it is about to build.
- Refine the planner time scrubber. The current React planner uses a compact
  in-card slider with URL-backed `at` state; the final interaction should add
  clearer time ticks, stronger mobile styling, debounced updates if needed, and
  smart presets such as next slack, peak ebb, and peak flood once the backend
  exposes those timestamps.
- Move NYC local water-movement guidance into typed location config. The React
  copy pass now uses Grimaldo's-centered swimmer language, flood/ebb current
  guidance, CIBBOWS context, and wider drift bars with trend text in the
  instrument header. The remaining architecture work is to stop hardcoding NYC
  landmarks and caveats in the frontend: add typed backend config for the
  reference point, flood/ebb landmark directions, route-planning advice, and
  local caveats, NYC current source override, map credits, and harbor-chart
  source links; expose those through bootstrap or another static app-config
  contract and render them generically.
- Preserve NYC as a derived current location if current prediction sources
  change. NYC currently estimates local swim-area current by averaging nearby
  NOAA current prediction station velocity curves. A future source upgrade
  should keep that explicit virtual-location model: build or fetch a prediction
  curve per source station, normalize them onto a common timestamped velocity
  series, blend the station curves into one location curve, and only then derive
  phase, strength, trend, slack/peak range, and API/UI state.
- Evaluate whether the tide/current detail chart should remain a backend
  Matplotlib SVG or move to a frontend-rendered chart. A frontend charting pass
  could improve responsive layout, hover/inspection, and label collision
  handling, but it is a product/architecture decision because it would move
  presentation logic and possibly chart data contracts into the React app.

### Conditions And Temperature Experience

- Explore richer swimmer-focused condition summaries. The current app should
  stay deterministic, but a future version could generate a concise
  natural-language summary from water temperature, tide/current state, weather,
  and possibly webcam imagery. Treat this as a product/reliability project, not
  a small copy tweak.
- Add backend-derived temperature trend context for the app summary, such as
  `up 1.2°F in 24h`, `down 3.0°F this week`, or `near seasonal range`. Do this
  from structured historical/live temperature data rather than inferring it from
  plot images or adding filler copy in the frontend.
- Add a lightweight water-temperature projection cone. Start with a deterministic
  forecast rather than a heavy model: build a seasonal climatology from prior
  years, calculate historical spread/quantiles by day-of-year, compare the
  current year's recent actuals to the seasonal baseline, decay that anomaly
  forward over the horizon, and widen the uncertainty cone with horizon and
  observed volatility.

### Installable App And Durable Web

- Decide whether to add an install-app butter bar for the launched React app.
  The local preference store already records `installPrompt.organicVisitCount`
  for this purpose; use it only as on-device UI state. The prompt should appear
  only after enough organic use, should not appear in standalone/PWA display
  mode, and should support permanent dismissal via the same local preference
  store.
- Consider richer durable app HTML only after there is a clear consumer need.
  The current FastAPI-rendered app shell already provides location-aware
  title/meta/canonical tags, JSON API discovery links, compact no-JavaScript
  fallback content, and conservative JSON-LD. Deferred work should stay narrow,
  such as adding structured condition summaries once freshness, safety language,
  and duplicate rendering ownership are explicitly designed.

## API And Data Model

### Condition And Current API Shape

- Consider aligning point-in-time condition data under symmetric API structures,
  for example `tides.state` and `currents.state`. The existing `current` response
  is effectively current-state data already, but its top-level naming is less
  consistent with the planned tide state.
- Reassess whether `/api/{location}/currents` is still needed outside legacy
  compatibility. The React Water Movement card now uses
  `/api/{location}/conditions?at=...` as the single source for shifted tide and
  current state; if legacy pages are removed or migrated, mark the currents
  endpoint and `NavigationInfo` response model as removal candidates.
- Remove legacy currents navigation fields after the React planner fully owns
  time controls. `NavigationInfo.next_hour`, `prev_hour`, and `current_api_url`
  are deprecated compatibility/convenience fields; the planner should derive
  previous/next times and currents API URLs from local app state. Keep `shift`,
  `at`, and `plot_url` unless a later frontend pass removes the backend-rendered
  tide/current plot dependency too.
- Clean up API timestamp field types. Several existing API Pydantic models use
  `str` plus manual `.isoformat()` serialization for timestamps. Prefer
  `datetime.datetime` fields so Pydantic/OpenAPI can expose proper `date-time`
  schemas, then migrate route serialization and generated frontend types in one
  coordinated pass.
- Review whether the newer `units` fields are consistently useful in public API
  responses. Keep them where they clarify mixed-unit locations; remove or
  normalize them where they add noise.

### Tide Curve Source Upgrade

Consider adding a NOAA-derived tide-height prediction curve instead of deriving
the current tide height from high/low tide events.

Current state:

- The tide feed requests NOAA CO-OPS `product=predictions` with `interval=hilo`,
  so it stores only high/low tide events.
- The tide/current plot creates an approximate tide curve by interpolating those
  high/low event heights.
- The conditions API currently exposes previous and next tide events, not a
  current predicted tide-height state.

Potential upgrade:

- Keep the existing `hilo` feed for previous/next high/low tide events.
- Add a separate tide-height prediction feed using NOAA CO-OPS
  `product=predictions` with a fixed interval such as `interval=6` or
  `interval=15`.
- Use that denser NOAA prediction series for current predicted tide height,
  rising/falling state, normalized tide height, and any tide bar/instrument UI.
- Fall back to local interpolation from high/low events if a station only
  supports high/low predictions.

Notes:

- NOAA harmonic tide prediction stations can generally provide interval
  predictions; subordinate prediction stations may only support high/low
  predictions.
- This is a data-source/modeling change and should be handled separately from
  near-term frontend polish.

### Bootstrap And Static Metadata

- Review user-facing station and location names. Some source station names, such
  as "The Battery", may be accurate data-source labels but confusing as primary
  swim-condition copy.
- Make Windy forecast configuration location-aware instead of hardcoding one
  frontend embed contract. Coastal locations can keep wave/swell-oriented
  defaults, while inland, river, or future international locations should be able
  to choose more relevant overlay/product/unit settings through typed static
  config.

## Data Sources And Quality

### Validation And Conditioning

- Add stronger frame validation for feed data: missing values, valid value
  ranges, monotonic timestamps, timezone-naive internal indexes, and
  source-specific invariants where appropriate.
- Extract historical temperature conditioning from `plot.py` into a named core
  layer if it becomes more than plot-local presentation logic. The likely
  architecture is `feed fetch -> conditioning/derivation -> serving/plotting`:
  feeds preserve normalized source measurements and configured known-bad
  outliers; conditioning owns named derived products such as a visualization
  plot series, forecast training series, or seasonal baseline; plotting and
  serving only render or serialize those products.
- Improve historical temperature plot artifact auditability before tuning gets
  serious. The inspection script currently reports suppressed points by
  stage/year; add segment-level output with start/end, duration, min/max source
  temperature, residual range, and neighboring valid context so a contiguous bad
  tail is distinguishable from scattered isolated artifacts.
- Sort out realtime versus historical modes explicitly. Live condition serving,
  historical plot inputs, seasonal baselines, and future projection products
  should not quietly share assumptions just because they all start from
  temperature frames.

### Source Client Work

- Evaluate replacing the `ndbc-api` dependency with a first-party async NDBC
  client. The current client wrapper still carries thread-bound behavior and
  dependency-update friction; a local client may be simpler to own.
- Investigate San Diego station availability before changing any station
  configuration. Do not change production stations without explicit approval.
- Investigate whether SDF should add USGS discharge-rate context in addition to
  the existing observed river-current feed. Parameter `72294` may support a more
  useful river-current presentation, but should be evaluated before changing the
  configured source.
- Investigate an alternative Chicago temperature source for year-round coverage,
  such as the daily NOAA/NWS marine observation text product, before changing
  configured data sources.

## Operations And Infrastructure

### Monitoring And Alerting

- Set up Cloud Monitoring coverage for `/api/status`. The status endpoint already
  exposes per-feed `is_healthy`, `is_expired`, and `age_seconds`; production
  monitoring should retain that feed-level visibility rather than relying only on
  the coarse service health check.
- Add alerting policies for stale or unhealthy critical feeds. Start with NYC
  feeds as the highest-priority alerts, then use lower-priority policies for
  other locations once the signal/noise balance is understood.
- Build a feed-health visibility dashboard that tracks `age_seconds`,
  `is_healthy`, and `is_expired` over time across all feeds. Use the observed
  outage patterns to guide future health-policy decisions rather than guessing
  which feeds should be mandatory for each location.
- Revisit stricter location health rules after monitoring data exists. The
  current app treats a location as having data if any feed works; future policy
  may need per-location critical feeds, but that should be driven by production
  data and user impact.
- Evaluate dead-link monitoring for configured source, swim-location, webcam,
  and citation URLs. Keep it separate from data-feed health so broken reference
  links do not page like production data outages.

### Runtime And Deployment

- Evaluate Cloud Run second generation for performance, startup behavior, and
  operational simplicity. Compare with the current generation under a realistic
  startup and plotting workload before changing production.
- Profile production-like CPU and memory usage, including process-pool sizing and
  aiohttp connection behavior. Add debug visibility only if it is useful for
  operational decisions; avoid building a broad `/api/debug` surface by default.
- Evaluate infrastructure-as-code for Cloud Run and related GCP resources. Keep
  credential material and project-specific secrets out of the repo, but consider
  whether service configuration, monitoring resources, and deployment topology
  should be reviewable as code.
- Evaluate multi-region deployment for international location support. If the app
  adds substantial non-US location coverage, consider whether an additional Cloud
  Run region improves latency or resilience enough to justify operational
  complexity.
- Reduce verbose logs that do not help diagnose data outages, retries, startup,
  or user-facing failures. Keep enough context for feed health and station
  debugging without making normal logs noisy.

## Codebase Maintenance

### Architecture And Ownership

- Align internal domain types and external API response models. Keep Pandas
  frames at feed/derivation boundaries, but avoid leaking inconsistent ad hoc
  shapes into route handlers and generated frontend types.
- Clean up processing ownership between clients, feeds, queries, and plotting.
  Clients should stay source-specific and side-effect-light, feeds should own
  fetch/cache/validation, core queries should own derived condition semantics,
  and plotting should not become the only place where data conditioning exists.
- Clean up feed time-window constants. Several code paths still encode similar
  multi-day fetch windows directly; replace those with named policy constants or
  source-specific configuration where it clarifies behavior.
- Revisit the overloaded "interval" concept in feeds and health checks. Separate
  fetch cadence, freshness threshold, source request window, and retry behavior
  if the current naming keeps causing confusion.
- Remove defensive `hasattr` patterns and similar dynamic checks where typed
  interfaces or explicit protocols would make the boundary clearer.

### Test And Tooling Maintenance

- Split tide/current query and manager tests out of `tests/test_data.py` once the
  tide-state API work lands. That file is becoming a broad data-layer catch-all;
  future coverage would be easier to maintain in focused modules such as
  `tests/test_tide_queries.py`, `tests/test_current_queries.py`, and/or
  manager-specific tests.
- Expand joint frontend/backend stack coverage as React planner behavior grows.
  The first optional Python Playwright test now runs the built React app against
  real FastAPI routes with mocked data managers and verifies that URL `at` state,
  rendered tide/current bars, detail plot URL, and backend calls all agree. Add
  more cases here when new cross-stack behavior lands, such as smart presets,
  non-NYC capability behavior, predicted water temperature, and user-facing API
  error states.
- Triage existing `TODO` and `XXX` comments in the codebase. Convert durable
  work into this file, fix small stale notes inline, and remove comments that no
  longer describe real work.
- Do a holistic documentation-vs-implementation audit across `README.md`,
  `ARCHITECTURE.md`, `AGENTS.md`, `frontend/DESIGN.md`, and route/API tests.
  As part of that work, add a repo-local Codex skill for repeatable docs drift
  reviews so future audits check routes, commands, API ownership, frontend
  serving behavior, tests, and deployment assumptions consistently.
- Revisit URL formatting in static config files. `ruff format` wraps some long
  configured URLs in `config/locations.py`, which makes them harder to inspect
  and copy/paste. Decide whether to use narrow `# fmt: off/on` blocks around
  URL-heavy location presentation config or adjust formatter settings if this
  becomes a broader repo-wide readability issue.

## Location Backlog

- Keep a low-priority candidate backlog for additional swim locations. New
  locations still need reliable data sources, station approval, frontend
  capability modeling, and tests before production configuration changes.
- Domestic candidates: Lake Tahoe, Jacksonville FL, St. Pete/Clearwater, and
  other established open-water swim groups with reliable public data sources.
- Canadian candidates: Kingston, Vancouver area, and Oakville/Toronto.
- International candidates: Dover, UK and Sandycove, Ireland.
- Add support needed for non-US locations before committing to international
  production coverage: non-NOAA/non-USGS data clients, Celsius-first display
  defaults where appropriate, timezone handling, source citations, and location
  capability flags that do not assume US data products.

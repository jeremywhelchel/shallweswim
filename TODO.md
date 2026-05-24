# TODO

## Frontend Location Parity

### 1. Webcam Follow-Ups

- SDF EarthCam parity is in place and verified on `shallweswim.today`. Keep
  provider scripts out of the React runtime unless there is no contained
  alternative; SDF uses the direct EarthCam player iframe URL because the legacy
  script depends on `document.write` and fails when dynamically mounted by
  React.
- After the first feature-parity pass, revisit SDF EarthCam specifically. Keep
  the named EarthCam provider, but verify whether a more official iframe URL or
  provider-supported embed contract exists.
- Localhost is not a reliable SDF EarthCam playback test because EarthCam
  whitelists allowed referrers. Use production-domain verification for future
  EarthCam changes.

### 2. All Locations Parity

- Do a systematic parity smoke pass across all React location pages before any
  launch work:
  - `nyc`: temperature, water movement, planner, YouTube webcam, transit,
    Windy, plots, sources.
  - `chi`: temperature, iframe webcam, Windy, plots, sources, and no water
    movement/planner.
  - `sdf`: temperature, observed water movement, EarthCam iframe webcam, Windy,
    plots, sources, and no planner unless river projections are explicitly
    added.
  - `san`, `sfo`, `bos`, `sea`: temperature, tide-only water movement, planner,
    Windy, plots, sources.
  - `aus`: temperature, Windy, plots, sources, and no water movement/planner.

### 3. Coverage

- Add backend and frontend coverage for the parity work: bootstrap integration
  config by location, NYC YouTube webcam, CHI iframe webcam, SDF EarthCam iframe
  behavior, feature-driven absence of unsupported sections, and all-location
  status behavior.

## Launch Parking Lot

- Do not start launch work until explicitly requested. "Launch" means moving
  the React app out of `/app` and making it the default root-location
  experience.
- Add an explicit location-code alias mechanism if we want friendly alternate
  URLs. Model aliases in typed `LocationConfig`, keep bootstrap/location order
  canonical, and redirect alias routes to the canonical location URL rather than
  rendering duplicate canonical pages.
- When launch is requested, move Jinja pages under a temporary `/legacy`
  namespace, change canonical routes from `/app/{loc}` to `/{loc}`, make `/`
  load the default/saved location dashboard, make `/locations` the React
  all-locations/status page, and update Vite base, React Router basename,
  FastAPI route ordering, manifest `start_url`/`scope`, canonical/meta tags,
  persisted-location behavior, and tests. Reuse the existing manifest/config
  generation path where possible; the launch should change emitted manifest
  values rather than add a parallel manifest system. Keep `/api/...`,
  `/static/...`, `/legacy/...`, and frontend asset routes explicitly reserved so
  root-mounted React routing does not mask them. Remove the `/app` route after
  root launch rather than keeping it as a long-lived alias.

## Tech Debt

- Revisit URL formatting in static config files. `ruff format` wraps some long
  configured URLs in `config/locations.py`, which makes them harder to inspect
  and copy/paste. Decide whether to use narrow `# fmt: off/on` blocks around
  URL-heavy location presentation config or adjust formatter settings if this
  becomes a broader repo-wide readability issue.
- Split tide/current query and manager tests out of `tests/test_data.py` once the
  tide-state API work lands. That file is becoming a broad data-layer catch-all;
  future coverage would be easier to maintain in focused modules such as
  `tests/test_tide_queries.py`, `tests/test_current_queries.py`, and/or
  manager-specific tests.
- Consider aligning point-in-time condition data under symmetric API structures,
  for example `tides.state` and `currents.state`. The existing `current` response
  is effectively current-state data already, but its top-level naming is less
  consistent with the planned tide state. Because the app is deployed as one
  consolidated frontend/backend, this can be a coordinated client migration
  rather than a long-lived compatibility burden.
- Clean up API timestamp field types. Several existing API Pydantic models use
  `str` plus manual `.isoformat()` serialization for timestamps. Prefer
  `datetime.datetime` fields so Pydantic/OpenAPI can expose proper `date-time`
  schemas, then migrate route serialization and generated frontend types in one
  coordinated pass.
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
- Explore richer swimmer-focused condition summaries. The current app should
  stay deterministic, but a future version could generate a concise
  natural-language summary from water temperature, tide/current state, weather,
  and possibly webcam imagery. Treat this as a product/reliability project, not
  a small copy tweak.
- Add backend-derived temperature trend context for the app summary, such as
  `up 1.2°F in 24h`, `down 3.0°F this week`, or `near seasonal range`. Do this
  from structured historical/live temperature data rather than inferring it from
  plot images or adding filler copy in the frontend.
- Extract historical temperature conditioning from `plot.py` into a named core
  layer if it becomes more than plot-local presentation logic. The likely
  architecture is `feed fetch -> conditioning/derivation -> serving/plotting`:
  feeds preserve normalized source measurements and configured known-bad
  outliers; conditioning owns named derived products such as a visualization
  plot series, forecast training series, or seasonal baseline; plotting and
  serving only render or serialize those products. Do this before reusing the
  current plot-only visual artifact suppression in API responses, summaries,
  frontend-rendered charts, or other product behavior. When this graduates,
  keep the current plot artifact masks separate from modeling/forecast
  conditioning so unusual but real years still inform seasonal spread and tail
  risk.
- Improve historical temperature plot artifact auditability before tuning gets
  serious. The inspection script currently reports suppressed points by
  stage/year; add segment-level output with start/end, duration, min/max source
  temperature, residual range, and neighboring valid context so a contiguous bad
  tail is distinguishable from scattered isolated artifacts. Consider requiring
  a minimum number of comparison years before applying cross-year artifact
  suppression, and add per-location/source threshold overrides only once
  repeated tuning pressure shows the global defaults are insufficient.
- Add a lightweight water-temperature projection cone. Start with a deterministic
  forecast rather than a heavy model: build a seasonal climatology from prior
  years, calculate historical spread/quantiles by day-of-year, compare the
  current year's recent actuals to the seasonal baseline, decay that anomaly
  forward over the horizon, and widen the uncertainty cone with horizon and
  observed volatility. This should use a modeling-specific conditioning product,
  not the private plot artifact suppression frame, so legitimate extreme years
  still inform the tails while known-bad source measurements remain excluded.
  Likely supporting work: named historical temperature conditioning module,
  per-location/source QC thresholds, explicit segment-level audit output,
  structured API data for the frontend chart, and tests for sparse-year,
  current-year, and unusually hot/cold-year behavior.
- Preserve NYC as a derived current location if we upgrade current prediction
  sources. NYC currently estimates local swim-area current by averaging nearby
  NOAA current prediction station velocity curves. A future NOAA curve/covariate
  upgrade should keep that explicit virtual-location model: build or fetch a
  prediction curve per source station, normalize them onto a common timestamped
  velocity series, blend the station curves into one location curve, and only
  then derive phase, strength, trend, slack/peak range, and API/UI state. Avoid
  blending harmonic constituents, offsets, or other model inputs up front unless
  we have a defensible physical model for doing so.
- Remove legacy currents navigation fields after the React planner owns time
  controls. `NavigationInfo.next_hour`, `prev_hour`, and `current_api_url` are
  deprecated compatibility/convenience fields; the planner should derive
  previous/next times and currents API URLs from local app state. Keep
  `shift`, `at`, and `plot_url` unless a later frontend pass removes the
  backend-rendered tide/current plot dependency too.
- Reassess whether `/api/{location}/currents` is still needed outside the legacy
  HTML currents page. The React Water Movement card now uses
  `/api/{location}/conditions?at=...` as the single source for shifted tide and
  current state; if the legacy page is removed or migrated, mark the currents
  endpoint and `NavigationInfo` response model as removal candidates.
- Treat durable app HTML as its own future migration project, not an incidental
  test-only task. Before replacing the Jinja location pages or removing the
  `/app` prefix, define and implement a progressively enhanced React page that is
  useful when fetched without JavaScript: location-aware HTML, canonical/meta
  tags, links to structured JSON APIs, and machine-readable data such as JSON-LD
  or embedded bootstrap JSON. Add curl/no-JS acceptance tests only when that
  feature work is being implemented so the normal suite never carries expected
  failures.
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
  local caveats, expose it through bootstrap, and render it generically. Keep
  collecting swimmer feedback after launch to tune wording such as "east toward
  Manhattan Beach" versus a more Brighton-specific eastbound landmark.
- Expand joint frontend/backend stack coverage as React planner behavior grows.
  The first optional Python Playwright test now runs the built React app against
  real FastAPI routes with mocked data managers and verifies that URL `at` state,
  rendered tide/current bars, detail plot URL, and backend calls all agree. Add
  more cases here when new cross-stack behavior lands, such as smart presets,
  non-NYC capability behavior, predicted water temperature, and user-facing API
  error states.
- Add the install-app butter bar once the React app is ready to promote as an
  installable experience. The local preference store already records
  `installPrompt.organicVisitCount` for this purpose; use it only as on-device UI
  state, not analytics. The prompt should appear only after enough organic use,
  should not appear in standalone/PWA display mode, and should support permanent
  dismissal via the same local preference store.

## Future Tide Curve Source Upgrade

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
  the current React migration milestones.

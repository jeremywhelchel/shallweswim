# TODO

## Tech Debt

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
- Deepen the NYC local water-movement detail soon, with an explicit swimmer
  guidance review. The current React detail panel has the right raw ingredients
  — Grimaldo's Chair direction guidance, local current maps, tide/current timing
  notes, and historic harbor charts — but needs a focused product/copy pass to
  decide whether the wording is actually practical for swimmers making a plan.
  Show the current guidance to local swimmers, collect feedback on what is
  useful/confusing, and spend time on the exact language. Candidate improvements:
  show "flood carries east / ebb carries west" closer to the current instrument,
  tighten map/chart captions, make slack/flip guidance more explicit, and better
  connect the local map and harbor chart to the selected planner time.
- Expand joint frontend/backend stack coverage as React planner behavior grows.
  The first optional Python Playwright test now runs the built React app against
  real FastAPI routes with mocked data managers and verifies that URL `at` state,
  rendered tide/current bars, detail plot URL, and backend calls all agree. Add
  more cases here when new cross-stack behavior lands, such as smart presets,
  non-NYC fallback behavior, predicted water temperature, and user-facing API
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

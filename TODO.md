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

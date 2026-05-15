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

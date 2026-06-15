# Scripts

One-off operational and data-investigation scripts live here. Run them from the
repository root with `uv run python -m shallweswim.scripts.<module>`.

## Irish Lights Temperature Fetches

`debug_irish_lights_fetch.py` exercises the Irish Lights MetOcean client used by
Cork/Sandycove temperature feeds.

```bash
# Recent Cork buoy observations.
uv run python -m shallweswim.scripts.debug_irish_lights_fetch --location cor

# Year-by-year historical fetch checks.
uv run python -m shallweswim.scripts.debug_irish_lights_fetch \
  --location cor \
  --start-year 2024 \
  --end-year 2026
```

The script reports row counts, local timestamp bounds, Fahrenheit min/max
values, failures, and elapsed time. It uses the same configured MMSI and
source-specific plausible Celsius bounds as the runtime feeds.

## Dover Harmonic Tide Fitting

`derive_harmonic_tide_model.py` is an offline investigation/build script for
deriving compact local harmonic tide coefficients from observed Dover tide-gauge
history. It is not part of app startup.

Current archive workflow:

```bash
# Fetch/update a filtered Dover cache. Existing cached days are skipped unless
# --force-fetch is provided. Cached days outside the requested window are
# preserved, so this is safe to run incrementally over different windows.
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --fetch \
  --archive-start 2025-12-01 \
  --archive-end 2026-06-01 \
  --cache /tmp/dover_ea_local.csv

# Fit a compact harmonic model from the filtered cache.
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --fit \
  --cache /tmp/dover_ea_local.csv \
  --output /tmp/dov_harmonics.json

# Evaluate residuals. Without --eval-start/--eval-end this evaluates the clean
# cache rows, which may be in-sample if the model was fit from the same cache.
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --eval \
  --cache /tmp/dover_ea_local.csv \
  --model /tmp/dov_harmonics.json

# Evaluate a holdout window explicitly.
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --eval \
  --eval-start 2026-05-01 \
  --eval-end 2026-06-01 \
  --cache /tmp/dover_ea_local.csv \
  --model /tmp/dov_harmonics.json

# Run a rolling backtest. Each split fits on the preceding training window,
# evaluates the following test window, and reports high/low timing errors.
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --backtest \
  --backtest-train-days 365 \
  --backtest-test-days 30 \
  --backtest-step-days 30 \
  --cache /tmp/dover_ea_local.csv

# Full workflow in one command.
uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
  --fetch --fit --eval \
  --archive-start 2025-12-01 \
  --archive-end 2026-06-01 \
  --cache /tmp/dover_ea_local.csv \
  --output /tmp/dov_harmonics.json
```

The cache is the canonical local artifact for this workflow. It stores only
filtered Dover rows with `dateTime`, `measure`, and `value`, not the large
all-station Environment Agency archive files. Fetching is idempotent for a
given cache: the script merges newly fetched days into the existing cache and
does not drop cached rows outside the requested fetch window. Each fetched day
is persisted with an atomic cache rewrite before the script moves to the next
day, so interrupted long archive pulls can be resumed.

Sizing notes from June 2026 investigation:

- The Environment Agency daily archive files are all-station CSVs, roughly
  58 MB per day for the sampled dates.
- The script streams each daily archive response and keeps only Dover rows, so
  retained local data is tiny: 42 days produced 4,014 Dover rows and a 109 KiB
  filtered cache.
- A 6-month window from 2025-12-01 through 2026-05-31 transferred 8.32 GB for
  140 uncached archive days, reused 42 already-filtered days, retained 17,439
  Dover rows, and wrote a 488 KiB filtered CSV. That run took 610 seconds.
- A full year is expected to transfer roughly 21 GB and should still retain
  only a few MB of Dover rows.

Model notes:

- Recent near-real-time Dover readings around early June 2026 showed degraded
  tidal range and should not be used blindly for fitting.
- Archive days before the degradation showed plausible Dover tide ranges.
- A 6-month clean-day fit excluded 11 low-range days, trained on 13,709 rows
  from 2025-12-01 through 2026-04-30, and held out 2,676 clean May rows. The
  simple 11-constituent model had about 0.277 m RMSE on both train and clean
  May holdout, and compared to the current NTSLF Dover table at about 8.6
  minutes median absolute high/low timing error across the sampled window.
- Residual RMSE is a secondary validation target for this app. The swimmer-facing
  question is high/low timing, so `--backtest` extracts observed extrema from
  the gauge series and reports predicted-vs-observed timing errors for highs
  and lows separately. Observed event times are limited by the source cadence
  of the gauge observations, usually 15-minute means for the Dover EA archive.
  Observed extrema are extracted within continuous observation segments so data
  gaps do not create artificial cross-gap peaks.
- Fit output includes the harmonic design matrix condition number plus fitted
  constituent amplitude and phase diagnostics. Treat unstable conditioning or
  implausible constituent diagnostics as reasons to fetch more history or reduce
  the constituent set before using a model in the app.
- The NTSLF comparison is an external spot-check against a public Dover tide
  table, not training ground truth. Treat it as a sanity check for high/low
  timing, especially because the fitted model is derived from Environment
  Agency gauge observations rather than NTSLF predictions.
- Validate fitted high/low timing against a trusted Dover tide table before
  using any generated model in location configuration.

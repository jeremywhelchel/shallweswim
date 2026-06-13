"""Fetch, fit, and evaluate local harmonic tide models.

This is offline tooling for deriving compact harmonic tide coefficients from
observed gauge history. It is intentionally separate from app startup.

Common workflows:

    # Update a filtered Dover cache from Environment Agency daily archives.
    uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
      --fetch --archive-start 2025-12-01 --archive-end 2026-06-01 \
      --cache /tmp/dover_ea_local.csv

    # Fit a model from the filtered cache.
    uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
      --fit --cache /tmp/dover_ea_local.csv --output /tmp/dov_harmonics.json

    # Evaluate residuals and compare generated events with NTSLF Dover tides.
    uv run python -m shallweswim.scripts.derive_harmonic_tide_model \
      --eval --cache /tmp/dover_ea_local.csv --model /tmp/dov_harmonics.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from shallweswim.scripts import harmonic_tides

DOVER_LOCAL_DATUM_MEASURE = "E71624-level-tidal_level-Mean-15_min-m"
DOVER_AOD_MEASURE = "E71639-level-tidal_level-Mean-15_min-mAOD"
EA_FLOOD_MONITORING_ROOT = "https://environment.data.gov.uk/flood-monitoring"
NTSLF_DOVER_TIDE_FRAGMENT_URL = (
    "https://ntslf.org/files/ntslf_php/tidepred.php?port=Dover"
)
NTSLF_DOVER_TIDE_PAGE_URL = "https://ntslf.org/tides/uk-network/tidepred?port=Dover"

DEFAULT_MODEL_CONSTITUENTS = tuple(
    harmonic_tides.DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR
)


class Urlopen(Protocol):
    """Protocol for dependency-injecting network access in tests."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Open a URL-like object."""


@dataclass(frozen=True)
class FetchStats:
    """Summary of filtered cache update work."""

    cache: Path
    fetched_days: int
    skipped_days: int
    rows_written: int
    archive_content_length_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class QualityFilter:
    """Quality thresholds for archive days used in model fitting."""

    min_daily_rows: int
    min_daily_span: float


@dataclass(frozen=True)
class FitResult:
    """Fitted model plus the observations used to fit it."""

    model: harmonic_tides.HarmonicTideModel
    observations: pd.DataFrame
    daily_summary: pd.DataFrame
    excluded_days: pd.DataFrame


@dataclass(frozen=True)
class ResidualMetrics:
    """Basic residual metrics for a fitted model."""

    count: int
    rmse_m: float
    mae_m: float
    bias_m: float
    p95_abs_m: float


@dataclass(frozen=True)
class EventTimingMetrics:
    """High/low event timing metrics for predicted vs observed extrema."""

    timing_errors_min: tuple[float, ...]
    matched_events: int
    reference_events: int
    predicted_events: int
    missed_events: int
    extra_events: int
    median_abs_min: float
    mean_abs_min: float
    p95_abs_min: float
    max_abs_min: float


@dataclass(frozen=True)
class BacktestSplitResult:
    """One rolling backtest split."""

    train_start: datetime.date
    train_end: datetime.date
    test_start: datetime.date
    test_end: datetime.date
    train_rows: int
    test_rows: int
    train_coverage_days: float
    test_coverage_days: float
    residual_metrics: ResidualMetrics
    timing_metrics: dict[str, EventTimingMetrics]


@dataclass(frozen=True)
class BacktestResult:
    """Rolling backtest result across all splits."""

    splits: tuple[BacktestSplitResult, ...]
    train_days: int
    test_days: int
    step_days: int


class _NtslfTableParser(HTMLParser):
    """Minimal HTML table parser for the NTSLF Dover tide fragment."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.cell = ""
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track tide table and cell boundaries."""
        if tag == "table":
            self.in_table = True
        if self.in_table and tag in ("td", "th"):
            self.in_cell = True
            self.cell = ""

    def handle_data(self, data: str) -> None:
        """Collect text in the current cell."""
        if self.in_cell:
            self.cell += data

    def handle_endtag(self, tag: str) -> None:
        """Finish table cells and rows."""
        if self.in_table and tag in ("td", "th"):
            self.in_cell = False
            self.row.append(" ".join(self.cell.split()))
        if self.in_table and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.row = []
        if tag == "table":
            self.in_table = False


def main() -> None:
    """Run selected offline tide model workflow steps."""
    args = _parse_args()
    quality_filter = QualityFilter(
        min_daily_rows=args.min_daily_rows,
        min_daily_span=args.min_daily_span,
    )

    fitted_model: harmonic_tides.HarmonicTideModel | None = None
    if args.fetch:
        stats = update_archive_cache(
            cache=Path(args.cache),
            measure=args.measure,
            start=_parse_date(args.archive_start),
            end=_parse_date(args.archive_end),
            force=args.force_fetch,
        )
        print_fetch_stats(stats)

    if args.fit:
        fit_result = fit_from_cache(
            cache=Path(args.cache),
            measure=args.measure,
            quality_filter=quality_filter,
            model_name=args.name,
            constituents=_parse_constituents(args.constituents),
        )
        fitted_model = fit_result.model
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(fitted_model.to_dict(), indent=2) + "\n")
        print_fit_summary(fit_result, output=output)

    if args.eval:
        model_path = Path(args.model or args.output)
        model = fitted_model or harmonic_tides.load_model(str(model_path))
        observations = load_filtered_cache(Path(args.cache), measure=args.measure)
        observations = select_observation_window(
            observations,
            start=_parse_optional_date(args.eval_start),
            end=_parse_optional_date(args.eval_end),
        )
        clean_observations, daily_summary, excluded_days = quality_filter_observations(
            observations,
            quality_filter=quality_filter,
        )
        print_quality_summary(
            clean_observations,
            daily_summary=daily_summary,
            excluded_days=excluded_days,
        )
        print_residual_metrics(
            model,
            observations=clean_observations,
            label=_eval_label(args.eval_start, args.eval_end),
        )
        if not args.skip_ntslf:
            comparison = compare_model_to_ntslf(
                model,
                limit=args.ntslf_limit,
            )
            print_ntslf_comparison(comparison)

    if args.backtest:
        result = run_rolling_backtest(
            cache=Path(args.cache),
            measure=args.measure,
            quality_filter=quality_filter,
            model_name=args.name,
            constituents=_parse_constituents(args.constituents),
            train_days=args.backtest_train_days,
            test_days=args.backtest_test_days,
            step_days=args.backtest_step_days,
            start=_parse_optional_date(args.backtest_start),
            end=_parse_optional_date(args.backtest_end),
            match_tolerance_minutes=args.event_match_tolerance_min,
        )
        print_backtest_summary(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_argument_group("workflow modes")
    modes.add_argument("--fetch", action="store_true", help="Update filtered cache.")
    modes.add_argument("--fit", action="store_true", help="Fit a model from cache.")
    modes.add_argument("--eval", action="store_true", help="Evaluate model quality.")
    modes.add_argument(
        "--backtest",
        action="store_true",
        help="Run rolling train/test timing validation from cache.",
    )

    parser.add_argument(
        "--cache",
        default="/tmp/dover_ea_local.csv",
        help="Filtered Dover-row cache CSV path.",
    )
    parser.add_argument(
        "--measure",
        default=DOVER_LOCAL_DATUM_MEASURE,
        choices=[DOVER_LOCAL_DATUM_MEASURE, DOVER_AOD_MEASURE],
        help="Environment Agency Dover tide gauge measure.",
    )
    parser.add_argument(
        "--archive-start",
        help="First EA archive date to fetch, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--archive-end",
        help="Exclusive EA archive end date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Refetch days even if they are already present in the cache.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/dov_harmonics.json",
        help="Output model JSON path for --fit.",
    )
    parser.add_argument(
        "--model",
        help="Existing model JSON path for --eval. Defaults to --output.",
    )
    parser.add_argument(
        "--name",
        default="Dover local harmonic tide timing model",
        help="Human-readable model name stored in generated JSON.",
    )
    parser.add_argument(
        "--constituents",
        default=",".join(DEFAULT_MODEL_CONSTITUENTS),
        help="Comma-separated tidal constituents to fit.",
    )
    parser.add_argument(
        "--min-daily-rows",
        type=int,
        default=90,
        help="Minimum rows required for a day to be included in fitting/eval.",
    )
    parser.add_argument(
        "--min-daily-span",
        type=float,
        default=3.0,
        help="Minimum daily tide range in metres for inclusion in fitting/eval.",
    )
    parser.add_argument(
        "--skip-ntslf",
        action="store_true",
        help="Skip live NTSLF comparison during --eval.",
    )
    parser.add_argument(
        "--ntslf-limit",
        type=int,
        default=20,
        help="Maximum NTSLF high/low events to compare.",
    )
    parser.add_argument(
        "--eval-start",
        help="Optional first cache date for residual evaluation, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--eval-end",
        help="Optional exclusive cache end date for residual evaluation, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--backtest-start",
        help=(
            "Optional first test-window date for rolling backtest, YYYY-MM-DD. "
            "Defaults to first cache date plus --backtest-train-days."
        ),
    )
    parser.add_argument(
        "--backtest-end",
        help=(
            "Optional exclusive last date for rolling backtest test windows, "
            "YYYY-MM-DD. Defaults to the day after the last cache observation."
        ),
    )
    parser.add_argument(
        "--backtest-train-days",
        type=int,
        default=365,
        help="Training window length in days for each rolling backtest split.",
    )
    parser.add_argument(
        "--backtest-test-days",
        type=int,
        default=30,
        help="Test window length in days for each rolling backtest split.",
    )
    parser.add_argument(
        "--backtest-step-days",
        type=int,
        default=30,
        help="Days to advance between rolling backtest splits.",
    )
    parser.add_argument(
        "--event-match-tolerance-min",
        type=float,
        default=120.0,
        help="Maximum predicted-vs-observed event match distance in minutes.",
    )
    args = parser.parse_args()

    if not (args.fetch or args.fit or args.eval or args.backtest):
        parser.error(
            "At least one of --fetch, --fit, --eval, or --backtest is required."
        )
    if args.fetch and not (args.archive_start and args.archive_end):
        parser.error("--fetch requires --archive-start and --archive-end.")
    if args.eval and not (args.fit or args.model or args.output):
        parser.error("--eval requires --model or --output.")
    if args.backtest and (
        args.backtest_train_days <= 0
        or args.backtest_test_days <= 0
        or args.backtest_step_days <= 0
    ):
        parser.error("--backtest window and step day values must be positive.")
    return args


def update_archive_cache(
    *,
    cache: Path,
    measure: str,
    start: datetime.date,
    end: datetime.date,
    force: bool = False,
    urlopen: Urlopen = urllib.request.urlopen,
) -> FetchStats:
    """Update a filtered EA archive cache with Dover rows only."""
    started = time.monotonic()
    existing = (
        load_cache_rows_by_day(cache, default_measure=measure)
        if cache.exists() and not force
        else {}
    )
    merged_rows = {day: list(rows) for day, rows in existing.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    fetched_days = 0
    skipped_days = 0
    archive_content_length_bytes = 0

    current = start
    while current < end:
        if current in existing:
            skipped_days += 1
            print(
                f"skip {current.isoformat()} rows={len(existing[current])}", flush=True
            )
        else:
            print(f"fetch {current.isoformat()}", flush=True)
            rows, content_length = fetch_archive_day(
                current,
                measure=measure,
                urlopen=urlopen,
            )
            merged_rows[current] = rows
            fetched_days += 1
            archive_content_length_bytes += content_length
            rows_written = write_archive_cache(cache, merged_rows)
            print(
                f"  kept rows={len(rows)} content_length={content_length}",
                flush=True,
            )
        current += datetime.timedelta(days=1)

    rows_written = write_archive_cache(cache, merged_rows)

    return FetchStats(
        cache=cache,
        fetched_days=fetched_days,
        skipped_days=skipped_days,
        rows_written=rows_written,
        archive_content_length_bytes=archive_content_length_bytes,
        elapsed_seconds=time.monotonic() - started,
    )


def write_archive_cache(
    cache: Path,
    rows_by_day: dict[datetime.date, list[dict[str, str]]],
) -> int:
    """Atomically write filtered cache rows grouped by day."""
    rows_written = 0
    temp_cache = cache.with_name(f".{cache.name}.tmp")
    with temp_cache.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dateTime", "measure", "value"])
        writer.writeheader()
        for day in sorted(rows_by_day):
            rows_written += write_cache_rows(writer, rows_by_day[day])
    temp_cache.replace(cache)
    return rows_written


def load_cache_rows_by_day(
    cache: Path, *, default_measure: str
) -> dict[datetime.date, list[dict[str, str]]]:
    """Load existing filtered cache rows grouped by UTC date."""
    rows_by_day: dict[datetime.date, list[dict[str, str]]] = {}
    with cache.open(newline="") as f:
        for row in csv.DictReader(f):
            timestamp = pd.Timestamp(row["dateTime"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(datetime.UTC)
            day = timestamp.tz_convert(datetime.UTC).date()
            row.setdefault("measure", default_measure)
            rows_by_day.setdefault(day, []).append(row)
    return rows_by_day


def fetch_archive_day(
    day: datetime.date,
    *,
    measure: str,
    urlopen: Urlopen = urllib.request.urlopen,
) -> tuple[list[dict[str, str]], int]:
    """Fetch and filter one EA daily archive file."""
    url = f"{EA_FLOOD_MONITORING_ROOT}/archive/readings-{day.isoformat()}.csv"
    rows: list[dict[str, str]] = []
    with urlopen(url, timeout=120) as response:  # type: ignore[attr-defined]
        content_length = int(getattr(response, "headers", {}).get("content-length", 0))
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
        for row in reader:
            if strict_row_matches_measure(row, measure):
                rows.append(
                    {
                        "dateTime": row["dateTime"],
                        "measure": measure,
                        "value": row["value"],
                    }
                )
    return rows, content_length


def write_cache_rows(
    writer: csv.DictWriter[str],
    rows: Sequence[dict[str, str]],
) -> int:
    """Write filtered cache rows in stable schema order."""
    for row in rows:
        writer.writerow(
            {
                "dateTime": row["dateTime"],
                "measure": row.get("measure") or DOVER_LOCAL_DATUM_MEASURE,
                "value": row["value"],
            }
        )
    return len(rows)


def fit_from_cache(
    *,
    cache: Path,
    measure: str,
    quality_filter: QualityFilter,
    model_name: str,
    constituents: Sequence[str],
) -> FitResult:
    """Fit a harmonic tide model from a filtered cache."""
    observations = load_filtered_cache(cache, measure=measure)
    return fit_from_observations(
        observations,
        measure=measure,
        quality_filter=quality_filter,
        model_name=model_name,
        constituents=constituents,
        cache=cache,
    )


def fit_from_observations(
    observations: pd.DataFrame,
    *,
    measure: str,
    quality_filter: QualityFilter,
    model_name: str,
    constituents: Sequence[str],
    cache: Path | None = None,
) -> FitResult:
    """Fit a harmonic tide model from a provided observation window."""
    clean_observations, daily_summary, excluded_days = quality_filter_observations(
        observations,
        quality_filter=quality_filter,
    )
    metadata = {
        "derived_at_utc": datetime.datetime.now(datetime.UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "measure": measure,
        "source_url": measure_url(measure),
        "observation_count": len(clean_observations),
        "first_observation_utc": clean_observations.index.min().isoformat(),
        "last_observation_utc": clean_observations.index.max().isoformat(),
        "excluded_day_count": len(excluded_days),
        "min_daily_rows": quality_filter.min_daily_rows,
        "min_daily_span": quality_filter.min_daily_span,
        "note": (
            "Model is for swimmer-condition timing context, not navigation. "
            "Predicted heights use the fitted source datum, not necessarily Chart Datum."
        ),
    }
    if cache is not None:
        metadata["cache"] = str(cache)
    speeds = {
        name: harmonic_tides.DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR[name]
        for name in constituents
    }
    model = harmonic_tides.fit_harmonic_model(
        clean_observations,
        name=model_name,
        source=f"Environment Agency Dover tide gauge measure {measure}",
        height_units="m",
        height_datum=height_datum(measure),
        constituent_speeds=speeds,
        metadata=metadata,
    )
    return FitResult(
        model=model,
        observations=clean_observations,
        daily_summary=daily_summary,
        excluded_days=excluded_days,
    )


def load_filtered_cache(cache: Path, *, measure: str) -> pd.DataFrame:
    """Load filtered cache CSV into a UTC-indexed observations dataframe."""
    if not cache.exists():
        raise FileNotFoundError(f"Filtered cache does not exist: {cache}")
    frame = pd.read_csv(cache)
    if frame.empty:
        raise ValueError(f"Filtered cache is empty: {cache}")
    if "measure" in frame.columns:
        frame = frame[
            frame["measure"].map(
                lambda value: cache_row_matches_measure({"measure": value}, measure)
            )
        ]
    frame["dateTime"] = pd.to_datetime(frame["dateTime"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).drop_duplicates(subset=["dateTime"])
    if frame.empty:
        raise ValueError(f"No usable rows found in filtered cache: {cache}")
    return frame.set_index("dateTime").sort_index()[["value"]]


def select_observation_window(
    observations: pd.DataFrame,
    *,
    start: datetime.date | None,
    end: datetime.date | None,
) -> pd.DataFrame:
    """Select an optional date window from cached UTC observations."""
    selected = observations
    if start is not None:
        start_at = pd.Timestamp(start, tz=datetime.UTC)
        selected = selected[selected.index >= start_at]
    if end is not None:
        end_at = pd.Timestamp(end, tz=datetime.UTC)
        selected = selected[selected.index < end_at]
    if selected.empty:
        raise ValueError("No observations found in requested evaluation window")
    return selected


def quality_filter_observations(
    observations: pd.DataFrame,
    *,
    quality_filter: QualityFilter,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop days with too few rows or suspiciously low tidal range."""
    daily = pd.DataFrame(
        observations["value"].resample("D").agg(["count", "min", "max"])
    )
    daily["span"] = daily["max"] - daily["min"]
    include = (daily["count"] >= quality_filter.min_daily_rows) & (
        daily["span"] >= quality_filter.min_daily_span
    )
    included_days = set(daily[include].index.date)
    day_mask = pd.Series(observations.index.date, index=observations.index).isin(
        included_days
    )
    clean = observations[day_mask.to_numpy()]
    if clean.empty:
        raise ValueError("No observations remain after quality filtering")
    excluded = pd.DataFrame(daily.loc[~include])
    return clean, daily, excluded


def residual_metrics(
    model: harmonic_tides.HarmonicTideModel,
    observations: pd.DataFrame,
) -> ResidualMetrics:
    """Compute residual metrics for model predictions against observations."""
    predictions = model.predict_utc(
        np.array([ts.to_pydatetime() for ts in observations.index], dtype=object)
    )
    residuals = observations["value"].to_numpy(dtype=float) - predictions
    return ResidualMetrics(
        count=len(observations),
        rmse_m=float(np.sqrt(np.mean(residuals * residuals))),
        mae_m=float(np.mean(np.abs(residuals))),
        bias_m=float(np.mean(residuals)),
        p95_abs_m=float(np.percentile(np.abs(residuals), 95)),
    )


def observed_high_low_events(
    observations: pd.DataFrame,
    *,
    min_separation_hours: float = 4.0,
    max_gap_factor: float = 1.5,
    min_segment_hours: float = 12.0,
) -> pd.DataFrame:
    """Extract observed high/low events from a UTC-indexed gauge series.

    The returned event times are limited by the observation cadence. For the
    Dover EA archive this is usually 15-minute mean water level data, so these
    observed extrema are validation targets for swimmer-facing timing, not
    navigation-quality harmonic constants.
    """
    clean = observations[["value"]].dropna().sort_index()
    if len(clean) < 3:
        return _empty_event_frame()

    index = clean.index
    if index.tz is None:
        index = index.tz_localize(datetime.UTC)
    else:
        index = index.tz_convert(datetime.UTC)
    clean = clean.copy()
    clean.index = index

    rows: list[dict[str, object]] = []
    for segment in continuous_observation_segments(
        clean,
        max_gap_factor=max_gap_factor,
        min_segment_hours=min_segment_hours,
    ):
        values = segment["value"].to_numpy(dtype=float)
        median_step_minutes = _median_observation_step_minutes(segment)
        min_distance = max(1, int((min_separation_hours * 60.0) / median_step_minutes))
        high_indexes, _ = find_peaks(values, distance=min_distance)
        low_indexes, _ = find_peaks(-values, distance=min_distance)

        for peak_index in high_indexes:
            rows.append(_observed_event_row(segment, int(peak_index), "high"))
        for peak_index in low_indexes:
            rows.append(_observed_event_row(segment, int(peak_index), "low"))
    if not rows:
        return _empty_event_frame()
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def continuous_observation_segments(
    observations: pd.DataFrame,
    *,
    max_gap_factor: float = 1.5,
    min_segment_hours: float = 12.0,
) -> tuple[pd.DataFrame, ...]:
    """Split observations into continuous segments and drop too-short segments."""
    if len(observations) < 3:
        return ()
    clean = observations[["value"]].dropna().sort_index()
    if len(clean) < 3:
        return ()

    median_step_minutes = _median_observation_step_minutes(clean)
    max_gap = datetime.timedelta(minutes=median_step_minutes * max_gap_factor)
    segment_ids = clean.index.to_series().diff().gt(max_gap).cumsum()
    segments: list[pd.DataFrame] = []
    min_duration = datetime.timedelta(hours=min_segment_hours)
    for _, segment in clean.groupby(segment_ids):
        if len(segment) < 3:
            continue
        duration = (
            segment.index.max().to_pydatetime() - segment.index.min().to_pydatetime()
        )
        if duration < min_duration:
            continue
        segments.append(segment)
    return tuple(segments)


def predicted_high_low_events_for_window(
    model: harmonic_tides.HarmonicTideModel,
    *,
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    """Generate predicted UTC high/low event rows for a date window."""
    predicted = harmonic_tides.predict_high_low_events(
        model,
        start_utc=datetime.datetime.combine(
            start, datetime.time(), tzinfo=datetime.UTC
        ),
        end_utc=datetime.datetime.combine(end, datetime.time(), tzinfo=datetime.UTC),
        timezone=datetime.UTC,
        sample_minutes=1,
    ).reset_index()
    if predicted.empty:
        return _empty_event_frame()
    predicted["time"] = pd.to_datetime(predicted["time"], utc=True)
    predicted = predicted.rename(columns={"prediction": "height_m"})
    return predicted[["time", "height_m", "type"]]


def event_timing_metrics(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    max_timing_error_minutes: float,
) -> dict[str, EventTimingMetrics]:
    """Compute one-to-one predicted-vs-observed event timing metrics by type."""
    return {
        event_type: _event_timing_metrics_for_type(
            observed,
            predicted,
            event_type=event_type,
            max_timing_error_minutes=max_timing_error_minutes,
        )
        for event_type in ("high", "low")
    }


def run_rolling_backtest(
    *,
    cache: Path,
    measure: str,
    quality_filter: QualityFilter,
    model_name: str,
    constituents: Sequence[str],
    train_days: int,
    test_days: int,
    step_days: int,
    start: datetime.date | None,
    end: datetime.date | None,
    match_tolerance_minutes: float,
) -> BacktestResult:
    """Run rolling train/test backtests from a filtered observation cache."""
    observations = load_filtered_cache(cache, measure=measure)
    cache_start = observations.index.min().date()
    cache_end = observations.index.max().date() + datetime.timedelta(days=1)
    test_start = start or (cache_start + datetime.timedelta(days=train_days))
    backtest_end = end or cache_end
    first_train_start = test_start - datetime.timedelta(days=train_days)
    if first_train_start < cache_start:
        raise ValueError(
            "Backtest training window starts before cache coverage. "
            f"cache_start={cache_start}, train_start={first_train_start}"
        )
    if backtest_end > cache_end:
        raise ValueError(
            "Backtest end extends beyond cache coverage. "
            f"cache_end={cache_end}, backtest_end={backtest_end}"
        )

    split_results: list[BacktestSplitResult] = []
    while test_start + datetime.timedelta(days=test_days) <= backtest_end:
        train_start = test_start - datetime.timedelta(days=train_days)
        train_end = test_start
        test_end = test_start + datetime.timedelta(days=test_days)
        train_observations = select_observation_window(
            observations,
            start=train_start,
            end=train_end,
        )
        fit_result = fit_from_observations(
            train_observations,
            measure=measure,
            quality_filter=quality_filter,
            model_name=f"{model_name} backtest {test_start.isoformat()}",
            constituents=constituents,
        )
        test_observations = select_observation_window(
            observations,
            start=test_start,
            end=test_end,
        )
        clean_test, _, _ = quality_filter_observations(
            test_observations,
            quality_filter=quality_filter,
        )
        observed_events = observed_high_low_events(clean_test)
        predicted_events = predicted_high_low_events_for_window(
            fit_result.model,
            start=test_start,
            end=test_end,
        )
        split_results.append(
            BacktestSplitResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_rows=len(fit_result.observations),
                test_rows=len(clean_test),
                train_coverage_days=observation_coverage_days(fit_result.observations),
                test_coverage_days=observation_coverage_days(clean_test),
                residual_metrics=residual_metrics(fit_result.model, clean_test),
                timing_metrics=event_timing_metrics(
                    observed_events,
                    predicted_events,
                    max_timing_error_minutes=match_tolerance_minutes,
                ),
            )
        )
        test_start += datetime.timedelta(days=step_days)

    if not split_results:
        raise ValueError(
            "No backtest splits generated. Fetch a longer cache or reduce "
            "--backtest-train-days/--backtest-test-days."
        )
    return BacktestResult(
        splits=tuple(split_results),
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )


def observation_coverage_days(observations: pd.DataFrame) -> float:
    """Return elapsed observation coverage in days."""
    if observations.empty:
        return 0.0
    return float(
        (
            observations.index.max().to_pydatetime()
            - observations.index.min().to_pydatetime()
        ).total_seconds()
        / 86400.0
    )


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "height_m", "type"])


def _observed_event_row(
    observations: pd.DataFrame,
    peak_index: int,
    event_type: str,
) -> dict[str, object]:
    timestamp = observations.index[peak_index].to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=datetime.UTC)
    return {
        "time": timestamp.astimezone(datetime.UTC),
        "height_m": float(observations["value"].iloc[peak_index]),
        "type": event_type,
    }


def _median_observation_step_minutes(observations: pd.DataFrame) -> float:
    diffs = observations.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return 15.0
    return max(1.0, float(diffs.median()))


def _event_timing_metrics_for_type(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    event_type: str,
    max_timing_error_minutes: float,
) -> EventTimingMetrics:
    observed_events = _events_of_type(observed, event_type)
    predicted_events = _events_of_type(predicted, event_type)
    used_predicted: set[int] = set()
    errors: list[float] = []

    for observed_row in observed_events.itertuples():
        best_index: int | None = None
        best_error: float | None = None
        for predicted_row in predicted_events.itertuples():
            predicted_index = int(predicted_row.Index)
            if predicted_index in used_predicted:
                continue
            error = (
                pd.Timestamp(predicted_row.time) - pd.Timestamp(observed_row.time)
            ).total_seconds() / 60.0
            if abs(error) > max_timing_error_minutes:
                continue
            if best_error is None or abs(error) < abs(best_error):
                best_error = float(error)
                best_index = predicted_index
        if best_index is not None and best_error is not None:
            used_predicted.add(best_index)
            errors.append(best_error)

    abs_errors = np.abs(np.array(errors, dtype=float))
    return EventTimingMetrics(
        timing_errors_min=tuple(errors),
        matched_events=len(errors),
        reference_events=len(observed_events),
        predicted_events=len(predicted_events),
        missed_events=len(observed_events) - len(errors),
        extra_events=len(predicted_events) - len(used_predicted),
        median_abs_min=_nan_percentile(abs_errors, 50),
        mean_abs_min=float(np.mean(abs_errors)) if len(abs_errors) else float("nan"),
        p95_abs_min=_nan_percentile(abs_errors, 95),
        max_abs_min=float(np.max(abs_errors)) if len(abs_errors) else float("nan"),
    )


def _events_of_type(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    if events.empty:
        return _empty_event_frame()
    filtered = events[events["type"].astype(str) == event_type].copy()
    if filtered.empty:
        return _empty_event_frame()
    filtered["time"] = pd.to_datetime(filtered["time"], utc=True)
    return filtered.sort_values("time")


def _nan_percentile(values: np.ndarray, percentile: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def compare_model_to_ntslf(
    model: harmonic_tides.HarmonicTideModel,
    *,
    limit: int,
    urlopen: Urlopen = urllib.request.urlopen,
) -> pd.DataFrame:
    """Compare model high/low events against the current NTSLF Dover table."""
    ntslf_events = fetch_ntslf_dover_events(urlopen=urlopen)
    reference = ntslf_events.head(limit)
    start = reference["time"].min().to_pydatetime() - datetime.timedelta(hours=3)
    end = reference["time"].max().to_pydatetime() + datetime.timedelta(hours=3)
    predicted = harmonic_tides.predict_high_low_events(
        model,
        start_utc=start,
        end_utc=end,
        timezone=datetime.UTC,
        sample_minutes=1,
    ).reset_index()
    predicted["time"] = pd.to_datetime(predicted["time"], utc=True)

    rows: list[dict[str, object]] = []
    for row in reference.itertuples(index=False):
        candidates = predicted[predicted["type"].astype(str) == row.type].copy()
        if candidates.empty:
            continue
        candidates["dt_min"] = (candidates["time"] - row.time).dt.total_seconds() / 60
        match = candidates.iloc[candidates["dt_min"].abs().argmin()]
        rows.append(
            {
                "type": row.type,
                "ntslf_time": row.time,
                "model_time": match["time"],
                "dt_min": float(match["dt_min"]),
                "ntslf_height_m": row.height_m,
                "model_height": float(match["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def fetch_ntslf_dover_events(
    *,
    urlopen: Urlopen = urllib.request.urlopen,
) -> pd.DataFrame:
    """Fetch and parse the NTSLF Dover high/low table fragment."""
    request = urllib.request.Request(
        NTSLF_DOVER_TIDE_FRAGMENT_URL,
        headers={
            "User-Agent": "shallweswim tide model evaluation",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": NTSLF_DOVER_TIDE_PAGE_URL,
        },
    )
    with urlopen(request, timeout=30) as response:  # type: ignore[attr-defined]
        body = response.read().decode("utf-8", "replace")
    return parse_ntslf_dover_events(body)


def parse_ntslf_dover_events(body: str) -> pd.DataFrame:
    """Parse NTSLF Dover high/low HTML table into UTC event rows."""
    parser = _NtslfTableParser()
    parser.feed(body)
    month_map = {
        month: index
        for index, month in enumerate(
            (
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ),
            1,
        )
    }
    events: list[dict[str, object]] = []
    current_year: int | None = None
    current_month: int | None = None

    for row in parser.rows:
        if not row:
            continue
        date_match = re.search(
            r"(\w{3})\s+(\d+)(?:st|nd|rd|th)([A-Za-z]{3})?\s*(\d{4})?",
            row[0],
        )
        if not date_match:
            continue
        day = int(date_match.group(2))
        if date_match.group(3):
            current_month = month_map[date_match.group(3)]
        if date_match.group(4):
            current_year = int(date_match.group(4))
        if current_month is None or current_year is None:
            continue
        for cell in row[1:]:
            event_match = re.search(r"(\d{2}:\d{2})\s+([0-9.]+)m\s+([HL])", cell)
            if not event_match:
                continue
            timestamp = datetime.datetime.fromisoformat(
                f"{current_year:04d}-{current_month:02d}-{day:02d}"
                f"T{event_match.group(1)}:00+00:00"
            )
            events.append(
                {
                    "time": timestamp,
                    "height_m": float(event_match.group(2)),
                    "type": "high" if event_match.group(3) == "H" else "low",
                }
            )
    if not events:
        raise ValueError("No NTSLF Dover tide events parsed")
    return pd.DataFrame(events).sort_values("time")


def print_fetch_stats(stats: FetchStats) -> None:
    """Print fetch stats in a stable, grep-friendly form."""
    print("fetch_summary:")
    print(f"  cache={stats.cache}")
    print(f"  fetched_days={stats.fetched_days}")
    print(f"  skipped_days={stats.skipped_days}")
    print(f"  rows_written={stats.rows_written}")
    print(f"  archive_content_length_bytes={stats.archive_content_length_bytes}")
    print(f"  cache_size_bytes={stats.cache.stat().st_size}")
    print(f"  elapsed_seconds={stats.elapsed_seconds:.1f}")


def print_fit_summary(fit_result: FitResult, *, output: Path) -> None:
    """Print fit inputs and output path."""
    print("fit_summary:")
    print(f"  output={output}")
    print(f"  observations={len(fit_result.observations)}")
    print(f"  first={fit_result.observations.index.min()}")
    print(f"  last={fit_result.observations.index.max()}")
    print(f"  excluded_days={len(fit_result.excluded_days)}")
    condition_number = fit_result.model.metadata.get("fit_condition_number")
    if condition_number is not None:
        print(f"  fit_condition_number={float(condition_number):.3e}")
    print_constituent_summary(fit_result.model)
    if not fit_result.excluded_days.empty:
        print("excluded_days:")
        print(fit_result.excluded_days[["count", "span"]].to_string())


def print_quality_summary(
    observations: pd.DataFrame,
    *,
    daily_summary: pd.DataFrame,
    excluded_days: pd.DataFrame,
) -> None:
    """Print quality filtering summary."""
    print("quality_summary:")
    print(f"  clean_rows={len(observations)}")
    print(f"  clean_first={observations.index.min()}")
    print(f"  clean_last={observations.index.max()}")
    print(f"  total_days={len(daily_summary)}")
    print(f"  excluded_days={len(excluded_days)}")
    print(f"  daily_span_median={daily_summary['span'].median():.3f}")
    print(f"  daily_span_min={daily_summary['span'].min():.3f}")
    print(f"  daily_span_max={daily_summary['span'].max():.3f}")
    if not excluded_days.empty:
        print("excluded_days:")
        print(excluded_days[["count", "span"]].to_string())


def print_constituent_summary(model: harmonic_tides.HarmonicTideModel) -> None:
    """Print fitted constituent amplitude and phase diagnostics."""
    print("constituents:")
    for constituent in model.constituents:
        diagnostics = harmonic_tides.constituent_amplitude_phase(constituent)
        print(
            "  "
            f"{diagnostics['name']}: "
            f"amplitude={float(diagnostics['amplitude']):.6f} "
            f"phase_degrees={float(diagnostics['phase_degrees']):.3f}"
        )


def print_residual_metrics(
    model: harmonic_tides.HarmonicTideModel,
    *,
    observations: pd.DataFrame,
    label: str,
) -> None:
    """Print residual metrics for one observation set."""
    metrics = residual_metrics(model, observations)
    print("residuals:")
    print(f"  label={label}")
    print(f"  count={metrics.count}")
    print(f"  rmse_m={metrics.rmse_m:.6f}")
    print(f"  mae_m={metrics.mae_m:.6f}")
    print(f"  bias_m={metrics.bias_m:.6f}")
    print(f"  p95_abs_m={metrics.p95_abs_m:.6f}")


def print_ntslf_comparison(comparison: pd.DataFrame) -> None:
    """Print model-vs-NTSLF event comparison."""
    print("ntslf_comparison:")
    print("  label=external_spot_check_not_training_ground_truth")
    print(comparison.to_string(index=False))
    if not comparison.empty:
        print(f"  abs_timing_median_min={comparison['dt_min'].abs().median():.3f}")
        print(f"  abs_timing_mean_min={comparison['dt_min'].abs().mean():.3f}")
        print(f"  abs_timing_max_min={comparison['dt_min'].abs().max():.3f}")


def print_backtest_summary(result: BacktestResult) -> None:
    """Print rolling backtest timing and residual summaries."""
    print("backtest_summary:")
    print(f"  splits={len(result.splits)}")
    print(f"  train_window_days={result.train_days}")
    print(f"  test_window_days={result.test_days}")
    print(f"  step_days={result.step_days}")
    residual_rmses = np.array(
        [split.residual_metrics.rmse_m for split in result.splits],
        dtype=float,
    )
    print(f"  residual_rmse_median={np.median(residual_rmses):.6f}")
    print(f"  residual_rmse_max={np.max(residual_rmses):.6f}")
    for event_type in ("high", "low"):
        metrics = aggregate_timing_metrics(result, event_type=event_type)
        print(f"  {event_type}_matched_events={metrics.matched_events}")
        print(f"  {event_type}_missed_events={metrics.missed_events}")
        print(f"  {event_type}_extra_events={metrics.extra_events}")
        print(f"  {event_type}_median_abs_timing_min={metrics.median_abs_min:.3f}")
        print(f"  {event_type}_mean_abs_timing_min={metrics.mean_abs_min:.3f}")
        print(f"  {event_type}_p95_abs_timing_min={metrics.p95_abs_min:.3f}")
        print(f"  {event_type}_max_abs_timing_min={metrics.max_abs_min:.3f}")
    print("backtest_splits:")
    for split in result.splits:
        high = split.timing_metrics["high"]
        low = split.timing_metrics["low"]
        print(
            "  "
            f"test={split.test_start}:{split.test_end} "
            f"train_rows={split.train_rows} test_rows={split.test_rows} "
            f"train_coverage_days={split.train_coverage_days:.1f} "
            f"test_coverage_days={split.test_coverage_days:.1f} "
            f"rmse_m={split.residual_metrics.rmse_m:.6f} "
            f"high_median_abs_min={high.median_abs_min:.3f} "
            f"low_median_abs_min={low.median_abs_min:.3f} "
            f"missed={high.missed_events + low.missed_events} "
            f"extra={high.extra_events + low.extra_events}"
        )


def aggregate_timing_metrics(
    result: BacktestResult,
    *,
    event_type: str,
) -> EventTimingMetrics:
    """Aggregate event timing metrics across rolling backtest splits."""
    split_metrics = [split.timing_metrics[event_type] for split in result.splits]
    errors = tuple(
        error for metrics in split_metrics for error in metrics.timing_errors_min
    )
    abs_errors = np.abs(np.array(errors, dtype=float))
    return EventTimingMetrics(
        timing_errors_min=errors,
        matched_events=sum(metrics.matched_events for metrics in split_metrics),
        reference_events=sum(metrics.reference_events for metrics in split_metrics),
        predicted_events=sum(metrics.predicted_events for metrics in split_metrics),
        missed_events=sum(metrics.missed_events for metrics in split_metrics),
        extra_events=sum(metrics.extra_events for metrics in split_metrics),
        median_abs_min=_nan_percentile(abs_errors, 50),
        mean_abs_min=float(np.mean(abs_errors)) if len(abs_errors) else float("nan"),
        p95_abs_min=_nan_percentile(abs_errors, 95),
        max_abs_min=float(np.max(abs_errors)) if len(abs_errors) else float("nan"),
    )


def measure_url(measure: str) -> str:
    """Return the Environment Agency measure URL."""
    return f"{EA_FLOOD_MONITORING_ROOT}/id/measures/{measure}"


def strict_row_matches_measure(row: dict[str, str], measure: str) -> bool:
    """Return whether a raw EA archive row has the requested measure id."""
    row_measure = row.get("measure")
    if row_measure is None:
        return False
    return cache_measure_matches(row_measure, measure)


def cache_row_matches_measure(row: dict[str, str], measure: str) -> bool:
    """Return whether a filtered cache row belongs to a requested measure id."""
    row_measure = row.get("measure")
    if row_measure is None or pd.isna(row_measure):
        return True
    return cache_measure_matches(row_measure, measure)


def cache_measure_matches(row_measure: object, measure: str) -> bool:
    """Return whether an EA CSV row belongs to the requested measure id."""
    if not isinstance(row_measure, str):
        return False
    return row_measure == measure or row_measure.rstrip("/").endswith(f"/{measure}")


def height_datum(measure: str) -> str:
    """Return the datum label for a configured Dover measure."""
    if measure == DOVER_AOD_MEASURE:
        return "Environment Agency mAOD"
    return "Environment Agency local tide gauge datum"


def _parse_constituents(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [
        name
        for name in names
        if name not in harmonic_tides.DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR
    ]
    if unknown:
        raise SystemExit(f"Unknown constituents: {', '.join(unknown)}")
    return names


def _parse_date(raw: str) -> datetime.date:
    return datetime.date.fromisoformat(raw)


def _parse_optional_date(raw: str | None) -> datetime.date | None:
    if raw is None:
        return None
    return _parse_date(raw)


def _eval_label(start: str | None, end: str | None) -> str:
    if start is None and end is None:
        return "cache_clean_days_in_sample_unless_model_was_fit_elsewhere"
    return f"cache_clean_days_window[{start or '-inf'}:{end or '+inf'}]"


if __name__ == "__main__":
    main()

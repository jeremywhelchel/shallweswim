"""Unit tests for the offline harmonic tide fitting script."""

import datetime
import io
import json
import urllib.request
from pathlib import Path
from types import TracebackType

import numpy as np
import pandas as pd

from shallweswim.scripts import derive_harmonic_tide_model as script
from shallweswim.scripts import harmonic_tides


class FakeResponse(io.BytesIO):
    """Binary response object compatible with urllib responses."""

    def __init__(self, body: str, *, content_length: int | None = None) -> None:
        super().__init__(body.encode())
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def synthetic_model() -> harmonic_tides.HarmonicTideModel:
    """Return a simple single-constituent tide model."""
    return harmonic_tides.HarmonicTideModel(
        name="Synthetic tide",
        epoch_utc=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        intercept=2.0,
        constituents=(
            harmonic_tides.HarmonicConstituent(
                name="M2",
                speed_degrees_per_hour=28.9841042,
                cos=1.0,
                sin=0.0,
            ),
        ),
        height_units="m",
        height_datum="synthetic datum",
        source="synthetic source",
        metadata={},
    )


def write_synthetic_cache(
    path: Path,
    *,
    days: int = 30,
    freq: str = "h",
) -> pd.DataFrame:
    """Write synthetic observations to a filtered cache CSV."""
    model = synthetic_model()
    times = pd.date_range(
        "2026-01-01T00:00:00Z",
        end=pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=days),
        inclusive="left",
        freq=freq,
    )
    values = model.predict_utc(np.array([ts.to_pydatetime() for ts in times]))
    frame = pd.DataFrame(
        {
            "dateTime": [ts.isoformat().replace("+00:00", "Z") for ts in times],
            "measure": script.DOVER_LOCAL_DATUM_MEASURE,
            "value": values,
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_measure_matching_distinguishes_raw_archive_from_filtered_cache() -> None:
    """Raw rows must name a measure; legacy cache rows may omit it."""
    row = {
        "measure": (
            "http://environment.data.gov.uk/flood-monitoring/id/measures/"
            f"{script.DOVER_LOCAL_DATUM_MEASURE}"
        )
    }

    assert script.strict_row_matches_measure(row, script.DOVER_LOCAL_DATUM_MEASURE)
    assert not script.strict_row_matches_measure({}, script.DOVER_LOCAL_DATUM_MEASURE)
    assert script.cache_row_matches_measure({}, script.DOVER_LOCAL_DATUM_MEASURE)
    assert not script.cache_measure_matches(
        "http://environment.data.gov.uk/flood-monitoring/id/measures/"
        f"prefix-{script.DOVER_LOCAL_DATUM_MEASURE}-suffix",
        script.DOVER_LOCAL_DATUM_MEASURE,
    )


def test_update_archive_cache_preserves_cached_days_outside_requested_window(
    tmp_path: Path,
) -> None:
    """Filtered cache updates merge fetched days without truncating other days."""
    cache = tmp_path / "dover.csv"
    cache.write_text(
        "dateTime,measure,value\n"
        f"2026-01-01T00:00:00Z,{script.DOVER_LOCAL_DATUM_MEASURE},1.0\n"
        f"2026-01-05T00:00:00Z,{script.DOVER_LOCAL_DATUM_MEASURE},5.0\n"
    )
    calls: list[str] = []

    def fake_urlopen(
        request: str | urllib.request.Request,
        timeout: float | None = None,
    ) -> FakeResponse:
        calls.append(str(request))
        body = (
            "dateTime,measure,value\n"
            "2026-01-02T00:00:00Z,"
            f"http://environment.data.gov.uk/flood-monitoring/id/measures/{script.DOVER_LOCAL_DATUM_MEASURE},"
            "2.0\n"
            "2026-01-02T00:00:00Z,"
            "http://environment.data.gov.uk/flood-monitoring/id/measures/other,"
            "9.0\n"
        )
        return FakeResponse(body, content_length=123)

    stats = script.update_archive_cache(
        cache=cache,
        measure=script.DOVER_LOCAL_DATUM_MEASURE,
        start=datetime.date(2026, 1, 1),
        end=datetime.date(2026, 1, 3),
        urlopen=fake_urlopen,
    )

    assert stats.skipped_days == 1
    assert stats.fetched_days == 1
    assert stats.rows_written == 3
    assert stats.archive_content_length_bytes == 123
    assert len(calls) == 1

    cached = pd.read_csv(cache)
    assert cached["value"].tolist() == [1.0, 2.0, 5.0]


def test_update_archive_cache_persists_each_fetched_day_before_later_failure(
    tmp_path: Path,
) -> None:
    """Long fetches are resumable because each fetched day is written."""
    cache = tmp_path / "dover.csv"
    calls = 0

    def fake_urlopen(
        request: str | urllib.request.Request,
        timeout: float | None = None,
    ) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("synthetic later-day failure")
        body = (
            "dateTime,measure,value\n"
            "2026-01-01T00:00:00Z,"
            f"http://environment.data.gov.uk/flood-monitoring/id/measures/{script.DOVER_LOCAL_DATUM_MEASURE},"
            "1.0\n"
        )
        return FakeResponse(body, content_length=123)

    try:
        script.update_archive_cache(
            cache=cache,
            measure=script.DOVER_LOCAL_DATUM_MEASURE,
            start=datetime.date(2026, 1, 1),
            end=datetime.date(2026, 1, 3),
            urlopen=fake_urlopen,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("Expected synthetic failure")

    cached = pd.read_csv(cache)
    assert cached["value"].tolist() == [1.0]


def test_select_observation_window_uses_exclusive_end() -> None:
    """Evaluation windows are explicit and use an exclusive end date."""
    observations = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
            ],
            utc=True,
        ),
    )

    selected = script.select_observation_window(
        observations,
        start=datetime.date(2026, 1, 2),
        end=datetime.date(2026, 1, 3),
    )

    assert selected["value"].tolist() == [2.0]


def test_quality_filter_observations_excludes_low_range_days() -> None:
    """Suspiciously flat days are excluded from model fitting."""
    index = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="h")
    observations = pd.DataFrame(
        {
            "value": [0.0, 4.0] * 12 + [1.0] * 24,
        },
        index=index,
    )

    clean, daily, excluded = script.quality_filter_observations(
        observations,
        quality_filter=script.QualityFilter(
            min_daily_rows=20,
            min_daily_span=3.0,
        ),
    )

    assert len(daily) == 2
    assert len(excluded) == 1
    assert clean.index.min().date() == datetime.date(2026, 1, 1)
    assert clean.index.max().date() == datetime.date(2026, 1, 1)


def test_fit_from_cache_recovers_synthetic_model(tmp_path: Path) -> None:
    """Fitting from a filtered cache creates a usable harmonic model."""
    cache = tmp_path / "synthetic.csv"
    write_synthetic_cache(cache)

    result = script.fit_from_cache(
        cache=cache,
        measure=script.DOVER_LOCAL_DATUM_MEASURE,
        quality_filter=script.QualityFilter(
            min_daily_rows=20,
            min_daily_span=1.0,
        ),
        model_name="Fitted synthetic tide",
        constituents=("M2",),
    )
    metrics = script.residual_metrics(result.model, result.observations)

    assert metrics.rmse_m < 1e-10
    assert result.model.metadata["observation_count"] == len(result.observations)
    assert result.model.metadata["fit_condition_number"] > 0


def test_observed_high_low_events_extracts_timing_targets(tmp_path: Path) -> None:
    """Observed extrema become explicit timing validation targets."""
    cache = tmp_path / "synthetic.csv"
    write_synthetic_cache(cache, days=5, freq="15min")
    observations = script.load_filtered_cache(
        cache,
        measure=script.DOVER_LOCAL_DATUM_MEASURE,
    )

    events = script.observed_high_low_events(observations)

    assert set(events["type"]) == {"high", "low"}
    assert len(events) >= 8
    assert pd.to_datetime(events["time"], utc=True).is_monotonic_increasing


def test_observed_high_low_events_splits_across_large_gaps() -> None:
    """Observed extrema are extracted per continuous segment, not across gaps."""
    first_segment = pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
    second_segment = pd.date_range("2026-01-03T00:00:00Z", periods=96, freq="15min")
    index = first_segment.append(second_segment)
    values = np.sin(np.linspace(0.0, 8.0 * np.pi, len(index)))
    observations = pd.DataFrame({"value": values}, index=index)

    segments = script.continuous_observation_segments(observations)

    assert len(segments) == 2
    assert all(len(segment) == 96 for segment in segments)


def test_rolling_backtest_reports_event_timing_metrics(tmp_path: Path) -> None:
    """Rolling backtests validate high/low timing on future holdout windows."""
    cache = tmp_path / "synthetic.csv"
    write_synthetic_cache(cache, days=80, freq="15min")

    result = script.run_rolling_backtest(
        cache=cache,
        measure=script.DOVER_LOCAL_DATUM_MEASURE,
        quality_filter=script.QualityFilter(
            min_daily_rows=80,
            min_daily_span=1.0,
        ),
        model_name="Synthetic backtest",
        constituents=("M2",),
        train_days=30,
        test_days=10,
        step_days=10,
        start=None,
        end=None,
        match_tolerance_minutes=60.0,
    )

    high = script.aggregate_timing_metrics(result, event_type="high")
    low = script.aggregate_timing_metrics(result, event_type="low")
    assert len(result.splits) == 5
    assert high.missed_events == 0
    assert low.missed_events == 0
    assert high.median_abs_min <= 15.0
    assert low.median_abs_min <= 15.0


def test_rolling_backtest_rejects_windows_outside_cache_coverage(
    tmp_path: Path,
) -> None:
    """Explicit rolling windows cannot silently train on partial cache coverage."""
    cache = tmp_path / "synthetic.csv"
    write_synthetic_cache(cache, days=40, freq="15min")

    try:
        script.run_rolling_backtest(
            cache=cache,
            measure=script.DOVER_LOCAL_DATUM_MEASURE,
            quality_filter=script.QualityFilter(
                min_daily_rows=80,
                min_daily_span=1.0,
            ),
            model_name="Synthetic backtest",
            constituents=("M2",),
            train_days=30,
            test_days=10,
            step_days=10,
            start=datetime.date(2026, 1, 15),
            end=None,
            match_tolerance_minutes=60.0,
        )
    except ValueError as exc:
        assert "before cache coverage" in str(exc)
    else:
        raise AssertionError("Expected cache coverage validation failure")


def test_parse_ntslf_dover_events_handles_omitted_month_and_year() -> None:
    """NTSLF rows omit repeated month/year after the first day row."""
    body = """
    <table>
      <tr><td>Sat 13thJun 2026</td><td>03:52 1.34m L</td><td>08:56 6.22m H</td></tr>
      <tr><td>Sun 14th</td><td>04:54 1.17m L</td><td>09:50 6.41m H</td></tr>
    </table>
    """

    events = script.parse_ntslf_dover_events(body)

    assert events["type"].tolist() == ["low", "high", "low", "high"]
    assert events["height_m"].tolist() == [1.34, 6.22, 1.17, 6.41]
    assert events["time"].iloc[2] == datetime.datetime(
        2026, 6, 14, 4, 54, tzinfo=datetime.UTC
    )


def test_compare_model_to_ntslf_returns_timing_deltas(tmp_path: Path) -> None:
    """NTSLF comparison returns per-event time deltas."""
    model_path = tmp_path / "model.json"
    model = synthetic_model()
    model_path.write_text(json.dumps(model.to_dict()))
    first_high = harmonic_tides.predict_high_low_events(
        model,
        start_utc=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        end_utc=datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC),
        timezone=datetime.UTC,
        sample_minutes=1,
    ).reset_index()
    event = first_high.iloc[0]
    event_type = "H" if str(event["type"]) == "high" else "L"
    body = (
        "<table><tr><td>Thu 1stJan 2026</td>"
        f"<td>{event['time'].strftime('%H:%M')} 1.00m {event_type}</td>"
        "</tr></table>"
    )

    def fake_urlopen(
        request: str | urllib.request.Request,
        timeout: float | None = None,
    ) -> FakeResponse:
        return FakeResponse(body)

    comparison = script.compare_model_to_ntslf(
        harmonic_tides.load_model(str(model_path)),
        limit=1,
        urlopen=fake_urlopen,
    )

    assert len(comparison) == 1
    assert abs(comparison["dt_min"].iloc[0]) < 1.0

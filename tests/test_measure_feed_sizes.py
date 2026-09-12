"""Unit tests for the pure formatting/aggregation helpers in measure_feed_sizes.

These tests use only in-memory DataFrames constructed by hand; they never
touch the network or a real location config.
"""

import os

import pandas as pd
import pytest

from shallweswim.scripts.measure_feed_sizes import (
    FeedMeasurement,
    LocationReport,
    PlotMeasurement,
    YearPartitionMeasurement,
    _disable_archive_capture,
    _human_bytes,
    _parquet_bytes,
    _year_partitions,
    format_text,
    reports_to_dict,
    summarize,
)


def test_disable_archive_capture_removes_archive_bucket_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "prod-archive-bucket")

    _disable_archive_capture()

    assert "SHALLWESWIM_ARCHIVE_BUCKET" not in os.environ


def test_disable_archive_capture_is_a_noop_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)

    _disable_archive_capture()  # Must not raise.

    assert "SHALLWESWIM_ARCHIVE_BUCKET" not in os.environ


def _temp_frame(timestamps: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps), name="time")
    return pd.DataFrame(
        {"water_temp": [60.0 + i for i in range(len(timestamps))]}, index=index
    )


def test_parquet_bytes_returns_positive_size_for_nonempty_frame() -> None:
    df = _temp_frame(["2024-01-01", "2024-01-02", "2024-01-03"])
    size = _parquet_bytes(df)
    assert size > 0
    assert isinstance(size, int)


def test_year_partitions_splits_by_local_calendar_year() -> None:
    df = _temp_frame(["2023-12-31", "2024-01-01", "2024-06-01", "2025-01-01"])

    partitions = _year_partitions("nyc", df)

    assert [p.year for p in partitions] == [2023, 2024, 2025]
    assert [p.rows for p in partitions] == [1, 2, 1]
    assert all(p.location == "nyc" for p in partitions)
    assert all(p.parquet_bytes > 0 for p in partitions)
    # Sorted ascending regardless of row order.
    assert partitions == sorted(partitions, key=lambda p: p.year)


def test_year_partitions_empty_frame_returns_no_partitions() -> None:
    df = _temp_frame([])
    assert _year_partitions("nyc", df) == []


def test_human_bytes_formats_expected_units() -> None:
    assert _human_bytes(0) == "0B"
    assert _human_bytes(512) == "512B"
    assert _human_bytes(2048) == "2.0KiB"
    assert _human_bytes(5 * 1024 * 1024) == "5.0MiB"
    assert _human_bytes(3 * 1024 * 1024 * 1024) == "3.0GiB"


def _sample_reports() -> list[LocationReport]:
    return [
        LocationReport(
            location="nyc",
            feeds=[
                FeedMeasurement(
                    location="nyc",
                    feed="tides",
                    source="Coney Island, NY",
                    rows=100,
                    memory_bytes=1000,
                    parquet_bytes=200,
                ),
                FeedMeasurement(
                    location="nyc",
                    feed="historic_temps",
                    source="The Battery, NY",
                    rows=100_000,
                    memory_bytes=900_000,
                    parquet_bytes=300_000,
                ),
            ],
            year_partitions=[
                YearPartitionMeasurement(
                    location="nyc", year=2024, rows=8_760, parquet_bytes=20_000
                ),
            ],
            plots=[
                PlotMeasurement(location="nyc", plot="live_temps", svg_bytes=5_000),
                PlotMeasurement(
                    location="nyc",
                    plot="historic_temps_12mo",
                    svg_bytes=0,
                    error="ValueError: boom",
                ),
            ],
        ),
        LocationReport(
            location="san",
            feeds=[
                FeedMeasurement(
                    location="san",
                    feed="live_temps",
                    source=None,
                    rows=0,
                    memory_bytes=0,
                    parquet_bytes=0,
                    error="StationUnavailableError: no data",
                ),
            ],
        ),
    ]


def test_summarize_aggregates_totals_and_skips_errors() -> None:
    summary = summarize(_sample_reports())

    assert summary["locations_measured"] == 2
    assert summary["total_feed_rows"] == 100_100
    assert summary["total_feed_parquet_bytes"] == 200 + 300_000
    assert summary["total_feed_memory_bytes"] == 1000 + 900_000
    assert summary["total_plot_bytes"] == 5_000
    assert summary["grand_total_bytes"] == 200 + 300_000 + 5_000
    assert summary["feed_errors"] == 1
    assert summary["plot_errors"] == 1
    assert summary["feed_totals"]["tides"]["rows"] == 100
    assert summary["feed_totals"]["historic_temps"]["parquet_bytes"] == 300_000
    assert summary["largest_combined_historic_frame"] == {
        "location": "nyc",
        "parquet_bytes": 300_000,
    }


def test_summarize_with_no_historic_feeds_reports_no_largest_frame() -> None:
    reports = [
        LocationReport(
            location="chi",
            feeds=[
                FeedMeasurement(
                    location="chi",
                    feed="live_temps",
                    source="Chicago Buoy",
                    rows=10,
                    memory_bytes=100,
                    parquet_bytes=50,
                )
            ],
        )
    ]
    summary = summarize(reports)
    assert summary["largest_combined_historic_frame"] is None
    assert summary["feed_errors"] == 0
    assert summary["plot_errors"] == 0


def test_format_text_includes_rows_years_plots_and_totals() -> None:
    text = format_text(_sample_reports(), summarize(_sample_reports()))

    assert "nyc" in text
    assert "tides" in text
    assert "historic_temps/2024" in text
    assert "plot:live_temps" in text
    assert "ERROR" in text
    assert "boom" in text
    assert "no data" in text
    assert "Locations measured: 2" in text
    assert "WARNING: 1 feed error(s), 1 plot error(s)" in text


def test_reports_to_dict_round_trips_measurement_fields() -> None:
    reports = _sample_reports()
    payload = reports_to_dict(reports, summarize(reports))

    assert payload["summary"]["locations_measured"] == 2
    nyc = payload["locations"][0]
    assert nyc["location"] == "nyc"
    assert nyc["feeds"][0]["feed"] == "tides"
    assert nyc["year_partitions"][0]["year"] == 2024
    assert nyc["plots"][0]["plot"] == "live_temps"

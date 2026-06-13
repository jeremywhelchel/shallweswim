"""Tests for historical temperature visual artifact inspection script helpers."""

import pandas as pd

from shallweswim import plot
from shallweswim.scripts import inspect_historic_temp_plot_artifacts as inspect


def test_build_visual_artifact_outputs_reports_suppressed_stages() -> None:
    index = pd.date_range("2022-01-01", periods=30 * 24, freq="h")
    hist_temps = pd.DataFrame({"water_temp": [50.0] * len(index)}, index=index)
    hist_temps.loc[pd.Timestamp("2022-01-15 12:00:00"), "water_temp"] = 80.0

    plot_suppressed_points, final_plot_frame, counts = (
        inspect._build_visual_artifact_outputs(
            hist_temps, plot.DEFAULT_HISTORIC_TEMP_PLOT_POLICY
        )
    )

    assert counts["raw"] == {"2022": 1}
    assert list(plot_suppressed_points["stage"].unique()) == ["raw"]
    assert plot_suppressed_points.iloc[0]["source_temp_f"] == 80.0
    assert not final_plot_frame[2022].isna().all()


def test_plot_suppressed_points_frame_reconstructs_original_timestamp() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="h")
    source_frame = pd.DataFrame({2025: [50.0, 51.0, 80.0, 52.0]}, index=index)
    smoothed_frame = pd.DataFrame({2025: [50.0, 51.0, 52.0, 52.0]}, index=index)
    raw_mask = pd.DataFrame({2025: [False, False, True, False]}, index=index)
    empty_mask = pd.DataFrame({2025: [False] * len(index)}, index=index)

    plot_suppressed_points = inspect._plot_suppressed_points_frame(
        source_frame=source_frame,
        smoothed_frame=smoothed_frame,
        raw_mask=raw_mask,
        cross_year_mask=empty_mask,
        volatility_mask=empty_mask,
        short_segment_mask=empty_mask,
    )

    assert plot_suppressed_points.to_dict("records") == [
        {
            "stage": "raw",
            "year": 2025,
            "pivot_timestamp": pd.Timestamp("2020-01-01 02:00:00"),
            "original_timestamp": "2025-01-01 02:00:00",
            "source_temp_f": 80.0,
            "suppressed_value_f": 80.0,
            "seasonal_median_f": 52.0,
            "seasonal_residual_f": 28.0,
        }
    ]


def test_original_timestamp_leap_day_for_non_leap_year_is_blank() -> None:
    assert inspect._original_timestamp(pd.Timestamp("2020-02-29"), 2025) == ""

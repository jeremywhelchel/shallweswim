"""Tests for plotting validation helpers."""

import datetime
import io
import logging

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time
from matplotlib.figure import Figure

from shallweswim import config as config_lib
from shallweswim import plot, types, util


def test_save_fig_rejects_non_static_paths() -> None:
    fig = Figure()

    with pytest.raises(ValueError, match="Plot path must start with 'static/'"):
        plot.save_fig(fig, "tmp/plot.svg")


def test_generate_live_temp_plot_requires_two_rows() -> None:
    live_temps = pd.DataFrame(
        {"water_temp": [55.0]},
        index=pd.DatetimeIndex([datetime.datetime(2025, 4, 22, 10, 0, 0)]),
    )

    with pytest.raises(ValueError, match="Insufficient temperature data"):
        plot.generate_live_temp_plot(live_temps, "tst", None)


def test_generate_historic_temp_plots_requires_ten_rows() -> None:
    hist_temps = pd.DataFrame(
        {"water_temp": [55.0]},
        index=pd.DatetimeIndex([datetime.datetime(2025, 4, 22, 10, 0, 0)]),
    )

    with pytest.raises(ValueError, match="Insufficient historical temperature data"):
        plot.generate_historic_temp_plots(hist_temps, "tst", None)


def test_historic_yearly_plot_breaks_long_missing_temperature_gaps() -> None:
    early_index = pd.date_range("2025-01-01", periods=48, freq="h")
    late_index = pd.date_range("2025-04-01", periods=48, freq="h")
    hist_temps = pd.DataFrame(
        {"water_temp": [58.0] * len(early_index) + [62.0] * len(late_index)},
        index=early_index.append(late_index),
    )

    fig = plot.create_historic_yearly_plot(hist_temps, "Test Station")
    ax = fig.axes[0]
    y_values = np.asarray(ax.lines[-1].get_ydata(), dtype=float)

    assert np.isnan(y_values).any()
    assert not np.isinf(y_values).any()


def test_multi_year_plot_styles_current_year_as_primary() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="h")
    current_year = util.utc_now().year
    df = pd.DataFrame(
        {
            current_year - 2: [50.0, 51.0, 52.0, 53.0],
            current_year - 1: [51.0, 52.0, 53.0, 54.0],
            current_year: [52.0, 53.0, 54.0, 55.0],
        },
        index=index,
    )

    ax = plot.multi_year_plot(df, plot.create_standard_figure(), "Title", "Subtitle")

    data_lines = ax.lines[:3]
    assert data_lines[0].get_linestyle() != "-"
    assert data_lines[1].get_linestyle() != "-"
    assert data_lines[0].get_linewidth() < data_lines[-1].get_linewidth()
    assert data_lines[1].get_alpha() < data_lines[-1].get_alpha()
    assert data_lines[-1].get_linestyle() == "-"
    assert data_lines[-1].get_linewidth() == 3
    assert data_lines[-1].get_color() == "r"
    legend = ax.get_legend()
    assert legend is not None
    assert legend._loc == 1


def test_multi_year_plot_styles_same_year_consistently_when_columns_differ() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="h")
    current_year = util.utc_now().year
    full_df = pd.DataFrame(
        {
            current_year - 3: [50.0, 51.0, 52.0, 53.0],
            current_year - 2: [51.0, 52.0, 53.0, 54.0],
            current_year: [52.0, 53.0, 54.0, 55.0],
        },
        index=index,
    )
    subset_df = full_df[[current_year - 2, current_year]]

    full_ax = plot.multi_year_plot(
        full_df, plot.create_standard_figure(), "Title", "Subtitle"
    )
    subset_ax = plot.multi_year_plot(
        subset_df, plot.create_standard_figure(), "Title", "Subtitle"
    )

    full_historic_line = full_ax.lines[1]
    subset_historic_line = subset_ax.lines[0]
    assert full_historic_line.get_color() == subset_historic_line.get_color()
    assert full_historic_line.get_linestyle() == subset_historic_line.get_linestyle()


def test_historic_temperature_plot_frame_bridges_short_gaps() -> None:
    index = pd.date_range("2020-01-01", periods=10 * 24, freq="h")
    water_temp_by_year = pd.DataFrame({2025: [60.0] * len(index)}, index=index)
    short_gap = pd.date_range("2020-01-04", periods=24, freq="h")
    water_temp_by_year.loc[short_gap, 2025] = np.nan

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert not plot_frame.loc[short_gap, 2025].isna().any()


def test_historic_temperature_plot_frame_preserves_long_gaps() -> None:
    index = pd.date_range("2020-01-01", periods=120 * 24, freq="h")
    water_temp_by_year = pd.DataFrame({2025: [60.0] * len(index)}, index=index)
    long_gap = pd.date_range("2020-02-01", periods=60 * 24, freq="h")
    water_temp_by_year.loc[long_gap, 2025] = np.nan

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert plot_frame.loc[long_gap, 2025].isna().all()


def test_historic_temperature_plot_frame_suppresses_isolated_spike_artifacts() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame({2025: [50.0] * len(index)}, index=index)
    spike_time = pd.Timestamp("2020-01-15 12:00:00")
    water_temp_by_year.loc[spike_time, 2025] = 80.0

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert plot_frame.loc[spike_time, 2025] == pytest.approx(50.0)


def test_historic_temperature_plot_spike_artifact_counts_by_year() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame(
        {
            2024: [50.0] * len(index),
            2025: [51.0] * len(index),
        },
        index=index,
    )
    water_temp_by_year.loc[pd.Timestamp("2020-01-15 12:00:00"), 2024] = 80.0
    water_temp_by_year.loc[pd.Timestamp("2020-01-16 12:00:00"), 2025] = 20.0
    water_temp_by_year.loc[pd.Timestamp("2020-01-17 12:00:00"), 2025] = 82.0

    counts = plot._historic_temperature_plot_spike_artifact_counts(water_temp_by_year)

    assert counts == {"2024": 1, "2025": 2}


def test_historic_temperature_plot_logs_visual_artifact_counts_by_year(
    caplog: pytest.LogCaptureFixture,
) -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame({2025: [50.0] * len(index)}, index=index)
    water_temp_by_year.loc[pd.Timestamp("2020-01-15 12:00:00"), 2025] = 80.0

    with caplog.at_level(logging.INFO):
        plot._log_historic_temperature_plot_artifact_counts("bos", water_temp_by_year)

    assert (
        "[bos] Historical temperature plot visual artifact suppression flagged "
        "1 raw points by year: {'2025': 1}; "
        "0 cross-year smoothed points by year: {}; "
        "0 volatile smoothed points by year: {}; "
        "0 short-segment smoothed points by year: {}"
    ) in caplog.text


def test_historic_temperature_plot_frame_suppresses_cross_year_artifacts() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame(
        {
            2022: [50.0] * len(index),
            2023: [51.0] * len(index),
            2024: [49.0] * len(index),
            2025: [50.5] * len(index),
        },
        index=index,
    )
    artifact = pd.date_range("2020-01-15", periods=7 * 24, freq="h")
    water_temp_by_year.loc[artifact, 2025] = 65.0

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert plot_frame.loc[artifact, 2025].isna().all()
    assert not plot_frame.loc[artifact, 2022].isna().any()


def test_historic_temperature_plot_cross_year_artifact_counts() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame(
        {
            2022: [50.0] * len(index),
            2023: [51.0] * len(index),
            2024: [49.0] * len(index),
            2025: [50.5] * len(index),
        },
        index=index,
    )
    artifact = pd.date_range("2020-01-15", periods=7 * 24, freq="h")
    water_temp_by_year.loc[artifact, 2025] = 65.0

    counts = plot._historic_temperature_plot_cross_year_artifact_counts(
        water_temp_by_year
    )

    assert counts == {"2025": 182}


def test_historic_temperature_plot_frame_suppresses_volatile_smoothed_segments() -> (
    None
):
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame(
        {
            2022: [50.0] * len(index),
            2023: [51.0] * len(index),
            2024: [49.0] * len(index),
            2025: [50.5] * len(index),
        },
        index=index,
    )
    volatile_segment = pd.date_range("2020-01-15", periods=48, freq="h")
    water_temp_by_year.loc[volatile_segment[:24], 2025] = 42.0
    water_temp_by_year.loc[volatile_segment[24:], 2025] = 58.0

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert plot_frame.loc[volatile_segment, 2025].isna().all()


def test_historic_temperature_plot_volatility_artifact_counts() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame(
        {
            2022: [50.0] * len(index),
            2023: [51.0] * len(index),
            2024: [49.0] * len(index),
            2025: [50.5] * len(index),
        },
        index=index,
    )
    volatile_segment = pd.date_range("2020-01-15", periods=48, freq="h")
    water_temp_by_year.loc[volatile_segment[:24], 2025] = 42.0
    water_temp_by_year.loc[volatile_segment[24:], 2025] = 58.0

    counts = plot._historic_temperature_plot_volatility_artifact_counts(
        water_temp_by_year
    )

    assert counts == {"2025": 83}


def test_historic_temperature_plot_frame_removes_short_orphan_segments() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    water_temp_by_year = pd.DataFrame({2025: [60.0] * len(index)}, index=index)
    orphan = pd.date_range("2020-01-15", periods=4, freq="h")
    water_temp_by_year.loc[: orphan[0] - pd.Timedelta(hours=1), 2025] = np.nan
    water_temp_by_year.loc[orphan[-1] + pd.Timedelta(hours=1) :, 2025] = np.nan

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert plot_frame.loc[orphan, 2025].isna().all()


def test_historic_temperature_short_segment_counts() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    plot_frame = pd.DataFrame({2025: [np.nan] * len(index)}, index=index)
    orphan = pd.date_range("2020-01-15", periods=4, freq="h")
    plot_frame.loc[orphan, 2025] = 60.0

    mask = plot._short_historic_temperature_plot_segment_mask(plot_frame)

    assert int(mask[2025].sum()) == 4


def test_historic_temperature_plot_frame_preserves_gradual_temperature_moves() -> None:
    index = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    values = np.linspace(40.0, 70.0, len(index))
    water_temp_by_year = pd.DataFrame({2025: values}, index=index)

    plot_frame = plot._historic_temperature_plot_frame(water_temp_by_year)

    assert plot_frame[2025].max() > 65.0
    assert plot_frame[2025].min() < 45.0


def test_create_tide_current_plot_requires_enough_tides() -> None:
    location_config = config_lib.LocationConfig(
        code="tst",
        name="Test Location",
        description="Test location for tests",
        latitude=40.7128,
        longitude=-74.0060,
        timezone=datetime.UTC,
        swim_location="Test Beach",
        swim_location_link="https://example.com/test-beach",
    )
    tides = pd.DataFrame(
        {"prediction": [4.2]},
        index=pd.DatetimeIndex([datetime.datetime(2025, 4, 22, 10, 0, 0)]),
    )
    currents = pd.DataFrame(
        {"velocity": [0.5, 0.6]},
        index=pd.DatetimeIndex(
            [
                datetime.datetime(2025, 4, 22, 10, 0, 0),
                datetime.datetime(2025, 4, 22, 11, 0, 0),
            ]
        ),
    )

    with pytest.raises(ValueError, match="Insufficient tide data"):
        plot.create_tide_current_plot(
            tides,
            currents,
            datetime.datetime(2025, 4, 22, 10, 0, 0),
            location_config,
        )


def _test_location_config() -> config_lib.LocationConfig:
    return config_lib.LocationConfig(
        code="tst",
        name="Test Location",
        description="Test location for tests",
        latitude=40.7128,
        longitude=-74.0060,
        timezone=datetime.UTC,
        swim_location="Test Beach",
        swim_location_link="https://example.com/test-beach",
    )


@freeze_time("2026-05-24T12:00:00Z")
def test_create_tide_current_plot_handles_window_without_tide_extrema() -> None:
    location_config = _test_location_config()
    tides = pd.DataFrame(
        {
            "prediction": [0.2, 5.0],
            "type": [types.TideCategory.LOW.value, types.TideCategory.HIGH.value],
        },
        index=pd.DatetimeIndex(
            [
                datetime.datetime(2026, 5, 24, 6, 0, 0),
                datetime.datetime(2026, 5, 24, 18, 0, 0),
            ]
        ),
    )
    current_index = pd.date_range("2026-05-24T06:00:00", periods=13, freq="h")
    currents = pd.DataFrame(
        {"velocity": np.sin(np.linspace(-np.pi, np.pi, len(current_index)))},
        index=current_index,
    )

    fig = plot.create_tide_current_plot(
        tides,
        currents,
        datetime.datetime(2026, 5, 24, 12, 0, 0),
        location_config,
    )

    fig_to_svg = io.StringIO()
    fig.savefig(fig_to_svg, format="svg")
    assert "<svg" in fig_to_svg.getvalue()


@freeze_time("2026-05-24T12:00:00Z")
def test_create_tide_current_plot_handles_non_minute_aligned_tides() -> None:
    location_config = _test_location_config()
    tides = pd.DataFrame(
        {
            "prediction": [0.2, 5.0, 0.9],
            "type": [
                types.TideCategory.LOW.value,
                types.TideCategory.HIGH.value,
                types.TideCategory.LOW.value,
            ],
        },
        index=pd.DatetimeIndex(
            [
                datetime.datetime(2026, 5, 24, 6, 0, 14, 123456),
                datetime.datetime(2026, 5, 24, 15, 0, 14, 123456),
                datetime.datetime(2026, 5, 24, 21, 0, 14, 123456),
            ]
        ),
    )
    current_index = pd.date_range("2026-05-24T06:00:00", periods=20, freq="h")
    currents = pd.DataFrame(
        {"velocity": np.sin(np.linspace(-np.pi, np.pi, len(current_index)))},
        index=current_index,
    )

    fig = plot.create_tide_current_plot(
        tides,
        currents,
        datetime.datetime(2026, 5, 24, 12, 0, 0),
        location_config,
    )

    fig_to_svg = io.StringIO()
    fig.savefig(fig_to_svg, format="svg")
    assert "<svg" in fig_to_svg.getvalue()


@freeze_time("2026-05-24T12:00:00Z")
def test_create_tide_current_plot_places_high_tide_labels_above_peaks() -> None:
    location_config = _test_location_config()
    tides = pd.DataFrame(
        {
            "prediction": [0.2, 5.0, 0.9],
            "type": [
                types.TideCategory.LOW.value,
                types.TideCategory.HIGH.value,
                types.TideCategory.LOW.value,
            ],
        },
        index=pd.DatetimeIndex(
            [
                datetime.datetime(2026, 5, 24, 6, 0, 0),
                datetime.datetime(2026, 5, 24, 15, 0, 0),
                datetime.datetime(2026, 5, 24, 21, 0, 0),
            ]
        ),
    )
    current_index = pd.date_range("2026-05-24T06:00:00", periods=20, freq="h")
    currents = pd.DataFrame(
        {"velocity": np.sin(np.linspace(-np.pi, np.pi, len(current_index)))},
        index=current_index,
    )

    fig = plot.create_tide_current_plot(
        tides,
        currents,
        datetime.datetime(2026, 5, 24, 12, 0, 0),
        location_config,
    )

    tide_annotations = {
        text.get_text(): text for text in fig.axes[1].texts if " ft " in text.get_text()
    }

    assert tide_annotations["5.0 ft high"].xyann == (0, 14)
    assert tide_annotations["5.0 ft high"].get_va() == "bottom"
    assert tide_annotations["0.9 ft low"].xyann == (0, -14)
    assert tide_annotations["0.9 ft low"].get_va() == "top"


def test_tide_current_plot_window_preserves_default_forward_horizon() -> None:
    now = datetime.datetime(2026, 5, 24, 9, 0, 0)
    marker_time = now + datetime.timedelta(hours=12)

    start_time, end_time = plot._tide_current_plot_window(now, marker_time)

    assert start_time == now - datetime.timedelta(hours=3)
    assert end_time == now + datetime.timedelta(hours=21)


def test_tide_current_plot_window_keeps_now_and_late_planner_time_visible() -> None:
    now = datetime.datetime(2026, 5, 24, 9, 0, 0)
    marker_time = now + datetime.timedelta(hours=24)

    start_time, end_time = plot._tide_current_plot_window(now, marker_time)

    assert start_time == now - datetime.timedelta(hours=3)
    assert end_time == marker_time + datetime.timedelta(hours=3)


@pytest.mark.parametrize(
    ("offset_hours", "expected_start_hours", "expected_end_hours"),
    [(-3, -6, 21), (21, -3, 24)],
)
def test_tide_current_plot_window_tracks_planner_edges_without_collapsing_horizon(
    offset_hours: int,
    expected_start_hours: int,
    expected_end_hours: int,
) -> None:
    now = datetime.datetime(2026, 5, 24, 9, 0, 0)
    marker_time = now + datetime.timedelta(hours=offset_hours)

    start_time, end_time = plot._tide_current_plot_window(now, marker_time)

    assert start_time == now + datetime.timedelta(hours=expected_start_hours)
    assert end_time == now + datetime.timedelta(hours=expected_end_hours)


@pytest.mark.parametrize(
    "offset",
    [
        datetime.timedelta(hours=-3, seconds=-30),
        datetime.timedelta(hours=21, seconds=30),
    ],
)
def test_tide_current_plot_window_tracks_request_time_drift_smoothly(
    offset: datetime.timedelta,
) -> None:
    now = datetime.datetime(2026, 5, 24, 9, 0, 0)
    marker_time = now + offset

    start_time, end_time = plot._tide_current_plot_window(now, marker_time)

    assert start_time == min(now, marker_time) - datetime.timedelta(hours=3)
    assert end_time == max(
        now + datetime.timedelta(hours=21),
        marker_time + datetime.timedelta(hours=3),
    )


def test_create_current_chart_rejects_out_of_range_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude_bin must be between"):
        plot.create_current_chart("flood", 101)

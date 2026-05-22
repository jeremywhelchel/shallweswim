"""Tests for plotting validation helpers."""

import datetime

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from shallweswim import config as config_lib
from shallweswim import plot


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


def test_create_current_chart_rejects_out_of_range_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude_bin must be between"):
        plot.create_current_chart("flood", 101)

"""Tests for plotting validation helpers."""

import datetime

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

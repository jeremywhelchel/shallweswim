# pylint: disable=duplicate-code
import datetime

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from shallweswim import util
from shallweswim.api_types import DataFrameSummary

# Constants
EXPECTED_COLUMNS = ["value", "flag"]
EXPECTED_DTYPES: dict[str, type | str] = {
    "value": np.float64,
    "flag": "int64",
}


def test_now() -> None:
    """Test that utc_now() returns naive datetime without timezone information."""
    now = util.utc_now()
    assert isinstance(now, datetime.datetime)
    assert now.tzinfo is None  # Should be naive (no timezone info)

    # Test that the time is within a reasonable range of UTC time
    # Get naive UTC time for comparison
    utc_now = datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)
    time_diff = abs((utc_now - now).total_seconds())
    assert time_diff < 1  # Should be less than 1 second difference


@pytest.mark.parametrize(
    "fahrenheit,expected_celsius",
    [
        (32, 0),  # Freezing point
        (212, 100),  # Boiling point
        (98.6, 37),  # Body temperature (rounded)
        (-40, -40),  # Same in both scales
        (68, 20),  # Room temperature
    ],
)
def test_f2c(fahrenheit: float, expected_celsius: float) -> None:
    """Test Fahrenheit to Celsius conversion."""
    result = util.f_to_c(fahrenheit)
    assert abs(result - expected_celsius) < 0.1  # within 0.1 degree


@pytest.mark.parametrize(
    "celsius,expected_fahrenheit",
    [
        (0, 32),  # Freezing point
        (100, 212),  # Boiling point
        (37, 98.6),  # Body temperature
        (-40, -40),  # Same in both scales
        (20, 68),  # Room temperature
    ],
)
def test_c2f(celsius: float, expected_fahrenheit: float) -> None:
    """Test Celsius to Fahrenheit conversion."""
    result = util.c_to_f(celsius)
    assert abs(result - expected_fahrenheit) < 0.1  # within 0.1 degree


def test_pivot_year() -> None:
    df = pd.DataFrame(
        {"air_temp": 60, "water_temp": 50},
        index=pd.date_range("2011-01-01", "2023-12-31", freq="D"),
    )
    got = util.pivot_year(df)

    pd.testing.assert_index_equal(pd.date_range("2020-01-01", "2020-12-31"), got.index)

    cols = ["air_temp", "water_temp"]
    years = pd.Series(range(2011, 2023 + 1), dtype="int32")
    pd.testing.assert_index_equal(
        pd.MultiIndex.from_product([cols, years], names=[None, "year"]), got.columns
    )
    # Check that all values in air_temp are either 60 or NaN
    for year in range(2011, 2023 + 1):
        # Use bool() to explicitly convert the result to a boolean to satisfy the type checker
        assert bool(got[("air_temp", year)].isin([60, np.nan]).all())
        assert bool(got[("water_temp", year)].isin([50, np.nan]).all())


def test_latest_time_value() -> None:
    """Test that latest_time_value correctly extracts the timestamp from a DataFrame."""
    # No longer testing None input as the function now requires a non-None input

    # Test with a DataFrame with naive datetime index
    dates = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)
    result = util.latest_time_value(df)
    assert result == datetime.datetime(2025, 1, 10)
    assert result.tzinfo is None  # Should be naive

    # Test with a DataFrame with timezone-aware datetime index - should raise ValueError
    tz = datetime.timezone(datetime.timedelta(hours=-5))  # EST
    dates_tz = pd.date_range("2025-01-01", "2025-01-10", freq="D", tz=tz)
    df_tz = pd.DataFrame({"value": range(len(dates_tz))}, index=dates_tz)
    with pytest.raises(ValueError, match="Index contains timezone info"):
        util.latest_time_value(df_tz)


# Test data setup for summarize_dataframe
basic_dates = pd.to_datetime(
    ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]
)
basic_df = pd.DataFrame(
    {"temp": [10, 11, 12], "humidity": [50, 51, 52]}, index=basic_dates
)
nan_df = pd.DataFrame(
    {"temp": [10, np.nan, 12], "humidity": [np.nan, 51, np.nan]}, index=basic_dates
)
empty_df = pd.DataFrame()
non_dt_index_df = pd.DataFrame({"temp": [10, 11, 12], "humidity": [50, 51, 52]})

summarize_test_cases = [
    pytest.param(
        basic_df,
        DataFrameSummary(
            length=3,
            width=2,
            column_names=["temp", "humidity"],
            index_oldest=datetime.datetime(2024, 1, 1, 10, 0),
            index_newest=datetime.datetime(2024, 1, 1, 12, 0),
            missing_values={"temp": 0, "humidity": 0},
            index_frequency="h",  # Hourly frequency (lowercase from pandas)
            memory_usage_bytes=72,
        ),
        id="basic_datetime_index",
    ),
    pytest.param(
        nan_df,
        DataFrameSummary(
            length=3,
            width=2,
            column_names=["temp", "humidity"],
            index_oldest=datetime.datetime(2024, 1, 1, 10, 0),
            index_newest=datetime.datetime(2024, 1, 1, 12, 0),
            missing_values={"temp": 1, "humidity": 2},
            index_frequency="h",  # Hourly frequency (lowercase from pandas)
            memory_usage_bytes=72,
        ),
        id="with_nan",
    ),
    pytest.param(
        empty_df,
        DataFrameSummary(
            length=0,
            width=0,
            column_names=[],
            index_oldest=None,
            index_newest=None,
            missing_values={},
            index_frequency=None,
            memory_usage_bytes=0,
        ),
        id="empty_dataframe",
    ),
    pytest.param(
        None,
        DataFrameSummary(
            length=0,
            width=0,
            column_names=[],
            index_oldest=None,
            index_newest=None,
            missing_values={},
            index_frequency=None,
            memory_usage_bytes=0,
        ),
        id="none_input",
    ),
    pytest.param(
        non_dt_index_df,
        DataFrameSummary(
            length=3,
            width=2,
            column_names=["temp", "humidity"],
            index_oldest=None,
            index_newest=None,
            missing_values={"temp": 0, "humidity": 0},
            index_frequency=None,  # Not a DatetimeIndex
            memory_usage_bytes=180,
        ),
        id="non_datetime_index",
    ),
]


@freeze_time("2024-01-15 12:00:00 UTC")
@pytest.mark.parametrize("df_input, expected_summary", summarize_test_cases)
def test_summarize_dataframe(
    df_input: pd.DataFrame | None, expected_summary: DataFrameSummary
) -> None:
    """Test summarize_dataframe with various inputs using parametrization."""
    summary = util.summarize_dataframe(df_input)
    assert summary == expected_summary


def test_fps_to_knots() -> None:
    """Test conversion from feet per second to knots."""
    # Test zero
    assert util.fps_to_knots(0.0) == pytest.approx(0.0)

    # Test base conversion factor
    assert util.fps_to_knots(1.68781) == pytest.approx(1.0)

    # Test another value
    assert util.fps_to_knots(5.0) == pytest.approx(5.0 / 1.68781)

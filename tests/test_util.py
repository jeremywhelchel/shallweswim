# pylint: disable=duplicate-code
import datetime
import re
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from shallweswim import util
from shallweswim.types import DataFrameSummary
from shallweswim.util import DATETIME_INDEX_NAME, DataFrameValidationError

# Constants
EXPECTED_COLUMNS = ["value", "flag"]
EXPECTED_DTYPES: Dict[str, Union[type, str]] = {
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
    utc_now = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
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
    assert got["air_temp"].isin([60, np.nan]).all().all()
    assert got["water_temp"].isin([50, np.nan]).all().all()


def test_latest_time_value() -> None:
    """Test that latest_time_value correctly extracts the timestamp from a DataFrame."""
    # Test with None input
    assert util.latest_time_value(None) is None

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
    with pytest.raises(ValueError, match="DataFrame index contains timezone info"):
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
        ),
        id="non_datetime_index",
    ),
]


@freeze_time("2024-01-15 12:00:00 UTC")
@pytest.mark.parametrize("df_input, expected_summary", summarize_test_cases)
def test_summarize_dataframe(
    df_input: Optional[pd.DataFrame], expected_summary: DataFrameSummary
) -> None:
    """Test summarize_dataframe with various inputs using parametrization."""
    summary = util.summarize_dataframe(df_input)
    assert summary == expected_summary


# --- Tests for validate_timeseries_dataframe ---


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Fixture for a valid timeseries DataFrame using ALLOWED columns/dtypes."""
    index = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"], utc=False)
    # Use columns and dtypes defined in util.py
    df = pd.DataFrame({"water_temp": [15.5, 16.0], "velocity": [5.1, 5.5]}, index=index)
    # Ensure correct dtypes
    df = df.astype(util.ALLOWED_TIMESERIES_DTYPES)
    df.index.name = DATETIME_INDEX_NAME
    return df


def test_validate_timeseries_dataframe_valid(valid_df: pd.DataFrame) -> None:
    """Test validation passes for a valid DataFrame."""
    try:
        util.validate_timeseries_dataframe(valid_df)
    except DataFrameValidationError as e:
        pytest.fail(f"Validation unexpectedly failed: {e}")


def test_validate_timeseries_dataframe_invalid_index_type(
    valid_df: pd.DataFrame,
) -> None:
    """Test validation fails for non-DatetimeIndex."""
    df = valid_df.reset_index()
    with pytest.raises(DataFrameValidationError, match="Index is not a DatetimeIndex"):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_unsorted_index(valid_df: pd.DataFrame) -> None:
    """Test validation fails for unsorted index."""
    df = valid_df.sort_index(ascending=False)
    with pytest.raises(DataFrameValidationError, match="Index is not sorted"):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_wrong_index_name(valid_df: pd.DataFrame) -> None:
    """Test validation fails for incorrect index name."""
    df = valid_df.copy()
    df.index.name = "wrong_name"
    # Match the specific error message format
    expected_msg = f"Index name is 'wrong_name', expected '{DATETIME_INDEX_NAME}'"
    with pytest.raises(DataFrameValidationError, match=re.escape(expected_msg)):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_tz_aware_index(valid_df: pd.DataFrame) -> None:
    """Test validation fails for timezone-aware index."""
    df = valid_df.tz_localize("UTC")
    with pytest.raises(DataFrameValidationError, match="Index is timezone-aware"):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_wrong_columns(valid_df: pd.DataFrame) -> None:
    """Test validation fails for incorrect columns."""
    # Test with missing allowed column
    df_missing = valid_df.drop(columns=["velocity"])
    expected_msg_missing = "DataFrame is missing required columns: ['velocity']"
    with pytest.raises(DataFrameValidationError, match=re.escape(expected_msg_missing)):
        util.validate_timeseries_dataframe(df_missing)

    # Test with extra disallowed column
    df_extra = valid_df.copy()
    df_extra["extra_col"] = 100
    expected_msg_extra = "DataFrame contains disallowed columns: ['extra_col']"
    with pytest.raises(DataFrameValidationError, match=re.escape(expected_msg_extra)):
        util.validate_timeseries_dataframe(df_extra)


def test_validate_timeseries_dataframe_wrong_dtype(valid_df: pd.DataFrame) -> None:
    """Test validation fails for incorrect column dtype."""
    # Change 'water_temp' to string
    df = valid_df.copy()
    df["water_temp"] = df["water_temp"].astype(str)
    expected_dtype = util.ALLOWED_TIMESERIES_DTYPES["water_temp"]
    # Match the specific error message format including actual and expected types
    expected_msg = f"Column 'water_temp' has incorrect dtype. Got: object, Expected: {expected_dtype}"
    with pytest.raises(DataFrameValidationError, match=re.escape(expected_msg)):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_empty_df() -> None:
    """Test validation with an empty DataFrame (should pass structural checks)."""
    index = pd.to_datetime([]).rename(DATETIME_INDEX_NAME)
    # Use columns and dtypes from util.py
    df = pd.DataFrame(columns=util.ALLOWED_TIMESERIES_COLUMNS, index=index)
    df = df.astype(util.ALLOWED_TIMESERIES_DTYPES)
    # Explicitly ensure index type after potential casting
    df.index = pd.to_datetime(df.index)
    df.index.name = DATETIME_INDEX_NAME

    try:
        util.validate_timeseries_dataframe(df)
    except DataFrameValidationError as e:
        pytest.fail(f"Validation unexpectedly failed for empty DataFrame: {e}")

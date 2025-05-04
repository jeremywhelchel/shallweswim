# pylint: disable=duplicate-code
import datetime
import re
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pytest
import pandera as pa
from freezegun import freeze_time

from shallweswim import util
from shallweswim.types import DataFrameSummary

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
    df_input: Optional[pd.DataFrame], expected_summary: DataFrameSummary
) -> None:
    """Test summarize_dataframe with various inputs using parametrization."""
    summary = util.summarize_dataframe(df_input)
    assert summary == expected_summary


# --- Tests for validate_timeseries_dataframe ---
@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Fixture for a valid timeseries DataFrame using ALLOWED columns/dtypes."""
    dates = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"])
    df = pd.DataFrame(
        {
            # Use columns defined in TimeSeriesDataModel
            "water_temp": [10.1, 11.2, 12.3],
            "velocity": [0.5, 0.6, 0.7],
            "prediction": [10.0, 11.0, 12.0],
            "type": ["low", "high", "low"],  # Added type column with valid values
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )
    # Ensure dtypes match schema expectations (float for numeric, str/object for type)
    df = df.astype(
        {
            "water_temp": np.float64,
            "velocity": np.float64,
            "prediction": np.float64,
            "type": str,
        }
    )
    return df


def test_validate_timeseries_dataframe_valid(valid_df: pd.DataFrame) -> None:
    """Test validation passes for a valid DataFrame."""
    try:
        util.validate_timeseries_dataframe(valid_df)
    except pa.errors.SchemaError as e:
        pytest.fail(f"Validation unexpectedly failed: {e}")


def test_validate_timeseries_dataframe_invalid_index_type(
    valid_df: pd.DataFrame,
) -> None:
    """Test validation fails for non-DatetimeIndex."""
    # Create df with a simple RangeIndex instead of DatetimeIndex
    df = valid_df.copy()
    df = df.reset_index(drop=True)
    # Schema fails because index name is wrong (expected 'time', found 'None')
    with pytest.raises(
        pa.errors.SchemaError, match=r"Expected .* to have name 'time', found 'None'"
    ):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_wrong_index_name(valid_df: pd.DataFrame) -> None:
    """Test validation fails for incorrect index name."""
    df = valid_df.copy()
    df.index.name = "wrong_name"
    # Pandera error message for wrong index name
    expected_msg = r"Expected .* name 'time', found 'wrong_name'"
    with pytest.raises(pa.errors.SchemaError, match=expected_msg):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_tz_aware_index(valid_df: pd.DataFrame) -> None:
    """Test validation fails for timezone-aware index."""
    # Create df with tz-aware index directly
    df = valid_df.copy()
    # Explicitly reset index first to ensure no prior duplicates interfere?
    # df = df.reset_index().set_index("time")
    df.index = df.index.tz_localize("UTC")  # Localize the valid index
    # Pandera fails because the dtype includes timezone information
    with pytest.raises(
        pa.errors.SchemaError,
        match=r"expected series 'time' to have type datetime64.* got datetime64.*UTC",
    ):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_wrong_columns(valid_df: pd.DataFrame) -> None:
    """Test validation fails for incorrect columns."""
    # Test with extra disallowed column (due to strict=True)
    df_extra = valid_df.copy()
    df_extra["extra_col"] = 100
    # Pandera's strict check error
    expected_msg_extra = "column 'extra_col' not in DataFrameSchema"
    with pytest.raises(pa.errors.SchemaError, match=re.escape(expected_msg_extra)):
        util.validate_timeseries_dataframe(df_extra)


def test_validate_timeseries_dataframe_wrong_dtype(valid_df: pd.DataFrame) -> None:
    """Test validation fails for incorrect column dtype."""
    df = valid_df.copy()
    df["water_temp"] = df["water_temp"].astype(str)
    # Pandera's dtype error message
    expected_msg = "expected series 'water_temp' to have type float"
    with pytest.raises(pa.errors.SchemaError, match=expected_msg):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_empty_df() -> None:
    """Test validation fails if the DataFrame is empty."""
    empty_df = pd.DataFrame(
        index=pd.DatetimeIndex([], name="time"),
        # Define columns matching the schema for an empty df check
        columns=["water_temp", "velocity", "prediction", "type"],
    ).astype(
        {  # Set dtypes to avoid dtype errors on empty df
            "water_temp": np.float64,
            "velocity": np.float64,
            "prediction": np.float64,
            "type": str,
        }
    )

    # For an empty DF, the column check 'not isnull().all()' fails first
    with pytest.raises(pa.errors.SchemaError, match=r"Column 'water_temp' .* all NaN"):
        util.validate_timeseries_dataframe(empty_df)


def test_validate_timeseries_index_duplicate_index(valid_df: pd.DataFrame) -> None:
    """Test validation fails for an index with duplicate timestamps."""
    df = valid_df.copy()
    duplicate_row = df.iloc[0:1].copy()
    df = pd.concat([df, duplicate_row])
    df.index.name = "time"  # Ensure index name is preserved
    # The pa.Field(unique=True) check now catches duplicates directly.
    with pytest.raises(
        pa.errors.SchemaError, match=r"series 'time' contains duplicate values"
    ):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_index_nat_index(valid_df: pd.DataFrame) -> None:
    """Test validation fails for an index with NaT values."""
    df = valid_df.copy()
    index_list = df.index.to_list()
    index_list[0] = pd.NaT
    df.index = pd.DatetimeIndex(index_list)
    df.index.name = "time"  # Ensure index name is preserved
    # Pandera raises this error because the index is non-nullable
    with pytest.raises(
        pa.errors.SchemaError, match=r"non-nullable series 'time' contains null values"
    ):
        util.validate_timeseries_dataframe(df)


def test_validate_timeseries_dataframe_columns_all_nan(valid_df: pd.DataFrame) -> None:
    """Test validation fails if a column contains only NaN values."""
    df = valid_df.copy()
    nan_col_name = "water_temp"
    df[nan_col_name] = np.nan
    df = df.astype({nan_col_name: np.float64})  # Ensure dtype stays float

    # Test the main validation function, match specific check error message
    with pytest.raises(
        pa.errors.SchemaError,
        match=f"{re.escape(nan_col_name)} all NaN",
    ):
        util.validate_timeseries_dataframe(df)


# Added test for invalid 'type' value
def test_validate_timeseries_dataframe_invalid_type_value(
    valid_df: pd.DataFrame,
) -> None:
    """Test validation fails if 'type' column has invalid values."""
    df = valid_df.copy()
    df.loc[df.index[0], "type"] = "medium"  # Invalid value

    # Match the specific error message format from pandera for isin check failures
    with pytest.raises(pa.errors.SchemaError, match=r"isin.*failure cases: medium"):
        util.validate_timeseries_dataframe(df)

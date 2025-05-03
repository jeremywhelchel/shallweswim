"""Shared utilities."""

import datetime
from typing import Dict, List, Optional, Union, cast
import pandas as pd
import numpy as np
from shallweswim.types import DataFrameSummary

# Constants
DATETIME_INDEX_NAME = (
    "time"  # Standard name for the datetime index in internal DataFrames
)

# Define the standard allowed columns and their expected dtypes for timeseries data
ALLOWED_TIMESERIES_COLUMNS: List[str] = [
    "water_temp",
    "velocity",
]
ALLOWED_TIMESERIES_DTYPES: Dict[str, Union[type, str]] = {
    "water_temp": np.float64,
    "velocity": np.float64,
}


class DataFrameValidationError(ValueError):
    """Custom exception for DataFrame validation errors."""


# Time shift limits for current predictions (in minutes)
MAX_SHIFT_LIMIT = 1440  # 24 hours forward
MIN_SHIFT_LIMIT = -1440  # 24 hours backward


def utc_now() -> datetime.datetime:
    """Returns the current time in UTC as a naive datetime (without timezone information).

    All timestamps in the application are naive datetimes in their respective timezones.
    For NOAA data, timestamps are in local time based on the station's location.
    """
    # Get timezone-aware UTC time, then strip the timezone to make it naive
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def effective_time(
    timezone: datetime.tzinfo, shift_minutes: int = 0
) -> datetime.datetime:
    """Calculate the effective time with an optional shift, in the specified timezone.

    Args:
        timezone: Timezone to convert the time to (required)
        shift_minutes: Number of minutes to shift from current time

    Returns:
        Naive datetime with the shift applied, in the specified timezone with tzinfo removed
    """
    # Get current time (as a timezone-aware datetime)
    now = datetime.datetime.now(datetime.timezone.utc)

    # Convert to the location's timezone
    now = now.astimezone(timezone)

    # Apply the time shift to the location's local time
    if shift_minutes:
        # Clamp the shift limit
        shift_minutes = max(MIN_SHIFT_LIMIT, min(shift_minutes, MAX_SHIFT_LIMIT))

        delta = datetime.timedelta(minutes=shift_minutes)
        now = now + delta

    # Remove timezone info to return naive datetime
    return now.replace(tzinfo=None)


def f_to_c(temp: float) -> float:
    """Convert Fahrenheit temp to Celsius."""
    return (5.0 / 9.0) * (temp - 32)


def c_to_f(temp: float) -> float:
    """Convert Celsius temp to Fahrenheit."""
    return (9.0 / 5.0) * temp + 32


def pivot_year(df: pd.DataFrame) -> pd.DataFrame:
    """Move year dimension to columns."""
    df = df.assign(year=df.index.year)
    df.index = pd.to_datetime(
        # Use 2020-indexing because it's a leap year
        df.index.strftime("2020-%m-%d %H:%M:%S")
    )
    return df.set_index("year", append=True).unstack("year")


def latest_time_value(df: Optional[pd.DataFrame]) -> Optional[datetime.datetime]:
    """Extract the timestamp of the most recent data point from a DataFrame.

    Args:
        df: DataFrame with DatetimeIndex, or None

    Returns:
        Timezone-naive datetime object of the last index value,
        or None if DataFrame is None

    Raises:
        ValueError: If the DataFrame index contains timezone information
    """
    if df is None:
        return None
    # Get the datetime from the DataFrame index
    dt = df.index[-1].to_pydatetime()
    # Assert that the datetime is already naive
    if dt.tzinfo is not None:
        raise ValueError(
            "DataFrame index contains timezone info; expected naive datetime"
        )
    return cast(datetime.datetime, dt)


def summarize_dataframe(df: Optional[pd.DataFrame]) -> DataFrameSummary:
    """Generates a summary object for a given pandas DataFrame.

    Args:
        df: The pandas DataFrame to summarize. Can be None.

    Returns:
        A DataFrameSummary object containing statistics about the DataFrame.
    """
    if df is None or df.empty:
        return DataFrameSummary(
            length=0,
            width=0,
            column_names=[],
            index_oldest=None,
            index_newest=None,
            missing_values={},
        )

    length = len(df)
    width = len(df.columns)
    column_names = df.columns.tolist()

    # Calculate missing values (convert Series result to dict)
    missing_values = df.isnull().sum().astype(int).to_dict()

    # Get index min/max, checking if index is DatetimeIndex
    index_oldest: Optional[datetime.datetime] = None
    index_newest: Optional[datetime.datetime] = None
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
        # Ensure pandas Timestamps are converted to standard Python datetimes
        # Use .to_pydatetime() which handles potential NaT values gracefully (returns None)
        min_ts = df.index.min()
        max_ts = df.index.max()
        index_oldest = min_ts.to_pydatetime() if pd.notna(min_ts) else None
        index_newest = max_ts.to_pydatetime() if pd.notna(max_ts) else None

    return DataFrameSummary(
        length=length,
        width=width,
        column_names=column_names,
        index_oldest=index_oldest,
        index_newest=index_newest,
        missing_values=missing_values,
    )


def validate_timeseries_dataframe(df: pd.DataFrame) -> None:
    """Validate the structure and content of an internal timeseries DataFrame.

    This function orchestrates checks for both the index and columns by calling:
        - validate_timeseries_index
        - validate_timeseries_dataframe_columns

    Args:
        df: The pandas DataFrame to validate.

    Raises:
        DataFrameValidationError: If any validation check fails.
    """
    validate_timeseries_index(df.index)
    validate_timeseries_dataframe_columns(df)


def validate_timeseries_index(index: pd.Index) -> None:
    """Validate the index of an internal timeseries DataFrame.

    Checks:
        - Index is a pd.DatetimeIndex
        - Index is monotonically increasing (sorted).
        - Index name is DATETIME_INDEX_NAME ('time').
        - Index is timezone-naive.

    Args:
        index: The pandas Index to validate.

    Raises:
        DataFrameValidationError: If any validation check fails.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise DataFrameValidationError("Index is not a DatetimeIndex.")

    if not index.empty and not index.is_monotonic_increasing:
        raise DataFrameValidationError("Index is not sorted monotonically increasing.")

    if index.name != DATETIME_INDEX_NAME:
        raise DataFrameValidationError(
            f"Index name is '{index.name}', expected '{DATETIME_INDEX_NAME}'."
        )

    if index.tz is not None:
        raise DataFrameValidationError(
            f"Index is timezone-aware ({index.tz}), expected timezone-naive."
        )


def validate_timeseries_dataframe_columns(df: pd.DataFrame) -> None:
    """Validate the columns and dtypes of an internal timeseries DataFrame.

    Checks against the predefined ALLOWED_TIMESERIES_COLUMNS and ALLOWED_TIMESERIES_DTYPES.
        - All DataFrame columns *present* must be in ALLOWED_TIMESERIES_COLUMNS.
        - Each *present* column's dtype must match the corresponding type in ALLOWED_TIMESERIES_DTYPES.

    Args:
        df: The pandas DataFrame to validate.

    Raises:
        DataFrameValidationError: If any validation check fails.
    """
    # Check columns - ensure all present columns are allowed
    present_columns = set(df.columns)
    allowed_columns_set = set(ALLOWED_TIMESERIES_COLUMNS)
    disallowed_columns = present_columns - allowed_columns_set
    if disallowed_columns:
        raise DataFrameValidationError(
            f"DataFrame contains disallowed columns: {sorted(list(disallowed_columns))}. "
            f"Allowed columns are: {ALLOWED_TIMESERIES_COLUMNS}"
        )

    # Check dtypes for present columns
    for col in df.columns:
        # Skip check if column somehow isn't in the DTYPES dict (shouldn't happen if disallowed check passed)
        if col not in ALLOWED_TIMESERIES_DTYPES:
            continue

        expected_dtype = ALLOWED_TIMESERIES_DTYPES[col]
        actual_dtype = df[col].dtype
        if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
            raise DataFrameValidationError(
                f"Column '{col}' has incorrect dtype. Got: {actual_dtype}, "
                f"Expected: {expected_dtype}"
            )

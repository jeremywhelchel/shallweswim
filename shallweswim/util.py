"""Shared utilities."""

# Standard library imports
import datetime
from typing import cast

# Third-party imports
import pandas as pd

# Local application imports
from shallweswim import api_types

# Time shift limits for current predictions (in minutes)
MAX_SHIFT_LIMIT = 1440  # 24 hours forward
MIN_SHIFT_LIMIT = -1440  # 24 hours backward


def utc_now() -> datetime.datetime:
    """Returns the current time in UTC as a naive datetime (without timezone information).

    All timestamps in the application are naive datetimes in their respective timezones.
    For NOAA data, timestamps are in local time based on the station's location.
    """
    # Get timezone-aware UTC time, then strip the timezone to make it naive
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


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
    now = datetime.datetime.now(datetime.UTC)

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


def fps_to_knots(speed_fps: float) -> float:
    """Convert speed from feet per second (fps) to knots."""
    # 1 knot = 1.68781 feet per second
    return speed_fps / 1.68781


def pivot_year(df: pd.DataFrame) -> pd.DataFrame:
    """Move year dimension to columns."""
    # Access year from DatetimeIndex in a way compatible with pandas 2.x
    assert isinstance(df.index, pd.DatetimeIndex), (
        "DataFrame index must be a DatetimeIndex"
    )
    df = df.assign(year=df.index.to_series().dt.year)
    df.index = pd.to_datetime(
        # Use 2020-indexing because it's a leap year
        df.index.to_series().dt.strftime("2020-%m-%d %H:%M:%S")
    )
    result = df.set_index("year", append=True).unstack("year")
    return cast(pd.DataFrame, result)


def latest_time_value(df: pd.DataFrame | pd.Series) -> datetime.datetime:
    """Extract the timestamp of the most recent data point from a DataFrame or Series.

    Args:
        df: DataFrame or Series with DatetimeIndex

    Returns:
        Timezone-naive datetime object of the last index value

    Raises:
        ValueError: If the index contains timezone information
        TypeError: If the index is not a DatetimeIndex
    """
    # Check if the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    # Get the last timestamp directly from the index
    # Use cast to tell the type checker to treat it as a datetime
    ts = df.index[-1]  # pandas.Timestamp
    dt = cast(datetime.datetime, ts)  # tell Pyright "treat this as datetime"

    # Assert that the datetime is already naive
    if dt.tzinfo is not None:
        raise ValueError("Index contains timezone info; expected naive datetime")
    return dt


def summarize_dataframe(df: pd.DataFrame | None) -> api_types.DataFrameSummary:
    """Generates a summary object for a given pandas DataFrame.

    Args:
        df: The pandas DataFrame to summarize. Can be None.

    Returns:
        A DataFrameSummary object containing statistics about the DataFrame.
    """
    if df is None or df.empty:
        return api_types.DataFrameSummary(
            length=0,
            index_frequency=None,
            width=0,
            column_names=[],
            index_oldest=None,
            index_newest=None,
            missing_values={},
            memory_usage_bytes=0,
        )

    length = len(df)
    width = len(df.columns)
    column_names = df.columns.tolist()

    # Infer index frequency
    index_frequency: str | None = None
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
        index_frequency = pd.infer_freq(df.index)

    # Calculate missing values (convert Series result to dict)
    missing_values = df.isnull().sum().astype(int).to_dict()

    # Get index min/max, checking if index is DatetimeIndex
    index_oldest: datetime.datetime | None = None
    index_newest: datetime.datetime | None = None
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
        # Get min and max timestamps from the index
        min_ts = df.index.min()
        max_ts = df.index.max()

        # Use cast to tell the type checker to treat pandas Timestamps as datetime objects
        # Handle NaT (Not a Time) values by checking if they're valid
        # pd.NaT is treated as False in boolean context
        index_oldest = cast(datetime.datetime, min_ts) if min_ts is not pd.NaT else None
        index_newest = cast(datetime.datetime, max_ts) if max_ts is not pd.NaT else None

    # Calculate memory usage (deep=True for accurate object dtype size)
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())

    return api_types.DataFrameSummary(
        length=length,
        index_frequency=index_frequency,
        width=width,
        column_names=column_names,
        index_oldest=index_oldest,
        index_newest=index_newest,
        missing_values=missing_values,
        memory_usage_bytes=memory_usage_bytes,
    )

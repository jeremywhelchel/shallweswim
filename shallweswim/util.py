"""Shared utilities."""

# Standard library imports
import datetime
from typing import Optional, cast

# Third-party imports
import pandas as pd
import pandera as pa
from pandera import Column, Check
from pandera.engines.pandas_engine import DateTime

# Local application imports
from shallweswim.types import DataFrameSummary

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
            index_frequency=None,
            width=0,
            column_names=[],
            index_oldest=None,
            index_newest=None,
            missing_values={},
        )

    length = len(df)
    width = len(df.columns)
    column_names = df.columns.tolist()

    # Infer index frequency
    index_frequency: Optional[str] = None
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
        index_frequency = pd.infer_freq(df.index)

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
        index_frequency=index_frequency,
        width=width,
        column_names=column_names,
        index_oldest=index_oldest,
        index_newest=index_newest,
        missing_values=missing_values,
    )


TIMESERIES_SCHEMA = pa.DataFrameSchema(
    columns={
        "water_temp": Column(
            float,
            nullable=True,
            checks=[Check(lambda s: not s.isna().all(), error="water_temp all NaN")],
            required=False,
        ),
        "velocity": Column(
            float,
            nullable=True,
            checks=[Check(lambda s: not s.isna().all(), error="velocity all NaN")],
            required=False,
        ),
        "prediction": Column(
            float,
            nullable=True,
            checks=[Check(lambda s: not s.isna().all(), error="prediction all NaN")],
            required=False,
        ),
        "type": Column(
            str,
            nullable=True,
            checks=[
                Check.isin(["high", "low"], ignore_na=False),
                Check(lambda s: not s.isna().all(), error="prediction all NaN"),
            ],
            required=False,
        ),
    },
    index=pa.Index(
        DateTime,
        name="time",
        nullable=False,
        coerce=False,
        checks=[
            Check(lambda idx: not idx.hasnans, error="NaT in index"),
            Check(lambda idx: idx.is_monotonic_increasing, error="Index not sorted"),
            Check(lambda idx: idx.is_unique, error="Index values must be unique"),
            Check(lambda idx: idx.dt.tz is None, error="Index must be timezone naive"),
            Check(
                lambda ts: ts.tz is None,
                element_wise=True,
                error="Index must be timezone naive",
            ),
        ],
    ),
    checks=[
        Check(lambda df: not df.empty, error="DataFrame must have at least one row")
    ],
    strict=True,
)


def validate_timeseries_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the structure and content of an internal timeseries DataFrame using Pandera.

    Uses the TIMESERIES_SCHEMA to perform validation checks.

    Args:
        df: The pandas DataFrame to validate.

    Returns:
        The validated DataFrame.

    Raises:
        pandera.errors.SchemaError: If any validation check fails.
    """
    # No need for try/except, just let Pandera raise SchemaError if validation fails
    return TIMESERIES_SCHEMA.validate(df)

"""Shared utilities."""

import datetime
import pandas as pd


# Time shift limits for current predictions (in minutes)
MAX_SHIFT_LIMIT = 1440  # 24 hours forward
MIN_SHIFT_LIMIT = -1440  # 24 hours backward


def UTCNow() -> datetime.datetime:
    """Returns the current time in UTC as a naive datetime (without timezone information).

    All timestamps in the application are naive datetimes in their respective timezones.
    For NOAA data, timestamps are in local time based on the station's location.
    """
    # Get timezone-aware UTC time, then strip the timezone to make it naive
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def EffectiveTime(
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


def F2C(temp: float) -> float:
    """Convert Fahrenheit temp to Celsius."""
    return (5.0 / 9.0) * (temp - 32)


def PivotYear(df: pd.DataFrame) -> pd.DataFrame:
    """Move year dimension to columns."""
    df = df.assign(year=df.index.year)
    df.index = pd.to_datetime(
        # Use 2020-indexing because it's a leap year
        df.index.strftime("2020-%m-%d %H:%M:%S")
    )
    return df.set_index("year", append=True).unstack("year")

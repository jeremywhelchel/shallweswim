"""Shared utilities."""

import datetime
import pytz
import pandas as pd


EASTERN_TZ = pytz.timezone("US/Eastern")

# Time shift limits for current predictions (in minutes)
MAX_SHIFT_LIMIT = 1440  # 24 hours forward
MIN_SHIFT_LIMIT = -1440  # 24 hours backward


# XXX Needs a parameter here...
def Now() -> datetime.datetime:
    return datetime.datetime.now(tz=EASTERN_TZ).replace(tzinfo=None)


def EffectiveTime(shift_minutes: int = 0) -> datetime.datetime:
    """Calculate the effective time with an optional shift.

    Args:
        shift_minutes: Number of minutes to shift from current time

    Returns:
        Effective datetime with the shift applied
    """
    # Use the app's Now() function to ensure timezone consistency
    now = Now()

    # Clamp the shift limit
    shift_minutes = max(MIN_SHIFT_LIMIT, min(shift_minutes, MAX_SHIFT_LIMIT))

    if shift_minutes:
        delta = datetime.timedelta(minutes=shift_minutes)
        return now + delta
    return now


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

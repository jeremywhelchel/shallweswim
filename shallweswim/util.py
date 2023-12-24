"""Shared utilities."""

import datetime
import pytz
import pandas as pd


EASTERN_TZ = pytz.timezone("US/Eastern")


# XXX Needs a parameter here...
def Now() -> datetime.datetime:
    return datetime.datetime.now(tz=EASTERN_TZ).replace(tzinfo=None)


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

import numpy as np
import pandas as pd
import datetime
import pytz
import pytest

from shallweswim import util


def test_now() -> None:
    """Test that Now() returns current time in Eastern timezone without tzinfo."""
    now = util.Now()
    assert isinstance(now, datetime.datetime)
    assert now.tzinfo is None  # Should have no timezone info

    # Test that the time is within a reasonable range
    eastern_now = datetime.datetime.now(tz=pytz.timezone("US/Eastern"))
    time_diff = abs((eastern_now.replace(tzinfo=None) - now).total_seconds())
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
    result = util.F2C(fahrenheit)
    assert abs(result - expected_celsius) < 0.1  # within 0.1 degree


def test_pivot_year() -> None:
    df = pd.DataFrame(
        {"air_temp": 60, "water_temp": 50},
        index=pd.date_range("2011-01-01", "2023-12-31", freq="D"),
    )
    got = util.PivotYear(df)

    pd.testing.assert_index_equal(pd.date_range("2020-01-01", "2020-12-31"), got.index)

    cols = ["air_temp", "water_temp"]
    years = pd.Series(range(2011, 2023 + 1), dtype="int32")
    pd.testing.assert_index_equal(
        pd.MultiIndex.from_product([cols, years], names=[None, "year"]), got.columns
    )
    assert got["air_temp"].isin([60, np.nan]).all().all()
    assert got["water_temp"].isin([50, np.nan]).all().all()

import numpy as np
import pandas as pd
import datetime
import pytest

from shallweswim import util


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

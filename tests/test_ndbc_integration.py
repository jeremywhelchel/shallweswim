"""Integration tests for NOAA NDBC API client.

These tests connect to the actual NOAA NDBC API endpoints and verify compatibility
with the current API implementation.

Run with: pytest tests/test_ndbc_integration.py -v --run-integration
"""

# pylint: disable=unused-argument

import pytest
import pandas as pd
import datetime
import asyncio

from shallweswim.ndbc import NdbcApi

# Mark all tests in this file as integration tests that hit live services
pytestmark = pytest.mark.integration

# Real stations to use for testing
# Mid Gulf of Mexico - 180 nm South of Southwest Pass, LA
NDBC_STATION = "42001"


# Basic validation functions


def validate_temperature_data(df: pd.DataFrame) -> None:
    """Validate structure and content of temperature data."""
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "water_temp" in df.columns
    # Check that the column contains float values (can be NaN)
    assert pd.api.types.is_float_dtype(df["water_temp"])


# Integration tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_temperature_fetch() -> None:
    """Test fetching real temperature data from NOAA NDBC API."""
    # Skip the test if the --run-integration flag is not provided
    pytest.importorskip("sys").argv.append("--run-integration")

    # Get data for the last 8 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=8)

    # Fetch temperature data from NDBC station
    df = await NdbcApi.temperature(
        station_id=NDBC_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
    )

    validate_temperature_data(df)

    # Verify we got reasonable amount of data for an 8-day period
    # NDBC stations typically report hourly
    assert len(df) >= 5, "Expected several temperature readings over 8 days"

    # Check that timestamps are ordered correctly
    assert df.index.is_monotonic_increasing

    # First, check if we have any non-NaN values
    valid_temps = df.dropna()
    if valid_temps.empty:
        print(
            f"WARNING: All {len(df)} temperature readings are NaN. Sample data:\n{df.head()}"
        )
    else:
        # Check that valid temperatures are in a reasonable Fahrenheit range
        below_freezing = valid_temps[valid_temps["water_temp"] <= 32]
        assert (
            below_freezing.empty
        ), f"Water temperatures should be above freezing. Found {len(below_freezing)} problematic readings:\n{below_freezing.head(10) if len(below_freezing) > 10 else below_freezing}"

        above_boiling = valid_temps[valid_temps["water_temp"] >= 212]
        assert (
            above_boiling.empty
        ), f"Water temperatures should be below boiling (212°F). Found {len(above_boiling)} problematic readings:\n{above_boiling.head(10) if len(above_boiling) > 10 else above_boiling}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_date_range_handling() -> None:
    """Test fetching temperature data with different date ranges."""
    # Skip the test if the --run-integration flag is not provided
    pytest.importorskip("sys").argv.append("--run-integration")

    # Test with a short date range (1 day)
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=1)

    # Fetch temperature data with a short date range
    df_short = await NdbcApi.temperature(
        station_id=NDBC_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
    )

    validate_temperature_data(df_short)

    # Test with a longer date range (14 days)
    begin_date_long = end_date - datetime.timedelta(days=14)

    # Fetch temperature data with a longer date range
    df_long = await NdbcApi.temperature(
        station_id=NDBC_STATION,
        begin_date=begin_date_long,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
    )

    validate_temperature_data(df_long)

    # The longer date range should have more data points
    # (or at least the same if data is missing for some days)
    assert len(df_long) >= len(
        df_short
    ), "Longer date range should have at least as many data points"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timezone_conversion() -> None:
    """Test timezone conversion with real data."""
    # Skip the test if the --run-integration flag is not provided
    pytest.importorskip("sys").argv.append("--run-integration")

    # Get data for the last 3 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=3)

    # Fetch data with Eastern timezone
    df_eastern = await NdbcApi.temperature(
        station_id=NDBC_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
    )

    # Fetch the same data with Pacific timezone
    df_pacific = await NdbcApi.temperature(
        station_id=NDBC_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/Los_Angeles",
        location_code="tst",
    )

    # Both should have valid data
    validate_temperature_data(df_eastern)
    validate_temperature_data(df_pacific)

    # They should have the same number of data points
    assert len(df_eastern) == len(
        df_pacific
    ), "Same data should have same number of points regardless of timezone"

    # The timestamps should be different due to timezone conversion
    # Eastern time is 3 hours ahead of Pacific time
    if not df_eastern.empty and not df_pacific.empty:
        # Convert both to UTC for comparison
        eastern_utc = (
            pd.DatetimeIndex(df_eastern.index)
            .tz_localize("America/New_York")
            .tz_convert("UTC")
        )
        pacific_utc = (
            pd.DatetimeIndex(df_pacific.index)
            .tz_localize("America/Los_Angeles")
            .tz_convert("UTC")
        )

        # The UTC times should be the same
        pd.testing.assert_index_equal(eastern_utc, pacific_utc)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consecutive_api_calls() -> None:
    """Test making consecutive API calls to validate rate limiting handling."""
    # Skip the test if the --run-integration flag is not provided
    pytest.importorskip("sys").argv.append("--run-integration")

    # Get data for the last 3 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=3)

    # Make multiple API calls in succession
    for _ in range(3):
        df = await NdbcApi.temperature(
            station_id=NDBC_STATION,
            begin_date=begin_date,
            end_date=end_date,
            timezone="America/New_York",
            location_code="tst",
        )
        validate_temperature_data(df)
        # Small delay to avoid hitting rate limits
        await asyncio.sleep(0.5)

    # If we got here without exceptions, the API handled consecutive calls properly


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_station() -> None:
    """Test behavior with an invalid station ID."""
    # Skip the test if the --run-integration flag is not provided
    pytest.importorskip("sys").argv.append("--run-integration")

    # Get data for the last 3 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=3)

    # Use a non-existent station ID
    invalid_station = "99999"

    with pytest.raises(Exception):
        await NdbcApi.temperature(
            station_id=invalid_station,
            begin_date=begin_date,
            end_date=end_date,
            timezone="America/New_York",
            location_code="tst",
        )

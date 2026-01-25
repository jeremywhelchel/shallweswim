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

from shallweswim.clients.ndbc import NdbcApi

# Mark all tests in this file as integration tests that hit live services
pytestmark = pytest.mark.integration

# Real stations to use for testing
# Mid Gulf of Mexico - 180 nm South of Southwest Pass, LA
NDBC_STDMET_STATION = "42001"
# Norrie Point, Hudson River Reserve, NY
NDBC_OCEAN_STATION = "NPQN6"

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
async def test_live_temperature_fetch_stdmet() -> None:
    """Test fetching real temperature data from NOAA NDBC API using stdmet mode."""

    # Get data for the last 8 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=8)

    # NOTE: NdbcApi(session=None) works because NdbcApi._execute_request currently
    # bypasses the BaseApiClient's aiohttp session. Instead, it creates its own
    # synchronous NdbcApiClient (from the ndbc-api library) and calls its methods
    # using asyncio.to_thread. If NdbcApi is refactored to use aiohttp directly,
    # this test will need a valid session fixture.
    ndbc_api = NdbcApi(session=None)

    # Fetch temperature data from NDBC station using stdmet mode
    df = await ndbc_api.temperature(
        station_id=NDBC_STDMET_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
        mode="stdmet",
    )

    validate_temperature_data(df)

    # Verify we got reasonable amount of data for an 8-day period
    assert len(df) > 0

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
        # Sea water freezes at ~28.4°F (-2°C) due to salt content
        below_freezing = valid_temps[valid_temps["water_temp"] < 28]
        assert (
            below_freezing.empty
        ), f"Water temperatures should be above sea water freezing point (~28°F). Found {len(below_freezing)} problematic readings:\n{below_freezing.head(10) if len(below_freezing) > 10 else below_freezing}"

        above_boiling = valid_temps[valid_temps["water_temp"] >= 212]
        assert (
            above_boiling.empty
        ), f"Water temperatures should be below boiling (212°F). Found {len(above_boiling)} problematic readings:\n{above_boiling.head(10) if len(above_boiling) > 10 else above_boiling}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_temperature_fetch_ocean() -> None:
    """Test fetching real temperature data from NOAA NDBC API using ocean mode."""

    # Get data for the last 8 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=8)

    ndbc_api = NdbcApi(session=None)

    try:
        # Fetch temperature data from NDBC station using ocean mode
        df = await ndbc_api.temperature(
            station_id=NDBC_OCEAN_STATION,
            begin_date=begin_date,
            end_date=end_date,
            timezone="America/New_York",
            location_code="tst",
            mode="ocean",
        )

        validate_temperature_data(df)

        # Verify we got reasonable amount of data for an 8-day period
        assert len(df) > 0

        # First, check if we have any non-NaN values
        valid_temps = df.dropna()
        if valid_temps.empty:
            print(
                f"WARNING: All {len(df)} temperature readings are NaN. Sample data:\n{df.head()}"
            )
        else:
            # Check that valid temperatures are in a reasonable Fahrenheit range
            # Sea water freezes at ~28.4°F (-2°C) due to salt content
            below_freezing = valid_temps[valid_temps["water_temp"] < 28]
            assert (
                below_freezing.empty
            ), f"Water temperatures should be above sea water freezing point (~28°F). Found {len(below_freezing)} problematic readings:\n{below_freezing.head(10) if len(below_freezing) > 10 else below_freezing}"

            above_boiling = valid_temps[valid_temps["water_temp"] >= 212]
            assert (
                above_boiling.empty
            ), f"Water temperatures should be below boiling (212°F). Found {len(above_boiling)} problematic readings:\n{above_boiling.head(10) if len(above_boiling) > 10 else above_boiling}"
    except Exception as e:
        # Some stations might not have oceanographic data
        # Fail the test if the station doesn't support ocean mode
        if "No water temperature data ('OTMP')" in str(e):
            assert (
                False
            ), f"Station {NDBC_OCEAN_STATION} should have oceanographic data but doesn't"
        else:
            # Re-raise any other exceptions
            raise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_date_range_handling() -> None:
    """Test fetching temperature data with different date ranges."""

    # Test with a short date range (1 day)
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=1)

    ndbc_api = NdbcApi(session=None)

    # Fetch temperature data with a short date range
    df_short = await ndbc_api.temperature(
        station_id=NDBC_STDMET_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
        mode="stdmet",
    )

    validate_temperature_data(df_short)

    # Test with a longer date range (14 days)
    begin_date_long = end_date - datetime.timedelta(days=14)

    # Fetch temperature data with a longer date range
    df_long = await ndbc_api.temperature(
        station_id=NDBC_STDMET_STATION,
        begin_date=begin_date_long,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
        mode="stdmet",
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

    # Get data for the last 3 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=3)

    ndbc_api = NdbcApi(session=None)

    # Fetch data with Eastern timezone
    df_eastern = await ndbc_api.temperature(
        station_id=NDBC_STDMET_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/New_York",
        location_code="tst",
        mode="stdmet",
    )

    # Fetch the same data with Pacific timezone
    df_pacific = await ndbc_api.temperature(
        station_id=NDBC_STDMET_STATION,
        begin_date=begin_date,
        end_date=end_date,
        timezone="America/Los_Angeles",
        location_code="tst",
        mode="stdmet",
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

    # Add explicit assertions that data exists
    assert not df_eastern.empty, "Eastern timezone data should not be empty"
    assert not df_pacific.empty, "Pacific timezone data should not be empty"

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

    # The UTC timestamps should be identical
    pd.testing.assert_index_equal(eastern_utc, pacific_utc)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consecutive_api_calls() -> None:
    """Test making consecutive API calls to validate rate limiting handling."""

    # Get data for the last 3 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=3)

    ndbc_api = NdbcApi(session=None)

    # Make multiple API calls in succession
    for _ in range(3):
        df = await ndbc_api.temperature(
            station_id=NDBC_STDMET_STATION,
            begin_date=begin_date,
            end_date=end_date,
            timezone="America/New_York",
            location_code="tst",
            mode="stdmet",
        )
        validate_temperature_data(df)
        # Small delay to avoid hitting rate limits
        await asyncio.sleep(0.5)

    # If we got here without exceptions, the API handled consecutive calls properly


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_station() -> None:
    """Test behavior with an invalid station ID."""

    # Get data for the last 3 days
    end_date = datetime.date.today()
    begin_date = end_date - datetime.timedelta(days=3)

    ndbc_api = NdbcApi(session=None)

    # Use a non-existent station ID
    invalid_station = "99999"

    with pytest.raises(Exception):
        await ndbc_api.temperature(
            station_id=invalid_station,
            begin_date=begin_date,
            end_date=end_date,
            timezone="America/New_York",
            location_code="tst",
            mode="stdmet",
        )

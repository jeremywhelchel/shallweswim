"""Integration tests for USGS NWIS API client.

These tests interact with the live USGS NWIS API and may fail if the API
is down or changes. They are skipped by default and only run when the
--run-integration flag is passed to pytest.
"""

# Standard library imports
import datetime
import logging

# Third-party imports
import aiohttp
import pandas as pd
import pytest

# Local imports
from shallweswim.clients.base import StationUnavailableError
from shallweswim.clients.nwis import NwisApi, NwisApiError, NwisDataError

# Mark all tests in this file as integration tests that hit live services
pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_temperature() -> None:
    """Integration test for fetching NWIS temperature data."""
    # Dates known to have data for this site
    begin_date = datetime.date(2024, 6, 1)
    end_date = datetime.date(2024, 6, 2)
    site_no = "01463500"  # Use Delaware River @ Trenton
    timezone = "America/New_York"
    location_code = "test_loc"

    # Create a real session and client instance
    async with aiohttp.ClientSession() as session:
        nwis_client = NwisApi(session)
        # Call on instance
        df = await nwis_client.temperature(
            site_no=site_no,
            begin_date=begin_date,
            end_date=end_date,
            timezone=timezone,
            location_code=location_code,
        )

    # Basic checks
    assert isinstance(df, pd.DataFrame)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_missing_temp_column() -> None:
    """Integration test for a site known to be missing the standard temp parameter."""
    # Dates known to have data for this site
    begin_date = datetime.date(2024, 6, 1)
    end_date = datetime.date(2024, 6, 2)
    site_no = (
        "01646500"  # Potomac River near Washington, D.C. - Known *not* to have 00010
    )
    timezone = "America/New_York"
    location_code = "test_loc_missing"

    # Use parameter 00011 (Fahrenheit) for site 01646500, expecting no data
    # With principled error handling, missing data raises StationUnavailableError
    with pytest.raises(StationUnavailableError, match="returned no data"):
        # Create a real session and client instance
        async with aiohttp.ClientSession() as session:
            nwis_client = NwisApi(session)
            # Call on instance
            await nwis_client.temperature(
                site_no=site_no,
                begin_date=begin_date,
                end_date=end_date,
                timezone=timezone,
                location_code=location_code,
                parameter_cd="00011",  # Explicitly request the parameter known to be missing
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_live_temperature_fetch() -> None:
    """Integration test for fetching temperature data from the live USGS NWIS API."""
    # Use a date range in the past to ensure data is available
    # (past 7 days should be safe)
    end_date = datetime.datetime.now().date()
    begin_date = end_date - datetime.timedelta(days=7)

    # Create a real session and client instance
    async with aiohttp.ClientSession() as session:
        nwis_client = NwisApi(session)
        # Call on instance
        df = await nwis_client.temperature(
            site_no="01463500",  # Delaware River @ Trenton
            parameter_cd="00010",  # Water temperature, Celsius
            begin_date=begin_date,
            end_date=end_date,
            timezone="US/Eastern",
            location_code="sdf",
        )

    # Log the results for debugging
    logging.info(f"Fetched {len(df)} temperature readings")
    if not df.empty:
        logging.info(
            f"Temperature range: {df['water_temp'].min():.1f}°F - {df['water_temp'].max():.1f}°F"
        )
        logging.info(f"Date range: {df.index.min()} - {df.index.max()}")

    # Verify the results
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "No temperature data returned"
    assert "water_temp" in df.columns
    assert df.index.name == "time"

    # Check that the temperatures are reasonable for water
    # Water shouldn't be below freezing (32°F) or above 100°F
    assert df["water_temp"].min() >= 32.0, (
        f"Water temperature below freezing: {df['water_temp'].min()}°F"
    )
    assert df["water_temp"].max() <= 100.0, (
        f"Water temperature too high: {df['water_temp'].max()}°F"
    )

    # Check that the timestamps are in the requested range
    # Allow a buffer of one day on each end
    buffer_begin = pd.Timestamp(begin_date) - pd.Timedelta(days=1)
    buffer_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    # Get the min and max timestamps from the DataFrame index
    # Use pandas' built-in comparison methods which handle type compatibility
    assert not df.empty, "DataFrame is empty, cannot check timestamp range"

    # Convert to strings for the assertion message to avoid type issues
    min_ts_str = str(df.index.min())
    max_ts_str = str(df.index.max())

    # Use pandas' built-in comparison which handles type compatibility
    assert all(df.index >= buffer_begin), (
        f"Earliest timestamp {min_ts_str} before requested begin date {begin_date}"
    )
    assert all(df.index <= buffer_end), (
        f"Latest timestamp {max_ts_str} after requested end date {end_date}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_live_temperature_fetch_with_parameter_cd() -> None:
    """Integration test for fetching temperature data with a specific parameter code."""
    # Use a date range in the past to ensure data is available
    end_date = datetime.datetime.now().date()
    begin_date = end_date - datetime.timedelta(days=7)

    # Create a real session and client instance
    async with aiohttp.ClientSession() as session:
        nwis_client = NwisApi(session)
        # Call on instance
        df = await nwis_client.temperature(
            site_no="01463500",  # Delaware River @ Trenton
            parameter_cd="00010",  # Water temperature in Celsius
            begin_date=begin_date,
            end_date=end_date,
            timezone="US/Eastern",
            location_code="sdf",
        )

    # Verify the results
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "No temperature data returned"
    assert "water_temp" in df.columns
    assert df.index.name == "time"

    # Check that the temperatures are reasonable for water
    # Water shouldn't be below freezing (32°F) or above 100°F
    assert df["water_temp"].min() >= 32.0, (
        f"Water temperature below freezing: {df['water_temp'].min()}°F"
    )
    assert df["water_temp"].max() <= 100.0, (
        f"Water temperature too high: {df['water_temp'].max()}°F"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_nwis_currents() -> None:
    """Test fetching current data (velocity) from a specific NWIS site."""
    site_no = "03292494"  # Ohio River at Louisville Water Tower (used in sdf config)
    parameter_cd = "72255"  # Stream velocity, ft/sec (used in sdf config)
    timezone = "America/New_York"  # Example timezone

    async with aiohttp.ClientSession() as session:
        client = NwisApi(session)
        try:
            df = await client.currents(
                site_no=site_no,
                parameter_cd=parameter_cd,
                timezone=timezone,
                location_code="test-currents",
            )

            # Basic checks
            assert isinstance(df, pd.DataFrame)
            assert not df.empty, (
                f"No current data returned for site {site_no}, param {parameter_cd}"
            )
            assert "velocity_fps" in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            # Note: NWIS 'iv' service might only return one row
            print(f"\nReceived {len(df)} current readings for {site_no}:")
            print(df.head())

        except NwisDataError as e:
            pytest.fail(f"NwisDataError encountered: {e}")
        except NwisApiError as e:
            pytest.fail(f"NwisApiError encountered: {e}")

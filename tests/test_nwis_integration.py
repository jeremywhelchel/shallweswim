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
from shallweswim.clients.nwis import NwisApi, NwisApiError

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

    # Use parameter 00011 (Fahrenheit) for site 01646500, expecting NwisDataError
    # Expect NwisApiError because the generic except block wraps the NwisDataError
    with pytest.raises(
        NwisApiError, match="Unexpected error.*NwisDataError: No data returned"
    ):
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
    assert (
        df["water_temp"].min() >= 32.0
    ), f"Water temperature below freezing: {df['water_temp'].min()}°F"
    assert (
        df["water_temp"].max() <= 100.0
    ), f"Water temperature too high: {df['water_temp'].max()}°F"

    # Check that the timestamps are in the requested range
    # Allow a buffer of one day on each end
    buffer_begin = pd.Timestamp(begin_date) - pd.Timedelta(days=1)
    buffer_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    assert (
        df.index.min() >= buffer_begin
    ), f"Earliest timestamp {df.index.min()} before requested begin date {begin_date}"
    assert (
        df.index.max() <= buffer_end
    ), f"Latest timestamp {df.index.max()} after requested end date {end_date}"


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
    assert (
        df["water_temp"].min() >= 32.0
    ), f"Water temperature below freezing: {df['water_temp'].min()}°F"
    assert (
        df["water_temp"].max() <= 100.0
    ), f"Water temperature too high: {df['water_temp'].max()}°F"

"""Integration tests for USGS NWIS API client.

These tests interact with the live USGS NWIS API and may fail if the API
is down or changes. They are skipped by default and only run when the
--run-integration flag is passed to pytest.
"""

# Standard library imports
import datetime
import logging

# Third-party imports
import pandas as pd
import pytest

# Local imports
from shallweswim.clients.nwis import NwisApi

# Mark all tests in this file as integration tests that hit live services
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_live_temperature_fetch() -> None:
    """Test fetching temperature data from the live USGS NWIS API.

    This test uses the Ohio River at Louisville site (03292494) which should
    have reliable water temperature data.
    """
    # Use a date range in the past to ensure data is available
    # (past 7 days should be safe)
    end_date = datetime.datetime.now().date()
    begin_date = end_date - datetime.timedelta(days=7)

    # Fetch temperature data
    df = await NwisApi.temperature(
        site_no="03292494",  # Ohio River at Louisville
        parameter_cd="00011",  # Water temperature
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
    assert df.index.name == "timestamp"

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


@pytest.mark.asyncio
async def test_live_temperature_fetch_with_parameter_cd() -> None:
    """Test fetching temperature data with a specific parameter code.

    This test uses parameter code 00011 (water temperature, Celsius) instead of
    the default 00010 (water temperature, Fahrenheit).
    """
    # Use a date range in the past to ensure data is available
    end_date = datetime.datetime.now().date()
    begin_date = end_date - datetime.timedelta(days=7)

    # Fetch temperature data with parameter code 00011
    df = await NwisApi.temperature(
        site_no="03292494",  # Ohio River at Louisville
        parameter_cd="00011",  # Water temperature in Celsius
        begin_date=begin_date,
        end_date=end_date,
        timezone="US/Eastern",
        location_code="sdf",
    )

    # Verify the results
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "No temperature data returned"
    assert "water_temp" in df.columns
    assert df.index.name == "timestamp"

    # Check that the temperatures are reasonable for water
    # Water shouldn't be below freezing (32°F) or above 100°F
    assert (
        df["water_temp"].min() >= 32.0
    ), f"Water temperature below freezing: {df['water_temp'].min()}°F"
    assert (
        df["water_temp"].max() <= 100.0
    ), f"Water temperature too high: {df['water_temp'].max()}°F"

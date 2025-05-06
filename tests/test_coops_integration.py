"""Integration tests for NOAA CO-OPS API client.

These tests connect to the actual NOAA CO-OPS API endpoints and verify compatibility
with the current API implementation.

Run with: pytest tests/test_coops_integration.py -v --run-integration
"""

# pylint: disable=unused-argument

import pytest
import pandas as pd
import datetime
import asyncio
import aiohttp
from typing import Literal

from shallweswim.clients.coops import CoopsApi

# Mark all tests in this file as integration tests that hit live services
pytestmark = pytest.mark.integration

# Real stations to use for testing
NYC_BATTERY = 8518750  # NYC Battery - excellent station with comprehensive data
# Use the same station for all tests
TIDE_STATION = NYC_BATTERY
CURRENT_STATION = "n03020"  # NY Harbor Entrance (nearby current station)
TEMP_STATION = NYC_BATTERY


# Basic validation functions


def validate_tide_data(df: pd.DataFrame) -> None:
    """Validate structure and content of tide data."""
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "prediction" in df.columns
    assert "type" in df.columns
    assert bool(df["type"].isin(["high", "low"]).all())
    assert pd.api.types.is_float_dtype(df["prediction"])


def validate_current_data(df: pd.DataFrame) -> None:
    """Validate structure and content of current data."""
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "velocity" in df.columns
    assert pd.api.types.is_float_dtype(df["velocity"])


def validate_temperature_data(
    df: pd.DataFrame, product: Literal["water_temperature", "air_temperature"]
) -> None:
    """Validate structure and content of temperature data."""
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_col = "water_temp" if product == "water_temperature" else "air_temp"
    assert expected_col in df.columns
    # Check that the column contains float values (can be NaN)
    assert pd.api.types.is_float_dtype(df[expected_col])


# Integration tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_tides_fetch() -> None:
    """Test fetching real tide data from NOAA CO-OPS API."""
    async with aiohttp.ClientSession() as session:
        client = CoopsApi(session)
        df = await client.tides(station=TIDE_STATION)
    validate_tide_data(df)

    # Verify we got multiple tide predictions
    # At least 4 tide events in a 3-day period (yesterday to tomorrow)
    assert len(df) >= 4

    # Check that timestamps are ordered correctly
    assert df.index.is_monotonic_increasing

    # Check that we have some predictions in the past and some in the future
    now = pd.Timestamp.now()
    assert (df.index < now).any(), "Should have at least one past prediction"
    assert (df.index > now).any(), "Should have at least one future prediction"

    # Verify that high and low tides alternate (or at least both exist)
    assert df["type"].value_counts()["high"] > 0, "Should have high tides"
    assert df["type"].value_counts()["low"] > 0, "Should have low tides"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_currents_fetch() -> None:
    """Test fetching real current data from NOAA CO-OPS API."""
    async with aiohttp.ClientSession() as session:
        client = CoopsApi(session)
        df = await client.currents(station=CURRENT_STATION)
    validate_current_data(df)

    # Verify we have interpolated current data
    # Should have many points over 3 days
    assert len(df) > 24

    # Check that timestamps are ordered correctly
    assert df.index.is_monotonic_increasing

    # Check that we have some predictions in the past and some in the future
    now = pd.Timestamp.now()
    assert (df.index < now).any(), "Should have at least one past prediction"
    assert (df.index > now).any(), "Should have at least one future prediction"

    # Verify we have both positive and negative currents (flood and ebb)
    assert (df["velocity"] > 0).any(), "Should have some flood currents"
    assert (df["velocity"] < 0).any(), "Should have some ebb currents"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("product", ["water_temperature", "air_temperature"])
async def test_live_temperature_fetch(
    product: Literal["water_temperature", "air_temperature"],
) -> None:
    """Test fetching real temperature data from NOAA CO-OPS API."""
    async with aiohttp.ClientSession() as session:
        client = CoopsApi(session)
        # Get data for the last 3 days
        end_date = datetime.date.today()
        begin_date = end_date - datetime.timedelta(days=3)

        # No try/except - let the test fail if there's an issue
        df = await client.temperature(
            station=TEMP_STATION,  # Use the station known to have temperature data
            product=product,
            begin_date=begin_date,
            end_date=end_date,
        )
    validate_temperature_data(df, product)
    assert len(df) > 0, f"Should have received {product} data"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_temperature_intervals() -> None:
    """Test temperature data with different interval settings."""
    async with aiohttp.ClientSession() as session:
        client = CoopsApi(session)
        # Get data for the last 3 days
        end_date = datetime.date.today()
        begin_date = end_date - datetime.timedelta(days=1)

        # No try/except - let the test fail if there's an issue
        # Get hourly data
        df_hourly = await client.temperature(
            station=TEMP_STATION,
            product="air_temperature",
            begin_date=begin_date,
            end_date=end_date,
            interval="h",  # hourly
        )

        # Get default 6-minute data
        df_default = await client.temperature(
            station=TEMP_STATION,
            product="air_temperature",
            begin_date=begin_date,
            end_date=end_date,
        )

    # Validate both datasets
    validate_temperature_data(df_hourly, "air_temperature")
    validate_temperature_data(df_default, "air_temperature")

    # Hourly should have fewer points than default 6-minute interval
    # Note: This assumes that the data is actually available at both intervals
    # If this test fails, it may be because the station only provides hourly data
    if len(df_default) > 0 and len(df_hourly) > 0:
        assert len(df_default) >= len(
            df_hourly
        ), "Default interval should have more points than hourly"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_retries() -> None:
    """Test API retry mechanism with real requests."""
    async with aiohttp.ClientSession() as session:
        client = CoopsApi(session)
        # Simply test that we can make a successful request
        # This is not a proper test of the retry logic, but it at least verifies
        # that the API client can connect to the API
        df = await client.tides(station=TIDE_STATION)
    validate_tide_data(df)
    assert len(df) > 0, "Should have received tide data"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consecutive_api_calls() -> None:
    """Test making consecutive API calls to validate rate limiting handling."""
    async with aiohttp.ClientSession() as session:
        client = CoopsApi(session)
        # Make multiple API calls in succession
        for _ in range(3):
            df = await client.tides(station=TIDE_STATION)
            validate_tide_data(df)
            # Small delay to avoid hitting rate limits
            await asyncio.sleep(0.5)

    # If we got here without exceptions, the API handled consecutive calls properly

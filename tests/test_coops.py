"""Tests for NOAA CO-OPS API client."""

# Standard library imports
import datetime
import io
from typing import Any, cast

# Third-party imports
import aiohttp
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Local imports
from shallweswim.clients.coops import (
    CoopsApi,
    CoopsConnectionError,
)
from pandas.testing import assert_frame_equal


# Type definitions for test data (assuming they are defined elsewhere or basic)
# Fixtures


@pytest.fixture
def mock_session() -> MagicMock:
    """Provides a mock aiohttp ClientSession."""
    # Using MagicMock as the session itself doesn't need async methods mocked here
    # The _Request method, which uses the session, will be mocked in tests
    return MagicMock(spec=aiohttp.ClientSession)


@pytest.fixture
def coops_client(mock_session: MagicMock) -> CoopsApi:
    """Provides an instance of CoopsApi with a mock session."""
    # Instantiate with the mocked session
    return CoopsApi(session=mock_session)


@pytest.fixture
def mock_tide_data() -> pd.DataFrame:
    """Mock tide prediction data."""
    return pd.DataFrame(
        {
            "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Prediction": [5.2, 1.3],
            " Type": ["H", "L"],
        }
    )


@pytest.fixture
def mock_current_data() -> pd.DataFrame:
    """Mock current prediction data."""
    return pd.DataFrame(
        {
            "Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Velocity_Major": [2.5, -1.8],
            " Depth": [10.0, 10.0],
            " Type": ["flood", "ebb"],
            " meanFloodDir": [45.0, 45.0],
            " Bin": [1, 1],
        }
    )


@pytest.fixture
def mock_temperature_data() -> pd.DataFrame:
    """Create a mock temperature DataFrame."""
    # Data *before* _FixTime processing (as returned by _Request mock)
    data = {
        "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
        " Water Temperature": [62.5, 63.2],  # Match raw column name from API
        " Air Temperature": [65.0, 66.0],  # Include air temp too
    }
    df = pd.DataFrame(data)
    return df


@pytest.mark.asyncio
async def test_tides_success(
    coops_client: CoopsApi, mock_tide_data: pd.DataFrame
) -> None:
    """Test successful tide prediction fetch."""
    # Mock the _execute_request method directly instead of trying to mock aiohttp
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        # Set up the mock to return the proper DataFrame directly
        # The mock should return what _execute_request returns (raw CSV data as DF)
        mock_df = pd.read_csv(io.StringIO(mock_tide_data.to_csv(index=False)))
        mock_request.return_value = mock_df

        df = await coops_client.tides(
            station=9414290,
            location_code="test_loc",
        )

    assert len(df) == 2
    assert list(df.columns) == ["prediction", "type"]
    assert df["type"].tolist() == ["high", "low"]
    assert df["prediction"].tolist() == [5.2, 1.3]


@pytest.mark.asyncio
async def test_currents_success(
    coops_client: CoopsApi, mock_current_data: pd.DataFrame
) -> None:
    """Test successful current prediction fetch."""
    # Mock the _execute_request method directly instead of trying to mock aiohttp
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        # Set up the mock to return the proper DataFrame directly
        mock_df = pd.read_csv(io.StringIO(mock_current_data.to_csv(index=False)))
        mock_request.return_value = mock_df

        df = await coops_client.currents(
            station="SFB1201",
            interpolate=False,
            location_code="test_loc",
        )

    assert len(df) == 2
    assert list(df.columns) == ["velocity"]
    assert df["velocity"].tolist() == [2.5, -1.8]


@pytest.mark.asyncio
async def test_temperature_success(
    coops_client: CoopsApi, mock_temperature_data: pd.DataFrame
) -> None:
    """Test successful temperature fetch."""
    # Mock the _execute_request method directly instead of trying to mock aiohttp
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        # Assign the fixture DataFrame directly
        mock_request.return_value = mock_temperature_data

        # Test water temperature
        df_water = await coops_client.temperature(
            station=9414290,
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            product="water_temperature",
        )
        expected_water_df = pd.DataFrame(
            {"water_temp": [62.5, 63.2], "air_temp": [65.0, 66.0]},
            index=pd.to_datetime(["2025-04-19 10:00:00", "2025-04-19 16:00:00"]),
        ).rename_axis(
            "time"
        )  # Match index name set by _FixTime
        # Client returns all temp columns found, test needs to select the relevant one
        assert_frame_equal(df_water[["water_temp"]], expected_water_df[["water_temp"]])

        # Test air temperature
        df_air = await coops_client.temperature(
            station=9414290,
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            product="air_temperature",
        )
        expected_air_df = pd.DataFrame(
            {"water_temp": [62.5, 63.2], "air_temp": [65.0, 66.0]},
            index=pd.to_datetime(["2025-04-19 10:00:00", "2025-04-19 16:00:00"]),
        ).rename_axis(
            "time"
        )  # Match index name set by _FixTime
        # Client returns all temp columns found, test needs to select the relevant one
        assert_frame_equal(df_air[["air_temp"]], expected_air_df[["air_temp"]])


@pytest.mark.asyncio
async def test_connection_error(coops_client: CoopsApi) -> None:
    """Test connection error handling."""
    # Mock _execute_request to raise the error that request_with_retry expects
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = CoopsConnectionError("Connection timed out")

        # request_with_retry should propagate the CoopsConnectionError after retries
        with pytest.raises(CoopsConnectionError, match="Connection timed out"):
            await coops_client.tides(
                station=9414290,
                location_code="test_conn_error",
            )


@pytest.mark.asyncio
async def test_data_error(coops_client: CoopsApi) -> None:
    """Test data error handling (e.g., missing column)."""
    # Mock _execute_request to return bad data
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        # Return a DataFrame missing the 'Prediction' column but *with* 'Date Time'
        # _execute_request returns the raw data before _FixTime
        mock_df_bad = pd.DataFrame({"Date Time": ["2025-04-19 10:00"], " Value": [5.2]})
        mock_request.return_value = mock_df_bad

        # The error should occur during processing *after* _execute_request returns
        # Specifically, the .rename() step ignores missing columns (' Prediction', ' Type')
        # The error occurs in the .assign() step trying to access the non-existent 'type' column.
        with pytest.raises(KeyError, match="'type'"):
            # Use tides for testing connection/data errors as it's simpler
            await coops_client.tides(
                station=9414290,
                location_code="test_data_error",
            )


@pytest.mark.asyncio
async def test_invalid_temperature_dates(coops_client: CoopsApi) -> None:
    """Test temperature fetch with invalid date range."""
    with pytest.raises(ValueError, match="begin_date must be <= end_date"):
        await coops_client.temperature(
            station=9414290,  # Change to int
            begin_date=datetime.date(2025, 4, 20),  # Correct arg name
            end_date=datetime.date(2025, 4, 19),
            product="air_temperature",
        )


@pytest.mark.asyncio
async def test_invalid_temperature_product(coops_client: CoopsApi) -> None:
    """Test temperature fetch with invalid product."""
    with pytest.raises(ValueError, match="Invalid product: water_level"):
        await coops_client.temperature(
            station=9414290,  # Change to int
            begin_date=datetime.date(2025, 4, 19),  # Correct arg name
            end_date=datetime.date(2025, 4, 19),
            product=cast(Any, "water_level"),  # Intentionally invalid, cast to Any
        )

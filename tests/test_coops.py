"""Tests for NOAA CO-OPS API client."""

# Standard library imports
import datetime
import io

# Third-party imports
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock

# Local imports
from shallweswim.coops import CoopsApi, CoopsConnectionError, CoopsDataError


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
    """Mock temperature data."""
    return pd.DataFrame(
        {
            "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Water Temperature": [62.5, 63.2],
            " Air Temperature": [68.0, 70.5],
            " X": [1, 1],
            " N": [1, 1],
            " R ": [1, 1],
        }
    )


@pytest.mark.asyncio
async def test_tides_success(mock_tide_data: pd.DataFrame) -> None:
    """Test successful tide prediction fetch."""
    # Mock the _Request method directly instead of trying to mock aiohttp
    with patch.object(CoopsApi, "_Request", new_callable=AsyncMock) as mock_request:
        # Set up the mock to return the proper DataFrame directly
        mock_df = pd.read_csv(io.StringIO(mock_tide_data.to_csv(index=False)))
        mock_request.return_value = mock_df

        df = await CoopsApi.tides(station=9414290)

    assert len(df) == 2
    assert list(df.columns) == ["prediction", "type"]
    assert df["type"].tolist() == ["high", "low"]
    assert df["prediction"].tolist() == [5.2, 1.3]


@pytest.mark.asyncio
async def test_currents_success(mock_current_data: pd.DataFrame) -> None:
    """Test successful current prediction fetch."""
    # Mock the _Request method directly instead of trying to mock aiohttp
    with patch.object(CoopsApi, "_Request", new_callable=AsyncMock) as mock_request:
        # Set up the mock to return the proper DataFrame directly
        mock_df = pd.read_csv(io.StringIO(mock_current_data.to_csv(index=False)))
        mock_request.return_value = mock_df

        df = await CoopsApi.currents(station="SFB1201", interpolate=False)

    assert len(df) == 2
    assert list(df.columns) == ["velocity"]
    assert df["velocity"].tolist() == [2.5, -1.8]


@pytest.mark.asyncio
async def test_temperature_success(mock_temperature_data: pd.DataFrame) -> None:
    """Test successful temperature fetch."""
    # Mock the _Request method directly instead of trying to mock aiohttp
    with patch.object(CoopsApi, "_Request", new_callable=AsyncMock) as mock_request:
        # Set up the mock to return the proper DataFrame directly
        mock_df = pd.read_csv(io.StringIO(mock_temperature_data.to_csv(index=False)))
        mock_request.return_value = mock_df

        df = await CoopsApi.temperature(
            station=9414290,
            product="water_temperature",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
        )

    assert len(df) == 2
    assert "water_temp" in df.columns
    assert df["water_temp"].tolist() == [62.5, 63.2]


@pytest.mark.asyncio
async def test_connection_error() -> None:
    """Test handling of connection errors."""
    with patch.object(CoopsApi, "_Request", new_callable=AsyncMock) as mock_request:
        # Simulate a connection error
        mock_request.side_effect = CoopsConnectionError(
            "Failed to connect to NOAA CO-OPS API: Network error"
        )

        with pytest.raises(
            CoopsConnectionError, match="Failed to connect to NOAA CO-OPS API"
        ):
            await CoopsApi.tides(station=9414290)


@pytest.mark.asyncio
async def test_data_error() -> None:
    """Test handling of API data errors."""
    with patch.object(CoopsApi, "_Request", new_callable=AsyncMock) as mock_request:
        # Mock error response
        mock_request.side_effect = CoopsDataError("Invalid station ID")

        with pytest.raises(CoopsDataError, match="Invalid station ID"):
            await CoopsApi.tides(station=9414290)


@pytest.mark.asyncio
async def test_invalid_temperature_dates() -> None:
    """Test validation of temperature date ranges."""
    with pytest.raises(ValueError, match="begin_date must be <= end_date"):
        await CoopsApi.temperature(
            station=9414290,
            product="water_temperature",
            begin_date=datetime.date(2025, 4, 20),
            end_date=datetime.date(2025, 4, 19),
        )


@pytest.mark.asyncio
async def test_invalid_temperature_product() -> None:
    """Test validation of temperature product type."""
    with pytest.raises(ValueError, match="Invalid product"):
        await CoopsApi.temperature(
            station=9414290,
            product="invalid_product",  # type: ignore[arg-type] # intentionally invalid for testing error case
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
        )


@pytest.mark.asyncio
async def test_current_interpolation(mock_current_data: pd.DataFrame) -> None:
    """Test current interpolation."""
    # Mock the _Request method directly instead of trying to mock aiohttp
    with patch.object(CoopsApi, "_Request", new_callable=AsyncMock) as mock_request:
        # Set up the mock to return the proper DataFrame directly
        mock_df = pd.read_csv(io.StringIO(mock_current_data.to_csv(index=False)))
        mock_request.return_value = mock_df

        df = await CoopsApi.currents(station="SFB1201", interpolate=True)

    # Should have many more points due to 60s interpolation
    assert len(df) > len(mock_current_data)
    # First and last values should match original
    assert df["velocity"].iloc[0] == mock_current_data[" Velocity_Major"].iloc[0]
    assert df["velocity"].iloc[-1] == mock_current_data[" Velocity_Major"].iloc[-1]

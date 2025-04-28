"""Tests for NOAA NDBC API client."""

# Standard library imports
import datetime

# Third-party imports
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Local imports
from shallweswim.ndbc import NdbcApi, NdbcApiError


@pytest.fixture
def mock_ndbc_temp_data() -> pd.DataFrame:
    """Mock NDBC temperature data."""
    # Create a MultiIndex with timestamp and station_id
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-04-19 10:00"), "44025"),
            (pd.Timestamp("2025-04-19 16:00"), "44025"),
        ],
        names=["timestamp", "station_id"],
    )

    # Create DataFrame with water temperature in Celsius
    df = pd.DataFrame(
        {
            "WTMP": [15.5, 16.2],  # Water temperature in Celsius
        },
        index=index,
    )

    return df


@pytest.mark.asyncio
async def test_temperature_success(mock_ndbc_temp_data: pd.DataFrame) -> None:
    """Test successful temperature fetch."""
    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Set up the mock to return the proper DataFrame directly
        mock_to_thread.return_value = mock_ndbc_temp_data

        # Mock the ndbc_api.NdbcApi class
        with patch("ndbc_api.NdbcApi") as mock_ndbc_api_class:
            # The instance doesn't need to do anything as we're mocking to_thread
            mock_ndbc_api_instance = MagicMock()
            mock_ndbc_api_class.return_value = mock_ndbc_api_instance

            df = await NdbcApi.temperature(
                station_id="44025",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="nyc",
            )

    assert len(df) == 2
    assert "water_temp" in df.columns
    # Check that temperatures were converted from Celsius to Fahrenheit
    assert round(df["water_temp"].iloc[0], 1) == 59.9  # 15.5°C = 59.9°F
    assert round(df["water_temp"].iloc[1], 1) == 61.2  # 16.2°C = 61.2°F


@pytest.mark.asyncio
async def test_api_error() -> None:
    """Test handling of API errors."""
    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Simulate an API error
        mock_to_thread.side_effect = Exception("NDBC API error")

        # Mock the ndbc_api.NdbcApi class
        with patch("ndbc_api.NdbcApi") as mock_ndbc_api_class:
            # The instance doesn't need to do anything as we're mocking to_thread
            mock_ndbc_api_instance = MagicMock()
            mock_ndbc_api_class.return_value = mock_ndbc_api_instance

            with pytest.raises(NdbcApiError, match="Error fetching NDBC data"):
                await NdbcApi.temperature(
                    station_id="44025",
                    begin_date=datetime.date(2025, 4, 19),
                    end_date=datetime.date(2025, 4, 19),
                    timezone="America/New_York",
                    location_code="nyc",
                )


@pytest.mark.asyncio
async def test_dictionary_result() -> None:
    """Test handling of dictionary result instead of DataFrame."""
    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Return a dictionary instead of a DataFrame
        mock_to_thread.return_value = {"error": "No data available"}

        # Mock the ndbc_api.NdbcApi class
        with patch("ndbc_api.NdbcApi") as mock_ndbc_api_class:
            # The instance doesn't need to do anything as we're mocking to_thread
            mock_ndbc_api_instance = MagicMock()
            mock_ndbc_api_class.return_value = mock_ndbc_api_instance

            with pytest.raises(NdbcApiError, match="Error fetching NDBC data"):
                await NdbcApi.temperature(
                    station_id="44025",
                    begin_date=datetime.date(2025, 4, 19),
                    end_date=datetime.date(2025, 4, 19),
                    timezone="America/New_York",
                    location_code="nyc",
                )


@pytest.mark.asyncio
async def test_missing_wtmp_column() -> None:
    """Test handling of missing WTMP column."""
    # Create DataFrame without WTMP column
    df = pd.DataFrame(
        {
            "ATMP": [20.5, 21.2],  # Air temperature instead of water
        },
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-04-19 10:00"), "44025"),
                (pd.Timestamp("2025-04-19 16:00"), "44025"),
            ],
            names=["timestamp", "station_id"],
        ),
    )

    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Return DataFrame without WTMP column
        mock_to_thread.return_value = df

        # Mock the ndbc_api.NdbcApi class
        with patch("ndbc_api.NdbcApi") as mock_ndbc_api_class:
            # The instance doesn't need to do anything as we're mocking to_thread
            mock_ndbc_api_instance = MagicMock()
            mock_ndbc_api_class.return_value = mock_ndbc_api_instance

            with pytest.raises(NdbcApiError, match="Error fetching NDBC data"):
                await NdbcApi.temperature(
                    station_id="44025",
                    begin_date=datetime.date(2025, 4, 19),
                    end_date=datetime.date(2025, 4, 19),
                    timezone="America/New_York",
                    location_code="nyc",
                )


@pytest.mark.asyncio
async def test_fix_time() -> None:
    """Test the _fix_time method."""
    # Create a DataFrame with UTC timestamps
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-04-19 14:00:00"),  # 10:00 AM EDT
            pd.Timestamp("2025-04-19 20:00:00"),  # 4:00 PM EDT
        ]
    )
    df = pd.DataFrame({"water_temp": [59.9, 61.2]}, index=index)

    # Convert timestamps to EDT
    result_df = NdbcApi._fix_time(df, "America/New_York")

    # Check that timestamps were converted correctly
    expected_times = [
        pd.Timestamp("2025-04-19 10:00:00"),  # 10:00 AM EDT
        pd.Timestamp("2025-04-19 16:00:00"),  # 4:00 PM EDT
    ]
    pd.testing.assert_index_equal(result_df.index, pd.DatetimeIndex(expected_times))

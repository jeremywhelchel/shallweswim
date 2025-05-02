"""Tests for USGS NWIS API client."""

# pylint: disable=duplicate-code

# Standard library imports
import datetime

# Third-party imports
import aiohttp
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Local imports
from shallweswim.clients.nwis import NwisApi, NwisDataError
from shallweswim.clients.base import (
    ClientConnectionError,
)


def create_mock_nwis_data(parameter_cd: str = "00010") -> pd.DataFrame:
    """Create mock NWIS data for testing.

    Args:
        parameter_cd: Parameter code for water temperature ('00010' or '00011')
    """
    # Create timezone-aware timestamps for the index (UTC)
    timestamps = [
        pd.Timestamp("2025-04-19 14:00:00", tz="UTC"),
        pd.Timestamp("2025-04-19 15:00:00", tz="UTC"),
    ]

    # Create a column name that includes the parameter code
    # Format similar to what the NWIS API returns: 'USGS:SITE_NO:00010:00000_00003'
    temp_column = f"USGS:03292494:{parameter_cd}:00000_00003"

    # Create DataFrame with water temperature data
    if parameter_cd == "00010":
        # For 00010, provide temperatures in Celsius (15.5°C ≈ 59.9°F, 16.2°C ≈ 61.2°F)
        df = pd.DataFrame(
            {
                temp_column: [15.5, 16.2],  # In Celsius
            },
            index=timestamps,
        )
    else:
        # For 00011 or any other parameter, provide temperatures in Fahrenheit
        df = pd.DataFrame(
            {
                temp_column: [59.9, 61.2],  # Already in Fahrenheit
            },
            index=timestamps,
        )

    return df


@pytest.fixture
def mock_session() -> MagicMock:
    """Provides a mock aiohttp ClientSession."""
    # NwisApi doesn't directly use the session for dataretrieval,
    # so a simple MagicMock is sufficient.
    session = MagicMock(spec=aiohttp.ClientSession)
    # Mock the __aenter__ and __aexit__ methods for async context management if needed
    # session.__aenter__.return_value = session
    # session.__aexit__.return_value = None
    return session


@pytest.fixture
def nwis_client(mock_session: MagicMock) -> NwisApi:
    """Provides an instance of NwisApi with a mock session."""
    return NwisApi(session=mock_session)


@pytest.mark.asyncio
async def test_temperature_success(nwis_client: NwisApi) -> None:
    """Test successful temperature fetch."""
    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Set up the mock to return the proper DataFrame directly
        mock_to_thread.return_value = create_mock_nwis_data(parameter_cd="00010")

        # Mock the dataretrieval.nwis module
        with patch("dataretrieval.nwis.get_record") as mock_get_record:
            # The function is mocked via asyncio.to_thread, so this doesn't need to do anything
            mock_get_record.return_value = None

            # Call on instance
            df = await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )

    assert len(df) == 2
    assert "water_temp" in df.columns

    # Check that temperatures are in Fahrenheit
    assert round(df["water_temp"].iloc[0], 1) == 59.9
    assert round(df["water_temp"].iloc[1], 1) == 61.2


@pytest.mark.asyncio
async def test_temperature_celsius_param(nwis_client: NwisApi) -> None:
    """Test temperature fetch with parameter 00010 (Celsius)."""
    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Set up the mock to return Celsius data
        mock_to_thread.return_value = create_mock_nwis_data(parameter_cd="00010")

        # Mock the dataretrieval.nwis module
        with patch("dataretrieval.nwis.get_record") as mock_get_record:
            # The function is mocked via asyncio.to_thread, so this doesn't need to do anything
            mock_get_record.return_value = None

            # Call on instance
            df = await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
                parameter_cd="00010",  # Celsius parameter
            )

    assert len(df) == 2
    assert "water_temp" in df.columns

    # Check that temperatures were converted from Celsius to Fahrenheit
    # 15.5°C → 59.9°F, 16.2°C → 61.2°F
    assert round(df["water_temp"].iloc[0], 1) == 59.9
    assert round(df["water_temp"].iloc[1], 1) == 61.2


@pytest.mark.asyncio
async def test_temperature_fahrenheit_param(nwis_client: NwisApi) -> None:
    """Test temperature fetch with parameter 00011 (Fahrenheit)."""
    # Mock the asyncio.to_thread function
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Set up the mock to return Fahrenheit data
        mock_to_thread.return_value = create_mock_nwis_data(parameter_cd="00011")

        # Mock the dataretrieval.nwis module
        with patch("dataretrieval.nwis.get_record") as mock_get_record:
            # The function is mocked via asyncio.to_thread, so this doesn't need to do anything
            mock_get_record.return_value = None

            # Call on instance
            df = await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
                parameter_cd="00011",  # Fahrenheit parameter
            )

    assert len(df) == 2
    assert "water_temp" in df.columns

    # Check that temperatures are unchanged (already in Fahrenheit)
    assert round(df["water_temp"].iloc[0], 1) == 59.9
    assert round(df["water_temp"].iloc[1], 1) == 61.2


@pytest.mark.asyncio
async def test_api_error(nwis_client: NwisApi) -> None:
    """Test handling of connection errors after retries fail."""
    # Mock request_with_retry to simulate exhausted retries
    with patch.object(
        nwis_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request_with_retry:
        mock_request_with_retry.side_effect = ClientConnectionError(
            "Simulated connection error after retries"
        )

        with pytest.raises(
            ClientConnectionError, match="Simulated connection error after retries"
        ):
            await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.asyncio
async def test_data_error(nwis_client: NwisApi) -> None:
    """Test handling of NwisDataError when API returns empty results."""
    # Mock request_with_retry to simulate _execute_request returning empty DataFrame
    with patch.object(
        nwis_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request_with_retry:
        # Simulate _execute_request raising NwisDataError because the result was empty
        # This happens *inside* the retry logic, so request_with_retry re-raises it.
        mock_request_with_retry.side_effect = NwisDataError(
            "No data returned from NWIS API for site 03292494, params ['00010']"
        )

        # Expect NwisDataError because the fetch itself returned no data
        with pytest.raises(
            NwisDataError, match="No data returned from NWIS API for site 03292494"
        ):
            await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.asyncio
async def test_missing_temp_column(nwis_client: NwisApi) -> None:
    # Create DataFrame without temperature column
    df = pd.DataFrame(
        {
            "USGS:03292494:00060:00000_00003": [
                20.5,
                21.2,
            ],  # Discharge instead of temperature
        },
        index=[
            pd.Timestamp("2025-04-19 10:00"),
            pd.Timestamp("2025-04-19 16:00"),
        ],
    )

    # Mock request_with_retry to return the DataFrame *without* the temperature column
    # This simulates a successful fetch, but with incorrect data for post-processing.
    with patch.object(
        nwis_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request_with_retry:
        mock_request_with_retry.return_value = df

        # Expect NwisDataError because the post-processing in temperature() fails
        # The expected column '...00010...' is missing.
        expected_message = r"No water temperature data \(parameter 00010\) available for NWIS site 03292494"
        with pytest.raises(NwisDataError, match=expected_message):
            await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.asyncio
async def test_fix_time(nwis_client: NwisApi) -> None:
    """Test the _fix_time method."""
    # Create a DataFrame with timezone-aware UTC timestamps
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-04-19 14:00:00", tz="UTC"),  # 10:00 AM EDT
            pd.Timestamp("2025-04-19 20:00:00", tz="UTC"),  # 4:00 PM EDT
        ]
    )
    df = pd.DataFrame({"water_temp": [59.9, 61.2]}, index=index)

    # Convert timestamps to EDT
    # Call on instance
    result_df = nwis_client._fix_time(df, "America/New_York")

    # Check that timestamps were converted correctly
    expected_times = [
        pd.Timestamp("2025-04-19 10:00:00"),  # 10:00 AM EDT
        pd.Timestamp("2025-04-19 16:00:00"),  # 4:00 PM EDT
    ]
    expected_index = pd.DatetimeIndex(expected_times, name="timestamp")
    pd.testing.assert_index_equal(result_df.index, expected_index)

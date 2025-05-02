"""Tests for NOAA NDBC API client."""

# pylint: disable=duplicate-code

# Standard library imports
import datetime

# Third-party imports
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock

# Local imports
from shallweswim.clients.ndbc import NdbcApi, NdbcDataError
from shallweswim.clients.base import RetryableClientError
from shallweswim.util import c_to_f


def create_mock_ndbc_data(mode: str = "stdmet") -> pd.DataFrame:
    """Create mock NDBC data for testing.

    Args:
        mode: Data mode to create mock data for ('stdmet' or 'ocean')
    """
    # Create MultiIndex with timestamp and station_id
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-04-19 14:00:00"), "44025"),
            (pd.Timestamp("2025-04-19 15:00:00"), "44025"),
        ],
        names=["timestamp", "station_id"],
    )

    # Create DataFrame with water temperature in Celsius
    # Use appropriate column name based on mode
    temp_column = "WTMP" if mode == "stdmet" else "OTMP"
    df = pd.DataFrame(
        {
            temp_column: [15.5, 16.2],  # ~60°F and ~61°F in Fahrenheit
        },
        index=index,
    )

    return df


@pytest.fixture
def ndbc_client() -> NdbcApi:
    """Provides an NdbcApi instance for testing."""
    # Session isn't used by ndbc_api, but pass None or a dummy object if BaseApiClient requires it
    client = NdbcApi(session=None)  # Pass None or a dummy session
    return client


# Mock data fixtures
@pytest.fixture
def mock_ndbc_stdmet_raw() -> pd.DataFrame:
    """Mock raw DataFrame for stdmet data from ndbc_api."""
    index = pd.MultiIndex.from_tuples(
        [
            ("44013", pd.Timestamp("2025-04-19 10:00:00")),
            ("44013", pd.Timestamp("2025-04-19 11:00:00")),
        ],
        names=["station_id", "timestamp"],
    )
    return pd.DataFrame({"WTMP": [15.0, 15.5]}, index=index)  # Celsius


@pytest.fixture
def mock_ndbc_ocean_raw() -> pd.DataFrame:
    """Mock raw DataFrame for ocean data from ndbc_api."""
    index = pd.MultiIndex.from_tuples(
        [
            ("44013", pd.Timestamp("2025-04-19 10:00:00")),
            ("44013", pd.Timestamp("2025-04-19 11:00:00")),
        ],
        names=["station_id", "timestamp"],
    )
    return pd.DataFrame({"OTMP": [14.8, 15.2]}, index=index)  # Celsius


@pytest.mark.asyncio
async def test_temperature_success(
    ndbc_client: NdbcApi, mock_ndbc_stdmet_raw: pd.DataFrame
) -> None:
    """Test successful temperature fetch using default stdmet."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_ndbc_stdmet_raw

        result = await ndbc_client.temperature(  # Use instance
            station_id="44013",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone="America/New_York",
        )

        assert not result.empty
        assert "water_temp" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result.index)
        # Check C to F conversion
        assert result["water_temp"].iloc[0] == pytest.approx(c_to_f(15.0))
        mock_request.assert_called_once()
        call_args = mock_request.call_args[1]
        assert call_args["station_id"] == "44013"
        assert call_args["mode"] == "stdmet"


@pytest.mark.asyncio
async def test_temperature_stdmet(
    ndbc_client: NdbcApi, mock_ndbc_stdmet_raw: pd.DataFrame
) -> None:
    """Test temperature fetch explicitly using stdmet mode."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_ndbc_stdmet_raw

        result = await ndbc_client.temperature(  # Use instance
            station_id="44013",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone="America/New_York",
            mode="stdmet",
        )

        assert not result.empty
        assert result["water_temp"].iloc[0] == pytest.approx(c_to_f(15.0))
        mock_request.assert_called_once()
        assert mock_request.call_args[1]["mode"] == "stdmet"


@pytest.mark.asyncio
async def test_temperature_ocean(
    ndbc_client: NdbcApi, mock_ndbc_ocean_raw: pd.DataFrame
) -> None:
    """Test temperature fetch using ocean mode."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_ndbc_ocean_raw

        result = await ndbc_client.temperature(  # Use instance
            station_id="44013",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone="America/New_York",
            mode="ocean",
        )

        assert not result.empty
        assert result["water_temp"].iloc[0] == pytest.approx(c_to_f(14.8))
        mock_request.assert_called_once()
        assert mock_request.call_args[1]["mode"] == "ocean"


@pytest.mark.asyncio
async def test_api_error(
    ndbc_client: NdbcApi,
) -> None:
    """Test handling of connection errors (RetryableClientError)."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = RetryableClientError("NDBC connection failed")

        with pytest.raises(RetryableClientError, match="NDBC connection failed"):
            await ndbc_client.temperature(  # Use instance
                station_id="44013",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
            )


@pytest.mark.asyncio
async def test_dictionary_result(
    ndbc_client: NdbcApi,
) -> None:
    """Test handling when API returns a dictionary instead of DataFrame."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        # Simulate _execute_request raising NdbcDataError
        mock_request.side_effect = NdbcDataError("NDBC API returned a dictionary")

        with pytest.raises(NdbcDataError, match="NDBC API returned a dictionary"):
            await ndbc_client.temperature(  # Use instance
                station_id="44013",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
            )


@pytest.mark.asyncio
async def test_missing_temp_column_stdmet(
    ndbc_client: NdbcApi,
) -> None:
    """Test handling when stdmet data misses WTMP column."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        # Return data missing the expected column
        index = pd.MultiIndex.from_tuples(
            [("44013", pd.Timestamp("2025-04-19 10:00:00"))],
            names=["station_id", "timestamp"],
        )
        mock_df_bad = pd.DataFrame({"ATMP": [20.0]}, index=index)
        mock_request.return_value = mock_df_bad

        with pytest.raises(
            NdbcDataError, match=r"No water temperature data \(\'WTMP\'\) available"
        ):
            await ndbc_client.temperature(  # Use instance
                station_id="44013",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                mode="stdmet",
            )


@pytest.mark.asyncio
async def test_missing_temp_column_ocean(
    ndbc_client: NdbcApi,
) -> None:
    """Test handling when ocean data misses OTMP column."""
    with patch.object(
        ndbc_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request:
        index = pd.MultiIndex.from_tuples(
            [("44013", pd.Timestamp("2025-04-19 10:00:00"))],
            names=["station_id", "timestamp"],
        )
        mock_df_bad = pd.DataFrame({"SomeOtherMetric": [10.0]}, index=index)
        mock_request.return_value = mock_df_bad

        with pytest.raises(
            NdbcDataError, match=r"No water temperature data \(\'OTMP\'\) available"
        ):
            await ndbc_client.temperature(  # Use instance
                station_id="44013",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                mode="ocean",
            )


def test_fix_time(ndbc_client: NdbcApi) -> None:
    """Test the _fix_time method for timezone conversion.

    Based on original implementation provided by user.
    Checks conversion from naive (assumed UTC) to local naive.
    """
    # Create a DataFrame with a naive DatetimeIndex (as might come from MultiIndex drop)
    # Original test used naive times assumed to be UTC
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-04-19 14:00:00"),  # Represents 10:00 AM EDT
            pd.Timestamp("2025-04-19 20:00:00"),  # Represents 4:00 PM EDT
        ]
    )
    df = pd.DataFrame({"water_temp": [15.0, 15.5]}, index=index)

    # Call instance method via fixture
    result_df = ndbc_client._fix_time(df, "America/New_York")

    # Check that timestamps were converted correctly and result is naive
    expected_times = [
        pd.Timestamp("2025-04-19 10:00:00"),  # Expected 10:00 AM EDT (naive)
        pd.Timestamp("2025-04-19 16:00:00"),  # Expected 4:00 PM EDT (naive)
    ]
    # Ensure result index is naive
    assert result_df.index.tz is None
    pd.testing.assert_index_equal(result_df.index, pd.DatetimeIndex(expected_times))

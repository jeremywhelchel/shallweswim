"""Tests for NOAA NDBC API client."""

# pylint: disable=duplicate-code

# Standard library imports
import contextlib
import datetime
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

# Third-party imports
import aiohttp
import pandas as pd
import pytest

from shallweswim.clients.base import RetryableClientError

# Local imports
from shallweswim.clients.ndbc import (
    NDBC_BASE_URL,
    NDBC_CURRENT_MONTH_FILE_EXTENSION,
    NDBC_DATA_PATH,
    NDBC_HISTORICAL_DATA_PATH,
    NDBC_HISTORICAL_VIEW_PATH,
    NDBC_MAX_CONCURRENT_REQUESTS,
    NDBC_REALTIME_PATH,
    NdbcApi,
    NdbcDataError,
)
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
    return NdbcApi(session=None)


# Mock data fixtures
@pytest.fixture
def mock_ndbc_stdmet_raw() -> pd.DataFrame:
    """Mock raw DataFrame for stdmet data from NDBC."""
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
    """Mock raw DataFrame for ocean data from NDBC."""
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
async def test_execute_request_retries_client_errors(
    ndbc_client: NdbcApi,
) -> None:
    """Transient fetch failures propagate as retryable NDBC errors."""
    with patch.object(
        ndbc_client,
        "_fetch_url",
        new_callable=AsyncMock,
        side_effect=RetryableClientError("bad response"),
    ):
        with pytest.raises(RetryableClientError, match="bad response"):
            await ndbc_client._execute_request(
                station_id="44013",
                mode="stdmet",
                start_time="2025-04-19",
                end_time="2025-04-20",
                location_code="test",
            )


@pytest.mark.asyncio
async def test_execute_request_includes_full_end_date_without_next_day(
    ndbc_client: NdbcApi,
) -> None:
    """Same-day requests include the full end date but not the next day."""
    response_body = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 06 11 00 00 250  2.6   MM   1.3    MM     6  MM 1017.3  12.4  14.9   7.9   11   MM    MM
2026 06 10 17 00 250  2.6   MM   1.3    MM     6  MM 1017.3  12.4  14.8   7.9   11   MM    MM
2026 06 10 00 00 250  3.6   MM   0.5    MM     6  MM 1014.2  13.0  14.1   8.1   11   MM    MM
2026 06 09 23 00 240  4.1   MM   0.6    MM     6  MM 1014.0  13.2  14.0   8.0   11   MM    MM
"""

    with patch.object(
        ndbc_client,
        "_fetch_url",
        new_callable=AsyncMock,
        return_value=type("Response", (), {"status": 200, "body": response_body})(),
    ):
        result = await ndbc_client._execute_request(
            station_id="62304",
            mode="stdmet",
            start_time="2026-06-10",
            end_time="2026-06-10",
            location_code="test",
        )

    timestamps = result.index.get_level_values("timestamp")
    assert list(timestamps) == [
        pd.Timestamp("2026-06-10 00:00:00"),
        pd.Timestamp("2026-06-10 17:00:00"),
    ]
    assert pd.Timestamp("2026-06-10 17:00:00") in timestamps
    assert pd.Timestamp("2026-06-09 23:00:00") not in timestamps
    assert pd.Timestamp("2026-06-11 00:00:00") not in timestamps


@pytest.mark.asyncio
async def test_fetch_url_with_session_uses_provider_request_gate(
    ndbc_client: NdbcApi,
) -> None:
    """NDBC HTTP calls pass through the shared provider request gate."""
    entered_gate = False

    @contextlib.asynccontextmanager
    async def mock_provider_request_slot(
        provider: str, max_concurrent_requests: int
    ) -> AsyncIterator[None]:
        nonlocal entered_gate
        entered_gate = True
        assert provider == "ndbc"
        assert max_concurrent_requests == NDBC_MAX_CONCURRENT_REQUESTS
        yield

    class MockResponse:
        status = 200

        async def __aenter__(self) -> "MockResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def text(self) -> str:
            return "body"

    session = AsyncMock(spec=aiohttp.ClientSession)
    session.get.return_value = MockResponse()

    with patch(
        "shallweswim.clients.ndbc.provider_request_slot",
        mock_provider_request_slot,
    ):
        response = await ndbc_client._fetch_url_with_session(
            session=session,
            url="https://example.test",
            station_id="44013",
            location_code="test",
            timeout=aiohttp.ClientTimeout(total=30),
        )

    assert entered_gate
    assert response.body == "body"


def test_build_request_urls_for_current_year_historical_range() -> None:
    """Current-year historical ranges include monthly and realtime NDBC URLs."""
    urls = NdbcApi._build_request_urls(
        station_id="44013",
        mode="stdmet",
        start_time=datetime.datetime(2026, 1, 1),
        end_time=datetime.datetime(2026, 6, 4),
        now=datetime.datetime(2026, 6, 4),
    )

    assert urls == [
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=4401312026.txt.gz&dir={NDBC_DATA_PATH}stdmet/Jan/",
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=4401322026.txt.gz&dir={NDBC_DATA_PATH}stdmet/Feb/",
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=4401332026.txt.gz&dir={NDBC_DATA_PATH}stdmet/Mar/",
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=4401342026.txt.gz&dir={NDBC_DATA_PATH}stdmet/Apr/",
        f"{NDBC_BASE_URL}{NDBC_DATA_PATH}stdmet/Apr/44013.txt",
        f"{NDBC_BASE_URL}{NDBC_REALTIME_PATH}44013.txt",
    ]
    assert urls[0].startswith(f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}")
    assert urls[0].endswith(f"&dir={NDBC_DATA_PATH}stdmet/Jan/")
    assert urls[-2].endswith(f"/44013{NDBC_CURRENT_MONTH_FILE_EXTENSION}")
    assert urls[-1].endswith("/44013.txt")


def test_build_request_urls_for_ocean_current_year_historical_range() -> None:
    """Ocean archive URLs use .txt current-month files and .ocean realtime files."""
    urls = NdbcApi._build_request_urls(
        station_id="npqn6",
        mode="ocean",
        start_time=datetime.datetime(2026, 1, 1),
        end_time=datetime.datetime(2026, 6, 4),
        now=datetime.datetime(2026, 6, 4),
    )

    assert urls == [
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=npqn612026.txt.gz&dir={NDBC_DATA_PATH}ocean/Jan/",
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=npqn622026.txt.gz&dir={NDBC_DATA_PATH}ocean/Feb/",
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=npqn632026.txt.gz&dir={NDBC_DATA_PATH}ocean/Mar/",
        f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}?filename=npqn642026.txt.gz&dir={NDBC_DATA_PATH}ocean/Apr/",
        f"{NDBC_BASE_URL}{NDBC_DATA_PATH}ocean/Apr/npqn6{NDBC_CURRENT_MONTH_FILE_EXTENSION}",
        f"{NDBC_BASE_URL}{NDBC_REALTIME_PATH}NPQN6.ocean",
    ]
    assert all(NDBC_HISTORICAL_DATA_PATH not in url for url in urls[:-2])
    assert urls[-2].endswith("/npqn6.txt")
    assert urls[-1].endswith("/NPQN6.ocean")


def test_parse_stdmet_response_body() -> None:
    """NDBC whitespace text responses parse into timestamp-indexed frames."""
    body = (
        "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
        "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft\n"
        "2025 01 01 00 00 169  4.7  5.9 99.00 99.00 99.00 999 1012.6   7.0   7.6   3.2 99.0 99.00\n"
        "2025 01 01 00 10 167  4.7  5.7  0.70  9.09  4.88  89 1012.6   7.0   MM   3.3 99.0 99.00\n"
    )

    result = NdbcApi._parse_response_body(body=body, station_id="44013", mode="stdmet")

    assert list(result.index) == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 00:10:00"),
    ]
    assert result["WTMP"].iloc[0] == pytest.approx(7.6)
    assert pd.isna(result["WTMP"].iloc[1])


@pytest.mark.asyncio
async def test_execute_request_retries_http_5xx(
    ndbc_client: NdbcApi,
) -> None:
    """Retryable HTTP statuses are transient for NDBC."""
    with patch.object(
        ndbc_client,
        "_fetch_url",
        new_callable=AsyncMock,
        side_effect=RetryableClientError("NDBC request returned HTTP 503"),
    ):
        with pytest.raises(RetryableClientError, match="HTTP 503"):
            await ndbc_client._execute_request(
                station_id="44013",
                mode="stdmet",
                start_time="2025-04-19",
                end_time="2025-04-20",
                location_code="test",
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
    # Use isinstance to check if it's a DatetimeIndex before accessing tz
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert result_df.index.tz is None
    expected_index = pd.DatetimeIndex(expected_times, name="time")
    pd.testing.assert_index_equal(result_df.index, expected_index)


# Example from user about potential data issue

"""Tests for NOAA CO-OPS API client."""

# Standard library imports
import contextlib
import datetime
import io
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

# Third-party imports
import aiohttp
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Local imports
from shallweswim.clients.base import RetryableClientError
from shallweswim.clients.coops import (
    CoopsApi,
    CoopsConnectionError,
)
from shallweswim.core.feeds import to_serving_index

# Station timezone used for every request window in these tests
EASTERN = "US/Eastern"


def request_query(mock_request: AsyncMock) -> dict[str, list[str]]:
    """Return the query parameters of the URL the client requested."""
    url = cast(str, mock_request.call_args.args[0])
    return urllib.parse.parse_qs(
        urllib.parse.urlparse(url).query, keep_blank_values=True
    )


def local_day_edge_utc(date: datetime.date, *, end_of_day: bool) -> str:
    """Return a US/Eastern day edge as a CO-OPS UTC request string."""
    wall = datetime.time(23, 59) if end_of_day else datetime.time.min
    local = datetime.datetime.combine(date, wall, tzinfo=ZoneInfo(EASTERN))
    return local.astimezone(datetime.UTC).strftime("%Y%m%d %H:%M")


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
    """Mock tide prediction data, with GMT timestamps as the API returns them."""
    return pd.DataFrame(
        {
            "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Prediction": [5.2, 1.3],
            " Type": ["H", "L"],
        }
    )


@pytest.fixture
def mock_current_data() -> pd.DataFrame:
    """Mock current prediction data, with GMT timestamps as the API returns them."""
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
    """Create a mock temperature DataFrame with GMT timestamps."""
    # Data *before* _FixTime processing (as returned by _Request mock)
    data = {
        "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
        " Water Temperature": [62.5, 63.2],  # Match raw column name from API
        " Air Temperature": [65.0, 66.0],  # Include air temp too
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_fall_back_temperature_data() -> pd.DataFrame:
    """GMT readings spanning the US/Eastern fall-back hour of 2025-11-02.

    05:00Z is 01:00 EDT and 06:00Z is 01:00 EST: one repeated wall time, two
    instants. The local-time product returns only one of them.
    """
    return pd.DataFrame(
        {
            "Date Time": [
                "2025-11-02 05:00",
                "2025-11-02 06:00",
                "2025-11-02 07:00",
            ],
            " Water Temperature": [60.0, 59.0, 58.0],
        }
    )


def test_build_url_merges_base_and_request_params(coops_client: CoopsApi) -> None:
    """CO-OPS URL construction applies default API params consistently."""
    url = coops_client._build_url(
        {
            "product": "water_temperature",
            "begin_date": "20250419",
            "end_date": "20250420",
            "station": 9414290,
            "interval": None,
        }
    )

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

    assert url.startswith(CoopsApi.BASE_URL + "?")
    assert query["application"] == ["shallweswim"]
    assert query["time_zone"] == ["gmt"]
    assert query["units"] == ["english"]
    assert query["format"] == ["csv"]
    assert query["product"] == ["water_temperature"]
    assert query["begin_date"] == ["20250419"]
    assert query["end_date"] == ["20250420"]
    assert query["station"] == ["9414290"]
    assert query["interval"] == ["None"]


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
            timezone=EASTERN,
            location_code="test_loc",
        )

    assert len(df) == 2
    assert list(df.columns) == ["prediction", "type"]
    assert df["type"].tolist() == ["high", "low"]
    assert df["prediction"].tolist() == [5.2, 1.3]
    assert str(df.index.tz) == "UTC"

    # The request window stays a span of local days, expressed in UTC.
    query = request_query(mock_request)
    assert query["time_zone"] == ["gmt"]
    today = datetime.date.today()
    assert query["begin_date"] == [
        local_day_edge_utc(today - datetime.timedelta(days=1), end_of_day=False)
    ]
    assert query["end_date"] == [
        local_day_edge_utc(today + datetime.timedelta(days=2), end_of_day=True)
    ]


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
            timezone=EASTERN,
            interpolate=False,
            location_code="test_loc",
        )

    assert len(df) == 2
    assert list(df.columns) == ["velocity"]
    assert df["velocity"].tolist() == [2.5, -1.8]
    assert str(df.index.tz) == "UTC"
    assert request_query(mock_request)["time_zone"] == ["gmt"]


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
            timezone=EASTERN,
            product="water_temperature",
        )
        expected_water_df = pd.DataFrame(
            {"water_temp": [62.5, 63.2], "air_temp": [65.0, 66.0]},
            index=pd.to_datetime(
                ["2025-04-19 10:00:00", "2025-04-19 16:00:00"], utc=True
            ),
        ).rename_axis("time")  # Match index name set by _FixTime
        # Client returns all temp columns found, test needs to select the relevant one
        assert_frame_equal(df_water[["water_temp"]], expected_water_df[["water_temp"]])

        # Test air temperature
        df_air = await coops_client.temperature(
            station=9414290,
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone=EASTERN,
            product="air_temperature",
        )
        expected_air_df = pd.DataFrame(
            {"water_temp": [62.5, 63.2], "air_temp": [65.0, 66.0]},
            index=pd.to_datetime(
                ["2025-04-19 10:00:00", "2025-04-19 16:00:00"], utc=True
            ),
        ).rename_axis("time")  # Match index name set by _FixTime
        # Client returns all temp columns found, test needs to select the relevant one
        assert_frame_equal(df_air[["air_temp"]], expected_air_df[["air_temp"]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("begin_date", "end_date", "expected_begin", "expected_end"),
    [
        # Wholly inside daylight time: local midnight is 04:00Z, the last local
        # minute of the day is 03:59Z the next morning.
        (
            datetime.date(2025, 4, 19),
            datetime.date(2025, 4, 19),
            "20250419 04:00",
            "20250420 03:59",
        ),
        # Across the 2025-11-02 fall-back day: the window opens on a daylight
        # time offset and closes on a standard time one.
        (
            datetime.date(2025, 11, 1),
            datetime.date(2025, 11, 3),
            "20251101 04:00",
            "20251104 04:59",
        ),
    ],
)
async def test_temperature_request_window_converts_local_days_to_utc(
    coops_client: CoopsApi,
    mock_temperature_data: pd.DataFrame,
    begin_date: datetime.date,
    end_date: datetime.date,
    expected_begin: str,
    expected_end: str,
) -> None:
    """Local window edges reach CO-OPS as the UTC instants they name."""
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_temperature_data

        await coops_client.temperature(
            station=9414290,
            product="water_temperature",
            begin_date=begin_date,
            end_date=end_date,
            timezone=EASTERN,
        )

    query = request_query(mock_request)
    assert query["time_zone"] == ["gmt"]
    assert query["begin_date"] == [expected_begin]
    assert query["end_date"] == [expected_end]


@pytest.mark.asyncio
async def test_temperature_keeps_both_folds_of_the_fall_back_hour(
    coops_client: CoopsApi, mock_fall_back_temperature_data: pd.DataFrame
) -> None:
    """A GMT fall-back hour arrives as two distinct instants, one 01:00 local.

    This is why a fall-back year has 8761 distinct hourly instants rather than
    8760: the repeated wall time is two readings. Serving keeps the first.
    """
    with patch.object(
        coops_client, "_execute_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = mock_fall_back_temperature_data

        df = await coops_client.temperature(
            station=8518750,
            product="water_temperature",
            begin_date=datetime.date(2025, 11, 2),
            end_date=datetime.date(2025, 11, 2),
            timezone=EASTERN,
        )

    assert str(df.index.tz) == "UTC"
    assert df.index.is_unique
    local = df.index.tz_convert(EASTERN)
    assert local.strftime("%H:%M").tolist() == ["01:00", "01:00", "02:00"]
    assert df["water_temp"].tolist() == [60.0, 59.0, 58.0]

    served = to_serving_index(df, ZoneInfo(EASTERN))
    assert served.index.tz is None
    assert served.index.is_unique
    assert served.index.strftime("%H:%M").tolist() == ["01:00", "02:00"]
    # The kept 01:00 reading is the daylight time fold, the earlier instant.
    assert served["water_temp"].tolist() == [60.0, 58.0]


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
                timezone=EASTERN,
                location_code="test_conn_error",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
async def test_execute_request_retries_transient_http_statuses(
    coops_client: CoopsApi, status_code: int
) -> None:
    """Transient CO-OPS HTTP statuses are classified as retryable."""

    class MockResponse:
        status = status_code

        async def __aenter__(self) -> "MockResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def text(self) -> str:
            return ""

    cast(MagicMock, coops_client._session.get).return_value = MockResponse()

    with pytest.raises(RetryableClientError, match=f"HTTP error {status_code}"):
        await coops_client._execute_request("https://example.test", "test")


@pytest.mark.asyncio
async def test_execute_request_uses_provider_request_gate(
    coops_client: CoopsApi,
) -> None:
    """CO-OPS HTTP calls pass through the shared provider request gate."""
    entered_gate = False

    @contextlib.asynccontextmanager
    async def mock_provider_request_slot(
        provider: str, max_concurrent_requests: int
    ) -> AsyncIterator[None]:
        nonlocal entered_gate
        entered_gate = True
        assert provider == "coops"
        assert max_concurrent_requests == 4
        yield

    class MockResponse:
        status = 200

        async def __aenter__(self) -> "MockResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def text(self) -> str:
            return "Date Time, Prediction, Type\n2025-04-19 10:00,5.2,H\n"

    cast(MagicMock, coops_client._session.get).return_value = MockResponse()

    with patch(
        "shallweswim.clients.coops.provider_request_slot",
        mock_provider_request_slot,
    ):
        await coops_client._execute_request("https://example.test", "test")

    assert entered_gate


@pytest.mark.asyncio
async def test_execute_request_keeps_non_retryable_http_statuses_terminal(
    coops_client: CoopsApi,
) -> None:
    """Client/request errors stay non-retryable for CO-OPS."""

    class MockResponse:
        status = 404

        async def __aenter__(self) -> "MockResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def text(self) -> str:
            return ""

    cast(MagicMock, coops_client._session.get).return_value = MockResponse()

    with pytest.raises(CoopsConnectionError, match="HTTP error 404"):
        await coops_client._execute_request("https://example.test", "test")


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
                timezone=EASTERN,
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
            timezone=EASTERN,
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
            timezone=EASTERN,
            product=cast(Any, "water_level"),  # Intentionally invalid, cast to Any
        )

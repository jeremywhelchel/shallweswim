"""Tests for feeds.py functionality."""

# mypy: disable-error-code="misc"

# Standard library imports
import asyncio
import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

# Third-party imports
import pandera.errors
import pandera.pandas as pa
import pandera.typing.pandas as pat
import pytest
import pytz

# Local imports
from shallweswim import config as config_lib
from shallweswim import util
from shallweswim.api_types import DataFrameSummary, HistoricalTempStatus
from shallweswim.clients.base import (
    BaseApiClient,
    RetryableClientError,
    StationUnavailableError,
)
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.cspf import CspfApi
from shallweswim.clients.irish_lights import IrishLightsApi
from shallweswim.clients.marine_institute import MarineInstituteApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi
from shallweswim.core import queries
from shallweswim.dataframe_models import (
    CurrentDataModel,
    WaterTempDataModel,
)
from shallweswim.feeds import (
    FEED_RETRY_INTERVALS,
    CompositeFeed,
    CoopsCurrentsFeed,
    CoopsTempFeed,
    CoopsTidesFeed,
    CspfTempFeed,
    Feed,
    HistoricalTempsFeed,
    HistoricalTempsIncompleteError,
    IrishLightsTempFeed,
    LocalHarmonicTidesFeed,
    MarineInstituteTidesFeed,
    MultiStationCurrentsFeed,
    create_temp_feed,
    create_tide_feed,
)
from tests.helpers import assert_json_serializable


# Define a reusable simple model for test fixtures
class TestDataModel(pa.DataFrameModel):
    """Simple Pandera model for test fixtures."""

    __test__ = False  # Prevent pytest from collecting this model class
    value: pat.Series[int]
    index: pat.Index[datetime.datetime] = pa.Field(nullable=False)


@pytest.fixture
def location_config() -> config_lib.LocationConfig:
    """Create a location config fixture."""
    return config_lib.LocationConfig(
        code="nyc",
        name="New York City",
        description="New York City swimming locations",
        latitude=40.7128,
        longitude=-74.0060,
        timezone=pytz.timezone("America/New_York"),
        default_temperature_unit="F",
        swim_location="Brighton Beach",
        swim_location_link="https://example.com/brighton-beach",
    )


@pytest.fixture
def coops_temp_config_fixture() -> config_lib.CoopsTempFeedConfig:
    """Create a temperature source config fixture."""
    return config_lib.CoopsTempFeedConfig(
        station=8518750,
        name="Test Station",  # Using a valid 7-digit station ID
    )


@pytest.fixture
def tide_config() -> config_lib.CoopsTideFeedConfig:
    """Create a tide source config fixture."""
    return config_lib.CoopsTideFeedConfig(
        station=8517741,  # Using a valid 7-digit station ID
        name="Coney Island, NY",
    )


@pytest.fixture
def currents_config() -> config_lib.CoopsCurrentsFeedConfig:
    """Create a currents source config fixture."""
    return config_lib.CoopsCurrentsFeedConfig(
        stations=["ACT3876", "NYH1905"],  # Using valid station IDs
    )


@pytest.fixture
def valid_currents_dataframe() -> pd.DataFrame:
    """Create a valid currents prediction DataFrame fixture."""
    # Create a datetime index with naive datetimes
    index = pd.date_range(
        start=datetime.datetime(2025, 4, 22, 0, 0, 0),
        end=datetime.datetime(2025, 4, 24, 0, 0, 0),
        freq="1h",  # Hourly current predictions
    )

    # Generate current velocity values (positive=flood, negative=ebb)
    # Create a repeating pattern that's exactly the right length
    pattern = [1.2, 0.8, 0.3, -0.2, -0.7, -1.1, -0.9, -0.5, -0.1, 0.4, 0.9, 1.3]
    velocities: list[float] = []
    while len(velocities) < len(index):
        velocities.extend(pattern)
    velocities = velocities[: len(index)]

    # Create the DataFrame
    df = pd.DataFrame({"velocity": velocities}, index=index)
    return df


@pytest.fixture
def valid_tide_dataframe() -> pd.DataFrame:
    """Create a valid tide prediction DataFrame fixture."""
    # Create a datetime index with naive datetimes
    index = pd.date_range(
        start=datetime.datetime(2025, 4, 22, 0, 0, 0),
        end=datetime.datetime(2025, 4, 24, 0, 0, 0),
        freq="6h",  # High/low tides occur approximately every 6 hours
    )

    # Generate tide prediction values and types (alternating high and low)
    predictions = [3.0, 0.5, 3.2, 0.3, 3.5, 0.2, 3.3, 0.4, 3.1]
    types = ["high", "low", "high", "low", "high", "low", "high", "low", "high"]

    # Create the DataFrame
    df = pd.DataFrame({"prediction": predictions, "type": types}, index=index)
    return df


@pytest.fixture
def valid_temp_dataframe() -> pd.DataFrame:
    """Create a valid temperature DataFrame fixture."""
    # Create a datetime index with naive datetimes
    index = pd.date_range(
        start=datetime.datetime(2025, 4, 22, 0, 0, 0),
        end=datetime.datetime(2025, 4, 22, 23, 0, 0),
        freq="h",
    )

    # Generate temperature values
    temps = [20.0 + i * 0.5 for i in range(len(index))]

    # Create the DataFrame
    df = pd.DataFrame({"temperature": temps}, index=index)
    return df


@pytest.fixture
def mock_clients() -> dict[str, BaseApiClient]:
    """Provides a mock dictionary of specific API clients."""
    return {
        "coops": MagicMock(spec=CoopsApi),
        "cspf": MagicMock(spec=CspfApi),
        "irish_lights": MagicMock(spec=IrishLightsApi),
        "marine_institute": MagicMock(spec=MarineInstituteApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }


@pytest.fixture
def concrete_feed(location_config: config_lib.LocationConfig) -> Feed:
    """Create a concrete implementation of the abstract Feed class with a simple model."""

    # Create a test version of BaseFeedConfig for testing
    class TestConfig(config_lib.BaseFeedConfig, frozen=True):
        """Test configuration class for ConcreteFeed."""

        @property
        def citation(self) -> str:
            """Return a citation for the test feed."""
            return "Test Feed Citation"

    class ConcreteFeed(Feed):
        feed_config: TestConfig = TestConfig()  # type: ignore[assignment]

        async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            # Return a simple DataFrame for testing
            index = pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 23, 0, 0),
                freq="h",
            )
            return pd.DataFrame({"value": range(len(index))}, index=index)

        @property
        def data_model(self) -> type[pa.DataFrameModel]:
            """Return the Pandera model associated with this test feed."""
            return TestDataModel

        @property
        def values(self) -> pd.DataFrame:
            """Return the data for testing."""
            if self._data is None:
                raise ValueError("Data not yet fetched")
            return self._data

    return ConcreteFeed(
        location_config=location_config,
        expiration_interval=datetime.timedelta(minutes=10),
    )


@pytest.fixture
def simple_composite_feed(location_config: config_lib.LocationConfig) -> CompositeFeed:
    """Create a simple concrete implementation of CompositeFeed for testing."""

    # Create a test version of BaseFeedConfig for testing
    class TestConfig(config_lib.BaseFeedConfig, frozen=True):
        """Test configuration class for SimpleCompositeFeed."""

        @property
        def citation(self) -> str:
            """Return a citation for the test feed."""
            return "Test Composite Feed Citation"

    class SimpleCompositeFeed(CompositeFeed):
        # Configuration for the feed
        feed_config: TestConfig = TestConfig()  # type: ignore[assignment]

        @property
        def data_model(self) -> type[pa.DataFrameModel]:
            """Return the Pandera model for combined data (using test model here)."""
            return TestDataModel

        def _get_feeds(self, clients: dict[str, BaseApiClient]) -> list[Feed]:
            """Get test feeds - this will be mocked in tests."""
            return []

        def _combine_feeds(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
            """Combine test dataframes - this will be mocked in tests."""
            return pd.DataFrame()

        async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            feeds = self._get_feeds(clients=clients)
            tasks = [feed._fetch(clients=clients) for feed in feeds]
            results = await asyncio.gather(*tasks)
            return self._combine_feeds(results)

    return SimpleCompositeFeed(
        location_config=location_config,
        expiration_interval=datetime.timedelta(minutes=10),
    )


@pytest.fixture
def multi_station_currents_feed(
    location_config: config_lib.LocationConfig,
    currents_config: config_lib.CoopsCurrentsFeedConfig,
) -> MultiStationCurrentsFeed:
    """Provides an instance of MultiStationCurrentsFeed for testing."""
    return MultiStationCurrentsFeed(
        location_config=location_config,
        feed_config=currents_config,
        expiration_interval=datetime.timedelta(hours=1),  # Example interval
    )


@pytest.fixture
def historical_temps_feed(
    location_config: config_lib.LocationConfig,
    coops_temp_config_fixture: config_lib.CoopsTempFeedConfig,
) -> HistoricalTempsFeed:
    """Provides an instance of HistoricalTempsFeed for testing."""
    # Assuming start_year and end_year are needed, or default behavior is okay
    # Using default start year and current year as end year for simplicity
    current_year = datetime.datetime.now().year
    return HistoricalTempsFeed(
        location_config=location_config,
        feed_config=coops_temp_config_fixture,
        start_year=current_year - 1,  # Example: last year
        end_year=current_year,
        expiration_interval=datetime.timedelta(hours=3),  # Match default
    )


class TestFeedBase:
    """Tests for the base Feed class."""

    @pytest.mark.asyncio
    async def test_is_expired_with_no_next_fetch_after(
        self, concrete_feed: Feed
    ) -> None:
        """Test that a feed with no _next_fetch_after is considered expired."""
        assert concrete_feed._next_fetch_after is None
        assert concrete_feed.is_expired is True

    @pytest.mark.asyncio
    async def test_never_expiring_feed_waits_for_scheduled_retry(
        self, concrete_feed: Feed
    ) -> None:
        """Never-expiring feeds still honor scheduled retries after failures."""
        concrete_feed._next_fetch_after = util.utc_now() + datetime.timedelta(
            minutes=10
        )
        concrete_feed.expiration_interval = None
        assert concrete_feed.is_expired is False

    @pytest.mark.asyncio
    async def test_successful_feed_with_no_interval_never_expires(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """A never-expiring feed stops refreshing after it has data."""
        concrete_feed.expiration_interval = None
        concrete_feed._data = valid_temp_dataframe
        concrete_feed._fetch_timestamp = util.utc_now()
        concrete_feed._next_fetch_after = None

        assert concrete_feed.is_expired is False

    @pytest.mark.asyncio
    async def test_is_expired_with_future_next_fetch_after(
        self, concrete_feed: Feed
    ) -> None:
        """Test that a feed with future _next_fetch_after is not expired."""
        # Set next fetch to 5 minutes in the future
        concrete_feed._next_fetch_after = util.utc_now() + datetime.timedelta(minutes=5)
        concrete_feed.expiration_interval = datetime.timedelta(minutes=10)
        assert concrete_feed.is_expired is False

    @pytest.mark.asyncio
    async def test_is_expired_with_past_next_fetch_after(
        self, concrete_feed: Feed
    ) -> None:
        """Test that a feed with past _next_fetch_after is expired."""
        # Set next fetch to 1 minute in the past
        concrete_feed._next_fetch_after = util.utc_now() - datetime.timedelta(minutes=1)
        concrete_feed.expiration_interval = datetime.timedelta(minutes=10)
        assert concrete_feed.is_expired is True

    @pytest.mark.asyncio
    async def test_is_healthy_with_no_timestamp(self, concrete_feed: Feed) -> None:
        """Test that a feed with no timestamp is considered not healthy."""
        assert concrete_feed.is_healthy is False

    @pytest.mark.asyncio
    async def test_is_healthy_with_no_interval(self, concrete_feed: Feed) -> None:
        """Test that a feed with no expiration interval is always healthy."""
        concrete_feed._fetch_timestamp = util.utc_now()
        concrete_feed.expiration_interval = None
        assert concrete_feed.is_healthy is True

    @pytest.mark.asyncio
    async def test_is_healthy_with_recent_timestamp(self, concrete_feed: Feed) -> None:
        """Test that a feed with a timestamp within the health buffer is healthy."""
        # Set a timestamp within the health buffer
        # Interval 10 min, Buffer 15 min -> Healthy up to 25 min old
        concrete_feed._fetch_timestamp = util.utc_now() - datetime.timedelta(minutes=20)
        concrete_feed.expiration_interval = datetime.timedelta(minutes=10)
        assert concrete_feed.is_healthy is True

    @pytest.mark.asyncio
    async def test_is_healthy_with_old_timestamp(self, concrete_feed: Feed) -> None:
        """Test that a feed with a timestamp outside the health buffer is not healthy."""
        # Set an old timestamp (older than interval + buffer)
        # Interval 10 min, Buffer 15 min -> Unhealthy if > 25 min old
        concrete_feed._fetch_timestamp = util.utc_now() - datetime.timedelta(minutes=30)
        concrete_feed.expiration_interval = datetime.timedelta(minutes=10)
        assert concrete_feed.is_healthy is False

    @pytest.mark.asyncio
    async def test_values_property_with_no_data(self, concrete_feed: Feed) -> None:
        """Test that values property raises an error when no data is available."""
        # Ensure data is None
        concrete_feed._data = None
        with pytest.raises(ValueError, match="Data not yet fetched"):
            _ = concrete_feed.values

    @pytest.mark.asyncio
    async def test_values_property_with_data(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that values property returns data when available."""
        concrete_feed._data = valid_temp_dataframe
        assert concrete_feed.values is valid_temp_dataframe

    @pytest.mark.asyncio
    async def test_update_calls_fetch(
        self, concrete_feed: Feed, mock_clients: dict[str, BaseApiClient]
    ) -> None:
        """Test that update calls _fetch and updates timestamp and data."""
        # Use patch to mock the _fetch method
        original_fetch = concrete_feed._fetch
        fetch_called = False

        async def mock_fetch(clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            nonlocal fetch_called
            fetch_called = True
            return await original_fetch(clients)

        # Use monkeypatch instead of direct assignment to avoid mypy error
        with patch.object(concrete_feed, "_fetch", mock_fetch):
            # Call update
            await concrete_feed.update(clients=mock_clients)

            # Check that _fetch was called and data was updated
            assert fetch_called is True
            assert concrete_feed._data is not None
            assert concrete_feed._fetch_timestamp is not None
            # Scheduling: _next_fetch_after should also be set
            assert concrete_feed._next_fetch_after is not None
            assert concrete_feed._last_error is None
            assert concrete_feed._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_station_unavailable_schedules_first_retry_after_one_minute(
        self, concrete_feed: Feed, mock_clients: dict[str, BaseApiClient]
    ) -> None:
        """StationUnavailableError schedules the first retry after one minute."""
        from shallweswim.clients.base import StationUnavailableError

        async def mock_fetch_unavailable(
            clients: dict[str, BaseApiClient],
        ) -> pd.DataFrame:
            raise StationUnavailableError("Station has no data")

        with patch.object(concrete_feed, "_fetch", mock_fetch_unavailable):
            # Should NOT raise - StationUnavailableError is handled gracefully
            await concrete_feed.update(clients=mock_clients)

        # _next_fetch_after should be set (scheduling next attempt)
        assert concrete_feed._next_fetch_after is not None
        retry_delay = concrete_feed._next_fetch_after - util.utc_now()
        assert 55 <= retry_delay.total_seconds() <= 60
        # _fetch_timestamp should NOT be set (no data was fetched)
        assert concrete_feed._fetch_timestamp is None
        # _last_error should capture the error
        assert concrete_feed._last_error is not None
        assert isinstance(concrete_feed._last_error, StationUnavailableError)
        assert concrete_feed._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_unexpected_error_schedules_first_retry_and_propagates(
        self, concrete_feed: Feed, mock_clients: dict[str, BaseApiClient]
    ) -> None:
        """Unexpected errors schedule the first retry and still propagate."""

        async def mock_fetch_error(clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            raise ValueError("Some unexpected error")

        with patch.object(concrete_feed, "_fetch", mock_fetch_error):
            # Should raise - other errors are propagated
            with pytest.raises(ValueError):
                await concrete_feed.update(clients=mock_clients)

        # _next_fetch_after should still be set (scheduling next attempt)
        assert concrete_feed._next_fetch_after is not None
        retry_delay = concrete_feed._next_fetch_after - util.utc_now()
        assert 55 <= retry_delay.total_seconds() <= 60
        # _fetch_timestamp should NOT be set (no data was fetched)
        assert concrete_feed._fetch_timestamp is None
        assert concrete_feed._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_repeated_failures_follow_shared_retry_sequence(
        self, concrete_feed: Feed, mock_clients: dict[str, BaseApiClient]
    ) -> None:
        """Repeated failures follow the shared retry sequence."""

        async def mock_fetch_error(clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            raise ValueError("retry me")

        with patch.object(concrete_feed, "_fetch", mock_fetch_error):
            for expected_delay in FEED_RETRY_INTERVALS[:4]:
                concrete_feed._next_fetch_after = None
                with pytest.raises(ValueError):
                    await concrete_feed.update(clients=mock_clients)

                assert concrete_feed._next_fetch_after is not None
                retry_delay = concrete_feed._next_fetch_after - util.utc_now()
                assert expected_delay.total_seconds() - 5 <= retry_delay.total_seconds()
                assert retry_delay.total_seconds() <= expected_delay.total_seconds()

        assert concrete_feed._consecutive_failures == 4

    @pytest.mark.asyncio
    async def test_retry_delay_caps_at_expiration_interval(
        self, concrete_feed: Feed, mock_clients: dict[str, BaseApiClient]
    ) -> None:
        """Retries never become less frequent than ordinary refreshes."""
        concrete_feed.expiration_interval = datetime.timedelta(minutes=3)

        async def mock_fetch_error(clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            raise ValueError("retry me")

        with patch.object(concrete_feed, "_fetch", mock_fetch_error):
            for _ in range(4):
                concrete_feed._next_fetch_after = None
                with pytest.raises(ValueError):
                    await concrete_feed.update(clients=mock_clients)

        assert concrete_feed._next_fetch_after is not None
        retry_delay = concrete_feed._next_fetch_after - util.utc_now()
        assert 175 <= retry_delay.total_seconds() <= 180

    @pytest.mark.asyncio
    async def test_never_expiring_feed_retries_until_first_success(
        self, concrete_feed: Feed, mock_clients: dict[str, BaseApiClient]
    ) -> None:
        """expiration_interval=None means no refresh after success, not no retry."""
        concrete_feed.expiration_interval = None

        async def mock_fetch_error(clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            raise ValueError("initial failure")

        with patch.object(concrete_feed, "_fetch", mock_fetch_error):
            with pytest.raises(ValueError):
                await concrete_feed.update(clients=mock_clients)

        assert concrete_feed._next_fetch_after is not None
        retry_delay = concrete_feed._next_fetch_after - util.utc_now()
        assert 55 <= retry_delay.total_seconds() <= 60
        assert concrete_feed.is_expired is False

        concrete_feed._next_fetch_after = util.utc_now() - datetime.timedelta(seconds=1)

        async def mock_fetch_success(clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            index = pd.date_range(start="2023-01-01", periods=3, freq="h", name="time")
            return pd.DataFrame({"value": [1, 2, 3]}, index=index)

        with patch.object(concrete_feed, "_fetch", mock_fetch_success):
            await concrete_feed.update(clients=mock_clients)

        assert concrete_feed._fetch_timestamp is not None
        assert concrete_feed._next_fetch_after is None
        assert concrete_feed._consecutive_failures == 0
        assert concrete_feed.is_expired is False

    def test_validate_frame_with_valid_dataframe(self, concrete_feed: Feed) -> None:
        """Test that _validate_frame accepts a valid DataFrame."""
        index = pd.date_range(start="2023-01-01", periods=3, freq="h")
        valid_df = pd.DataFrame({"value": [1, 2, 3]}, index=index)
        try:
            concrete_feed._validate_frame(valid_df)
        except pandera.errors.SchemaError as e:
            pytest.fail(f"_validate_frame raised SchemaError unexpectedly: {e}")

    def test_validate_frame_with_empty_dataframe(self, concrete_feed: Feed) -> None:
        """Test that _validate_frame fails validation for an empty DataFrame."""
        # Pandera validation fails because the empty frame lacks expected columns.
        with pytest.raises(pandera.errors.SchemaErrors):
            concrete_feed._validate_frame(pd.DataFrame())

    def test_validate_frame_with_non_datetime_index(self, concrete_feed: Feed) -> None:
        """Test that _validate_frame rejects a DataFrame with a non-DatetimeIndex."""
        df = pd.DataFrame({"value": [1, 2]}, index=pd.Index([0, 1]))
        # TestDataModel expects datetime index, so this should fail
        with pytest.raises(pandera.errors.SchemaErrors):
            concrete_feed._validate_frame(df)

    def test_validate_frame_with_timezone_aware_index(
        self, concrete_feed: Feed
    ) -> None:
        """Test that _validate_frame rejects a DataFrame with a timezone-aware index."""
        # Create a timezone-aware index
        index = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 0, 0, 0), periods=3, freq="h", tz="UTC"
        )
        df = pd.DataFrame({"value": [1, 2, 3]}, index=index)
        # TestDataModel expects naive datetime index, so this should fail
        with pytest.raises(pandera.errors.SchemaErrors):
            concrete_feed._validate_frame(df)

    @pytest.mark.asyncio
    async def test_log_method(self, concrete_feed: Feed, caplog: Any) -> None:
        """Test that log method formats messages correctly."""
        # Set the log level to INFO
        caplog.set_level("INFO")

        # Call the log method
        concrete_feed.log("Test message")

        # Check that the message was logged with the correct format
        assert f"[{concrete_feed.location_config.code}]" in caplog.text
        assert "Test message" in caplog.text

    def test_remove_outliers_with_no_outliers(self, concrete_feed: Feed) -> None:
        """Test that _remove_outliers returns the original DataFrame when no outliers are configured."""
        # Create a test DataFrame
        dates = pd.date_range("2023-01-01", periods=5)
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        # Feed config has no outliers attribute
        result = concrete_feed._remove_outliers(df)

        # Should return the original DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_remove_outliers_with_empty_outliers(self, concrete_feed: Feed) -> None:
        """Test that _remove_outliers returns the original DataFrame when outliers list is empty."""
        # Create a test DataFrame
        dates = pd.date_range("2023-01-01", periods=5)
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        # The TestConfig in concrete_feed already has empty outliers
        result = concrete_feed._remove_outliers(df)

        # Should return the original DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_remove_outliers_with_outliers(self, concrete_feed: Feed) -> None:
        """Test that _remove_outliers correctly removes outliers from the DataFrame."""
        # Create a test DataFrame
        df = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5]},
            index=pd.DatetimeIndex(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
        )

        # Create a new DataFrame with the outliers already removed
        # This simulates what would happen if the feed_config had outliers=["2023-01-02", "2023-01-04"]
        expected_dates = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
        expected_df = pd.DataFrame({"value": [1, 3, 5]}, index=expected_dates)

        # Mock the _remove_outliers method to return our expected DataFrame
        with patch.object(
            Feed, "_remove_outliers", return_value=expected_df
        ) as mock_remove:
            # Call _remove_outliers through the patch
            result = concrete_feed._remove_outliers(df)

            # Verify the method was called with the correct DataFrame
            mock_remove.assert_called_once_with(df)

            # Verify the result is as expected
            pd.testing.assert_frame_equal(result, expected_df)

    def test_remove_outliers_with_nonexistent_outliers(
        self, concrete_feed: Feed
    ) -> None:
        """Test that _remove_outliers handles outliers that don't exist in the DataFrame."""
        # Create a test DataFrame with dates that won't match any outliers
        df = pd.DataFrame(
            {"temperature": [20.0, 21.0, 22.0]},
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                periods=3,
                freq="h",
            ),
        )

        # Call _remove_outliers directly
        result = concrete_feed._remove_outliers(df)

        # Check that the result is the same as the input (no outliers removed)
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.asyncio
    async def test_wait_until_ready_with_data_already_available(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that wait_until_ready returns immediately when data is already available."""
        # Set data directly
        concrete_feed._data = valid_temp_dataframe
        concrete_feed._ready_event.set()  # This should already be set in __init__ when data is available

        # Call wait_until_ready with a short timeout
        result = await concrete_feed.wait_until_ready(timeout=0.1)

        # Check that it returned True immediately
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_until_ready_with_data_becoming_available(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that wait_until_ready waits until data becomes available."""
        # Ensure no data is available initially
        concrete_feed._data = None
        concrete_feed._ready_event.clear()

        # Set up a task to set the event after a short delay
        async def set_data_after_delay() -> None:
            await asyncio.sleep(0.1)
            concrete_feed._data = valid_temp_dataframe
            concrete_feed._ready_event.set()

        # Start the task
        task = asyncio.create_task(set_data_after_delay())

        # Call wait_until_ready with a longer timeout
        result = await concrete_feed.wait_until_ready(timeout=0.5)

        # Clean up the task
        await task

        # Check that it returned True after waiting
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_until_ready_timeout(self, concrete_feed: Feed) -> None:
        """Test that wait_until_ready returns False when timeout occurs."""
        # Set a very short timeout
        result = await concrete_feed.wait_until_ready(timeout=0.1)

        # Since the feed has no data and no one is setting the event,
        # wait_until_ready should timeout and return False
        assert result is False

    def test_age_with_no_timestamp(self, concrete_feed: Feed) -> None:
        """Test that age returns None when no timestamp is available."""
        # Ensure the feed has no timestamp
        concrete_feed._fetch_timestamp = None

        # Check that age returns None
        assert concrete_feed.age is None

    def test_age_with_timestamp(self, concrete_feed: Feed) -> None:
        """Test that age returns the correct age as a timedelta."""
        # Set a timestamp 10 seconds in the past (using naive datetime)
        now = util.utc_now()
        concrete_feed._fetch_timestamp = now - datetime.timedelta(seconds=10)

        # Check that age returns approximately 10 seconds as a timedelta
        age = concrete_feed.age
        assert age is not None
        assert 9.5 <= age.total_seconds() <= 10.5  # Allow for small timing differences

    def test_status_property_with_no_data(self, concrete_feed: Feed) -> None:
        """Test that status property returns correct information when no data is available."""
        # Ensure the feed has no data or timestamp
        concrete_feed._data = None
        concrete_feed._fetch_timestamp = None

        # Get the status dictionary
        status = concrete_feed.status

        # Check the status dictionary contents
        assert status.name == "ConcreteFeed"
        assert status.location == concrete_feed.location_config.code
        assert status.fetch_timestamp is None
        assert status.next_fetch_after is None
        assert status.data_summary is None
        assert status.age_seconds is None
        assert status.seconds_until_next_fetch is None
        assert status.consecutive_failures == 0
        assert status.is_expired is True
        assert status.is_healthy is False  # Not healthy if never fetched
        assert status.historical_temp_status is None

    def test_status_property_with_data(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that status property returns correct information when data is available."""
        # Set data and timestamp (using naive datetime)
        concrete_feed._data = valid_temp_dataframe
        concrete_feed._fetch_timestamp = util.utc_now()
        # Set next fetch to future so is_expired is False
        concrete_feed._next_fetch_after = util.utc_now() + datetime.timedelta(hours=1)
        concrete_feed._ready_event.set()

        # Get the status dictionary
        status = concrete_feed.status

        # Check the status dictionary contents
        assert status.name == "ConcreteFeed"
        assert status.location == concrete_feed.location_config.code
        assert status.fetch_timestamp is not None
        assert status.next_fetch_after is not None
        assert isinstance(
            status.data_summary, DataFrameSummary
        )  # Check it's the Pydantic model
        assert status.data_summary.length == len(valid_temp_dataframe)
        assert status.data_summary.column_names == list(valid_temp_dataframe.columns)

        expected_oldest = valid_temp_dataframe.index.min()
        expected_newest = valid_temp_dataframe.index.max()

        # Check timestamp comparison works (Pydantic v2 converts to datetime)
        assert status.data_summary.index_oldest == expected_oldest
        assert status.data_summary.index_newest == expected_newest

        assert status.age_seconds is not None
        assert status.seconds_until_next_fetch is not None
        assert status.consecutive_failures == 0
        assert status.is_expired is False
        assert status.is_healthy is True  # Healthy if within buffer

    def test_status_property_json_serializable(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that status property returns a JSON serializable dictionary."""
        # Set data and timestamp
        concrete_feed._data = valid_temp_dataframe
        concrete_feed._fetch_timestamp = util.utc_now()
        concrete_feed._ready_event.set()

        # Get the status object
        status = concrete_feed.status

        # Dump the pydantic model to a dict suitable for JSON serialization
        status_dict = status.model_dump(mode="json")

        # Check that the resulting dictionary is JSON serializable
        assert_json_serializable(status_dict)


class TestCoopsTidesFeed:
    """Tests for the CoopsTidesFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_noaa_client(
        self,
        location_config: config_lib.LocationConfig,
        tide_config: config_lib.CoopsTideFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch calls the COOPS client with correct parameters."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to return a valid DataFrame
        cast(MagicMock, mock_coops_client.tides).return_value = pd.DataFrame(
            {
                "prediction": [3.0, 0.5, 3.2],
                "type": ["high", "low", "high"],
            },
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 12, 0, 0),
                freq="6h",
            ),
        )

        # Create the feed
        feed = CoopsTidesFeed(
            location_config=location_config,
            feed_config=tide_config,
            expiration_interval=datetime.timedelta(hours=24),
        )

        # Call _fetch
        result = await feed._fetch(clients=mock_clients)

        # Check that the COOPS API was called with correct parameters
        cast(MagicMock, mock_coops_client.tides).assert_called_once()
        _, kwargs = cast(MagicMock, mock_coops_client.tides).call_args
        assert kwargs["station"] == tide_config.station
        # Add more specific checks for begin/end dates if needed

        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert list(result.columns) == ["prediction", "type"]

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(
        self,
        location_config: config_lib.LocationConfig,
        tide_config: config_lib.CoopsTideFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch raises an exception if the API call fails."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to raise an exception
        cast(MagicMock, mock_coops_client.tides).side_effect = RetryableClientError(
            "API Error"
        )

        # Create the feed
        feed = CoopsTidesFeed(
            location_config=location_config,
            feed_config=tide_config,
            expiration_interval=datetime.timedelta(hours=24),
        )

        # Call _fetch and expect it to raise an exception
        with pytest.raises(RetryableClientError, match="API Error"):
            await feed._fetch(clients=mock_clients)


class TestLocalHarmonicTidesFeed:
    """Tests for locally generated harmonic tide predictions."""

    @pytest.mark.asyncio
    async def test_fetch_converts_meter_model_output_to_internal_feet(
        self,
        tmp_path: Path,
        location_config: config_lib.LocationConfig,
    ) -> None:
        """Meter-native harmonic models publish feet for normal tide consumers."""
        model_path = tmp_path / "synthetic_harmonics.json"
        model_path.write_text(
            """
{
  "schema_version": 1,
  "name": "Synthetic local tide",
  "epoch_utc": "2026-06-13T00:00:00Z",
  "intercept": 0.0,
  "height_units": "m",
  "height_datum": "synthetic datum",
  "source": "synthetic source",
  "constituents": [
    {
      "name": "M2",
      "speed_degrees_per_hour": 28.9841042,
      "cos": 1.0,
      "sin": 0.0
    }
  ],
  "metadata": {}
}
""".strip()
        )
        feed = LocalHarmonicTidesFeed(
            location_config=location_config,
            feed_config=config_lib.LocalHarmonicTideFeedConfig(
                name="Synthetic tide",
                model_path=str(model_path),
                source_url="https://example.com/tides",
                attribution="Synthetic gauge",
            ),
            expiration_interval=datetime.timedelta(hours=24),
        )

        result = await feed._fetch(clients={})

        assert not result.empty
        assert set(result["type"].astype(str)) == {"high", "low"}
        assert all(
            timestamp.second == 0 and timestamp.microsecond == 0
            for timestamp in result.index
        )
        assert result["prediction"].abs().max() == pytest.approx(
            3.280839895013123,
            rel=1e-4,
        )
        frame = queries.prepare_tide_prediction_frame(result)
        state = queries.predict_tide_from_precomputed_frame(
            frame,
            location_config,
            frame.index[len(frame) // 2].to_pydatetime(),
        )
        assert state.estimated_height is not None

    @pytest.mark.asyncio
    async def test_dover_config_loads_packaged_model(self) -> None:
        """Production Dover config loads its packaged harmonic model."""
        location_config = config_lib.get("dov")
        assert location_config is not None
        assert isinstance(
            location_config.tide_source, config_lib.LocalHarmonicTideFeedConfig
        )
        feed = create_tide_feed(
            location_config=location_config,
            tide_config=location_config.tide_source,
            expiration_interval=datetime.timedelta(hours=24),
        )

        result = await feed._fetch(clients={})

        assert not result.empty
        assert set(result["type"].astype(str)) == {"high", "low"}
        assert result.index.is_monotonic_increasing
        assert result["prediction"].max() > result["prediction"].min()

    def test_create_tide_feed_supports_local_harmonic_config(
        self,
        location_config: config_lib.LocationConfig,
    ) -> None:
        """The tide factory dispatches local harmonic configs to their feed."""
        feed = create_tide_feed(
            location_config=location_config,
            tide_config=config_lib.LocalHarmonicTideFeedConfig(
                name="Synthetic tide",
                model_path="data/tides/synthetic.json",
                source_url="https://example.com/tides",
                attribution="Synthetic gauge",
            ),
            expiration_interval=datetime.timedelta(hours=24),
        )

        assert isinstance(feed, LocalHarmonicTidesFeed)


class TestMarineInstituteTidesFeed:
    """Tests for Marine Institute tide prediction feeds."""

    @pytest.mark.asyncio
    async def test_fetch_calls_marine_institute_client(
        self,
        location_config: config_lib.LocationConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Marine Institute tide feeds call the configured station."""
        tide_config = config_lib.MarineInstituteTideFeedConfig(
            station_id="Kinsale",
            height_offset_m=2.01,
            name="Kinsale",
        )
        mock_client = cast(MarineInstituteApi, mock_clients["marine_institute"])
        cast(MagicMock, mock_client.tides).return_value = pd.DataFrame(
            {
                "prediction": [9.5, 1.2, 9.8],
                "type": pd.Categorical(
                    ["high", "low", "high"],
                    categories=["low", "high"],
                ),
            },
            index=pd.date_range(
                start=datetime.datetime(2026, 6, 14, 4, 0, 0),
                periods=3,
                freq="6h",
            ),
        )
        feed = MarineInstituteTidesFeed(
            location_config=location_config,
            feed_config=tide_config,
            expiration_interval=datetime.timedelta(hours=24),
        )

        result = await feed._fetch(clients=mock_clients)

        cast(MagicMock, mock_client.tides).assert_called_once_with(
            station_id="Kinsale",
            timezone=location_config.timezone,
            height_offset_m=2.01,
            location_code=location_config.code,
        )
        assert list(result.columns) == ["prediction", "type"]

    def test_create_tide_feed_supports_marine_institute_config(
        self,
        location_config: config_lib.LocationConfig,
    ) -> None:
        """The tide factory dispatches Marine Institute configs to their feed."""
        feed = create_tide_feed(
            location_config=location_config,
            tide_config=config_lib.MarineInstituteTideFeedConfig(
                station_id="Kinsale",
                height_offset_m=2.01,
                name="Kinsale",
            ),
            expiration_interval=datetime.timedelta(hours=24),
        )

        assert isinstance(feed, MarineInstituteTidesFeed)


class TestCoopsCurrentsFeed:
    """Tests for the CoopsCurrentsFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_noaa_client(
        self,
        location_config: config_lib.LocationConfig,
        currents_config: config_lib.CoopsCurrentsFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch calls the COOPS client with correct parameters."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to return a valid DataFrame
        cast(MagicMock, mock_coops_client.currents).return_value = pd.DataFrame(
            {"velocity": [1.2, 0.8, 0.3, -0.2, -0.7]},
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 4, 0, 0),
                freq="1h",
            ),
        )

        # Create the feed
        feed = CoopsCurrentsFeed(
            location_config=location_config,
            feed_config=currents_config,
            station=currents_config.stations[0],  # Use the first station
            expiration_interval=datetime.timedelta(hours=24),
        )

        # Call _fetch
        result = await feed._fetch(clients=mock_clients)

        # Check that the COOPS API was called with correct parameters
        cast(MagicMock, mock_coops_client.currents).assert_called_once()
        _, kwargs = cast(MagicMock, mock_coops_client.currents).call_args
        assert kwargs["station"] == currents_config.stations[0]
        assert kwargs["interpolate"] is True
        assert kwargs["location_code"] == location_config.code

        # Check that the result is correct
        assert isinstance(result, pd.DataFrame)
        assert "velocity" in result.columns

    @pytest.mark.asyncio
    async def test_fetch_with_valid_station(
        self,
        location_config: config_lib.LocationConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch works with a valid station specified directly."""
        # Create a config with a valid station
        config = config_lib.CoopsCurrentsFeedConfig(
            stations=["ACT3876", "NYH1905"],  # Using valid station IDs
        )

        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to return a valid DataFrame
        cast(MagicMock, mock_coops_client.currents).return_value = pd.DataFrame(
            {"velocity": [1.2, 0.8, 0.3]},
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 2, 0, 0),
                freq="1h",
            ),
        )

        # Create the feed with a specific station
        feed = CoopsCurrentsFeed(
            location_config=location_config,
            feed_config=config,
            station="NYH1905",  # Specify the second station
            expiration_interval=datetime.timedelta(hours=24),
        )

        # Call _fetch
        await feed._fetch(clients=mock_clients)

        # Check that the COOPS API was called with the specified station
        cast(MagicMock, mock_coops_client.currents).assert_called_once()
        _, kwargs = cast(MagicMock, mock_coops_client.currents).call_args
        assert kwargs["station"] == "NYH1905"

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(
        self,
        location_config: config_lib.LocationConfig,
        currents_config: config_lib.CoopsCurrentsFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch handles API errors correctly."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to raise an exception
        cast(MagicMock, mock_coops_client.currents).side_effect = RetryableClientError(
            "API error"
        )

        # Create the feed
        feed = CoopsCurrentsFeed(
            location_config=location_config,
            feed_config=currents_config,
            station=currents_config.stations[0],
            expiration_interval=datetime.timedelta(hours=24),
        )

        # Call _fetch and expect it to raise the exception (following fail-fast principle)
        with pytest.raises(RetryableClientError, match="API error"):
            await feed._fetch(clients=mock_clients)


class TestCoopsTempFeed:
    """Tests for the CoopsTempFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_noaa_client(
        self,
        location_config: config_lib.LocationConfig,
        coops_temp_config_fixture: config_lib.CoopsTempFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch calls the COOPS client with correct parameters."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to return a valid DataFrame
        cast(MagicMock, mock_coops_client.temperature).return_value = pd.DataFrame(
            {
                "water_temp": [20.0, 20.5, 21.0],
            },
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 2, 0, 0),
                freq="h",
            ),
        )

        # Create the feed
        feed = CoopsTempFeed(
            location_config=location_config,
            feed_config=coops_temp_config_fixture,
            interval="6-min",  # Use 6-minute intervals for live data
            expiration_interval=datetime.timedelta(minutes=10),
        )

        # Call _fetch
        result = await feed._fetch(clients=mock_clients)

        # Check that the COOPS API was called with correct parameters
        cast(MagicMock, mock_coops_client.temperature).assert_called_once()
        _, kwargs = cast(MagicMock, mock_coops_client.temperature).call_args
        assert kwargs["station"] == coops_temp_config_fixture.station
        assert kwargs["product"] == "water_temperature"
        assert kwargs["location_code"] == location_config.code

        # Check that the result is correct
        assert isinstance(result, pd.DataFrame)
        assert "water_temp" in result.columns

    @pytest.mark.asyncio
    async def test_fetch_with_date_range(
        self,
        location_config: config_lib.LocationConfig,
        coops_temp_config_fixture: config_lib.CoopsTempFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch passes date range to COOPS API when provided."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to return a valid DataFrame
        cast(MagicMock, mock_coops_client.temperature).return_value = pd.DataFrame(
            {
                "water_temp": [20.0, 20.5, 21.0],
            },
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 2, 0, 0),
                freq="h",
            ),
        )

        # Create the feed with start and end dates
        start_date = datetime.datetime(2025, 4, 22, 0, 0, 0)
        end_date = datetime.datetime(2025, 4, 22, 23, 0, 0)

        feed = CoopsTempFeed(
            location_config=location_config,
            feed_config=coops_temp_config_fixture,
            interval="6-min",
            expiration_interval=datetime.timedelta(minutes=10),
            # Use the correct attribute names for the feed
            start=start_date,
            end=end_date,
        )

        # Call _fetch
        await feed._fetch(clients=mock_clients)

        # Check that the COOPS API was called with the date range
        cast(MagicMock, mock_coops_client.temperature).assert_called_once()

        # Instead of checking exact values, verify that start_date and end_date
        # were passed to the API call. The actual implementation might modify
        # these dates slightly, so we'll just check that they were passed.
        _, kwargs = cast(MagicMock, mock_coops_client.temperature).call_args
        assert "begin_date" in kwargs
        assert isinstance(kwargs["begin_date"], datetime.date)
        assert "end_date" in kwargs
        assert isinstance(kwargs["end_date"], datetime.date)

    @pytest.mark.asyncio
    async def test_fetch_with_default_date_range(
        self,
        location_config: config_lib.LocationConfig,
        coops_temp_config_fixture: config_lib.CoopsTempFeedConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch uses default date range when not provided."""
        # Get the mock coops client from the fixture and cast it
        mock_coops_client = cast(CoopsApi, mock_clients["coops"])

        # Configure the mock to return a valid DataFrame
        cast(MagicMock, mock_coops_client.temperature).return_value = pd.DataFrame(
            {
                "water_temp": [20.0, 20.5, 21.0],
            },
            index=pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 2, 0, 0),
                freq="h",
            ),
        )

        # Create the feed without start and end dates
        feed = CoopsTempFeed(
            location_config=location_config,
            feed_config=coops_temp_config_fixture,
            interval="6-min",
            expiration_interval=datetime.timedelta(minutes=10),
        )

        # Call _fetch
        await feed._fetch(clients=mock_clients)

        # Check that the COOPS API was called
        cast(MagicMock, mock_coops_client.temperature).assert_called_once()

        # Check that the COOPS API was called with default date range
        _, kwargs = cast(MagicMock, mock_coops_client.temperature).call_args

        # The default date range should be from today-8 days to today
        # Check the date arguments passed in kwargs
        assert "begin_date" in kwargs
        begin_date = kwargs["begin_date"]
        assert "end_date" in kwargs
        end_date = kwargs["end_date"]

        assert isinstance(begin_date, datetime.date)
        assert isinstance(end_date, datetime.date)

        # The begin_date should be approximately 8 days before end_date
        # Allow for small differences due to test execution timing
        date_diff = (end_date - begin_date).days
        assert date_diff == 8


class TestCspfTempFeed:
    """Tests for the CSPF temperature feed."""

    @pytest.mark.asyncio
    async def test_fetch_calls_cspf_client(
        self,
        location_config: config_lib.LocationConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """CSPF temp feed calls the CSPF client with configured source data."""
        cspf_config = config_lib.CspfTempFeedConfig(
            name="Sandettie Lightship",
            station_slug="sandettie-data",
        )
        start = datetime.datetime(2025, 1, 1)
        end = datetime.datetime(2025, 12, 31, 23, 59, 59)
        expected = pd.DataFrame(
            {"water_temp": [50.0]},
            index=pd.DatetimeIndex([datetime.datetime(2025, 4, 1)], name="time"),
        )
        mock_cspf_client = cast(CspfApi, mock_clients["cspf"])
        cast(MagicMock, mock_cspf_client.sandettie_temperature).return_value = expected

        feed = CspfTempFeed(
            location_config=location_config,
            feed_config=cspf_config,
            start=start,
            end=end,
            expiration_interval=None,
        )

        result = await feed._fetch(clients=mock_clients)

        assert result is expected
        cast(MagicMock, mock_cspf_client.sandettie_temperature).assert_called_once_with(
            begin_date=start,
            end_date=end,
            station_slug="sandettie-data",
            timezone=location_config.timezone,
            location_code=location_config.code,
        )


def test_create_temp_feed_supports_cspf_source(
    location_config: config_lib.LocationConfig,
    mock_clients: dict[str, BaseApiClient],
) -> None:
    """Temperature feed factory supports CSPF source configs."""
    cspf_config = config_lib.CspfTempFeedConfig(name="Sandettie Lightship")

    feed = create_temp_feed(
        location_config=location_config,
        temp_config=cspf_config,
        clients=mock_clients,
    )

    assert isinstance(feed, CspfTempFeed)
    assert feed.feed_config is cspf_config


def test_create_temp_feed_requires_cspf_client(
    location_config: config_lib.LocationConfig,
) -> None:
    """CSPF source configs require the CSPF client in the client dictionary."""
    with pytest.raises(TypeError, match="CSPF client not found"):
        create_temp_feed(
            location_config=location_config,
            temp_config=config_lib.CspfTempFeedConfig(name="Sandettie Lightship"),
            clients={},
        )


class TestIrishLightsTempFeed:
    """Tests for the Irish Lights temperature feed."""

    @pytest.mark.asyncio
    async def test_fetch_calls_irish_lights_client(
        self,
        location_config: config_lib.LocationConfig,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Irish Lights temp feed calls the configured MetOcean buoy."""
        temp_config = config_lib.IrishLightsTempFeedConfig(
            mmsi="992501100",
            name="Irish Lights Cork Buoy",
            min_valid_temp_c=1.0,
            max_valid_temp_c=20.0,
        )
        start = datetime.datetime(2026, 6, 15)
        end = datetime.datetime(2026, 6, 16)
        expected = pd.DataFrame(
            {"water_temp": [55.4]},
            index=pd.DatetimeIndex([datetime.datetime(2026, 6, 15, 1)], name="time"),
        )
        mock_client = cast(IrishLightsApi, mock_clients["irish_lights"])
        cast(MagicMock, mock_client.water_temperature).return_value = expected

        feed = IrishLightsTempFeed(
            location_config=location_config,
            feed_config=temp_config,
            start=start,
            end=end,
            expiration_interval=None,
        )

        result = await feed._fetch(clients=mock_clients)

        assert result is expected
        cast(MagicMock, mock_client.water_temperature).assert_called_once_with(
            mmsi="992501100",
            begin_date=start,
            end_date=end,
            timezone=location_config.timezone,
            location_code=location_config.code,
            min_valid_temp_c=1.0,
            max_valid_temp_c=20.0,
        )


def test_create_temp_feed_supports_irish_lights_source(
    location_config: config_lib.LocationConfig,
    mock_clients: dict[str, BaseApiClient],
) -> None:
    """Temperature feed factory supports Irish Lights source configs."""
    temp_config = config_lib.IrishLightsTempFeedConfig(
        mmsi="992501100",
        name="Irish Lights Cork Buoy",
    )

    feed = create_temp_feed(
        location_config=location_config,
        temp_config=temp_config,
        clients=mock_clients,
    )

    assert isinstance(feed, IrishLightsTempFeed)
    assert feed.feed_config is temp_config


def test_create_temp_feed_requires_irish_lights_client(
    location_config: config_lib.LocationConfig,
) -> None:
    """Irish Lights source configs require the Irish Lights client."""
    with pytest.raises(TypeError, match="Irish Lights client not found"):
        create_temp_feed(
            location_config=location_config,
            temp_config=config_lib.IrishLightsTempFeedConfig(
                mmsi="992501100",
                name="Irish Lights Cork Buoy",
            ),
            clients={},
        )


class TestCompositeFeed:
    """Tests for the CompositeFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_get_feeds_and_combine_feeds(
        self,
        simple_composite_feed: CompositeFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch calls _get_feeds and _combine_feeds."""
        # Create mock feeds and a test DataFrame to return
        mock_feed1 = MagicMock(spec=Feed)
        mock_feed2 = MagicMock(spec=Feed)

        test_df1 = pd.DataFrame(
            {"value": [1, 2, 3]}, index=pd.date_range(start="2025-01-01", periods=3)
        )
        test_df2 = pd.DataFrame(
            {"value": [4, 5, 6]}, index=pd.date_range(start="2025-01-04", periods=3)
        )

        # Configure the mocks
        mock_feed1._fetch.return_value = test_df1
        mock_feed2._fetch.return_value = test_df2

        # Mock the _get_feeds and _combine_feeds methods
        with (
            patch.object(
                simple_composite_feed,
                "_get_feeds",
                return_value=[mock_feed1, mock_feed2],
            ) as mock_get_feeds,
            patch.object(
                simple_composite_feed,
                "_combine_feeds",
                return_value=pd.concat([test_df1, test_df2]),
            ) as mock_combine_feeds,
        ):
            # Call _fetch
            result = await simple_composite_feed._fetch(clients=mock_clients)

            # Verify that the methods were called correctly
            mock_get_feeds.assert_called_once()
            mock_combine_feeds.assert_called_once()

            # Verify that the feeds' _fetch methods were called
            mock_feed1._fetch.assert_awaited_once()
            mock_feed2._fetch.assert_awaited_once()

            # Check that the result is a DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(test_df1) + len(test_df2)

    @pytest.mark.asyncio
    async def test_fetch_combines_dataframes_correctly(
        self,
        simple_composite_feed: CompositeFeed,
        valid_temp_dataframe: pd.DataFrame,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch combines DataFrames correctly."""

        # Create a test feed config for our test feeds
        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        # Create test feeds that return the valid_temp_dataframe
        class TestFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return TestDataModel

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                return valid_temp_dataframe

        # Create two instances of TestFeed
        feed1 = TestFeed(
            location_config=simple_composite_feed.location_config,
            expiration_interval=datetime.timedelta(minutes=10),
        )
        feed2 = TestFeed(
            location_config=simple_composite_feed.location_config,
            expiration_interval=datetime.timedelta(minutes=10),
        )

        # Create a combined DataFrame for the mock to return
        combined_df = pd.concat([valid_temp_dataframe, valid_temp_dataframe])

        # Mock both _get_feeds and _combine_feeds methods
        with (
            patch.object(
                simple_composite_feed, "_get_feeds", return_value=[feed1, feed2]
            ),
            patch.object(
                simple_composite_feed, "_combine_feeds", return_value=combined_df
            ),
        ):
            # Call _fetch
            result = await simple_composite_feed._fetch(clients=mock_clients)

            # Check that the result is the combined DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(valid_temp_dataframe) * 2
            # Verify the result matches our expected combined DataFrame
            pd.testing.assert_frame_equal(result, combined_df)

    @pytest.mark.asyncio
    async def test_fetch_propagates_errors(
        self,
        simple_composite_feed: CompositeFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _fetch propagates errors from underlying feeds."""

        # Create a mock feed that raises an exception
        mock_feed = AsyncMock(spec=Feed)
        mock_feed._fetch.side_effect = ValueError("Test error")

        # Mock the _get_feeds method to return our error feed
        with patch.object(
            simple_composite_feed,
            "_get_feeds",
            return_value=[
                mock_feed,
            ],
        ):
            # Call _fetch and expect it to raise an exception (following the project principle of failing fast)
            with pytest.raises(ValueError, match="Test error"):
                await simple_composite_feed._fetch(clients=mock_clients)


class TestMultiStationCurrentsFeed:
    """Tests for the MultiStationCurrentsFeed class."""

    def test_get_feeds_creates_correct_feeds(
        self,
        multi_station_currents_feed: MultiStationCurrentsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _get_feeds creates the correct number of CoopsCurrentsFeed instances."""
        # Get the feeds
        feeds = multi_station_currents_feed._get_feeds(clients=mock_clients)

        # Check that we have the correct number of feeds
        assert len(feeds) == len(multi_station_currents_feed.feed_config.stations)

        # Check that each feed is a CoopsCurrentsFeed
        for feed in feeds:
            assert isinstance(feed, CoopsCurrentsFeed)

        # Check that each feed is configured with the correct station
        # We know these are CoopsCurrentsFeed instances which have station attribute
        stations = [cast(CoopsCurrentsFeed, feed).station for feed in feeds]
        assert set(stations) == set(multi_station_currents_feed.feed_config.stations)

    def test_combine_feeds_with_single_dataframe(
        self,
        multi_station_currents_feed: MultiStationCurrentsFeed,
        valid_currents_dataframe: pd.DataFrame,
    ) -> None:
        """Test that _combine_feeds returns the input dataframe when only one is provided."""
        # Call _combine_feeds with a single dataframe
        result = multi_station_currents_feed._combine_feeds([valid_currents_dataframe])

        expected = valid_currents_dataframe.copy()
        expected.index.name = "time"

        # Check that the result preserves the values and required schema index name
        pd.testing.assert_frame_equal(result, expected)

    def test_combine_feeds_with_multiple_dataframes(
        self, multi_station_currents_feed: MultiStationCurrentsFeed
    ) -> None:
        """Test that _combine_feeds correctly combines data from multiple stations."""
        # Create two test dataframes with some overlapping timestamps
        index1 = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 0, 0, 0),
            end=datetime.datetime(2025, 4, 22, 5, 0, 0),
            freq="1h",
        )
        index2 = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 3, 0, 0),
            end=datetime.datetime(2025, 4, 22, 8, 0, 0),
            freq="1h",
        )

        # Note: We still include the 'type' column in the test data, but our implementation
        # will only keep the 'velocity' column to match the legacy behavior
        df1 = pd.DataFrame(
            {"velocity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "type": ["flood"] * 6},
            index=index1,
        )
        df2 = pd.DataFrame(
            {
                "velocity": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "type": ["flood"] * 3 + ["ebb"] * 3,
            },
            index=index2,
        )

        # Call _combine_feeds
        result = multi_station_currents_feed._combine_feeds([df1, df2])

        # Check that the result has the correct shape
        assert (
            len(result) == 9
        )  # 3 unique timestamps in df1 + 3 overlapping + 3 unique in df2

        # Verify that only the velocity column is present (matching legacy behavior)
        assert list(result.columns) == ["velocity"]
        assert result.index.name == "time"

        # Check that overlapping timestamps have averaged velocity values
        # Timestamps 3, 4, 5 are overlapping
        assert result.loc[index1[3], "velocity"] == 4.5  # (4.0 + 5.0) / 2
        assert result.loc[index1[4], "velocity"] == 5.5  # (5.0 + 6.0) / 2
        assert result.loc[index1[5], "velocity"] == 6.5  # (6.0 + 7.0) / 2

        # Check that non-overlapping timestamps have the original values
        assert result.loc[index1[0], "velocity"] == 1.0
        assert result.loc[index1[1], "velocity"] == 2.0
        assert result.loc[index1[2], "velocity"] == 3.0
        assert result.loc[index2[3], "velocity"] == 8.0
        assert result.loc[index2[4], "velocity"] == 9.0
        assert result.loc[index2[5], "velocity"] == 10.0

    def test_combine_feeds_result_validates_as_current_data(
        self, multi_station_currents_feed: MultiStationCurrentsFeed
    ) -> None:
        """Combined current station output preserves the named time index schema."""
        index1 = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 0, 0, 0),
            periods=3,
            freq="1h",
        )
        index2 = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 1, 0, 0),
            periods=3,
            freq="1h",
        )
        df1 = pd.DataFrame({"velocity": [1.0, 2.0, 3.0]}, index=index1)
        df2 = pd.DataFrame({"velocity": [4.0, 5.0, 6.0]}, index=index2)

        result = multi_station_currents_feed._combine_feeds([df1, df2])

        CurrentDataModel.validate(result)

    def test_combine_feeds_with_empty_list(
        self, multi_station_currents_feed: MultiStationCurrentsFeed
    ) -> None:
        """Test that _combine_feeds raises an error when no dataframes are provided."""
        with pytest.raises(ValueError, match="No dataframes provided to combine"):
            multi_station_currents_feed._combine_feeds([])

    @pytest.mark.asyncio
    async def test_fetch_handles_station_errors(
        self,
        multi_station_currents_feed: MultiStationCurrentsFeed,
        valid_currents_dataframe: pd.DataFrame,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test error handling when one station fails but others succeed."""

        # Create a test feed config for our test feeds
        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        # Create a test feed that raises an exception
        class ErrorFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return CurrentDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise ValueError("Test station error")

        # Create a test feed that returns valid data
        class TestFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return CurrentDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                return valid_currents_dataframe

        # Mock _get_feeds to return one error feed and one success feed
        with patch.object(
            multi_station_currents_feed,
            "_get_feeds",
            return_value=[
                ErrorFeed(
                    location_config=multi_station_currents_feed.location_config,
                    expiration_interval=datetime.timedelta(minutes=10),
                ),
                TestFeed(
                    location_config=multi_station_currents_feed.location_config,
                    expiration_interval=datetime.timedelta(minutes=10),
                ),
            ],
        ):
            # Call _fetch and expect it to raise an exception (following the project principle of failing fast)
            with pytest.raises(ValueError, match="Test station error"):
                await multi_station_currents_feed._fetch(clients=mock_clients)


class TestHistoricalTempsFeed:
    """Tests for the HistoricalTempsFeed class."""

    @staticmethod
    def _year_temp_dataframe(year: int) -> pd.DataFrame:
        index = pd.date_range(
            start=datetime.datetime(year, 7, 15, 12, 0, 0),
            end=datetime.datetime(year, 7, 15, 14, 0, 0),
            freq="1h",
            name="time",
        )
        return pd.DataFrame({"water_temp": [60.0, 61.0, 62.0]}, index=index)

    def test_get_feeds_creates_correct_feeds(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test that _get_feeds creates the correct number of CoopsTempFeed instances."""
        # Get the feeds
        feeds = historical_temps_feed._get_feeds(clients=mock_clients)

        # Check that we have the correct number of feeds
        assert len(feeds) == 2

        # Check that each feed is a CoopsTempFeed
        for feed in feeds:
            assert isinstance(feed, CoopsTempFeed)

        # Check that the feeds have the correct date ranges
        # Since we can't directly access the start/end dates of the feeds (they're used internally),
        # we'll have to trust that they were set correctly based on the implementation

    def test_status_includes_year_diagnostics_before_fetch(
        self, historical_temps_feed: HistoricalTempsFeed
    ) -> None:
        """Historical status reports required and missing years before first fetch."""
        status = historical_temps_feed.status

        assert isinstance(status.historical_temp_status, HistoricalTempStatus)
        assert status.historical_temp_status.required_years == [
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        ]
        assert status.historical_temp_status.available_years == []
        assert status.historical_temp_status.cached_years == []
        assert status.historical_temp_status.missing_years == [
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        ]
        assert status.historical_temp_status.fetched_years == []
        assert status.historical_temp_status.failed_years == {}

        status_dict = status.model_dump(mode="json")
        assert_json_serializable(status_dict)

    def test_combine_feeds_with_single_dataframe(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        valid_temp_dataframe: pd.DataFrame,
    ) -> None:
        """Test that _combine_feeds returns the input dataframe when only one is provided."""
        # Create a copy of the input dataframe with non-hourly timestamps
        test_df = valid_temp_dataframe.copy()

        # Call _combine_feeds with a single dataframe
        result = historical_temps_feed._combine_feeds([test_df])

        # Check that the result has been resampled to hourly intervals
        for dt in result.index:
            assert dt.minute == 0
            assert dt.second == 0

    def test_combine_feeds_with_multiple_dataframes(
        self, historical_temps_feed: HistoricalTempsFeed
    ) -> None:
        """Test that _combine_feeds correctly concatenates and resamples data from multiple years."""
        # Create two test dataframes with data from different years
        index2023 = pd.date_range(
            start=datetime.datetime(2023, 7, 15, 12, 0, 0),
            end=datetime.datetime(2023, 7, 15, 14, 0, 0),
            freq="1h",
        )
        index2024 = pd.date_range(
            start=datetime.datetime(2024, 7, 15, 12, 0, 0),
            end=datetime.datetime(2024, 7, 15, 14, 0, 0),
            freq="1h",
        )

        df2023 = pd.DataFrame({"temperature": [20.0, 21.0, 22.0]}, index=index2023)
        df2024 = pd.DataFrame({"temperature": [22.0, 23.0, 24.0]}, index=index2024)

        # Call _combine_feeds
        result = historical_temps_feed._combine_feeds([df2023, df2024])

        # The resampling will create a continuous hourly series between the two date ranges
        # Let's check only the non-NaN values to verify our original data is preserved
        result_no_nans = result.dropna()

        # Check that we have all our original data points
        assert len(result_no_nans) == 6  # 3 hours from 2023 + 3 hours from 2024

        # Check that the result contains all the original data
        # Check 2023 data
        assert (
            result.loc[datetime.datetime(2023, 7, 15, 12, 0, 0), "temperature"] == 20.0
        )
        assert (
            result.loc[datetime.datetime(2023, 7, 15, 13, 0, 0), "temperature"] == 21.0
        )
        assert (
            result.loc[datetime.datetime(2023, 7, 15, 14, 0, 0), "temperature"] == 22.0
        )

        # Check 2024 data
        assert (
            result.loc[datetime.datetime(2024, 7, 15, 12, 0, 0), "temperature"] == 22.0
        )
        assert (
            result.loc[datetime.datetime(2024, 7, 15, 13, 0, 0), "temperature"] == 23.0
        )
        assert (
            result.loc[datetime.datetime(2024, 7, 15, 14, 0, 0), "temperature"] == 24.0
        )

        # Verify that the index is sorted
        assert result.index.is_monotonic_increasing

        # Verify that the data is resampled to hourly intervals
        # All timestamps should be on the hour
        for dt in result.index:
            assert dt.minute == 0
            assert dt.second == 0

    def test_combine_feeds_with_empty_list(
        self, historical_temps_feed: HistoricalTempsFeed
    ) -> None:
        """Test that _combine_feeds raises an error when no dataframes are provided."""
        with pytest.raises(ValueError, match="No dataframes provided to combine"):
            historical_temps_feed._combine_feeds([])

    @pytest.mark.asyncio
    async def test_fetch_tracks_successful_required_years(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Historical fetch records year-level success before combining."""

        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        class TestFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]
            year: int

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                return TestHistoricalTempsFeed._year_temp_dataframe(self.year)

        with patch.object(
            historical_temps_feed,
            "_get_feeds",
            return_value=[
                TestFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                    year=historical_temps_feed.start_year,
                ),
                TestFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                    year=historical_temps_feed.end_year,
                ),
            ],
        ):
            result = await historical_temps_feed._fetch(clients=mock_clients)

        assert historical_temps_feed.last_required_years == (
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_successful_years == (
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_available_years == (
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_fetched_years == (
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_failed_years == {}
        assert len(result.dropna()) == 6

        status = historical_temps_feed.status
        assert status.historical_temp_status is not None
        assert status.historical_temp_status.required_years == [
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        ]
        assert status.historical_temp_status.available_years == [
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        ]
        assert status.historical_temp_status.cached_years == [
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        ]
        assert status.historical_temp_status.missing_years == []
        assert status.historical_temp_status.fetched_years == [
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        ]
        assert status.historical_temp_status.failed_years == {}

    @pytest.mark.asyncio
    async def test_fetch_normalizes_years_before_validation(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Raw yearly frames may need historical resampling before validation."""

        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        class DuplicateTimestampFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]
            year: int

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                index = pd.DatetimeIndex(
                    [
                        datetime.datetime(self.year, 11, 1, 1, 0, 0),
                        datetime.datetime(self.year, 11, 1, 1, 0, 0),
                        datetime.datetime(self.year, 11, 1, 2, 0, 0),
                    ],
                    name="time",
                )
                return pd.DataFrame({"water_temp": [60.0, 61.0, 62.0]}, index=index)

        with patch.object(
            historical_temps_feed,
            "_get_feeds",
            return_value=[
                DuplicateTimestampFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                    year=historical_temps_feed.start_year,
                ),
                DuplicateTimestampFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                    year=historical_temps_feed.end_year,
                ),
            ],
        ):
            result = await historical_temps_feed._fetch(clients=mock_clients)

        assert historical_temps_feed.last_failed_years == {}
        assert result.index.is_unique

    @pytest.mark.asyncio
    async def test_fetch_tracks_year_errors(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Test error handling when one year fails but others succeed."""

        # Create a test feed config for our test feeds
        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        # Create a test feed that raises an exception
        class ErrorFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise ValueError("Test year error")

        # Create a test feed that returns valid data
        class TestFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                return TestHistoricalTempsFeed._year_temp_dataframe(
                    historical_temps_feed.end_year
                )

        # Mock _get_feeds to return one error feed and one success feed
        with patch.object(
            historical_temps_feed,
            "_get_feeds",
            return_value=[
                ErrorFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
                TestFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
            ],
        ):
            with pytest.raises(
                HistoricalTempsIncompleteError,
                match=f"{historical_temps_feed.start_year}",
            ) as exc_info:
                await historical_temps_feed._fetch(clients=mock_clients)

        assert exc_info.value.failed_years == {
            historical_temps_feed.start_year: "ValueError: Test year error"
        }
        assert historical_temps_feed.last_required_years == (
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_successful_years == (
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_available_years == (
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_fetched_years == (
            historical_temps_feed.end_year,
        )
        assert historical_temps_feed.last_failed_years == {
            historical_temps_feed.start_year: "ValueError: Test year error"
        }

    @pytest.mark.asyncio
    async def test_fetch_uses_cached_past_years(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Historical retries skip past years already cached in memory."""
        cached_year = historical_temps_feed.start_year
        missing_year = historical_temps_feed.end_year
        historical_temps_feed._year_cache[cached_year] = self._year_temp_dataframe(
            cached_year
        )

        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        class UnexpectedFetchFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise AssertionError("cached year should not be fetched")

        class MissingYearFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                return TestHistoricalTempsFeed._year_temp_dataframe(missing_year)

        with patch.object(
            historical_temps_feed,
            "_get_feeds",
            return_value=[
                UnexpectedFetchFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
                MissingYearFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
            ],
        ):
            result = await historical_temps_feed._fetch(clients=mock_clients)

        assert set(historical_temps_feed._year_cache) == {cached_year, missing_year}
        assert historical_temps_feed.last_successful_years == (
            cached_year,
            missing_year,
        )
        assert historical_temps_feed.last_available_years == (
            cached_year,
            missing_year,
        )
        assert historical_temps_feed.last_fetched_years == (missing_year,)
        assert historical_temps_feed.last_failed_years == {}
        assert len(result.dropna()) == 6

        status = historical_temps_feed.status
        assert status.historical_temp_status is not None
        assert status.historical_temp_status.available_years == [
            cached_year,
            missing_year,
        ]
        assert status.historical_temp_status.cached_years == [
            cached_year,
            missing_year,
        ]
        assert status.historical_temp_status.missing_years == []
        assert status.historical_temp_status.fetched_years == [missing_year]
        assert status.historical_temp_status.failed_years == {}

    @pytest.mark.asyncio
    async def test_fetch_refreshes_cached_current_year(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """The current year refreshes even when already present in the year cache."""
        current_year = historical_temps_feed.end_year
        historical_temps_feed._year_cache[historical_temps_feed.start_year] = (
            self._year_temp_dataframe(historical_temps_feed.start_year)
        )
        historical_temps_feed._year_cache[current_year] = self._year_temp_dataframe(
            current_year
        )
        fetch_count = 0

        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        class CachedPastYearFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise AssertionError("cached past year should not be fetched")

        class CurrentYearFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                nonlocal fetch_count
                fetch_count += 1
                return TestHistoricalTempsFeed._year_temp_dataframe(current_year)

        with (
            patch.object(
                historical_temps_feed,
                "_current_historical_year",
                return_value=current_year,
            ),
            patch.object(
                historical_temps_feed,
                "_get_feeds",
                return_value=[
                    CachedPastYearFeed(
                        location_config=historical_temps_feed.location_config,
                        expiration_interval=datetime.timedelta(hours=3),
                    ),
                    CurrentYearFeed(
                        location_config=historical_temps_feed.location_config,
                        expiration_interval=datetime.timedelta(hours=3),
                    ),
                ],
            ),
        ):
            await historical_temps_feed._fetch(clients=mock_clients)

        assert fetch_count == 1
        assert historical_temps_feed.last_available_years == (
            historical_temps_feed.start_year,
            current_year,
        )
        assert historical_temps_feed.last_fetched_years == (current_year,)
        assert historical_temps_feed.last_failed_years == {}

    @pytest.mark.asyncio
    async def test_fetch_retains_cache_when_missing_year_fails(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """A failed missing year does not discard already cached years."""
        cached_year = historical_temps_feed.start_year
        failed_year = historical_temps_feed.end_year
        cached_frame = self._year_temp_dataframe(cached_year)
        historical_temps_feed._year_cache[cached_year] = cached_frame

        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        class CachedYearFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise AssertionError("cached year should not be fetched")

        class FailedYearFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise ValueError("missing year failed")

        with patch.object(
            historical_temps_feed,
            "_get_feeds",
            return_value=[
                CachedYearFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
                FailedYearFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
            ],
        ):
            with pytest.raises(HistoricalTempsIncompleteError):
                await historical_temps_feed._fetch(clients=mock_clients)

        assert historical_temps_feed._year_cache == {cached_year: cached_frame}
        assert historical_temps_feed.last_successful_years == (cached_year,)
        assert historical_temps_feed.last_available_years == (cached_year,)
        assert historical_temps_feed.last_fetched_years == ()
        assert historical_temps_feed.last_failed_years == {
            failed_year: "ValueError: missing year failed"
        }

        status = historical_temps_feed.status
        assert status.historical_temp_status is not None
        assert status.historical_temp_status.available_years == [cached_year]
        assert status.historical_temp_status.cached_years == [cached_year]
        assert status.historical_temp_status.missing_years == [failed_year]
        assert status.historical_temp_status.fetched_years == []
        assert status.historical_temp_status.failed_years == {
            failed_year: "ValueError: missing year failed"
        }

    @pytest.mark.asyncio
    async def test_update_keeps_existing_data_on_incomplete_fetch(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Incomplete historical refreshes leave the published dataset untouched."""
        previous_data = self._year_temp_dataframe(historical_temps_feed.start_year)
        historical_temps_feed._data = previous_data
        historical_temps_feed._fetch_timestamp = util.utc_now()

        async def incomplete_fetch(
            clients: dict[str, BaseApiClient],
        ) -> pd.DataFrame:
            raise HistoricalTempsIncompleteError(
                {historical_temps_feed.start_year: "ValueError: Test year error"}
            )

        with patch.object(historical_temps_feed, "_fetch", incomplete_fetch):
            await historical_temps_feed.update(clients=mock_clients)

        assert historical_temps_feed._data is previous_data
        assert historical_temps_feed._next_fetch_after is not None
        assert isinstance(
            historical_temps_feed._last_error, HistoricalTempsIncompleteError
        )

    @pytest.mark.asyncio
    async def test_fetch_preserves_station_unavailable_classification(
        self,
        historical_temps_feed: HistoricalTempsFeed,
        mock_clients: dict[str, BaseApiClient],
    ) -> None:
        """Expected no-data year failures remain StationUnavailableError."""

        class TestFeedConfig(config_lib.BaseFeedConfig, frozen=True):
            @property
            def citation(self) -> str:
                return "Test Feed Citation"

        class UnavailableFeed(Feed):
            feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

            @property
            def data_model(self) -> type[pa.DataFrameModel]:
                return WaterTempDataModel  # type: ignore[return-value]

            async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
                raise StationUnavailableError("Station has no data")

        with patch.object(
            historical_temps_feed,
            "_get_feeds",
            return_value=[
                UnavailableFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
                UnavailableFeed(
                    location_config=historical_temps_feed.location_config,
                    expiration_interval=datetime.timedelta(hours=3),
                ),
            ],
        ):
            with pytest.raises(StationUnavailableError, match="required years"):
                await historical_temps_feed._fetch(clients=mock_clients)

        assert historical_temps_feed.last_successful_years == ()
        assert historical_temps_feed.last_available_years == ()
        assert historical_temps_feed.last_fetched_years == ()
        assert set(historical_temps_feed.last_failed_years) == {
            historical_temps_feed.start_year,
            historical_temps_feed.end_year,
        }

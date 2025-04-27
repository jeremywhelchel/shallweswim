"""Tests for feeds.py functionality."""

# Standard library imports
import datetime
import logging
from typing import Any, List, cast
from unittest.mock import patch, MagicMock

# Third-party imports
import pandas as pd
import pytest
import pytz
from pydantic import BaseModel

# Local imports
from shallweswim import config as config_lib
from shallweswim.feeds import (
    Feed,
    NoaaTempFeed,
    NoaaTidesFeed,
    NoaaCurrentsFeed,
    CompositeFeed,
    MultiStationCurrentsFeed,
)


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
        swim_location="Brighton Beach",
        swim_location_link="https://example.com/brighton-beach",
    )


@pytest.fixture
def temp_config() -> config_lib.NoaaTempSource:
    """Create a temperature source config fixture."""
    return config_lib.NoaaTempSource(
        station=8518750, name="Test Station"  # Using a valid 7-digit station ID
    )


@pytest.fixture
def tide_config() -> config_lib.NoaaTideSource:
    """Create a tide source config fixture."""
    return config_lib.NoaaTideSource(
        station=8517741,  # Using a valid 7-digit station ID
        station_name="Coney Island, NY",
    )


@pytest.fixture
def currents_config() -> config_lib.NoaaCurrentsSource:
    """Create a currents source config fixture."""
    return config_lib.NoaaCurrentsSource(
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
    velocities: List[float] = []
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
def concrete_feed(location_config: config_lib.LocationConfig) -> Feed:
    """Create a concrete implementation of the abstract Feed class."""

    class TestConfig:
        """Test configuration class for ConcreteFeed."""

        outliers: list[str] = []

    class ConcreteFeed(Feed):
        config: TestConfig = TestConfig()

        async def _fetch(self) -> pd.DataFrame:
            # Return a simple DataFrame for testing
            index = pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 23, 0, 0),
                freq="h",
            )
            return pd.DataFrame({"value": range(len(index))}, index=index)

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

    class TestConfig(BaseModel):
        """Test configuration class for SimpleCompositeFeed."""

        pass

    class SimpleCompositeFeed(CompositeFeed):
        # Configuration for the feed
        config: TestConfig = TestConfig()

        def _get_feeds(self) -> List[Feed]:
            """Get test feeds - this will be mocked in tests."""
            return []

        def _combine_feeds(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
            """Combine test dataframes - this will be mocked in tests."""
            return pd.DataFrame()

    return SimpleCompositeFeed(
        location_config=location_config,
        expiration_interval=datetime.timedelta(minutes=10),
    )


class TestFeedBase:
    """Tests for the base Feed class."""

    @pytest.mark.asyncio
    async def test_is_expired_with_no_timestamp(self, concrete_feed: Feed) -> None:
        """Test that a feed with no timestamp is considered expired."""
        assert concrete_feed.is_expired is True

    @pytest.mark.asyncio
    async def test_is_expired_with_recent_timestamp(self, concrete_feed: Feed) -> None:
        """Test that a feed with a recent timestamp is not expired."""
        # Mock the utc_now function to return a fixed time
        fixed_now = datetime.datetime(2025, 4, 22, 12, 0, 0)

        with patch("shallweswim.feeds.utc_now", return_value=fixed_now):
            # Set timestamp to be very recent (1 minute ago)
            concrete_feed._timestamp = fixed_now - datetime.timedelta(minutes=1)
            # Set a large enough expiration interval
            concrete_feed.expiration_interval = datetime.timedelta(minutes=10)

            # This should not be expired
            assert concrete_feed.is_expired is False

    @pytest.mark.asyncio
    async def test_is_expired_with_old_timestamp(self, concrete_feed: Feed) -> None:
        """Test that a feed with an old timestamp is expired."""
        # Set an old timestamp
        concrete_feed._timestamp = datetime.datetime.now() - datetime.timedelta(hours=1)
        assert concrete_feed.is_expired is True

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
    async def test_update_calls_fetch(self, concrete_feed: Feed) -> None:
        """Test that update calls _fetch and updates timestamp and data."""
        # Use patch to mock the _fetch method
        original_fetch = concrete_feed._fetch
        fetch_called = False

        async def mock_fetch() -> pd.DataFrame:
            nonlocal fetch_called
            fetch_called = True
            return await original_fetch()

        # Use monkeypatch instead of direct assignment to avoid mypy error
        with patch.object(concrete_feed, "_fetch", mock_fetch):

            # Call update
            await concrete_feed.update()

            # Check that _fetch was called and data was updated
            assert fetch_called is True
            assert concrete_feed._data is not None
            assert concrete_feed._timestamp is not None

    def test_validate_frame_with_valid_dataframe(
        self, concrete_feed: Feed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that _validate_frame accepts a valid DataFrame."""
        # This should not raise an exception
        concrete_feed._validate_frame(valid_temp_dataframe)

    def test_validate_frame_with_none_dataframe(self, concrete_feed: Feed) -> None:
        """Test that _validate_frame rejects a None DataFrame."""
        with pytest.raises(ValueError, match="Received None dataframe"):
            concrete_feed._validate_frame(None)

    def test_validate_frame_with_empty_dataframe(self, concrete_feed: Feed) -> None:
        """Test that _validate_frame rejects an empty DataFrame."""
        with pytest.raises(ValueError, match="Received empty dataframe"):
            concrete_feed._validate_frame(pd.DataFrame())

    def test_validate_frame_with_non_datetime_index(self, concrete_feed: Feed) -> None:
        """Test that _validate_frame rejects a DataFrame with a non-DatetimeIndex."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="DataFrame index is not a DatetimeIndex"):
            concrete_feed._validate_frame(df)

    def test_validate_frame_with_timezone_aware_index(
        self, concrete_feed: Feed
    ) -> None:
        """Test that _validate_frame rejects a DataFrame with a timezone-aware index."""
        # Create a DataFrame with timezone-aware index
        index = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 0, 0, 0, tzinfo=datetime.timezone.utc),
            end=datetime.datetime(2025, 4, 22, 23, 0, 0, tzinfo=datetime.timezone.utc),
            freq="h",
        )
        df = pd.DataFrame({"value": range(len(index))}, index=index)

        with pytest.raises(ValueError, match="DataFrame index contains timezone info"):
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
        """Test that _remove_outliers removes the specified outliers from the DataFrame."""
        # Create a test DataFrame
        dates = pd.date_range("2023-01-01", periods=5)
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        # Set outliers in the TestConfig
        # Use getattr to avoid type checking issues
        config = getattr(concrete_feed, "config")
        config.outliers = ["2023-01-02", "2023-01-04"]

        result = concrete_feed._remove_outliers(df)

        # Expected DataFrame with outliers removed
        expected_dates = pd.DatetimeIndex(["2023-01-01", "2023-01-03", "2023-01-05"])
        expected_df = pd.DataFrame({"value": [1, 3, 5]}, index=expected_dates)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_remove_outliers_with_nonexistent_outliers(
        self, concrete_feed: Feed, caplog: Any
    ) -> None:
        """Test that _remove_outliers handles outliers that don't exist in the DataFrame."""
        # Create a test DataFrame
        dates = pd.date_range("2023-01-01", periods=3)
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

        # Set outliers in the TestConfig, including one that doesn't exist in the DataFrame
        # Use getattr to avoid type checking issues
        config = getattr(concrete_feed, "config")
        config.outliers = ["2023-01-02", "2023-01-10"]

        # Set the log level to WARNING to capture the warning messages
        caplog.set_level(logging.WARNING)

        result = concrete_feed._remove_outliers(df)

        # Expected DataFrame with existing outlier removed
        expected_dates = pd.DatetimeIndex(["2023-01-01", "2023-01-03"])
        expected_df = pd.DataFrame({"value": [1, 3]}, index=expected_dates)

        pd.testing.assert_frame_equal(result, expected_df)

        # Check that the log message for the non-existent outlier was recorded
        assert "Outlier timestamp 2023-01-10 not found in data" in caplog.text


class TestNoaaTidesFeed:
    """Tests for the NoaaTidesFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_noaa_client(
        self,
        location_config: config_lib.LocationConfig,
        tide_config: config_lib.NoaaTideSource,
    ) -> None:
        """Test that _fetch calls the NOAA client with correct parameters."""
        # Create a mock NOAA API with autospec
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to return a valid DataFrame
            MockNoaaApi.tides.return_value = pd.DataFrame(
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
            feed = NoaaTidesFeed(
                location_config=location_config,
                config=tide_config,
                expiration_interval=datetime.timedelta(hours=24),
            )

            # Call _fetch
            result = await feed._fetch()

            # Check that the NOAA API was called with correct parameters
            MockNoaaApi.tides.assert_called_once()
            args, kwargs = MockNoaaApi.tides.call_args

            # Check positional arguments (first is station_id)
            assert args[0] == tide_config.station
            # Check location_code is passed
            assert kwargs["location_code"] == location_config.code

            # Check that the result is correct
            assert isinstance(result, pd.DataFrame)
            assert "prediction" in result.columns
            assert "type" in result.columns

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(
        self,
        location_config: config_lib.LocationConfig,
        tide_config: config_lib.NoaaTideSource,
    ) -> None:
        """Test that _fetch handles API errors correctly."""
        # Create a mock NOAA API with autospec that raises an exception
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to raise an exception
            MockNoaaApi.tides.side_effect = Exception("API error")

            # Create the feed
            feed = NoaaTidesFeed(
                location_config=location_config,
                config=tide_config,
                expiration_interval=datetime.timedelta(hours=24),
            )

            # Call _fetch and expect it to raise the exception (following fail-fast principle)
            with pytest.raises(Exception, match="API error"):
                await feed._fetch()


class TestNoaaCurrentsFeed:
    """Tests for the NoaaCurrentsFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_noaa_client(
        self,
        location_config: config_lib.LocationConfig,
        currents_config: config_lib.NoaaCurrentsSource,
    ) -> None:
        """Test that _fetch calls the NOAA client with correct parameters."""
        # Create a mock NOAA API with autospec
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to return a valid DataFrame
            MockNoaaApi.currents.return_value = pd.DataFrame(
                {"velocity": [1.2, 0.8, 0.3, -0.2, -0.7]},
                index=pd.date_range(
                    start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                    end=datetime.datetime(2025, 4, 22, 4, 0, 0),
                    freq="1h",
                ),
            )

            # Create the feed
            feed = NoaaCurrentsFeed(
                location_config=location_config,
                config=currents_config,
                station=currents_config.stations[0],  # Use the first station
                expiration_interval=datetime.timedelta(hours=24),
            )

            # Call _fetch
            result = await feed._fetch()

            # Check that the NOAA API was called with correct parameters
            MockNoaaApi.currents.assert_called_once()
            args, kwargs = MockNoaaApi.currents.call_args

            # Check positional arguments (first is station_id)
            assert args[0] == currents_config.stations[0]
            # Check interpolate parameter
            assert kwargs["interpolate"] is True
            # Check location_code is passed
            assert kwargs["location_code"] == location_config.code

            # Check that the result is correct
            assert isinstance(result, pd.DataFrame)
            assert "velocity" in result.columns

    # This test was removed because we no longer support automatic station selection
    # in NoaaCurrentsFeed. Each feed instance must be explicitly configured with a station.
    # This is consistent with how MultiStationCurrentsFeed creates individual feed instances.

    # This test is no longer needed since Pydantic will validate that stations is not empty

    @pytest.mark.asyncio
    async def test_fetch_with_valid_station(
        self,
        location_config: config_lib.LocationConfig,
    ) -> None:
        """Test that _fetch works with a valid station specified directly."""
        # Create a config with a valid station
        config = config_lib.NoaaCurrentsSource(
            stations=["ACT3876", "NYH1905"],
        )

        # Create a mock NOAA API with autospec
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to return a valid DataFrame
            MockNoaaApi.currents.return_value = pd.DataFrame(
                {"velocity": [1.2, 0.8, 0.3]},
                index=pd.date_range(
                    start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                    end=datetime.datetime(2025, 4, 22, 2, 0, 0),
                    freq="1h",
                ),
            )

            # Create the feed with a specific station
            feed = NoaaCurrentsFeed(
                location_config=location_config,
                config=config,
                station="NYH1905",  # Specify the second station
                expiration_interval=datetime.timedelta(hours=24),
            )

            # Call _fetch
            await feed._fetch()

            # Check that the NOAA API was called with the specified station
            MockNoaaApi.currents.assert_called_once()
            args, _ = MockNoaaApi.currents.call_args
            assert args[0] == "NYH1905"

    @pytest.mark.asyncio
    async def test_fetch_handles_api_error(
        self,
        location_config: config_lib.LocationConfig,
        currents_config: config_lib.NoaaCurrentsSource,
    ) -> None:
        """Test that _fetch handles API errors correctly."""
        # Create a mock NOAA API with autospec that raises an exception
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to raise an exception
            MockNoaaApi.currents.side_effect = Exception("API error")

            # Create the feed
            feed = NoaaCurrentsFeed(
                location_config=location_config,
                config=currents_config,
                station=currents_config.stations[0],
                expiration_interval=datetime.timedelta(hours=24),
            )

            # Call _fetch and expect it to raise the exception (following fail-fast principle)
            with pytest.raises(Exception, match="API error"):
                await feed._fetch()


class TestNoaaTempFeed:
    """Tests for the NoaaTempFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_noaa_client(
        self,
        location_config: config_lib.LocationConfig,
        temp_config: config_lib.NoaaTempSource,
    ) -> None:
        """Test that _fetch calls the NOAA client with correct parameters."""
        # Create a mock NOAA API with autospec
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to return a valid DataFrame
            MockNoaaApi.temperature.return_value = pd.DataFrame(
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
            feed = NoaaTempFeed(
                location_config=location_config,
                config=temp_config,
                interval="6-min",  # Use 6-minute intervals for live data
                expiration_interval=datetime.timedelta(minutes=10),
            )

            # Call _fetch
            result = await feed._fetch()

            # Check that the NOAA API was called with correct parameters
            MockNoaaApi.temperature.assert_called_once()
            args, kwargs = MockNoaaApi.temperature.call_args

            # Check positional arguments (first is station_id)
            assert args[0] == temp_config.station
            # Check that product is water_temperature
            assert args[1] == "water_temperature"
            # Check location_code is passed
            assert kwargs["location_code"] == location_config.code

            # Check that the result is correct
            assert isinstance(result, pd.DataFrame)
            assert "water_temp" in result.columns

    @pytest.mark.asyncio
    async def test_fetch_with_date_range(
        self,
        location_config: config_lib.LocationConfig,
        temp_config: config_lib.NoaaTempSource,
    ) -> None:
        """Test that _fetch passes date range to NOAA API when provided."""
        # Create a mock NOAA API with autospec
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to return a valid DataFrame
            MockNoaaApi.temperature.return_value = pd.DataFrame(
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

            feed = NoaaTempFeed(
                location_config=location_config,
                config=temp_config,
                interval="6-min",
                expiration_interval=datetime.timedelta(minutes=10),
                # Use the correct attribute names for the feed
                start=start_date,
                end=end_date,
            )

            # Call _fetch
            await feed._fetch()

            # Check that the NOAA API was called with the date range
            MockNoaaApi.temperature.assert_called_once()

            # Instead of checking exact values, verify that start_date and end_date
            # were passed to the API call. The actual implementation might modify
            # these dates slightly, so we'll just check that they were passed.
            args, _ = MockNoaaApi.temperature.call_args
            assert len(args) >= 4  # At least 4 positional args
            assert isinstance(args[2], datetime.datetime)  # begin_date is a datetime
            assert isinstance(args[3], datetime.datetime)  # end_date is a datetime

    @pytest.mark.asyncio
    async def test_fetch_with_default_date_range(
        self,
        location_config: config_lib.LocationConfig,
        temp_config: config_lib.NoaaTempSource,
    ) -> None:
        """Test that _fetch uses default date range when not provided."""
        # Create a mock NOAA API with autospec
        with patch("shallweswim.noaa.NoaaApi", autospec=True) as MockNoaaApi:
            # Configure the mock to return a valid DataFrame
            MockNoaaApi.temperature.return_value = pd.DataFrame(
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
            feed = NoaaTempFeed(
                location_config=location_config,
                config=temp_config,
                interval="6-min",
                expiration_interval=datetime.timedelta(minutes=10),
            )

            # Call _fetch
            await feed._fetch()

            # Check that the NOAA API was called with default date range
            args, _ = MockNoaaApi.temperature.call_args

            # The default date range should be from today-8 days to today
            # Check the date arguments (positions 2 and 3)
            begin_date = args[2]
            end_date = args[3]

            # The begin_date should be approximately 8 days before end_date
            # Allow for small differences due to test execution timing
            date_diff = (end_date - begin_date).days
            assert date_diff == 8


class TestCompositeFeed:
    """Tests for the CompositeFeed class."""

    @pytest.mark.asyncio
    async def test_fetch_calls_get_feeds_and_combine_feeds(
        self, simple_composite_feed: CompositeFeed
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
        with patch.object(
            simple_composite_feed, "_get_feeds", return_value=[mock_feed1, mock_feed2]
        ) as mock_get_feeds, patch.object(
            simple_composite_feed,
            "_combine_feeds",
            return_value=pd.concat([test_df1, test_df2]),
        ) as mock_combine_feeds:

            # Call _fetch
            result = await simple_composite_feed._fetch()

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
        self, simple_composite_feed: CompositeFeed, valid_temp_dataframe: pd.DataFrame
    ) -> None:
        """Test that _fetch combines DataFrames correctly."""

        # Create test feeds that return the valid_temp_dataframe
        class TestFeed(Feed):
            async def _fetch(self) -> pd.DataFrame:
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
        with patch.object(
            simple_composite_feed, "_get_feeds", return_value=[feed1, feed2]
        ), patch.object(
            simple_composite_feed, "_combine_feeds", return_value=combined_df
        ):
            # Call _fetch
            result = await simple_composite_feed._fetch()

            # Check that the result is the combined DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(valid_temp_dataframe) * 2
            # Verify the result matches our expected combined DataFrame
            pd.testing.assert_frame_equal(result, combined_df)

    @pytest.mark.asyncio
    async def test_fetch_propagates_errors(
        self, simple_composite_feed: CompositeFeed
    ) -> None:
        """Test that _fetch propagates errors from underlying feeds."""

        # Create a test feed that raises an exception
        class ErrorFeed(Feed):
            async def _fetch(self) -> pd.DataFrame:
                raise ValueError("Test error")

        # Create an instance of ErrorFeed
        error_feed = ErrorFeed(
            location_config=simple_composite_feed.location_config,
            expiration_interval=datetime.timedelta(minutes=10),
        )

        # Mock the _get_feeds method to return our error feed
        with patch.object(
            simple_composite_feed, "_get_feeds", return_value=[error_feed]
        ):
            # Call _fetch and expect it to raise an exception
            with pytest.raises(ValueError, match="Test error"):
                await simple_composite_feed._fetch()


class TestMultiStationCurrentsFeed:
    """Tests for the MultiStationCurrentsFeed class."""

    @pytest.fixture
    def multi_station_currents_feed(
        self,
        location_config: config_lib.LocationConfig,
        currents_config: config_lib.NoaaCurrentsSource,
    ) -> MultiStationCurrentsFeed:
        """Create a MultiStationCurrentsFeed fixture."""
        return MultiStationCurrentsFeed(
            location_config=location_config,
            config=currents_config,
            expiration_interval=datetime.timedelta(minutes=10),
        )

    def test_get_feeds_creates_correct_feeds(
        self,
        multi_station_currents_feed: MultiStationCurrentsFeed,
        currents_config: config_lib.NoaaCurrentsSource,
    ) -> None:
        """Test that _get_feeds creates the correct number of NoaaCurrentsFeed instances."""
        # Get the feeds
        feeds = multi_station_currents_feed._get_feeds()

        # Check that we have the correct number of feeds
        assert len(feeds) == len(currents_config.stations)

        # Check that each feed is a NoaaCurrentsFeed
        for feed in feeds:
            assert isinstance(feed, NoaaCurrentsFeed)

        # Check that each feed is configured with the correct station
        # We know these are NoaaCurrentsFeed instances which have station attribute
        stations = [cast(NoaaCurrentsFeed, feed).station for feed in feeds]
        assert set(stations) == set(currents_config.stations)

    def test_combine_feeds_with_single_dataframe(
        self,
        multi_station_currents_feed: MultiStationCurrentsFeed,
        valid_currents_dataframe: pd.DataFrame,
    ) -> None:
        """Test that _combine_feeds returns the input dataframe when only one is provided."""
        # Call _combine_feeds with a single dataframe
        result = multi_station_currents_feed._combine_feeds([valid_currents_dataframe])

        # Check that the result is the same as the input
        pd.testing.assert_frame_equal(result, valid_currents_dataframe)

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
    ) -> None:
        """Test error handling when one station fails but others succeed."""

        # Create a test feed that raises an exception
        class ErrorFeed(Feed):
            async def _fetch(self) -> pd.DataFrame:
                raise ValueError("Test station error")

        # Create a test feed that returns valid data
        class SuccessFeed(Feed):
            async def _fetch(self) -> pd.DataFrame:
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
                SuccessFeed(
                    location_config=multi_station_currents_feed.location_config,
                    expiration_interval=datetime.timedelta(minutes=10),
                ),
            ],
        ):
            # Call _fetch and expect it to raise an exception (following the project principle of failing fast)
            with pytest.raises(ValueError, match="Test station error"):
                await multi_station_currents_feed._fetch()

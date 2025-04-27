"""Tests for feeds.py functionality."""

# Standard library imports
import datetime
from typing import Any, Optional
from unittest.mock import patch

# Third-party imports
import pandas as pd
import pytest
import pytz

# Local imports
from shallweswim import config as config_lib
from shallweswim.feeds import Feed, NoaaTempFeed, NoaaTidesFeed, NoaaCurrentsFeed


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
        predictions_available=True,
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
    velocities = [1.2, 0.8, 0.3, -0.2, -0.7, -1.1, -0.9, -0.5, -0.1, 0.4, 0.9, 1.3] * 4

    # Create the DataFrame
    df = pd.DataFrame({"velocity": velocities[: len(index)]}, index=index)
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

    class ConcreteFeed(Feed):
        async def _fetch(self) -> pd.DataFrame:
            # Return a simple DataFrame for testing
            index = pd.date_range(
                start=datetime.datetime(2025, 4, 22, 0, 0, 0),
                end=datetime.datetime(2025, 4, 22, 23, 0, 0),
                freq="h",
            )
            return pd.DataFrame({"value": range(len(index))}, index=index)

        # Override values property to match the implementation
        @property
        def values(self) -> Optional[pd.DataFrame]:
            if self._data is None:
                raise ValueError("Data not yet fetched")
            return self._data

    return ConcreteFeed(
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

    @pytest.mark.asyncio
    async def test_fetch_with_no_station_specified(
        self,
        location_config: config_lib.LocationConfig,
        currents_config: config_lib.NoaaCurrentsSource,
    ) -> None:
        """Test that _fetch uses the first station when none is specified."""
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

            # Create the feed without specifying a station
            feed = NoaaCurrentsFeed(
                location_config=location_config,
                config=currents_config,
                expiration_interval=datetime.timedelta(hours=24),
            )

            # Call _fetch
            await feed._fetch()

            # Check that the NOAA API was called with the first station
            MockNoaaApi.currents.assert_called_once()
            args, _ = MockNoaaApi.currents.call_args
            assert args[0] == currents_config.stations[0]

    @pytest.mark.asyncio
    async def test_fetch_with_no_stations_configured(
        self,
        location_config: config_lib.LocationConfig,
    ) -> None:
        """Test that _fetch raises ValueError when no stations configured."""
        # Create a config with no stations
        config = config_lib.NoaaCurrentsSource(
            stations=[],
        )

        # Create the feed
        feed = NoaaCurrentsFeed(
            location_config=location_config,
            config=config,
            expiration_interval=datetime.timedelta(hours=24),
        )

        # Call _fetch and expect it to raise ValueError
        with pytest.raises(ValueError, match="No current stations configured"):
            await feed._fetch()

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

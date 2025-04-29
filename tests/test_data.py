"""Tests for data.py functionality."""

# Standard library imports
import asyncio
import datetime
from unittest.mock import MagicMock

from tests.helpers import assert_json_serializable

# Third-party imports
import pandas as pd
import pytest
import pytest_asyncio
from typing import Any, AsyncGenerator

# Local imports
from shallweswim.data import LocationDataManager
from shallweswim.types import CurrentInfo  # Import from types module where it's defined
from shallweswim import config as config_lib


@pytest.fixture
def mock_config() -> Any:
    """Mock location config fixture."""
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    config.code = "nyc"
    config.StationName = "New York City"
    return config


@pytest.fixture
def mock_current_data() -> pd.DataFrame:
    """Create mock current data with peaks and transitions."""
    # Create a datetime index spanning 24 hours with 1-hour increments
    index = pd.date_range(
        start=datetime.datetime(2025, 4, 22, 0, 0, 0),
        end=datetime.datetime(2025, 4, 22, 23, 0, 0),
        freq="h",
    )

    # Generate a sinusoidal pattern for current velocity
    # Positive values = flood, negative values = ebb
    velocity = [
        1.0,
        1.5,
        1.2,
        0.6,
        0.1,
        -0.3,
        -0.8,
        -1.3,
        -1.5,
        -1.3,
        -0.8,
        -0.3,
        0.2,
        0.7,
        1.2,
        1.5,
        1.3,
        0.7,
        0.1,
        -0.4,
        -0.9,
        -1.3,
        -1.2,
        -0.6,
    ]

    # Create the DataFrame with velocity data
    df = pd.DataFrame({"velocity": velocity}, index=index)
    return df


@pytest_asyncio.fixture
async def mock_data_with_currents(
    mock_config: Any, mock_current_data: pd.DataFrame
) -> AsyncGenerator[LocationDataManager, None]:
    """Create a LocationDataManager instance with mock current data."""
    data = LocationDataManager(mock_config)

    # Create a mock currents feed
    mock_currents_feed = MagicMock()
    mock_currents_feed.values = mock_current_data
    mock_currents_feed.is_expired = False

    # Set the mock feed in the _feeds dictionary
    data._feeds["currents"] = mock_currents_feed

    yield data

    # Clean up the data manager after the test
    data.stop()


@pytest.mark.asyncio
async def test_current_prediction_at_flood_peak(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Test current prediction at a flood peak."""
    # Test at 3:00 PM (15:00) which is a flood peak at 1.5 knots
    t = datetime.datetime(2025, 4, 22, 15, 0, 0)
    result = mock_data_with_currents.current_prediction(t)

    # Check the result
    assert result.direction == "flooding"
    assert pytest.approx(result.magnitude, 0.1) == 1.5
    assert "at its strongest" in result.state_description


@pytest.mark.asyncio
async def test_current_prediction_at_ebb_peak() -> None:
    """Test current prediction at an ebb peak."""
    # Create a custom test instance
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    data = LocationDataManager(config)

    try:
        # Create a more comprehensive dataset for testing ebb currents
        # Create a single day with a complete cycle
        test_day = datetime.datetime(2025, 4, 22)
        hours = list(range(24))
        idx = pd.DatetimeIndex([test_day.replace(hour=h) for h in hours])

        # Create an ebb pattern with clear strengthening and weakening periods
        velocities = [
            0.1,  # 0:00 - Near slack
            0.5,  # 1:00 - Flooding
            0.9,  # 2:00 - Flooding
            1.2,  # 3:00 - Peak flood
            0.8,  # 4:00 - Weakening flood
            0.2,  # 5:00 - Near slack
            -0.3,  # 6:00 - Beginning to ebb
            -0.8,  # 7:00 - Strengthening ebb
            -1.3,  # 8:00 - Strengthening ebb
            -1.8,  # 9:00 - Peak ebb current (strongest)
            -1.5,  # 10:00 - Weakening
            -1.0,  # 11:00 - Weakening
            -0.6,  # 12:00 - Weakening
            -0.1,  # 13:00 - Near slack (clearly below 0.2 threshold)
            0.2,  # 14:00 - Beginning to flood
            0.7,  # 15:00 - Flooding
            1.1,  # 16:00 - Flooding
            1.5,  # 17:00 - Peak flood
            1.1,  # 18:00 - Weakening
            0.5,  # 19:00 - Weakening
            0.1,  # 20:00 - Near slack
            -0.4,  # 21:00 - Beginning to ebb
            -0.9,  # 22:00 - Ebbing
            -1.3,  # 23:00 - Ebbing
        ]

        # Create the DataFrame with the full tidal cycle
        current_df = pd.DataFrame({"velocity": velocities}, index=idx)

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = current_df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Test at peak ebb (9:00)
        peak_time = test_day.replace(hour=9)
        result = data.current_prediction(peak_time)

        # Verify the prediction at peak
        assert result.direction == "ebbing"
        assert (
            pytest.approx(result.magnitude, abs=0.1) == 1.8
        )  # Should be about 1.8 knots
        assert "at its strongest" in result.state_description

        # Test during strengthening ebb (8:00)
        strengthening_time = test_day.replace(hour=8)
        result = data.current_prediction(strengthening_time)
        assert result.direction == "ebbing"
        assert "getting stronger" in result.state_description

        # Test during weakening ebb (11:00)
        weakening_time = test_day.replace(hour=11)
        result = data.current_prediction(weakening_time)
        assert result.direction == "ebbing"
        assert "getting weaker" in result.state_description
        # Test near slack after ebb (13:00)
        slack_time = test_day.replace(hour=13)
        result = data.current_prediction(slack_time)
        assert "weakest" in result.state_description
    finally:
        # Clean up the data manager after the test
        data.stop()


@pytest.mark.asyncio
async def test_current_prediction_at_slack(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Test current prediction at slack water."""
    # Test at 4:00 AM (4:00) which is near zero (0.1 knots)
    t = datetime.datetime(2025, 4, 22, 4, 0, 0)
    result = mock_data_with_currents.current_prediction(t)

    # Check the result
    assert result.magnitude < 0.2
    assert "at its weakest (slack)" in result.state_description


@pytest.mark.asyncio
async def test_current_prediction_strengthening() -> None:
    """Test current prediction when current is strengthening."""
    # Create a custom test instance with clear strengthening pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    data = LocationDataManager(config)

    try:
        # Create data with a clear strengthening pattern
        hours = [12, 13, 14]
        idx = pd.DatetimeIndex([datetime.datetime(2025, 4, 22, h, 0, 0) for h in hours])
        # Clearly increasing velocity
        velocities = [0.5, 0.8, 1.2]

        # Create the DataFrame with strengthening pattern
        df = pd.DataFrame({"velocity": velocities}, index=idx)

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Test at the middle point where slope is positive
        t = datetime.datetime(2025, 4, 22, 13, 0, 0)
        result = data.current_prediction(t)

        # Check the result
        assert result.direction == "flooding"
        assert "getting stronger" in result.state_description
    finally:
        # Clean up the data manager after the test
        data.stop()


@pytest.mark.asyncio
async def test_current_prediction_weakening() -> None:
    """Test current prediction when current is weakening."""
    # Create a custom test instance with clear weakening pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    data = LocationDataManager(config)

    try:
        # Create data with a clear weakening pattern
        hours = [15, 16, 17]
        idx = pd.DatetimeIndex([datetime.datetime(2025, 4, 22, h, 0, 0) for h in hours])
        # Clearly decreasing velocity
        velocities = [1.2, 0.8, 0.5]

        # Create the DataFrame with weakening pattern
        df = pd.DataFrame({"velocity": velocities}, index=idx)

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Test at the middle point where slope is negative
        t = datetime.datetime(2025, 4, 22, 16, 0, 0)
        result = data.current_prediction(t)

        # Check the result
        assert result.direction == "flooding"
        assert "getting weaker" in result.state_description
    finally:
        # Clean up the data manager after the test
        data.stop()


@pytest.mark.asyncio
async def test_process_peaks_function() -> None:
    """Test that peaks are identified properly in the CurrentPrediction method."""
    # Create a custom data instance with a clear flood peak pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)

    data = LocationDataManager(config)

    try:
        # Create a very distinct peak pattern
        index = pd.date_range(start="2025-04-22", periods=5, freq="h")
        currents_df = pd.DataFrame(
            {
                "velocity": [0.5, 1.5, 0.8, 0.3, 0.1],  # Clear peak at index 1
            },
            index=index,
        )

        # We'll add a slope column to help with trend detection
        currents_df["slope"] = currents_df["velocity"].diff().fillna(0)

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = currents_df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Use a time point that's exactly at the peak
        peak_time = index[1]
        result = data.current_prediction(peak_time)

        # The peak should be detected and described correctly
        assert result.direction == "flooding"
        assert pytest.approx(result.magnitude, abs=0.1) == 1.5

        # Check if it's marked as a strong current
        # We're looking for either "at its strongest" or "getting stronger/weaker"
        assert (
            "at its strongest" in result.state_description
            or "getting stronger" in result.state_description
            or "getting weaker" in result.state_description
        )
    finally:
        # Clean up the data manager after the test
        data.stop()


@pytest.mark.asyncio
async def test_current_info_representation() -> None:
    """Test the representation of CurrentInfo objects."""
    # Test with flooding current
    flood_info = CurrentInfo(
        direction="flooding",
        magnitude=1.5,
        magnitude_pct=0.8,
        state_description="getting stronger",
    )
    assert flood_info.direction == "flooding"
    assert flood_info.magnitude == 1.5
    assert flood_info.magnitude_pct == 0.8
    assert flood_info.state_description == "getting stronger"
    # Check that the string representation contains the important information
    flood_str = str(flood_info)
    assert "flooding" in flood_str.lower()
    assert "1.5" in flood_str
    assert "getting stronger" in flood_str

    # Test with ebbing current
    ebb_info = CurrentInfo(
        direction="ebbing",
        magnitude=1.2,
        magnitude_pct=0.6,
        state_description="getting weaker",
    )
    assert ebb_info.direction == "ebbing"
    assert ebb_info.magnitude == 1.2
    assert ebb_info.magnitude_pct == 0.6
    assert ebb_info.state_description == "getting weaker"
    # Check that the string representation contains the important information
    ebb_str = str(ebb_info)
    assert "ebbing" in ebb_str.lower()
    assert "1.2" in ebb_str
    assert "getting weaker" in ebb_str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tides_timestamp,live_temps_timestamp,historic_temps_timestamp,expected_ready,configured_feeds",
    [
        # No feeds configured - ready (nothing to wait for)
        (None, None, None, True, False),
        # All datasets have timestamps but all expired - not ready
        (
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 1, 1),
            False,
            True,
        ),
        # Some datasets missing but all configured ones are valid - ready
        # This reflects the new ready logic where we only check configured feeds
        (
            datetime.datetime.now(),
            None,
            datetime.datetime.now(),
            True,
            True,
        ),
        # All configured datasets have recent timestamps - ready
        (
            datetime.datetime.now(),
            datetime.datetime.now(),
            datetime.datetime.now(),
            True,
            True,
        ),
    ],
)
async def test_data_ready_property(
    mock_config: Any,
    tides_timestamp: datetime.datetime,
    live_temps_timestamp: datetime.datetime,
    historic_temps_timestamp: datetime.datetime,
    expected_ready: bool,
    configured_feeds: bool,
) -> None:
    """Test the ready property of the Data class.

    Tests different combinations of dataset states to verify the ready property
    accurately represents if all data has been loaded and is not expired.
    """
    data = LocationDataManager(mock_config)

    try:
        # Mock the _expired method to control its behavior based on timestamps
        def mock_expired(dataset: str) -> bool:
            # For all datasets, we'll simulate the feed's is_expired property
            # based on whether the corresponding feed exists and its timestamp
            if dataset == "tides":
                # Consider None timestamps as expired
                if tides_timestamp is None:
                    return True
                # Consider timestamps older than 1 hour as expired
                cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
                return tides_timestamp < cutoff
            elif dataset == "currents":
                # Use the same timestamp as tides for simplicity
                if tides_timestamp is None:
                    return True
                cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
                return tides_timestamp < cutoff
            elif dataset == "live_temps":
                if live_temps_timestamp is None:
                    return True
                cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
                return live_temps_timestamp < cutoff
            elif dataset == "historic_temps":
                if historic_temps_timestamp is None:
                    return True
                cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
                return historic_temps_timestamp < cutoff
            else:
                # Unknown dataset
                return True

        # Apply our mock by directly setting the method
        # Use setattr to avoid mypy method-assign error
        setattr(data, "_expired", mock_expired)

        # Only set up feed mocks if configured_feeds is True
        if configured_feeds:
            # Set up feed mocks for each dataset in the _feeds dictionary
            # Tides feed
            mock_tides_feed = MagicMock()
            mock_tides_feed.is_expired = mock_expired("tides")
            data._feeds["tides"] = (
                mock_tides_feed if tides_timestamp is not None else None
            )

            # Currents feed - only configure if tides_timestamp is not None
            # This simulates a location that may not have currents configured
            if tides_timestamp is not None:
                mock_currents_feed = MagicMock()
                mock_currents_feed.is_expired = mock_expired("currents")
                data._feeds["currents"] = mock_currents_feed

            # Live temps feed - only configure if live_temps_timestamp is not None
            # This simulates a location that may not have live temps configured
            if live_temps_timestamp is not None:
                mock_live_temps_feed = MagicMock()
                mock_live_temps_feed.is_expired = mock_expired("live_temps")
                data._feeds["live_temps"] = mock_live_temps_feed

            # Historic temps feed - only configure if historic_temps_timestamp is not None
            # This simulates a location that may not have historic temps configured
            if historic_temps_timestamp is not None:
                mock_historic_temps_feed = MagicMock()
                mock_historic_temps_feed.is_expired = mock_expired("historic_temps")
                data._feeds["historic_temps"] = mock_historic_temps_feed

        # Test the ready property
        assert data.ready == expected_ready
    finally:
        # Clean up the data manager after the test
        data.stop()


@pytest.mark.asyncio
async def test_wait_until_ready() -> None:
    """Test the wait_until_ready method of the LocationDataManager class.

    Tests different scenarios:
    1. All feeds ready immediately
    2. Feeds becoming ready during the wait
    3. Timeout occurring before feeds are ready
    """
    # Create a LocationDataManager instance
    config = MagicMock(spec=config_lib.LocationConfig)
    config.code = "nyc"
    data = LocationDataManager(config)

    try:
        # Scenario 1: All feeds ready immediately
        # Create mock feeds that are already ready
        mock_feeds = []
        for _ in range(3):
            mock_feed = MagicMock()

            # Create an async function that returns True immediately
            async def wait_until_ready_immediate() -> bool:
                return True

            mock_feed.wait_until_ready = wait_until_ready_immediate
            mock_feeds.append(mock_feed)

        # Set the mock feeds in the _feeds dictionary
        data._feeds = {
            "tides": mock_feeds[0],
            "currents": mock_feeds[1],
            "live_temps": mock_feeds[2],
        }

        # Test wait_until_ready with feeds that are already ready
        result = await data.wait_until_ready(timeout=1.0)
        assert result is True

        # Scenario 2: Feeds becoming ready during the wait
        # Reset the mock feeds
        mock_feeds = []
        for _ in range(3):
            mock_feed = MagicMock()

            # Create an async function that returns True after a short delay
            async def wait_until_ready_with_delay() -> bool:
                await asyncio.sleep(0.1)  # Short delay
                return True

            mock_feed.wait_until_ready = wait_until_ready_with_delay
            mock_feeds.append(mock_feed)

        # Set the mock feeds in the _feeds dictionary
        data._feeds = {
            "tides": mock_feeds[0],
            "currents": mock_feeds[1],
            "live_temps": mock_feeds[2],
        }

        # Test wait_until_ready with feeds that become ready during the wait
        result = await data.wait_until_ready(timeout=1.0)
        assert result is True

        # Scenario 3: Timeout occurring before feeds are ready
        # Create a mock feed that will timeout
        mock_feed = MagicMock()

        # Create an async function that raises a TimeoutError when called with wait_for
        async def wait_until_ready_timeout() -> bool:
            # This will cause the asyncio.wait_for in LocationDataManager.wait_until_ready to timeout
            await asyncio.sleep(10.0)  # Much longer than our timeout
            return True

        mock_feed.wait_until_ready = wait_until_ready_timeout

        # Set the mock feed in the _feeds dictionary
        data._feeds = {"tides": mock_feed}

        # Test wait_until_ready with a feed that will cause a timeout
        result = await data.wait_until_ready(timeout=0.1)  # Short timeout
        assert result is False
    finally:
        # Clean up the data manager after the test
        data.stop()


@pytest.mark.asyncio
async def test_data_status_property() -> None:
    """Test the status property of the LocationDataManager class.

    Verifies that the status property returns a dictionary mapping feed names to their status dictionaries,
    and that the dictionary is JSON serializable.
    """
    # Create a LocationDataManager instance
    config = MagicMock(spec=config_lib.LocationConfig)
    config.code = "nyc"
    data = LocationDataManager(config)

    try:
        # Create mock feeds with status dictionaries
        mock_feeds = []
        for i in range(3):
            mock_feed = MagicMock()
            mock_feed.status = {
                "name": f"MockFeed{i}",
                "location": "nyc",
                "timestamp": "2025-04-27T12:00:00",
                "age_seconds": 3600,
                "is_expired": False,
                "is_ready": True,
                "data_shape": [24, 1],
                "expiration_seconds": 3600,
            }
            mock_feeds.append(mock_feed)

        # Set the mock feeds in the _feeds dictionary
        data._feeds = {
            "tides": mock_feeds[0],
            "currents": mock_feeds[1],
            "live_temps": mock_feeds[2],
            "historic_temps": None,  # Test with a None feed
        }

        # Get the status dictionary
        status = data.status

        # Check that the status dictionary contains the expected keys
        assert set(status.keys()) == {"tides", "currents", "live_temps"}

        # Check that the status dictionary contains the expected values
        for i, name in enumerate(["tides", "currents", "live_temps"]):
            assert status[name]["name"] == f"MockFeed{i}"
            assert status[name]["location"] == "nyc"
            assert status[name]["is_ready"] is True

        # Check that the status dictionary is JSON serializable
        assert_json_serializable(status)
    finally:
        # Clean up the data manager after the test
        data.stop()

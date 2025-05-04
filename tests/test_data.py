"""Tests for data.py functionality."""

# Standard library imports
import asyncio
import datetime
from unittest.mock import MagicMock, AsyncMock
from concurrent.futures import ProcessPoolExecutor
from typing import Any, AsyncGenerator

# Third-party imports
import pandas as pd
import pytest
import pytest_asyncio
from typing import Any, AsyncGenerator, Literal, get_args, cast
from freezegun import freeze_time
from pytest_mock import MockerFixture

# Local helpers
from tests.helpers import assert_json_serializable

# Local imports
from shallweswim.data import LocationDataManager
from shallweswim.types import (
    CurrentInfo,
    CurrentDirection,  # Import CurrentDirection Enum
    DataSourceType,  # Import DataSourceType
)  # Import from types module where it's defined
from shallweswim.api_types import FeedStatus
from shallweswim import config as config_lib
from shallweswim.feeds import Feed


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
async def process_pool() -> AsyncGenerator[ProcessPoolExecutor, None]:
    """Fixture to provide a ProcessPoolExecutor and ensure it's shut down."""
    pool = ProcessPoolExecutor()
    yield pool
    pool.shutdown(wait=False)  # Ensure pool is shut down after test


@pytest_asyncio.fixture
async def mock_data_with_currents(
    mock_config: Any, mock_current_data: pd.DataFrame, process_pool: ProcessPoolExecutor
) -> AsyncGenerator[LocationDataManager, None]:
    """Create a LocationDataManager instance with mock current data."""
    # Use the provided process_pool fixture
    data = LocationDataManager(mock_config, clients={}, process_pool=process_pool)

    # Create a mock currents feed
    mock_currents_feed = MagicMock()
    mock_currents_feed.values = mock_current_data
    mock_currents_feed.is_expired = False

    # Set the mock feed in the _feeds dictionary
    data._feeds["currents"] = mock_currents_feed

    yield data

    # Pool shutdown is handled by the process_pool fixture


@pytest.mark.asyncio
async def test_current_prediction_at_flood_peak(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Test current prediction at a flood peak."""
    # Test at 3:00 PM (15:00) which is a flood peak at 1.5 knots
    t = datetime.datetime(2025, 4, 22, 15, 0, 0)
    result = mock_data_with_currents.current_prediction(t)

    # Check the result
    assert result.direction == CurrentDirection.FLOODING  # Use Enum member
    assert pytest.approx(result.magnitude, 0.1) == 1.5
    assert "at its strongest" in result.state_description


@pytest.mark.asyncio
async def test_current_prediction_at_ebb_peak(
    process_pool: ProcessPoolExecutor,
) -> None:
    """Test current prediction at an ebb peak."""
    # Create a custom test instance
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    config.code = "tst"
    # Use the provided process_pool fixture
    data = LocationDataManager(config, clients={}, process_pool=process_pool)

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
        1.2,  # 3:00 - Flood peak
        0.8,  # 4:00 - Weakening flood
        0.2,  # 5:00 - Near slack
        -0.4,  # 6:00 - Strengthening ebb
        -0.9,  # 7:00 - Strengthening ebb
        -1.3,  # 8:00 - Ebb peak
        -1.0,  # 9:00 - Weakening ebb
        -0.5,  # 10:00 - Weakening ebb
        -0.1,  # 11:00 - Near slack
        0.3,  # 12:00 - Strengthening flood
        0.7,  # 13:00 - Strengthening flood
        1.1,  # 14:00 - Strengthening flood
        1.4,  # 15:00 - Flood peak
        1.0,  # 16:00 - Weakening flood
        0.4,  # 17:00 - Weakening flood
        -0.2,  # 18:00 - Near slack / turning ebb
        -0.7,  # 19:00 - Strengthening ebb
        -1.2,  # 20:00 - Strengthening ebb
        -1.5,  # 21:00 - Ebb peak
        -1.1,  # 22:00 - Weakening ebb
        -0.6,  # 23:00 - Weakening ebb
    ]
    mock_df = pd.DataFrame({"velocity": velocities}, index=idx)

    # Create a mock currents feed
    mock_currents_feed = MagicMock()
    mock_currents_feed.values = mock_df
    mock_currents_feed.is_expired = False

    # Set the mock feed in the _feeds dictionary
    data._feeds["currents"] = mock_currents_feed

    # Test at 9:00 PM (21:00) which is an ebb peak at -1.5 knots
    t = datetime.datetime(2025, 4, 22, 21, 0, 0)
    result = data.current_prediction(t)

    # Check the result
    assert result.direction == CurrentDirection.EBBING  # Use Enum member
    assert pytest.approx(result.magnitude, 0.1) == 1.5  # Magnitude is positive
    assert "at its strongest" in result.state_description
    # Pool shutdown is handled by the process_pool fixture


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

    # Provide an empty clients dict and a process pool for the test
    pool = ProcessPoolExecutor()
    data = LocationDataManager(config, clients={}, process_pool=pool)

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
        assert result.direction == CurrentDirection.FLOODING  # Use Enum member
        assert "getting stronger" in result.state_description
    finally:
        # Clean up the process pool
        pool.shutdown(wait=False)  # Don't wait in tests


@pytest.mark.asyncio
async def test_current_prediction_weakening() -> None:
    """Test current prediction when current is weakening."""
    # Create a custom test instance with clear weakening pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)

    # Provide an empty clients dict and a process pool for the test
    pool = ProcessPoolExecutor()
    data = LocationDataManager(config, clients={}, process_pool=pool)

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
        assert result.direction == CurrentDirection.FLOODING  # Use Enum member
        assert "getting weaker" in result.state_description
    finally:
        # Clean up the process pool
        pool.shutdown(wait=False)  # Don't wait in tests


@pytest.mark.asyncio
async def test_process_peaks_function() -> None:
    """Test that peaks are identified properly in the CurrentPrediction method."""
    # Create a custom data instance with a clear flood peak pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)

    # Provide an empty clients dict and a process pool for the test
    pool = ProcessPoolExecutor()
    data = LocationDataManager(config, clients={}, process_pool=pool)

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
        assert result.direction == CurrentDirection.FLOODING  # Use Enum member
        assert pytest.approx(result.magnitude, abs=0.1) == 1.5

        # Check if it's marked as a strong current
        # We're looking for either "at its strongest" or "getting stronger/weaker"
        assert (
            "at its strongest" in result.state_description
            or "getting stronger" in result.state_description
            or "getting weaker" in result.state_description
        )
    finally:
        # Clean up the process pool
        pool.shutdown(wait=False)  # Don't wait in tests


@pytest.mark.asyncio
async def test_current_info_representation() -> None:
    """Test the representation of CurrentInfo objects."""
    # Test with flooding current
    flood_info = CurrentInfo(
        source_type=DataSourceType.PREDICTION,
        direction=CurrentDirection.FLOODING,  # Use Enum member
        magnitude=1.5,
        magnitude_pct=0.8,
        state_description="getting stronger",
    )
    assert flood_info.direction == CurrentDirection.FLOODING  # Use Enum member
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
        source_type=DataSourceType.PREDICTION,
        direction=CurrentDirection.EBBING,  # Use Enum member
        magnitude=1.2,
        magnitude_pct=0.6,
        state_description="getting weaker",
    )
    assert ebb_info.direction == CurrentDirection.EBBING  # Use Enum member
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
    "tides_timestamp, live_temps_timestamp, historic_temps_timestamp, expected_ready, configured_feeds",
    [
        # --- All feeds configured ---
        # Case 1: All feeds ready (timestamps are recent)
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            True,  # Expected ready
            True,  # All feeds configured
        ),
        # Case 2: Tides expired
        (
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # Case 3: Live temps expired
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # Case 4: Historic temps expired
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 19, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # Case 5: Multiple feeds expired
        (
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # --- Only tides and historic configured ---
        # Case 6: Configured feeds ready
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            None,  # Live temps not configured
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            True,  # Expected ready
            False,  # Live temps not configured
        ),
        # Case 7: Configured tides expired
        (
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            None,
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            False,
        ),
        # Case 8: Configured historic temps expired
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            None,
            datetime.datetime(2025, 4, 19, 12, 0, 0),
            False,  # Expected not ready
            False,
        ),
    ],
)
@pytest.mark.asyncio
@freeze_time("2025-04-27 12:00:00")  # Freeze time for consistent expiration checks
# pylint: disable=unused-argument
async def test_data_ready_property(
    mock_config: Any,
    tides_timestamp: datetime.datetime | None,
    live_temps_timestamp: datetime.datetime | None,
    historic_temps_timestamp: datetime.datetime | None,
    expected_ready: bool,
    configured_feeds: bool,  # True if live temps feed is configured
    process_pool: ProcessPoolExecutor,
) -> None:
    """Test the ready property of the Data class.

    Tests different combinations of dataset states to verify the ready property
    accurately represents if all data has been loaded and is not expired.
    """
    # Use the provided process_pool fixture
    data = LocationDataManager(mock_config, clients={}, process_pool=process_pool)

    # Define the Literal type for feed names used as keys in data._feeds
    FeedName = Literal["tides", "currents", "live_temps", "historic_temps"]

    # --- Mock Feed Setup ---
    # Start with base feeds, type explicitly allows all FeedName possibilities
    feeds_to_configure: list[FeedName] = ["tides", "historic_temps"]
    if configured_feeds:
        feeds_to_configure.append("live_temps")  # Now compatible

    # Configure feeds based on the 'configured_feeds' parameter
    mock_config.feeds = feeds_to_configure

    # Define expiration times (adjust as needed for your feeds)
    expiration_map = {
        "tides": datetime.timedelta(hours=1),  # Example: 1 hour expiry
        "live_temps": datetime.timedelta(hours=1),  # Example: 1 hour expiry
        "historic_temps": datetime.timedelta(days=7),  # Example: 7 day expiry
    }

    # Initialize the dictionary; type is inferred or handled by mock/class def
    data._feeds = {}
    for feed_name in feeds_to_configure:
        mock_feed = MagicMock()
        timestamp = locals().get(f"{feed_name}_timestamp")

        if timestamp is not None:
            mock_feed.timestamp = timestamp
            mock_feed.expiration_seconds = expiration_map[feed_name].total_seconds()
            # Let the actual is_expired logic (using frozen time) work
            mock_feed.is_expired = (
                datetime.datetime(2025, 4, 27, 12, 0, 0) - timestamp
            ) > expiration_map[feed_name]
            mock_feed.values = pd.DataFrame({feed_name: [1]})  # Needs some data
        else:
            # Feed is configured but has no data yet
            mock_feed.timestamp = None
            mock_feed.is_expired = True  # Treat as expired if no timestamp
            mock_feed.values = None

        data._feeds[feed_name] = mock_feed

    # Ensure other potentially configured feeds are None if not in this test case
    # Iterate through all defined FeedName literals
    for feed_name in get_args(FeedName):
        if feed_name not in data._feeds:
            data._feeds[feed_name] = None

    # --- Test the ready property ---
    assert data.ready == expected_ready


@pytest.mark.asyncio
async def test_data_status_property(process_pool: ProcessPoolExecutor) -> None:
    """Test the status property of the LocationDataManager class.

    Verifies that the status property returns a dictionary mapping feed names to their status dictionaries,
    and that the dictionary is JSON serializable.
    """
    # Create a LocationDataManager instance
    config = MagicMock(spec=config_lib.LocationConfig)
    config.code = "nyc"
    # Use the provided process_pool fixture
    data = LocationDataManager(config, clients={}, process_pool=process_pool)

    # Create mock feeds with status dictionaries
    mock_feeds = []
    for i in range(3):
        mock_feed = MagicMock(spec=Feed)
        # Use FeedStatus model instead of dict
        mock_feed.status = FeedStatus(
            name=f"MockFeed{i}",
            location="nyc",
            fetch_timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
            age_seconds=3600,
            is_expired=False,
            expiration_seconds=3600,
            data_summary=None,  # Explicitly set optional fields if needed
            error=None,  # Explicitly set optional fields if needed
            is_healthy=True,  # Update field name and value
        )
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
    assert set(status.feeds.keys()) == {"tides", "currents", "live_temps"}

    # Check that the status dictionaries for each feed match the mock status
    assert status.feeds["tides"] == mock_feeds[0].status
    assert status.feeds["currents"] == mock_feeds[1].status
    assert status.feeds["live_temps"] == mock_feeds[2].status

    # Check that the status dictionary derived from the model is JSON serializable
    assert_json_serializable(status.model_dump(mode="json"))
    # Pool shutdown is handled by the process_pool fixture


# --- Tests for LocationDataManager.wait_until_ready ---


@pytest.fixture
def mock_data_manager(process_pool: ProcessPoolExecutor) -> LocationDataManager:
    """Fixture to create a LocationDataManager with mocked internals."""
    config = MagicMock(spec=config_lib.LocationConfig)
    config.code = "tst-wait"
    data = LocationDataManager(config, clients={}, process_pool=process_pool)
    data._ready_event = MagicMock(spec=asyncio.Event)
    data._update_task = MagicMock(spec=asyncio.Task)
    return data


@pytest.mark.asyncio
async def test_wait_until_ready_already_ready(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 1: Manager is already ready."""
    mock_data_manager._ready_event.is_set.return_value = True  # type: ignore[attr-defined]
    result = await mock_data_manager.wait_until_ready(timeout=0.1)
    assert result is True
    mock_data_manager._ready_event.wait.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_wait_until_ready_task_not_started(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 2: Update task hasn't been started (should raise RuntimeError)."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task = None
    with pytest.raises(RuntimeError, match="Update task has not been started"):
        await mock_data_manager.wait_until_ready(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_until_ready_task_already_failed(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 3: Update task already finished with an exception."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = True  # type: ignore[union-attr]
    mock_data_manager._update_task.exception.return_value = ValueError("Task failed")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="Task failed"):
        await mock_data_manager.wait_until_ready(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_until_ready_task_already_done_unexpectedly(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 4: Update task already finished ok, but ready event not set."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = True  # type: ignore[union-attr]
    mock_data_manager._update_task.exception.return_value = None  # type: ignore[union-attr]

    with pytest.raises(RuntimeError, match="finished unexpectedly"):
        await mock_data_manager.wait_until_ready(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_until_ready_event_set_during_wait(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 5: Ready event is set during the wait.

    Simulates the scenario where the ready event gets set *while* the
    wait_until_ready method is waiting.
    """
    # Initial setup: event not set, task not done
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    # Define the side effect for the event's wait() method
    async def wait_side_effect(*_args: Any, **_kwargs: Any) -> bool:
        # Simulate some delay during which the event might be set
        await asyncio.sleep(0)
        # Simulate the event being set externally
        mock_data_manager._ready_event.is_set.return_value = True  # type: ignore[attr-defined]
        # wait() should return True when the event is set
        return True

    # Mock event's wait() method with the async side effect
    mock_wait_method = mocker.patch.object(
        mock_data_manager._ready_event,
        "wait",
        new_callable=AsyncMock,
        side_effect=wait_side_effect,
    )

    # Call the method under test - use the real asyncio.wait
    result = await mock_data_manager.wait_until_ready(timeout=0.1)  # Short timeout

    # Assertions
    assert result is True
    mock_data_manager._update_task.exception.assert_not_called()  # type: ignore[union-attr]
    mock_wait_method.assert_awaited_once()  # Check the wait method was awaited


@pytest.mark.asyncio
async def test_wait_until_ready_task_fails_during_wait(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 6: Update task fails during the wait."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    async def mock_wait(
        waiters: set[asyncio.Future[Any]], **_kwargs: Any
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        update_task = mock_data_manager._update_task
        mock_data_manager._update_task.exception.return_value = ValueError(  # type: ignore[union-attr]
            "Task failed during wait"
        )
        done_set: set[asyncio.Future[Any]] = {cast(asyncio.Future[Any], update_task)}
        pending_set = waiters - done_set
        return done_set, pending_set

    mocker.patch("asyncio.wait", new=mock_wait)

    with pytest.raises(ValueError, match="Task failed during wait"):
        await mock_data_manager.wait_until_ready(timeout=1.0)


@pytest.mark.asyncio
async def test_wait_until_ready_task_done_unexpectedly_during_wait(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 7: Update task finishes ok during wait, but ready event not set."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    async def mock_wait(
        waiters: set[asyncio.Future[Any]], **_kwargs: Any
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        update_task = mock_data_manager._update_task
        mock_data_manager._update_task.exception.return_value = None  # type: ignore[union-attr]

        done_set: set[asyncio.Future[Any]] = {cast(asyncio.Future[Any], update_task)}
        pending_set = waiters - done_set
        return done_set, pending_set

    mocker.patch("asyncio.wait", new=mock_wait)

    with pytest.raises(RuntimeError, match="finished unexpectedly"):
        await mock_data_manager.wait_until_ready(timeout=1.0)


@pytest.mark.asyncio
async def test_wait_until_ready_timeout(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 8: Timeout occurs before ready or task completion."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    async def mock_wait(
        waiters: set[asyncio.Future[Any]], **_kwargs: Any
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        return set(), waiters  # Return empty done, all pending

    mocker.patch("asyncio.wait", new=mock_wait)
    mocker.patch.object(
        mock_data_manager._ready_event, "wait", side_effect=asyncio.TimeoutError
    )  # type: ignore[attr-defined, unused-ignore]

    result = await mock_data_manager.wait_until_ready(timeout=0.1)
    assert result is False

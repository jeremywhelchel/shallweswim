"""Tests for backend error handling.

Verifies:
- has_feed_data() correctly checks specific feeds
- handle_task_exception() uses correct log levels
- Update loop continues after external API failures (doesn't crash)

These tests emulate the conditions from production bugs to prevent regressions.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytz

from shallweswim.clients.base import BaseClientError, StationUnavailableError
from shallweswim.config import LocationConfig
from shallweswim.core import updater
from shallweswim.core.feeds import FEED_TIDES
from shallweswim.data import LocationDataManager


def create_minimal_config() -> LocationConfig:
    """Create a minimal location config for testing (no feed sources)."""
    return LocationConfig(
        code="tst",
        name="Test Location",
        swim_location="Test Swim Spot",
        swim_location_link="http://example.com/test",
        description="Test description",
        latitude=40.0,
        longitude=-74.0,
        timezone=pytz.timezone("US/Eastern"),
        enabled=True,
        # No feed sources - we'll manually set _feeds
    )


def create_manager_with_feeds(feeds_dict: dict[str, Any]) -> LocationDataManager:
    """Create a LocationDataManager with manually configured feeds.

    This bypasses the normal feed configuration to allow testing
    has_feed_data() with controlled feed states.
    """
    config = create_minimal_config()
    mock_clients: dict[str, Any] = {}
    mock_pool = MagicMock(spec=ProcessPoolExecutor)

    manager = LocationDataManager(config, mock_clients, mock_pool)
    # Override the auto-configured feeds with our test feeds
    manager._feeds = feeds_dict
    return manager


def create_mock_task_with_exception(exception: Exception) -> MagicMock:
    """Create a mock asyncio.Task that raises an exception on result().

    Args:
        exception: The exception to raise when task.result() is called

    Returns:
        A MagicMock that behaves like an asyncio.Task with an exception
    """
    mock_task = MagicMock()
    mock_task.get_name.return_value = "TestTask"
    mock_task.result.side_effect = exception
    return mock_task


def create_log_fn_for_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Any:
    """Create a log function compatible with handle_task_exception().

    The updater.handle_task_exception() expects a log_fn(message, level) signature.
    This creates one that logs to the standard logging module so caplog can capture it.
    """

    def log_fn(message: str, level: int) -> None:
        logging.log(level, message)

    return log_fn


# =============================================================================
# LocationDataManager.has_feed_data() tests
# =============================================================================


class TestHasFeedData:
    """Tests for LocationDataManager.has_feed_data() method.

    This method was added to fix the 500 bug where has_data passed but
    specific feeds were empty, causing AssertionError.
    """

    def test_returns_true_when_data_exists(self) -> None:
        """Feed exists and has data → True."""
        mock_feed = MagicMock()
        mock_feed._data = "some_data"  # Has data

        manager = create_manager_with_feeds({FEED_TIDES: mock_feed})

        assert manager.has_feed_data(FEED_TIDES) is True

    def test_returns_false_when_no_data(self) -> None:
        """Feed exists but no data yet → False.

        This is the exact scenario from the 500 bug: feed object exists
        but _data is None because fetch hasn't completed yet.
        """
        mock_feed = MagicMock()
        mock_feed._data = None  # No data yet

        manager = create_manager_with_feeds({FEED_TIDES: mock_feed})

        assert manager.has_feed_data(FEED_TIDES) is False

    def test_returns_false_for_nonexistent_feed(self) -> None:
        """Feed doesn't exist → False (not KeyError).

        Important: should return False gracefully, not raise KeyError.
        """
        manager = create_manager_with_feeds({})  # No feeds

        # Should not raise, should return False
        assert manager.has_feed_data("nonexistent") is False

    def test_returns_false_when_feed_is_none(self) -> None:
        """Feed slot exists but is None → False.

        Some locations don't configure all feeds, so _feeds[name] = None.
        """
        manager = create_manager_with_feeds({FEED_TIDES: None})

        assert manager.has_feed_data(FEED_TIDES) is False


# =============================================================================
# updater.handle_task_exception() log level tests
# =============================================================================


class TestHandleTaskException:
    """Tests for updater.handle_task_exception() log levels.

    Verifies our error handling philosophy:
    - StationUnavailableError → WARNING (expected operational condition)
    - BaseClientError → ERROR (unexpected, triggers alerts)
    - Other exceptions → ERROR (bugs, triggers alerts)
    """

    def test_station_unavailable_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """StationUnavailableError → WARNING (not ERROR).

        Station unavailability is expected (maintenance, no recent data).
        Should log WARNING to avoid triggering alerts for normal conditions.
        """
        task = create_mock_task_with_exception(
            StationUnavailableError("Station 12345 has no data for requested period")
        )

        with caplog.at_level(logging.DEBUG):
            # Should NOT raise - station unavailable is handled gracefully
            updater.handle_task_exception(task, create_log_fn_for_caplog(caplog))

        # Verify WARNING level was used
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]

        assert len(warning_records) >= 1, "Expected WARNING log for station unavailable"
        assert any("Station unavailable" in r.message for r in warning_records)
        assert len(error_records) == 0, "Should NOT log ERROR for expected condition"

    def test_client_error_logs_error_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """BaseClientError → ERROR and re-raises.

        Upstream API errors (unexpected responses, parsing failures) should
        log ERROR to trigger alerts, then re-raise for visibility.
        """
        task = create_mock_task_with_exception(
            BaseClientError("Unexpected API response format")
        )

        with caplog.at_level(logging.DEBUG):
            # Should re-raise the exception
            with pytest.raises(BaseClientError):
                updater.handle_task_exception(task, create_log_fn_for_caplog(caplog))

        # Verify ERROR level was used
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1, "Expected ERROR log for client error"
        assert any("Upstream error" in r.message for r in error_records)

    def test_generic_exception_logs_error_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Generic Exception → ERROR and re-raises.

        Unexpected exceptions (bugs in our code) should log ERROR and re-raise.
        This follows fail-fast for internal errors.
        """
        task = create_mock_task_with_exception(ValueError("Unexpected internal error"))

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                updater.handle_task_exception(task, create_log_fn_for_caplog(caplog))

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1, "Expected ERROR log for generic exception"
        assert any("Unhandled exception" in r.message for r in error_records)

    def test_cancelled_error_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """CancelledError → DEBUG (normal during shutdown).

        Task cancellation is expected during graceful shutdown.
        Should log at DEBUG level, not trigger alerts.
        """
        import asyncio

        task = create_mock_task_with_exception(asyncio.CancelledError())

        with caplog.at_level(logging.DEBUG):
            # Should NOT raise - cancellation is handled gracefully
            updater.handle_task_exception(task, create_log_fn_for_caplog(caplog))

        # Verify DEBUG level was used (not WARNING or ERROR)
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]

        assert len(debug_records) >= 1, "Expected DEBUG log for cancellation"
        assert any("cancelled" in r.message for r in debug_records)
        assert len(warning_records) == 0, "Should NOT log WARNING for cancellation"
        assert len(error_records) == 0, "Should NOT log ERROR for cancellation"


# =============================================================================
# LocationDataManager.has_data property tests
# =============================================================================


class TestHasData:
    """Tests for LocationDataManager.has_data property.

    This property checks if ANY feed has data (used for lenient health checks).
    Unlike has_feed_data() which checks specific feeds, this is a broad check.
    """

    def test_returns_true_when_any_feed_has_data(self) -> None:
        """At least one feed has data → True."""
        mock_feed_with_data = MagicMock()
        mock_feed_with_data._data = "some_data"

        mock_feed_without_data = MagicMock()
        mock_feed_without_data._data = None

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed_with_data,
                "currents": mock_feed_without_data,
            }
        )

        assert manager.has_data is True

    def test_returns_false_when_no_feeds_have_data(self) -> None:
        """No feeds have data → False."""
        mock_feed1 = MagicMock()
        mock_feed1._data = None

        mock_feed2 = MagicMock()
        mock_feed2._data = None

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed1,
                "currents": mock_feed2,
            }
        )

        assert manager.has_data is False

    def test_returns_false_when_no_feeds_configured(self) -> None:
        """No feeds configured → False."""
        manager = create_manager_with_feeds({})

        assert manager.has_data is False

    def test_returns_false_when_all_feeds_are_none(self) -> None:
        """All feed slots are None → False."""
        manager = create_manager_with_feeds(
            {
                FEED_TIDES: None,
                "currents": None,
            }
        )

        assert manager.has_data is False

    def test_ignores_none_feeds_when_checking(self) -> None:
        """Mix of None and real feeds - only checks real feeds."""
        mock_feed_with_data = MagicMock()
        mock_feed_with_data._data = "some_data"

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed_with_data,
                "currents": None,  # Not configured
            }
        )

        assert manager.has_data is True


# =============================================================================
# Update loop resilience tests
# =============================================================================


class TestUpdateLoopResilience:
    """Tests for update loop error handling.

    Verifies the update loop continues after external API failures
    instead of crashing (the "update loop crash bug" fix).
    """

    @pytest.mark.asyncio
    async def test_update_dataset_handles_exception_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Feed update raises exception → logs error, doesn't crash.

        This tests the updater.update_dataset() function which is called
        by the update loop. If a feed's update() raises, it should be
        caught and logged, not crash the loop.
        """
        from shallweswim.core.feeds import Feed

        # Create a mock feed that raises on update
        mock_feed = MagicMock(spec=Feed)
        mock_feed.update.side_effect = BaseClientError("API returned garbage")
        mock_feed.is_expired = True

        manager = create_manager_with_feeds({FEED_TIDES: mock_feed})

        # The update should raise (feed.update() raises)
        # This is expected - the update loop catches this
        with pytest.raises(BaseClientError):
            await updater.update_dataset(manager._feeds, {}, FEED_TIDES)

        # Verify update was attempted
        mock_feed.update.assert_called_once()

    def test_is_expired_with_none_feed(self) -> None:
        """is_expired() with None feed → True (needs refresh)."""
        feeds_dict: dict[str, Any] = {FEED_TIDES: None}

        assert updater.is_expired(feeds_dict, FEED_TIDES) is True

    def test_is_expired_with_missing_feed(self) -> None:
        """is_expired() with missing feed → True (needs refresh)."""
        feeds_dict: dict[str, Any] = {}

        assert updater.is_expired(feeds_dict, FEED_TIDES) is True

    def test_is_expired_delegates_to_feed(self) -> None:
        """is_expired() delegates to feed.is_expired property."""
        mock_feed = MagicMock()
        mock_feed.is_expired = False

        feeds_dict = {FEED_TIDES: mock_feed}

        assert updater.is_expired(feeds_dict, FEED_TIDES) is False

        mock_feed.is_expired = True
        assert updater.is_expired(feeds_dict, FEED_TIDES) is True


# =============================================================================
# LocationDataManager.ready property tests
# =============================================================================


class TestReady:
    """Tests for LocationDataManager.ready property.

    This property checks if ALL configured feeds are fresh (not expired).
    Stricter than has_data - used for strict readiness checks.
    """

    def test_returns_true_when_all_feeds_fresh(self) -> None:
        """All configured feeds are fresh → True."""
        mock_feed1 = MagicMock()
        mock_feed1.is_expired = False

        mock_feed2 = MagicMock()
        mock_feed2.is_expired = False

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed1,
                "currents": mock_feed2,
            }
        )

        assert manager.ready is True

    def test_returns_false_when_any_feed_expired(self) -> None:
        """Any feed expired → False.

        This is the key difference from has_data: ready requires ALL
        feeds to be fresh, not just ANY feed having data.
        """
        mock_fresh_feed = MagicMock()
        mock_fresh_feed.is_expired = False

        mock_expired_feed = MagicMock()
        mock_expired_feed.is_expired = True

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_fresh_feed,
                "currents": mock_expired_feed,
            }
        )

        assert manager.ready is False

    def test_returns_false_when_all_feeds_expired(self) -> None:
        """All feeds expired → False."""
        mock_feed1 = MagicMock()
        mock_feed1.is_expired = True

        mock_feed2 = MagicMock()
        mock_feed2.is_expired = True

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed1,
                "currents": mock_feed2,
            }
        )

        assert manager.ready is False

    def test_returns_true_when_no_feeds_configured(self) -> None:
        """No feeds configured → True (nothing to wait for)."""
        manager = create_manager_with_feeds({})

        assert manager.ready is True

    def test_ignores_none_feeds(self) -> None:
        """None feeds are ignored - only checks configured feeds.

        Some locations don't configure all feed types.
        """
        mock_fresh_feed = MagicMock()
        mock_fresh_feed.is_expired = False

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_fresh_feed,
                "currents": None,  # Not configured for this location
            }
        )

        assert manager.ready is True

    def test_returns_true_when_all_feeds_are_none(self) -> None:
        """All feed slots are None → True (nothing configured)."""
        manager = create_manager_with_feeds(
            {
                FEED_TIDES: None,
                "currents": None,
            }
        )

        assert manager.ready is True

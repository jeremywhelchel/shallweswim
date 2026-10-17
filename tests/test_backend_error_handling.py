"""Tests for backend error handling.

Verifies:
- has_feed_data() correctly checks specific feeds
- A feed update failure propagates out of update_dataset()

These tests emulate the conditions from production bugs to prevent regressions.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytz

from shallweswim.clients.base import BaseClientError, StationUnavailableError
from shallweswim.config import LocationConfig
from shallweswim.core import updater
from shallweswim.core.feeds import FEED_TIDES, FeedName
from shallweswim.core.manager import LocationDataManager


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
        default_temperature_unit="F",
        enabled=True,
        # No feed sources - we'll manually set _feeds
    )


def create_manager_with_feeds(feeds_dict: dict[FeedName, Any]) -> LocationDataManager:
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
        mock_feed_with_data.has_data = True

        mock_feed_without_data = MagicMock()
        mock_feed_without_data.has_data = False

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed_with_data,
                FeedName.CURRENTS: mock_feed_without_data,
            }
        )

        assert manager.has_data is True

    def test_returns_false_when_no_feeds_have_data(self) -> None:
        """No feeds have data → False."""
        mock_feed1 = MagicMock()
        mock_feed1.has_data = False

        mock_feed2 = MagicMock()
        mock_feed2.has_data = False

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed1,
                FeedName.CURRENTS: mock_feed2,
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
                FeedName.CURRENTS: None,
            }
        )

        assert manager.has_data is False

    def test_ignores_none_feeds_when_checking(self) -> None:
        """Mix of None and real feeds - only checks real feeds."""
        mock_feed_with_data = MagicMock()
        mock_feed_with_data.has_data = True

        manager = create_manager_with_feeds(
            {
                FEED_TIDES: mock_feed_with_data,
                FeedName.CURRENTS: None,  # Not configured
            }
        )

        assert manager.has_data is True


# =============================================================================
# Serving cycle resilience tests
# =============================================================================


class TestServingCycleResilience:
    """Tests for the serving cycle's update helpers.

    serve_location()'s handling of a propagated failure is tested in
    tests/test_manager_update_once.py.
    """

    @pytest.mark.asyncio
    async def test_update_dataset_propagates_feed_failure(self) -> None:
        """Feed update raises → update_dataset() re-raises it to the cycle."""
        from shallweswim.core.feeds import Feed

        # Create a mock feed that raises on update
        mock_feed = MagicMock(spec=Feed)
        mock_feed.update.side_effect = BaseClientError("API returned garbage")
        mock_feed.is_expired = True

        manager = create_manager_with_feeds({FEED_TIDES: mock_feed})

        # The update should raise (feed.update() raises)
        # This is expected - serve_location() catches this
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

        feeds_dict: dict[str, Any] = {FEED_TIDES: mock_feed}

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
                FeedName.CURRENTS: mock_feed2,
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
                FeedName.CURRENTS: mock_expired_feed,
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
                FeedName.CURRENTS: mock_feed2,
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
                FeedName.CURRENTS: None,  # Not configured for this location
            }
        )

        assert manager.ready is True

    def test_returns_true_when_all_feeds_are_none(self) -> None:
        """All feed slots are None → True (nothing configured)."""
        manager = create_manager_with_feeds(
            {
                FEED_TIDES: None,
                FeedName.CURRENTS: None,
            }
        )

        assert manager.ready is True


# =============================================================================
# Runaway retry prevention tests
# =============================================================================


def create_test_feed(expiration_minutes: int = 10) -> Any:
    """Create a test feed for runaway retry tests."""
    import datetime

    import pandas as pd
    from pandera.pandas import DataFrameModel

    from shallweswim.clients.base import BaseApiClient
    from shallweswim.config import BaseFeedConfig
    from shallweswim.core.feeds import Feed
    from shallweswim.dataframe_models import WaterTempDataModel

    class TestFeedConfig(BaseFeedConfig, frozen=True):
        """Test configuration for TestFeed."""

        @property
        def citation(self) -> str:
            return "Test Feed Citation"

        @property
        def citation_key(self) -> str:
            return "test-feed"

    class TestFeed(Feed):
        """Concrete Feed for testing retry behavior."""

        feed_config: TestFeedConfig = TestFeedConfig()  # type: ignore[assignment]

        @property
        def data_model(self) -> type[DataFrameModel]:
            """Return a simple data model for validation."""
            return WaterTempDataModel  # type: ignore[return-value]

        async def _fetch(self, clients: dict[str, BaseApiClient]) -> pd.DataFrame:
            # Default implementation - tests override with mock
            index = pd.date_range(start="2023-01-01", periods=3, freq="h")
            return pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=index)

    config = create_minimal_config()
    return TestFeed(
        location_config=config,
        expiration_interval=datetime.timedelta(minutes=expiration_minutes),
    )


class TestRunawayRetryPrevention:
    """Tests for runaway retry prevention.

    Verifies the fix for the production bug where stations returning "no data"
    caused 1500+ retries over 25 minutes because _fetch_timestamp wasn't set,
    making is_expired return True immediately.

    Key behaviors:
    - StationUnavailableError: Handled gracefully, schedules retry, NOT propagated
    - Other errors: Schedules retry, IS propagated
    - Both: is_expired returns False immediately after (no runaway retry)
    """

    @pytest.mark.asyncio
    async def test_station_unavailable_prevents_immediate_retry(self) -> None:
        """StationUnavailableError → is_expired is False (no immediate retry).

        This is the core fix for the runaway retry bug. When a station has no
        data, we schedule the next fetch but don't mark as expired immediately.
        """
        feed = create_test_feed()

        # Initially expired (never fetched)
        assert feed.is_expired is True
        assert feed._next_fetch_after is None

        # Mock _fetch to raise StationUnavailableError
        async def mock_fetch_unavailable(
            clients: dict[str, Any],
        ) -> None:
            raise StationUnavailableError("Station 12345 has no data for period")

        feed._fetch = mock_fetch_unavailable  # type: ignore[method-assign]

        # Update should NOT raise
        await feed.update(clients={})

        # CRITICAL: is_expired should be False now (prevents runaway retry)
        assert feed.is_expired is False, (
            "Feed should NOT be expired immediately after StationUnavailableError. "
            "This would cause runaway retries."
        )

        # Verify scheduling state
        assert feed._next_fetch_after is not None, "Should schedule next attempt"
        assert feed._fetch_timestamp is None, "No data fetched, timestamp not set"
        assert isinstance(feed._last_error, StationUnavailableError)

    @pytest.mark.asyncio
    async def test_other_error_prevents_immediate_retry_but_propagates(self) -> None:
        """Other errors → is_expired is False, but error IS propagated.

        Unlike StationUnavailableError, other errors (network issues, parsing
        failures) are propagated for visibility, but still schedule retry to
        prevent runaway retries in the serving cycle.
        """
        feed = create_test_feed()

        # Mock _fetch to raise a different error
        async def mock_fetch_error(clients: dict[str, Any]) -> None:
            raise BaseClientError("Unexpected API response format")

        feed._fetch = mock_fetch_error  # type: ignore[method-assign]

        # Update SHOULD raise (error is propagated)
        with pytest.raises(BaseClientError):
            await feed.update(clients={})

        # CRITICAL: is_expired should still be False (prevents runaway retry)
        assert feed.is_expired is False, (
            "Feed should NOT be expired immediately after error. "
            "This would cause runaway retries in the serving cycle."
        )

        # Verify scheduling state
        assert feed._next_fetch_after is not None, "Should schedule next attempt"
        assert feed._fetch_timestamp is None, "No data fetched, timestamp not set"

    @pytest.mark.asyncio
    async def test_success_sets_both_timestamps(self) -> None:
        """Successful fetch → both _fetch_timestamp and _next_fetch_after set.

        On success, we track when data was fetched (for age monitoring) AND
        when to try again (for scheduling).
        """
        import pandas as pd

        feed = create_test_feed()

        # Mock _fetch to return valid data (must match WaterTempDataModel schema)
        async def mock_fetch_success(clients: dict[str, Any]) -> pd.DataFrame:
            index = pd.date_range(start="2023-01-01", periods=3, freq="h", name="time")
            return pd.DataFrame({"water_temp": [15.0, 16.0, 17.0]}, index=index)

        feed._fetch = mock_fetch_success  # type: ignore[method-assign]

        await feed.update(clients={})

        # Both timestamps should be set
        assert feed._fetch_timestamp is not None, "Should track when data was fetched"
        assert feed._next_fetch_after is not None, "Should schedule next fetch"
        assert feed.is_expired is False, "Should not be expired immediately"
        assert feed._data is not None, "Should have data"

    @pytest.mark.asyncio
    async def test_error_does_not_affect_existing_data(self) -> None:
        """Errors don't clear existing data - stale data is still served.

        If we have cached data and a refresh fails, we keep serving the
        stale data rather than returning nothing.
        """
        import datetime

        import pandas as pd

        from shallweswim import util

        feed = create_test_feed()

        # Set up existing data
        index = pd.date_range(start="2023-01-01", periods=3, freq="h")
        existing_data = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=index)
        feed._data = existing_data
        feed._fetch_timestamp = util.utc_now()
        # Set _next_fetch_after to past to make feed expired
        feed._next_fetch_after = util.utc_now() - datetime.timedelta(hours=1)

        assert feed.is_expired is True  # Ready for refresh

        # Mock _fetch to raise error
        async def mock_fetch_error(clients: dict[str, Any]) -> None:
            raise StationUnavailableError("Station offline")

        feed._fetch = mock_fetch_error  # type: ignore[method-assign]

        await feed.update(clients={})

        # Data should still be present (stale data preserved)
        assert feed._data is not None, "Existing data should be preserved"
        assert len(feed._data) == 3, "Data should not be modified"
        # is_expired should be False (next attempt scheduled)
        assert feed.is_expired is False

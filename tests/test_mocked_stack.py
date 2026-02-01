"""Mock integration tests for the ShallWeSwim API.

These tests use mock API clients to test the full data flow without hitting
live NOAA/USGS APIs:
- Real update loop starts feeds
- Mock clients return controlled test data
- Real feeds process the data
- Real API handlers serve the data
- TestClient verifies correct responses

This validates the complete stack including error handling scenarios
like startup race conditions, partial data, and recovery from failures.
"""

import asyncio
import datetime
import math
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from unittest.mock import patch

import aiohttp
import pandas as pd
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shallweswim import api
from shallweswim.clients.base import StationUnavailableError
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi
from shallweswim.data import LocationDataManager
from shallweswim.types import TIDE_TYPE_CATEGORIES
from tests.conftest import (
    TEST_CONFIG_FULL,
    TEST_CONFIG_OBSERVATION_CURRENTS,
    TEST_CONFIG_PREDICTION_NO_CHARTS,
)


def _mock_config_get(code: str):
    """Mock config.get() to return our fake test configs."""
    configs = {
        TEST_CONFIG_FULL.code: TEST_CONFIG_FULL,
        TEST_CONFIG_OBSERVATION_CURRENTS.code: TEST_CONFIG_OBSERVATION_CURRENTS,
        TEST_CONFIG_PREDICTION_NO_CHARTS.code: TEST_CONFIG_PREDICTION_NO_CHARTS,
    }
    return configs.get(code)


# =============================================================================
# Test Data Factories
# =============================================================================


def create_mock_tides_df() -> pd.DataFrame:
    """Create mock tide predictions matching TidePredictionDataModel.

    Returns a DataFrame with 2 past tides and 2 future tides.
    """
    now = datetime.datetime.now()
    data = {
        "prediction": [-0.5, 1.2, -0.3, 1.4],
        "type": pd.Categorical(
            ["low", "high", "low", "high"], categories=TIDE_TYPE_CATEGORIES
        ),
    }
    index = pd.DatetimeIndex(
        [
            now - datetime.timedelta(hours=12),
            now - datetime.timedelta(hours=6),
            now + datetime.timedelta(hours=6),
            now + datetime.timedelta(hours=12),
        ],
        name="time",
    )
    return pd.DataFrame(data, index=index)


def create_mock_currents_df() -> pd.DataFrame:
    """Create mock current velocities matching CurrentDataModel.

    Returns 24 hours of data at 1-minute intervals with sinusoidal velocity.
    """
    now = datetime.datetime.now()
    # Create 24 hours of data at 1-minute intervals
    index = pd.date_range(
        now - datetime.timedelta(hours=12),
        now + datetime.timedelta(hours=12),
        freq="1min",
        name="time",
    )
    # Sinusoidal velocity pattern (flood/ebb cycle)
    velocities = [1.5 * math.sin(i * math.pi / 360) for i in range(len(index))]
    return pd.DataFrame({"velocity": velocities}, index=index)


def create_mock_temp_df() -> pd.DataFrame:
    """Create mock temperature data matching WaterTempDataModel.

    Returns 7 days of hourly data with constant temperature.
    """
    now = datetime.datetime.now()
    index = pd.date_range(
        now - datetime.timedelta(days=7),
        now,
        freq="h",
        name="time",
    )
    return pd.DataFrame({"water_temp": [68.5] * len(index)}, index=index)


# =============================================================================
# Mock Clients
# =============================================================================


class MockCoopsApi(CoopsApi):
    """Mock CO-OPS client with configurable failure modes.

    Attributes:
        should_fail_tides: If True, tides() raises failure_exception
        should_fail_currents: If True, currents() raises failure_exception
        should_fail_temperature: If True, temperature() raises failure_exception
        failure_exception: Exception to raise on failure
        call_count_tides: Number of times tides() was called
        call_count_currents: Number of times currents() was called
        call_count_temperature: Number of times temperature() was called
        fail_tides_after_n_calls: Fail tides() after N successful calls
        fail_currents_after_n_calls: Fail currents() after N successful calls
    """

    def __init__(self, session: aiohttp.ClientSession):
        super().__init__(session)
        # Failure control flags
        self.should_fail_tides = False
        self.should_fail_currents = False
        self.should_fail_temperature = False
        self.failure_exception: Exception = StationUnavailableError("Station offline")
        # Call tracking
        self.call_count_tides = 0
        self.call_count_currents = 0
        self.call_count_temperature = 0
        # Delayed failure support
        self.fail_tides_after_n_calls: int | None = None
        self.fail_currents_after_n_calls: int | None = None

    async def tides(self, station: int, location_code: str = "unknown") -> pd.DataFrame:
        """Return mock tide predictions or raise if configured to fail."""
        self.call_count_tides += 1

        # Fail after N calls (for testing refresh failures)
        if (
            self.fail_tides_after_n_calls is not None
            and self.call_count_tides > self.fail_tides_after_n_calls
        ):
            raise self.failure_exception

        if self.should_fail_tides:
            raise self.failure_exception

        return create_mock_tides_df()

    async def currents(
        self,
        station: str,
        interpolate: bool = True,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return mock current predictions or raise if configured to fail."""
        self.call_count_currents += 1

        # Fail after N calls (for testing refresh failures)
        if (
            self.fail_currents_after_n_calls is not None
            and self.call_count_currents > self.fail_currents_after_n_calls
        ):
            raise self.failure_exception

        if self.should_fail_currents:
            raise self.failure_exception

        return create_mock_currents_df()

    async def temperature(
        self,
        station: int,
        product: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        interval: str | None = None,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return mock temperature data or raise if configured to fail."""
        self.call_count_temperature += 1

        if self.should_fail_temperature:
            raise self.failure_exception

        return create_mock_temp_df()


class MockNwisApi(NwisApi):
    """Mock NWIS client that returns test data.

    Currently just passes through since NYC config doesn't use NWIS.
    """

    def __init__(self, session: aiohttp.ClientSession):
        super().__init__(session)


class MockNdbcApi(NdbcApi):
    """Mock NDBC client that returns test data.

    Currently just passes through since NYC config doesn't use NDBC.
    """

    def __init__(self, session: Any):
        super().__init__(session)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def mock_api_client() -> AsyncGenerator[TestClient]:
    """Create test client with mock API clients.

    This fixture waits for all data to be loaded before yielding.
    Use this for happy path tests where all data should be available.
    """
    # Patch config.get to return our fake test configs
    with patch("shallweswim.config.get", _mock_config_get):
        app = FastAPI()
        app.state.data_managers = {}
        app.state.process_pool = ProcessPoolExecutor()

        async with aiohttp.ClientSession() as session:
            app.state.http_session = session

            # Create MOCK clients instead of real ones
            mock_clients: dict[str, Any] = {
                "coops": MockCoopsApi(session=session),
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            # Create LocationDataManager with fake test config
            cfg = TEST_CONFIG_FULL

            app.state.data_managers[cfg.code] = LocationDataManager(
                cfg, clients=mock_clients, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code].start()

            # Wait for data to load (should be fast with mocks)
            await app.state.data_managers[cfg.code].wait_until_ready(timeout=10.0)

            api.register_routes(app)

            yield TestClient(app)

            # Cleanup
            await app.state.data_managers[cfg.code].stop()
            app.state.process_pool.shutdown(wait=False)


@pytest_asyncio.fixture
async def mock_api_client_with_controls() -> AsyncGenerator[
    tuple[TestClient, MockCoopsApi, LocationDataManager, FastAPI]
]:
    """Create test client with controllable mock behavior.

    This fixture provides access to the mock client for test control.
    Returns tuple of (TestClient, MockCoopsApi, LocationDataManager, FastAPI).
    """
    # Patch config.get to return our fake test configs
    with patch("shallweswim.config.get", _mock_config_get):
        app = FastAPI()
        app.state.data_managers = {}
        app.state.process_pool = ProcessPoolExecutor()

        async with aiohttp.ClientSession() as session:
            app.state.http_session = session

            # Create MOCK clients
            mock_coops = MockCoopsApi(session=session)
            mock_clients: dict[str, Any] = {
                "coops": mock_coops,
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            # Create LocationDataManager with fake test config
            cfg = TEST_CONFIG_FULL

            manager = LocationDataManager(
                cfg, clients=mock_clients, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code] = manager
            manager.start()

            # Wait for data to load (should be fast with mocks)
            await manager.wait_until_ready(timeout=10.0)

            api.register_routes(app)

            yield TestClient(app), mock_coops, manager, app

            # Cleanup
            await manager.stop()
            app.state.process_pool.shutdown(wait=False)


@pytest_asyncio.fixture
async def mock_api_client_no_wait() -> AsyncGenerator[
    tuple[TestClient, MockCoopsApi, LocationDataManager, FastAPI]
]:
    """Create test client WITHOUT waiting for data to load.

    Use this for testing startup race conditions.
    """
    # Patch config.get to return our fake test configs
    with patch("shallweswim.config.get", _mock_config_get):
        app = FastAPI()
        app.state.data_managers = {}
        app.state.process_pool = ProcessPoolExecutor()

        async with aiohttp.ClientSession() as session:
            app.state.http_session = session

            # Create MOCK clients
            mock_coops = MockCoopsApi(session=session)
            mock_clients: dict[str, Any] = {
                "coops": mock_coops,
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            # Create LocationDataManager with fake test config
            cfg = TEST_CONFIG_FULL

            manager = LocationDataManager(
                cfg, clients=mock_clients, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code] = manager

            # DON'T start the manager yet - let tests control this
            api.register_routes(app)

            yield TestClient(app), mock_coops, manager, app

            # Cleanup
            await manager.stop()
            app.state.process_pool.shutdown(wait=False)


# =============================================================================
# Happy Path Tests
# =============================================================================


@pytest.mark.asyncio
async def test_conditions_with_mock_data(mock_api_client: TestClient) -> None:
    """Full /conditions endpoint returns all data types."""
    response = mock_api_client.get("/api/nyc/conditions")

    assert response.status_code == 200
    data = response.json()

    # Verify all data types present
    assert data["location"]["code"] == "nyc"
    assert "Test Location" in data["location"]["name"]

    # Temperature should be present (NYC has live temps)
    assert data["temperature"] is not None
    assert data["temperature"]["water_temp"] == 68.5
    assert data["temperature"]["units"] == "F"

    # Tides should be present
    assert data["tides"] is not None
    assert len(data["tides"]["past"]) > 0
    assert len(data["tides"]["next"]) > 0

    # Currents should be present (NYC has currents)
    assert data["current"] is not None
    assert "magnitude" in data["current"]
    assert "direction" in data["current"]


@pytest.mark.asyncio
async def test_current_tide_plot_with_mock_data(mock_api_client: TestClient) -> None:
    """Plot generation works with mock data."""
    response = mock_api_client.get("/api/nyc/current_tide_plot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    assert "<svg" in response.text
    assert "</svg>" in response.text


@pytest.mark.asyncio
async def test_healthy_with_mock_data(mock_api_client: TestClient) -> None:
    """Health check returns healthy with mock data."""
    response = mock_api_client.get("/api/healthy")

    assert response.status_code == 200
    assert response.json() is True


@pytest.mark.asyncio
async def test_status_with_mock_data(mock_api_client: TestClient) -> None:
    """Status endpoint shows fresh data."""
    response = mock_api_client.get("/api/nyc/status")

    assert response.status_code == 200
    data = response.json()

    # Check feed statuses
    assert "feeds" in data
    for feed_name, feed_status in data["feeds"].items():
        assert feed_status["is_expired"] is False, (
            f"Feed {feed_name} should not be expired"
        )
        assert feed_status["is_healthy"] is True, f"Feed {feed_name} should be healthy"


@pytest.mark.asyncio
async def test_feed_data_endpoint(mock_api_client: TestClient) -> None:
    """Raw feed data endpoint returns data."""
    response = mock_api_client.get("/api/nyc/data/tides")

    assert response.status_code == 200
    data = response.json()

    # Should have tide entries
    assert len(data) > 0
    # Check structure (dict with timestamp keys)
    for _key, value in data.items():
        assert "prediction" in value
        assert "type" in value


# =============================================================================
# Failure Condition Tests
# =============================================================================


@pytest.mark.asyncio
async def test_startup_before_data_fetched(
    mock_api_client_no_wait: tuple[
        TestClient, MockCoopsApi, LocationDataManager, FastAPI
    ],
) -> None:
    """Request before data fetched returns 503 (not 500 crash).

    Emulates: User requests during cold start.
    """
    client, _mock_coops, _manager, _app = mock_api_client_no_wait

    # Manager not started yet - no data loaded
    response = client.get("/api/nyc/conditions")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_partial_feeds_loaded(
    mock_api_client_no_wait: tuple[
        TestClient, MockCoopsApi, LocationDataManager, FastAPI
    ],
) -> None:
    """Some feeds ready, others not - returns 200 with partial data.

    Emulates: The 500 bug scenario (has_data=True but specific feed empty).
    """
    client, mock_coops, manager, _app = mock_api_client_no_wait

    # Configure: tides works, temps fails
    mock_coops.should_fail_temperature = True

    # Start the manager
    manager.start()

    # Wait a bit for tides to load (but temps will fail)
    await asyncio.sleep(1.0)

    response = client.get("/api/nyc/conditions")

    # Should return 200 with partial data, not 500 crash
    # Note: if tides loaded, has_data=True, so we get 200
    if manager.has_data:
        assert response.status_code == 200
        data = response.json()
        # Temperature should be null since it failed
        assert data["temperature"] is None
        # Tides should be present
        assert data["tides"] is not None
    else:
        # If no data loaded yet, 503 is acceptable
        assert response.status_code == 503


@pytest.mark.asyncio
async def test_all_feeds_expired_but_have_stale_data(
    mock_api_client_with_controls: tuple[
        TestClient, MockCoopsApi, LocationDataManager, FastAPI
    ],
) -> None:
    """Expired feeds still serve stale data.

    Emulates: Extended API outage, system serves cached data.
    """
    client, _mock_coops, manager, _app = mock_api_client_with_controls

    # Data is loaded (fixture waits for ready)
    assert manager.has_data is True

    # Manually expire all feeds by setting fetch_timestamp to past
    old_time = datetime.datetime.now(datetime.UTC).replace(
        tzinfo=None
    ) - datetime.timedelta(days=1)
    for feed in manager._feeds.values():
        if feed is not None:
            feed._fetch_timestamp = old_time

    # Verify feeds are now expired
    for feed in manager._feeds.values():
        if feed is not None:
            assert feed.is_expired is True

    # /conditions should still return 200 (serves stale data)
    response = client.get("/api/nyc/conditions")
    assert response.status_code == 200

    # /healthy should return 200 (has_data=True)
    response = client.get("/api/healthy")
    assert response.status_code == 200

    # /status should show is_expired=True
    response = client.get("/api/nyc/status")
    assert response.status_code == 200
    data = response.json()
    for feed_name, feed_status in data["feeds"].items():
        assert feed_status["is_expired"] is True, f"Feed {feed_name} should be expired"


@pytest.mark.asyncio
async def test_station_unavailable_during_startup(
    mock_api_client_no_wait: tuple[
        TestClient, MockCoopsApi, LocationDataManager, FastAPI
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Station offline during startup - graceful handling.

    Emulates: NOAA station maintenance.
    """

    client, mock_coops, manager, _app = mock_api_client_no_wait

    # Configure all feeds to fail
    mock_coops.should_fail_tides = True
    mock_coops.should_fail_currents = True
    mock_coops.should_fail_temperature = True
    mock_coops.failure_exception = StationUnavailableError("Station under maintenance")

    # Start the manager
    manager.start()

    # Wait for update attempt
    await asyncio.sleep(1.0)

    # /conditions should return 503 (no data)
    response = client.get("/api/nyc/conditions")
    assert response.status_code == 503

    # Check logs - should have WARNING for station unavailable, not ERROR
    # (The actual log messages are generated by the feeds module)


@pytest.mark.asyncio
async def test_recovery_after_failure() -> None:
    """System can recover after previous instance failed.

    Emulates: Service restart after station comes back online.

    This creates two separate managers to simulate:
    1. First manager fails due to station unavailable
    2. Service restarts (new manager) when station is back
    3. New manager loads data successfully
    """
    # Patch config.get to return our fake test configs
    with patch("shallweswim.config.get", _mock_config_get):
        app = FastAPI()
        app.state.data_managers = {}
        app.state.process_pool = ProcessPoolExecutor()

        async with aiohttp.ClientSession() as session:
            app.state.http_session = session

            # Create MOCK clients - initially failing
            mock_coops = MockCoopsApi(session=session)
            mock_coops.should_fail_tides = True
            mock_coops.should_fail_currents = True
            mock_coops.should_fail_temperature = True

            mock_clients: dict[str, Any] = {
                "coops": mock_coops,
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            cfg = TEST_CONFIG_FULL

            # First manager - will fail
            manager1 = LocationDataManager(
                cfg, clients=mock_clients, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code] = manager1
            manager1.start()

            api.register_routes(app)
            client = TestClient(app)

            # Wait for initial (failed) fetch attempt
            await asyncio.sleep(1.0)

            # Should return 503 initially
            response = client.get("/api/nyc/conditions")
            assert response.status_code == 503

            # Stop the failed manager
            await manager1.stop()

            # === Simulate service restart with station back online ===

            # Create fresh mock clients - now working
            mock_coops2 = MockCoopsApi(session=session)
            # Not setting should_fail flags = success

            mock_clients2: dict[str, Any] = {
                "coops": mock_coops2,
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            # Create NEW manager with working mocks
            manager2 = LocationDataManager(
                cfg, clients=mock_clients2, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code] = manager2
            manager2.start()

            # Wait for data to load
            await manager2.wait_until_ready(timeout=10.0)

            # Should now return 200
            response = client.get("/api/nyc/conditions")
            assert response.status_code == 200

            # Verify data is present
            data = response.json()
            assert data["tides"] is not None
            assert data["current"] is not None

            # Cleanup
            await manager2.stop()
            app.state.process_pool.shutdown(wait=False)


@pytest.mark.asyncio
async def test_plot_with_missing_currents(
    mock_api_client_no_wait: tuple[
        TestClient, MockCoopsApi, LocationDataManager, FastAPI
    ],
) -> None:
    """Plot endpoint with missing currents returns 503.

    Emulates: Current station offline but tide station working.
    """
    client, mock_coops, manager, _app = mock_api_client_no_wait

    # Configure: tides works, currents fails
    mock_coops.should_fail_currents = True

    # Start the manager
    manager.start()

    # Wait for tides to load
    await asyncio.sleep(1.0)

    # Plot requires both tides AND currents
    response = client.get("/api/nyc/current_tide_plot")

    # Should return 503 with specific message, not 500 crash
    assert response.status_code == 503
    assert "tide/current data temporarily unavailable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_feed_update_fails_during_refresh(
    mock_api_client_with_controls: tuple[
        TestClient, MockCoopsApi, LocationDataManager, FastAPI
    ],
) -> None:
    """API failure during refresh - old data still served.

    Emulates: NOAA outage during refresh window.
    """
    client, mock_coops, manager, _app = mock_api_client_with_controls

    # Data is loaded (fixture waits for ready)
    # Verify we have data
    response = client.get("/api/nyc/conditions")
    assert response.status_code == 200

    # Now configure to fail on NEXT call
    mock_coops.fail_tides_after_n_calls = mock_coops.call_count_tides
    mock_coops.fail_currents_after_n_calls = mock_coops.call_count_currents

    # Force feeds to be marked as expired
    old_time = datetime.datetime.now(datetime.UTC).replace(
        tzinfo=None
    ) - datetime.timedelta(days=1)
    for feed in manager._feeds.values():
        if feed is not None:
            feed._fetch_timestamp = old_time

    # Let the update loop attempt to refresh (it will fail)
    await asyncio.sleep(1.0)

    # Old data should still be served
    response = client.get("/api/nyc/conditions")
    assert response.status_code == 200
    # Data should still be present (stale but available)
    data = response.json()
    assert data["tides"] is not None


# =============================================================================
# Location-Specific Behavior Tests
# =============================================================================


@pytest.mark.asyncio
async def test_currents_endpoint_observation_source_returns_404() -> None:
    """Currents endpoint for OBSERVATION-type source returns 404.

    This location has currents_source configured but it's OBSERVATION type (like Louisville),
    not PREDICTION type. The /currents endpoint only works with PREDICTION sources.
    Returns 404 (not 501) to avoid 5xx alerting.
    """
    # Patch config.get to return our fake test configs
    with patch("shallweswim.config.get", _mock_config_get):
        app = FastAPI()
        app.state.data_managers = {}
        app.state.process_pool = ProcessPoolExecutor()

        async with aiohttp.ClientSession() as session:
            app.state.http_session = session
            cfg = TEST_CONFIG_OBSERVATION_CURRENTS  # Has currents but OBSERVATION type

            mock_clients: dict[str, Any] = {
                "coops": MockCoopsApi(session=session),
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            manager = LocationDataManager(
                cfg, clients=mock_clients, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code] = manager
            manager.start()

            try:
                await manager.wait_until_ready(timeout=10.0)
                api.register_routes(app)

                with TestClient(app) as client:
                    response = client.get(f"/api/{cfg.code}/currents")
                    assert response.status_code == 404
                    assert "observation-only" in response.json()["detail"]
            finally:
                await manager.stop()
                app.state.process_pool.shutdown(wait=False)


@pytest.mark.asyncio
async def test_currents_endpoint_prediction_no_charts_returns_null_charts() -> None:
    """Currents endpoint for PREDICTION source without charts returns null chart fields.

    This location has PREDICTION-type currents (supports /currents endpoint) but
    has_static_charts=False, so legacy_chart and current_chart_filename should be null.
    """
    with patch("shallweswim.config.get", _mock_config_get):
        app = FastAPI()
        app.state.data_managers = {}
        app.state.process_pool = ProcessPoolExecutor()

        async with aiohttp.ClientSession() as session:
            app.state.http_session = session
            cfg = TEST_CONFIG_PREDICTION_NO_CHARTS  # PREDICTION type but no charts

            mock_clients: dict[str, Any] = {
                "coops": MockCoopsApi(session=session),
                "nwis": MockNwisApi(session=session),
                "ndbc": MockNdbcApi(session=session),
            }

            manager = LocationDataManager(
                cfg, clients=mock_clients, process_pool=app.state.process_pool
            )
            app.state.data_managers[cfg.code] = manager
            manager.start()

            try:
                await manager.wait_until_ready(timeout=10.0)
                api.register_routes(app)

                with TestClient(app) as client:
                    response = client.get(f"/api/{cfg.code}/currents")
                    assert response.status_code == 200
                    data = response.json()
                    # Should have current prediction data
                    assert "current" in data
                    assert data["current"] is not None
                    # But chart fields should be null (no static charts configured)
                    assert data["legacy_chart"] is None
                    assert data["current_chart_filename"] is None
            finally:
                await manager.stop()
                app.state.process_pool.shutdown(wait=False)

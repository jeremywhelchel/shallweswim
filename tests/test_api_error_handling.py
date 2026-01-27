"""Tests for API endpoint error handling.

Verifies routes.py handles missing/partial data gracefully:
- No data → 503 (with WARNING log, not ERROR)
- Partial data → partial response with nulls (not 500)

These tests emulate the conditions from production bugs to prevent regressions.
"""

import datetime
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytz
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shallweswim import types as sw_types
from shallweswim.api import register_routes
from shallweswim.config import (
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    LocationConfig,
)
from shallweswim.core.feeds import FEED_CURRENTS, FEED_TIDES
from shallweswim.data import LocationDataManager


def create_nyc_config() -> LocationConfig:
    """Create a NYC location config for testing."""
    return LocationConfig(
        code="nyc",
        name="New York City",
        swim_location="Test Swim Spot NYC",
        swim_location_link="http://example.com/nyc",
        description="Mock NYC description",
        latitude=40.7128,
        longitude=-74.0060,
        timezone=pytz.timezone("US/Eastern"),
        temp_source=CoopsTempFeedConfig(
            station=8518750, name="The Battery", live_enabled=True
        ),
        tide_source=CoopsTideFeedConfig(station=8518750, name="The Battery"),
        currents_source=CoopsCurrentsFeedConfig(
            stations=["NYH1914"], name="Narrows North"
        ),
        enabled=True,
    )


def create_mock_tide_info() -> sw_types.TideInfo:
    """Create mock tide info for testing."""
    mock_dt = datetime.datetime(2025, 5, 4, 12, 0, 0)
    past_tide = sw_types.TideEntry(
        time=mock_dt - datetime.timedelta(hours=6),
        type=sw_types.TideCategory.LOW,
        prediction=-0.5,
    )
    next_tide = sw_types.TideEntry(
        time=mock_dt + datetime.timedelta(hours=6),
        type=sw_types.TideCategory.HIGH,
        prediction=1.2,
    )
    return sw_types.TideInfo(past=[past_tide], next=[next_tide])


def create_mock_current_info() -> sw_types.CurrentInfo:
    """Create mock current info for testing."""
    return sw_types.CurrentInfo(
        timestamp=datetime.datetime(2025, 5, 4, 12, 0, 0),
        magnitude=0.75,
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.65,
        direction=sw_types.CurrentDirection.FLOODING,
        state_description="getting stronger",
    )


@pytest.fixture
def app_with_mock_manager() -> Any:
    """Create app with controllable mock data manager.

    Yields:
        Tuple of (app, mock_manager) where mock_manager can be configured
        for different test scenarios.
    """
    app = FastAPI()
    app.state.data_managers = {}
    register_routes(app)

    mock_manager = MagicMock(spec=LocationDataManager)
    mock_manager.config = create_nyc_config()
    # Default: no data
    mock_manager.has_data = False
    mock_manager.has_feed_data.return_value = False

    with patch("shallweswim.config.get") as mock_get:
        mock_get.return_value = create_nyc_config()
        app.state.data_managers["nyc"] = mock_manager
        yield app, mock_manager


# =============================================================================
# /api/{location}/conditions endpoint tests
# =============================================================================


def test_conditions_no_data_returns_503(app_with_mock_manager: Any) -> None:
    """No data available → 503 (not 500).

    Emulates startup before any data has been loaded.
    The endpoint should return 503 "temporarily unavailable", not crash.
    """
    app, mock_manager = app_with_mock_manager
    mock_manager.has_data = False

    client = TestClient(app)
    response = client.get("/api/nyc/conditions")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_conditions_no_data_logs_warning(
    app_with_mock_manager: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """No data available logs WARNING (not ERROR).

    503 for missing data is an expected operational condition during startup
    or refresh windows, so it should log at WARNING level to avoid triggering
    alerts.
    """
    app, mock_manager = app_with_mock_manager
    mock_manager.has_data = False

    with caplog.at_level(logging.DEBUG):
        client = TestClient(app)
        client.get("/api/nyc/conditions")

    # Should log at WARNING level, not ERROR
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]

    assert len(warning_records) >= 1, "Expected at least one WARNING log"
    assert any("No data available" in r.message for r in warning_records)
    # Should NOT have ERROR logs for this expected condition
    assert len(error_records) == 0, f"Unexpected ERROR logs: {error_records}"


def test_conditions_partial_data_returns_200(app_with_mock_manager: Any) -> None:
    """Partial data (tides available, temps not) → 200 with temp=null.

    This is the exact regression test for the 500 bug:
    - has_data was True (some feed had data)
    - But specific feed (live_temps) had no data
    - Code tried to access it → AssertionError → 500

    With the fix, we check has_feed_data() before accessing each feed,
    returning null for unavailable feeds instead of crashing.
    """
    app, mock_manager = app_with_mock_manager
    mock_manager.has_data = True  # Some feed has data

    # Configure: tides available, temps and currents not
    def has_feed_data_side_effect(feed_name: str) -> bool:
        return feed_name == FEED_TIDES

    mock_manager.has_feed_data.side_effect = has_feed_data_side_effect
    mock_manager.get_current_tide_info.return_value = create_mock_tide_info()

    client = TestClient(app)
    response = client.get("/api/nyc/conditions")

    # Should return 200 with partial data, not 500
    assert response.status_code == 200
    data = response.json()
    assert data["temperature"] is None  # Not available - no crash
    assert data["tides"] is not None  # Available
    assert data["current"] is None  # Not available - no crash


def test_conditions_all_feeds_missing_specific_data(
    app_with_mock_manager: Any,
) -> None:
    """has_data=True but all specific feeds missing → 200 with all nulls.

    Edge case: has_data passes (technically true - feeds exist) but
    none of the specific feeds have data yet. Should still return 200
    with null fields, not crash.
    """
    app, mock_manager = app_with_mock_manager
    mock_manager.has_data = True  # Passes the initial check
    mock_manager.has_feed_data.return_value = False  # But no specific feed has data

    client = TestClient(app)
    response = client.get("/api/nyc/conditions")

    assert response.status_code == 200
    data = response.json()
    assert data["temperature"] is None
    assert data["tides"] is None
    assert data["current"] is None
    # Location info should still be present
    assert data["location"]["code"] == "nyc"
    assert data["location"]["name"] == "New York City"


# =============================================================================
# /api/{location}/current_tide_plot endpoint tests
# =============================================================================


def test_plot_missing_tides_returns_503(app_with_mock_manager: Any) -> None:
    """Plot endpoint with missing tides data → 503.

    The plot endpoint requires both tides AND currents data.
    If tides is missing, should return 503, not crash.
    """
    app, mock_manager = app_with_mock_manager
    mock_manager.has_data = True

    # Currents available, tides not
    def has_feed_data_side_effect(feed_name: str) -> bool:
        return feed_name == FEED_CURRENTS

    mock_manager.has_feed_data.side_effect = has_feed_data_side_effect

    client = TestClient(app)
    response = client.get("/api/nyc/current_tide_plot")

    assert response.status_code == 503
    assert "tide/current data temporarily unavailable" in response.json()["detail"]


def test_plot_missing_currents_returns_503(app_with_mock_manager: Any) -> None:
    """Plot endpoint with missing currents data → 503.

    The plot endpoint requires both tides AND currents data.
    If currents is missing, should return 503, not crash.
    """
    app, mock_manager = app_with_mock_manager
    mock_manager.has_data = True

    # Tides available, currents not
    def has_feed_data_side_effect(feed_name: str) -> bool:
        return feed_name == FEED_TIDES

    mock_manager.has_feed_data.side_effect = has_feed_data_side_effect

    client = TestClient(app)
    response = client.get("/api/nyc/current_tide_plot")

    assert response.status_code == 503
    assert "tide/current data temporarily unavailable" in response.json()["detail"]


# =============================================================================
# /api/healthy endpoint tests
# =============================================================================


def test_healthy_no_locations_returns_503() -> None:
    """No locations configured → 503.

    Service should be unhealthy if no locations are set up.
    """
    app = FastAPI()
    app.state.data_managers = {}  # No locations
    register_routes(app)

    client = TestClient(app)
    response = client.get("/api/healthy")

    assert response.status_code == 503
    assert "no locations configured" in response.json()["detail"]


def test_healthy_no_location_has_data_returns_503() -> None:
    """All locations have no data → 503.

    Service should be unhealthy if no location can serve data.
    """
    app = FastAPI()
    app.state.data_managers = {}
    register_routes(app)

    # Create two mock managers with no data
    mock_manager1 = MagicMock(spec=LocationDataManager)
    mock_manager1.has_data = False

    mock_manager2 = MagicMock(spec=LocationDataManager)
    mock_manager2.has_data = False

    app.state.data_managers["nyc"] = mock_manager1
    app.state.data_managers["sfo"] = mock_manager2

    client = TestClient(app)
    response = client.get("/api/healthy")

    assert response.status_code == 503
    assert "no location has data" in response.json()["detail"]


def test_healthy_at_least_one_location_has_data_returns_200() -> None:
    """At least one location has data → 200 (healthy).

    Service is healthy if at least one location can serve data.
    This is a lenient check - single station outages shouldn't
    mark the entire service unhealthy.
    """
    app = FastAPI()
    app.state.data_managers = {}
    register_routes(app)

    # One location with data, one without
    mock_manager_with_data = MagicMock(spec=LocationDataManager)
    mock_manager_with_data.has_data = True

    mock_manager_without_data = MagicMock(spec=LocationDataManager)
    mock_manager_without_data.has_data = False

    app.state.data_managers["nyc"] = mock_manager_with_data
    app.state.data_managers["sfo"] = mock_manager_without_data

    client = TestClient(app)
    response = client.get("/api/healthy")

    assert response.status_code == 200
    assert response.json() is True


def test_healthy_all_locations_have_data_returns_200() -> None:
    """All locations have data → 200 (healthy)."""
    app = FastAPI()
    app.state.data_managers = {}
    register_routes(app)

    mock_manager1 = MagicMock(spec=LocationDataManager)
    mock_manager1.has_data = True

    mock_manager2 = MagicMock(spec=LocationDataManager)
    mock_manager2.has_data = True

    app.state.data_managers["nyc"] = mock_manager1
    app.state.data_managers["sfo"] = mock_manager2

    client = TestClient(app)
    response = client.get("/api/healthy")

    assert response.status_code == 200
    assert response.json() is True

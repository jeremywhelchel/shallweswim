"""Tests for the API status endpoints."""

# pylint: disable=duplicate-code,unused-argument

# Standard library imports
import datetime
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pandas as pd

# Third-party imports
import pytest
import pytz
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from shallweswim import types as sw_types

# Local imports
from shallweswim.api import register_routes
from shallweswim.api_types import FeedStatus, LocationConditions, LocationStatus
from shallweswim.config import (
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    LocationConfig,
)
from shallweswim.data import LocationDataManager
from tests.helpers import assert_json_serializable


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI application for testing."""
    app_instance = FastAPI()
    # Initialize app.state.data_managers
    app_instance.state.data_managers = {}
    register_routes(app_instance)
    return app_instance


@pytest.fixture
def test_client(app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_data_managers(
    app: FastAPI,
) -> Generator[dict[str, LocationConfig]]:
    """Create mock data managers for testing."""
    # Create real location configs
    nyc_config = LocationConfig(
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

    sf_config = LocationConfig(
        code="sfo",  # Use 3-letter code 'sfo'
        name="San Francisco",
        swim_location="Test Swim Spot SF",
        swim_location_link="http://example.com/sf",
        description="Mock SF description",
        latitude=37.7749,
        longitude=-122.4194,
        timezone=pytz.timezone("US/Pacific"),
        tide_source=CoopsTideFeedConfig(
            station=9414290, name="San Francisco"
        ),  # Example tide source
        enabled=True,
        # No temp or current sources for this mock
    )

    # Create mock data managers
    nyc_data = MagicMock(spec=LocationDataManager)
    nyc_data.config = nyc_config
    nyc_data._feeds = {}  # Initialize _feeds attribute
    nyc_data.status = LocationStatus(
        feeds={
            "tides": FeedStatus(
                name="NoaaTidesFeed",
                location="nyc",
                fetch_timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
                age_seconds=3600,
                is_expired=False,
                is_healthy=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            ),
            "currents": FeedStatus(
                name="MultiStationCurrentsFeed",
                location="nyc",
                fetch_timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
                age_seconds=3600,
                is_expired=False,
                is_healthy=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            ),
        }
    )
    nyc_data.ready = True

    sf_data = MagicMock(spec=LocationDataManager)
    sf_data.config = sf_config
    sf_data.status = LocationStatus(
        feeds={
            "tides": FeedStatus(
                name="NoaaTidesFeed",
                location="sfo",
                fetch_timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
                age_seconds=3600,
                is_expired=False,
                is_healthy=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            )
        }
    )
    sf_data.ready = True

    # Patch the config.get function to return our mock configs
    try:
        with patch("shallweswim.config.get") as mock_get:
            mock_get.side_effect = lambda code: {
                "nyc": nyc_config,
                "sfo": sf_config,
            }.get(code)

            # Initialize and add data managers to the app state
            app.state.data_managers = {}  # Ensure the dictionary exists
            app.state.data_managers["nyc"] = nyc_data
            app.state.data_managers["sfo"] = sf_data

            yield {"nyc": nyc_config, "sfo": sf_config}

    finally:
        # Clean up state after tests
        app.state.data_managers = {}


def test_location_status_endpoint(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:  # pylint: disable=unused-argument
    """Test the location status endpoint."""
    # Test the NYC location status endpoint
    response = test_client.get("/api/nyc/status")
    assert response.status_code == 200

    # Check that the response is a valid JSON object
    status_data = response.json()
    assert_json_serializable(status_data)

    # Check that the response contains the expected data
    assert "feeds" in status_data
    assert "tides" in status_data["feeds"]
    assert "currents" in status_data["feeds"]
    assert status_data["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["feeds"]["tides"]["location"] == "nyc"

    # Test the SF location status endpoint
    response = test_client.get("/api/sfo/status")
    assert response.status_code == 200

    # Check that the response is a valid JSON object
    status_data = response.json()
    assert_json_serializable(status_data)

    # Check that the response contains the expected data
    assert "feeds" in status_data
    assert "tides" in status_data["feeds"]
    assert status_data["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["feeds"]["tides"]["location"] == "sfo"

    # Test a non-existent location
    response = test_client.get("/api/nonexistent/status")
    assert response.status_code == 404


def test_all_locations_status_endpoint(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:  # pylint: disable=unused-argument
    """Test the all locations status endpoint."""
    # Test the all locations status endpoint
    response = test_client.get("/api/status")
    assert response.status_code == 200

    # Check that the response is a valid JSON object
    status_data = response.json()
    assert_json_serializable(status_data)

    # Check that the response contains the expected data
    assert "nyc" in status_data
    assert "sfo" in status_data

    # Check the NYC data
    assert "feeds" in status_data["nyc"]
    assert "tides" in status_data["nyc"]["feeds"]
    assert "currents" in status_data["nyc"]["feeds"]
    assert status_data["nyc"]["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["nyc"]["feeds"]["tides"]["location"] == "nyc"

    # Check the SF data
    assert "feeds" in status_data["sfo"]
    assert "tides" in status_data["sfo"]["feeds"]
    assert status_data["sfo"]["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["sfo"]["feeds"]["tides"]["location"] == "sfo"


def test_get_location_conditions(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test the /api/{location}/conditions endpoint with all data types present."""
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    mock_manager = test_client.app.state.data_managers["nyc"]

    # --- 1. Define Mock Data ---
    mock_dt = datetime.datetime(2025, 5, 4, 12, 0, 0)
    # Create mock temperature data
    mock_temp_value = 18.5
    # Correct mock_tide_info structure
    mock_past_tide_entry = sw_types.TideEntry(
        time=mock_dt - datetime.timedelta(hours=6),
        type=sw_types.TideCategory.LOW,
        prediction=-0.5,
    )
    mock_next_tide_entry_1 = sw_types.TideEntry(
        time=mock_dt + datetime.timedelta(hours=6),
        type=sw_types.TideCategory.HIGH,
        prediction=1.2,
    )
    mock_next_tide_entry_2 = sw_types.TideEntry(
        time=mock_dt + datetime.timedelta(hours=12),
        type=sw_types.TideCategory.LOW,
        prediction=-0.3,
    )
    mock_tide_info = sw_types.TideInfo(
        past=[mock_past_tide_entry],
        next=[mock_next_tide_entry_1, mock_next_tide_entry_2],
    )
    mock_current_info = sw_types.CurrentInfo(
        timestamp=mock_dt,  # Use datetime here as CurrentInfo expects it
        magnitude=0.753,  # Test rounding
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.65,
        direction=sw_types.CurrentDirection.FLOODING,
        state_description="getting stronger",
    )

    # --- 2. Mock Manager Methods and Attributes ---
    # Mock the return value for get_current_temperature method
    mock_manager.get_current_temperature.return_value = sw_types.TemperatureReading(
        timestamp=mock_dt, temperature=mock_temp_value
    )
    # Mock the return value for get_current_flow_info method
    mock_manager.get_current_flow_info.return_value = mock_current_info
    # Mock the return value for predict_flow_at_time method
    mock_manager.predict_flow_at_time.return_value = mock_current_info
    # Mock the return value for get_current_tide_info method
    mock_manager.get_current_tide_info.return_value = mock_tide_info

    # --- 3. Call API Endpoint ---
    response = test_client.get("/api/nyc/conditions")

    # --- 4. Assert Response ---
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert_json_serializable(data)  # Ensure it's valid JSON

    # Validate structure with Pydantic model
    try:
        LocationConditions.model_validate(data)
    except Exception as e:
        pytest.fail(f"Response failed Pydantic validation: {e}\nData: {data}")

    # Assert Temperature
    assert "temperature" in data
    assert data["temperature"] is not None
    # Check timestamp in the response
    assert data["temperature"]["timestamp"] == mock_dt.isoformat()
    assert data["temperature"]["water_temp"] == mock_temp_value
    assert data["temperature"]["units"] == "F"  # Default unit in TemperatureInfo
    nyc_config = mock_data_managers["nyc"]
    assert nyc_config.temp_source is not None  # Help mypy
    assert (
        data["temperature"]["station_name"] == nyc_config.temp_source.name
    )  # Check station name from config

    # Correct Assert Tides
    assert "tides" in data
    assert data["tides"] is not None
    assert "past" in data["tides"]
    assert "next" in data["tides"]
    assert isinstance(data["tides"]["past"], list)
    assert len(data["tides"]["past"]) == 1
    assert isinstance(data["tides"]["next"], list)
    assert len(data["tides"]["next"]) == 2

    # Check past tide entry
    past_entry_data = data["tides"]["past"][0]
    assert past_entry_data["time"] == mock_past_tide_entry.time.isoformat()
    assert past_entry_data["type"] == mock_past_tide_entry.type.value
    assert past_entry_data["prediction"] == mock_past_tide_entry.prediction

    # Check next tide entries
    next_entry_data_1 = data["tides"]["next"][0]
    assert next_entry_data_1["time"] == mock_next_tide_entry_1.time.isoformat()
    assert next_entry_data_1["type"] == mock_next_tide_entry_1.type.value
    assert next_entry_data_1["prediction"] == mock_next_tide_entry_1.prediction

    next_entry_data_2 = data["tides"]["next"][1]
    assert next_entry_data_2["time"] == mock_next_tide_entry_2.time.isoformat()
    assert next_entry_data_2["type"] == mock_next_tide_entry_2.type.value
    assert next_entry_data_2["prediction"] == mock_next_tide_entry_2.prediction

    # Assert Currents
    assert "current" in data  # Note: API key is 'current'
    assert data["current"] is not None
    assert mock_current_info.timestamp is not None  # Help mypy
    assert data["current"]["timestamp"] == mock_current_info.timestamp.isoformat()
    assert (
        data["current"]["magnitude"] == 0.753
    )  # API currently returns unrounded value
    assert data["current"]["source_type"] == mock_current_info.source_type.value
    assert data["current"]["magnitude_pct"] == mock_current_info.magnitude_pct
    assert mock_current_info.direction is not None  # Help mypy
    assert data["current"]["direction"] == mock_current_info.direction.value
    assert data["current"]["state_description"] == mock_current_info.state_description


def test_get_location_conditions_missing_data(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test conditions endpoint when data sources are missing or return errors.

    NYC has all data sources configured, so if any source is unavailable,
    the API should fail immediately (fail fast and loud).
    """
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    mock_manager = test_client.app.state.data_managers["nyc"]

    # --- Setup: Mock temperature data to raise an error ---
    # This simulates a case where temperature data is unavailable
    mock_manager.get_current_temperature.side_effect = ValueError("No temperature data")

    # --- Call API and Assert Exception ---
    # Expect a ValueError because the API should fail fast when data is missing
    with pytest.raises(ValueError):
        test_client.get("/api/nyc/conditions")


def test_get_feed_data_success(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test the /api/{location}/data/{feed_type} endpoint for a successful fetch."""
    # Define mock feed data
    mock_feed_data = pd.DataFrame({"value": [1, 2, 3]})
    mock_feed_name = "MockFeed"

    # Get the mock data manager
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    mock_data_managers_dict = test_client.app.state.data_managers

    # Ensure the _feeds dictionary exists and add a mock feed
    if (
        not hasattr(mock_data_managers_dict["nyc"], "_feeds")
        or mock_data_managers_dict["nyc"]._feeds is None
    ):
        mock_data_managers_dict["nyc"]._feeds = {}
    mock_data_managers_dict["nyc"]._feeds["testfeed"] = MagicMock(
        values=mock_feed_data, name=mock_feed_name
    )

    # Call the API endpoint
    response = test_client.get("/api/nyc/data/testfeed")

    # Assert the response
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # The default serialization seems to be orient='index'
    # e.g., {'0': {'value': 1}, '1': {'value': 2}, ...}
    assert isinstance(data, dict)
    # Check if keys look like indices and values contain the 'value' key
    assert len(data) == len(mock_feed_data)
    for key, value_dict in data.items():
        assert key.isdigit()  # Check if key looks like an index
        assert isinstance(value_dict, dict)
        assert "value" in value_dict


def test_get_feed_data_location_not_found(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test the feed data endpoint with an invalid location."""
    response = test_client.get("/api/invalid_loc/data/testfeed")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert (
        "Location 'invalid_loc' not found or data not loaded"
        in response.json()["detail"]
    )


def test_get_feed_data_feed_not_found(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test the feed data endpoint when the feed type is invalid for the location."""
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    # Ensure the specific feed doesn't exist for the mock manager
    mock_data_managers_dict = test_client.app.state.data_managers
    # Safely remove 'badfeed' if it exists in _feeds
    if (
        hasattr(mock_data_managers_dict["nyc"], "_feeds")
        and "badfeed" in mock_data_managers_dict["nyc"]._feeds
    ):
        del mock_data_managers_dict["nyc"]._feeds["badfeed"]

    response = test_client.get("/api/nyc/data/badfeed")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Feed 'badfeed' not found for location 'nyc'" in response.json()["detail"]


def test_currents_endpoint_non_nyc_returns_404(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that currents endpoint returns 404 for non-NYC locations.

    Current predictions are only fully implemented for NYC. Other locations
    should return 404 (not 501) to avoid triggering 5xx alerting.

    See also: test_mocked_stack.py::test_currents_endpoint_non_nyc_returns_404
    which tests a location WITH currents_source configured (but not NYC).
    """
    # sfo doesn't have currents_source configured, so it gets 404
    response = test_client.get("/api/sfo/currents")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    # Either "does not support" (no currents_source) or "not available" (has source but not NYC)
    detail = response.json()["detail"]
    assert "not support" in detail or "not available" in detail


def test_invalid_location_returns_404(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that API requests for invalid locations return 404 errors.

    This is a pure code path test - doesn't need real external APIs.
    Moved from test_api_integration.py since it doesn't require live data.
    """
    # Test conditions endpoint
    response = test_client.get("/api/invalid_location/conditions")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    error_data = response.json()
    assert "detail" in error_data
    assert "not found" in error_data["detail"].lower()

    # Test currents endpoint
    response = test_client.get("/api/invalid_location/currents")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    error_data = response.json()
    assert "detail" in error_data
    assert "not found" in error_data["detail"].lower()

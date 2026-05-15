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
from shallweswim.api import initialize_location_data, register_routes
from shallweswim.api_types import FeedStatus, LocationConditions, LocationStatus
from shallweswim.config import (
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    LocationConfig,
)
from shallweswim.core.feeds import FEED_TIDES
from shallweswim.core.queries import DataUnavailableError
from shallweswim.data import LocationDataManager
from tests.helpers import assert_json_serializable, create_test_app


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI application for testing."""
    app_instance = create_test_app()
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

    def nyc_has_feed(feed_name: str) -> bool:
        return feed_name in nyc_data._feeds

    def nyc_get_feed_values(feed_name: str) -> pd.DataFrame:
        if feed_name not in nyc_data._feeds:
            raise KeyError(feed_name)
        feed = nyc_data._feeds[feed_name]
        if feed is None:
            raise DataUnavailableError(f"Feed '{feed_name}' data not available")
        return feed.values

    nyc_data.has_feed.side_effect = nyc_has_feed
    nyc_data.get_feed_values.side_effect = nyc_get_feed_values
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
    sf_data._feeds = {}
    sf_data.has_feed.side_effect = lambda feed_name: feed_name in sf_data._feeds
    sf_data.get_feed_values.side_effect = lambda feed_name: (
        sf_data._feeds[feed_name].values
    )
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


def test_app_bootstrap_endpoint(test_client: TestClient) -> None:
    """Frontend bootstrap endpoint returns presentation metadata."""
    response = test_client.get("/api/app/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Shall We Swim"
    assert data["short_name"] == "Swim"
    assert data["default_location_code"] == "nyc"
    assert "nyc" in data["location_order"]
    assert data["manifest"]["start_url"] == "/app"

    nyc = data["locations"]["nyc"]
    assert nyc["metadata"]["code"] == "nyc"
    assert nyc["metadata"]["features"]["temperature"] is True
    assert nyc["metadata"]["features"]["webcam"] is True
    assert data["source_code_link"]["url"].endswith("/shallweswim")
    assert nyc["integrations"]["youtube_live"]["channel_id"]
    assert nyc["integrations"]["webcam_alternative"]["label"] == (
        "Earth Cam Coney Island"
    )
    assert nyc["integrations"]["webcam_source"]["description"] == (
        "thanks to David K and Karol L"
    )
    assert nyc["integrations"]["transit_source"]["url"] == "https://goodservice.io"
    assert nyc["integrations"]["transit_routes"][0]["goodservice_route_id"] == "B"


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
    mock_tide_state = sw_types.TideState(
        timestamp=mock_dt,
        estimated_height=0.8,
        units="ft",
        trend=sw_types.TideTrend.RISING,
        height_pct=0.6,
    )
    mock_current_info = sw_types.CurrentInfo(
        timestamp=mock_dt,  # Use datetime here as CurrentInfo expects it
        magnitude=0.753,  # Test rounding
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.65,
        direction=sw_types.CurrentDirection.FLOODING,
        phase=sw_types.CurrentPhase.FLOOD,
        strength=sw_types.CurrentStrength.MODERATE,
        trend=sw_types.CurrentTrend.BUILDING,
        state_description="moderate flood and building",
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
    mock_manager.predict_tide_at_time.return_value = mock_tide_state

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
    assert "state" in data["tides"]
    assert isinstance(data["tides"]["past"], list)
    assert len(data["tides"]["past"]) == 1
    assert isinstance(data["tides"]["next"], list)
    assert len(data["tides"]["next"]) == 2
    assert data["tides"]["state"] is not None
    assert data["tides"]["state"]["timestamp"] == mock_tide_state.timestamp.isoformat()
    assert (
        data["tides"]["state"]["estimated_height"] == mock_tide_state.estimated_height
    )
    assert data["tides"]["state"]["units"] == "ft"
    assert data["tides"]["state"]["trend"] == "rising"
    assert data["tides"]["state"]["height_pct"] == mock_tide_state.height_pct

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
    assert mock_current_info.phase is not None  # Help mypy
    assert data["current"]["phase"] == mock_current_info.phase.value
    assert mock_current_info.strength is not None  # Help mypy
    assert data["current"]["strength"] == mock_current_info.strength.value
    assert mock_current_info.trend is not None  # Help mypy
    assert data["current"]["trend"] == mock_current_info.trend.value
    assert data["current"]["state_description"] == mock_current_info.state_description


def test_openapi_documents_condition_state_enums(test_client: TestClient) -> None:
    """OpenAPI schema exposes compact current and tide state enum values."""
    response = test_client.get("/openapi.json")

    assert response.status_code == status.HTTP_200_OK
    schemas = response.json()["components"]["schemas"]

    assert schemas["CurrentPhase"]["enum"] == [
        "flood",
        "ebb",
        "slack_before_flood",
        "slack_before_ebb",
        "slack",
    ]
    assert schemas["CurrentStrength"]["enum"] == ["light", "moderate", "strong"]
    assert schemas["CurrentTrend"]["enum"] == ["building", "easing", "steady"]
    assert schemas["TideTrend"]["enum"] == ["rising", "falling", "steady"]
    assert "phase" in schemas["CurrentInfo"]["properties"]
    assert "strength" in schemas["CurrentInfo"]["properties"]
    assert "trend" in schemas["CurrentInfo"]["properties"]
    assert "state" in schemas["TideInfo"]["properties"]
    tide_state_timestamp = schemas["TideState"]["properties"]["timestamp"]
    assert tide_state_timestamp["type"] == "string"
    assert tide_state_timestamp["format"] == "date-time"


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


@pytest.mark.asyncio
async def test_initialize_location_data_missing_http_session() -> None:
    """Startup initialization fails explicitly when HTTP session state is missing."""
    app = create_test_app()
    app.state.process_pool = MagicMock()

    with pytest.raises(RuntimeError, match="HTTP session not found"):
        await initialize_location_data(["nyc"], app)


@pytest.mark.asyncio
async def test_initialize_location_data_missing_process_pool() -> None:
    """Startup initialization fails explicitly when process pool state is missing."""
    app = create_test_app()
    app.state.http_session = MagicMock()

    with pytest.raises(RuntimeError, match="Process pool not found"):
        await initialize_location_data(["nyc"], app)


@pytest.mark.asyncio
async def test_initialize_location_data_unknown_location() -> None:
    """Startup initialization reports invalid location codes without assert."""
    app = create_test_app()
    app.state.http_session = MagicMock()
    app.state.process_pool = MagicMock()

    with patch("shallweswim.config.get") as mock_get:
        mock_get.return_value = None
        with pytest.raises(ValueError, match="Config for location 'bad' not found"):
            await initialize_location_data(["bad"], app)


def test_get_feed_data_success(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Debug feed data endpoint returns raw feed cache data."""
    # Define mock feed data
    mock_feed_data = pd.DataFrame({"value": [1, 2, 3]})
    mock_feed_name = "MockFeed"

    # Get the mock data manager
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    mock_data_managers_dict = test_client.app.state.data_managers

    mock_data_managers_dict["nyc"]._feeds[FEED_TIDES] = MagicMock(
        values=mock_feed_data, name=mock_feed_name
    )

    # Call the API endpoint
    response = test_client.get(f"/api/nyc/data/{FEED_TIDES}")

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


def test_feed_data_endpoint_excluded_from_openapi(test_client: TestClient) -> None:
    """Raw feed data is a debug endpoint, not public OpenAPI surface."""
    response = test_client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK

    paths = response.json()["paths"]
    assert "/api/{loc}/data/{feed_name}" not in paths


def test_get_feed_data_location_not_found(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test the feed data endpoint with an invalid location."""
    response = test_client.get("/api/invalid_loc/data/testfeed")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Location 'invalid_loc' not found" in response.json()["detail"]


def test_get_feed_data_configured_location_missing_manager_returns_500() -> None:
    """Configured location with no data manager is an internal init error."""
    app = create_test_app()
    app.state.data_managers = {}
    register_routes(app)

    config = LocationConfig(
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

    with patch("shallweswim.config.get") as mock_get:
        mock_get.return_value = config
        client = TestClient(app)
        response = client.get("/api/nyc/data/tides")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "data manager missing" in response.json()["detail"]


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


def test_currents_endpoint_no_currents_source_returns_404(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that currents endpoint returns 404 for locations without currents_source.

    Locations without currents_source configured get 404 "does not support".
    Returns 404 (not 501) to avoid triggering 5xx alerting.

    See also: test_mocked_stack.py::test_currents_endpoint_observation_source_returns_404
    which tests a location WITH currents_source configured but OBSERVATION type.
    """
    # sfo doesn't have currents_source configured, so it gets 404
    response = test_client.get("/api/sfo/currents")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "does not support" in response.json()["detail"]


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


# =============================================================================
# DataUnavailableError → HTTP 503 tests
# =============================================================================


def test_currents_endpoint_returns_503_when_data_unavailable(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that currents endpoint returns 503 when feed data is unavailable.

    This tests the DataUnavailableError → HTTP 503 conversion in the API layer.
    When a station has no data (e.g., StationUnavailableError during fetch),
    querying for data raises DataUnavailableError, which should return 503.
    """
    from shallweswim.core.queries import DataUnavailableError

    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]

    # Mock predict_flow_at_time to raise DataUnavailableError
    mock_manager.predict_flow_at_time.side_effect = DataUnavailableError(
        "Feed 'currents' data not available"
    )

    response = test_client.get("/api/nyc/currents")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    error_data = response.json()
    assert "detail" in error_data
    assert "currents" in error_data["detail"].lower()
    assert "not available" in error_data["detail"].lower()


def test_currents_endpoint_returns_503_when_chart_data_unavailable(
    app: FastAPI, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that currents endpoint returns 503 when chart data is unavailable.

    Even if current prediction succeeds, get_chart_info() can fail if tide
    data is missing. This should also return 503.

    This test requires has_static_charts=True to trigger the chart code path.
    """
    from shallweswim.core.queries import DataUnavailableError

    # Create a config with has_static_charts=True
    nyc_config_with_charts = LocationConfig(
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
            stations=["NYH1914"], name="Narrows North", has_static_charts=True
        ),
        enabled=True,
    )

    # Patch config.get to return our custom config
    with patch("shallweswim.config.get") as mock_get:
        mock_get.return_value = nyc_config_with_charts

        mock_manager = app.state.data_managers["nyc"]

        # Mock predict_flow_at_time to succeed
        mock_dt = datetime.datetime(2025, 5, 4, 12, 0, 0)
        mock_manager.predict_flow_at_time.return_value = sw_types.CurrentInfo(
            timestamp=mock_dt,
            magnitude=0.5,
            source_type=sw_types.DataSourceType.PREDICTION,
            magnitude_pct=0.5,
            direction=sw_types.CurrentDirection.FLOODING,
            phase=sw_types.CurrentPhase.FLOOD,
            strength=sw_types.CurrentStrength.MODERATE,
            trend=sw_types.CurrentTrend.STEADY,
            state_description="moderate flood and steady",
        )

        # Mock get_chart_info to raise DataUnavailableError
        mock_manager.get_chart_info.side_effect = DataUnavailableError(
            "Feed 'tides' data not available"
        )

        test_client = TestClient(app)
        response = test_client.get("/api/nyc/currents")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        error_data = response.json()
        assert "detail" in error_data
        assert "tides" in error_data["detail"].lower()
        assert "not available" in error_data["detail"].lower()


# =============================================================================
# /api/locations endpoint tests
# =============================================================================


def test_list_locations_endpoint(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that /api/locations returns all configured locations with summary info."""
    from shallweswim import config as config_lib
    from shallweswim.api_types import LocationSummary

    response = test_client.get("/api/locations")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert_json_serializable(data)

    # Should be a list with all configured locations
    assert isinstance(data, list)
    assert len(data) == len(config_lib.CONFIGS)

    # Validate each location with Pydantic
    for loc_data in data:
        try:
            LocationSummary.model_validate(loc_data)
        except Exception as e:
            pytest.fail(f"Location failed Pydantic validation: {e}\nData: {loc_data}")

    # Find NYC and verify fields match config
    nyc_data = next((loc for loc in data if loc["code"] == "nyc"), None)
    assert nyc_data is not None
    nyc_config = config_lib.CONFIGS["nyc"]
    assert nyc_data["name"] == nyc_config.name
    assert nyc_data["swim_location"] == nyc_config.swim_location
    assert nyc_data["latitude"] == nyc_config.latitude
    assert nyc_data["longitude"] == nyc_config.longitude
    # has_data should be True for nyc since it's in mock_data_managers with data
    assert nyc_data["has_data"] is True


# =============================================================================
# NaN HANDLING TESTS
# =============================================================================


def test_conditions_endpoint_handles_nan_in_current_data(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Test that /api/{location}/conditions handles NaN values from data layer.

    This regression test ensures that numpy.nan values from pandas operations
    don't cause JSON serialization errors. The bug manifested as:
    ValueError: Out of range float values are not JSON compliant: nan

    FastAPI 0.130+ uses Pydantic's native JSON serialization which converts
    NaN to null automatically when response_model is set.
    """
    import numpy as np

    mock_manager = test_client.app.state.data_managers["nyc"]
    mock_dt = datetime.datetime(2026, 3, 12, 7, 9, 44)

    # Mock temperature (required for the endpoint to work)
    mock_manager.get_current_temperature.return_value = sw_types.TemperatureReading(
        timestamp=mock_dt, temperature=65.0
    )

    # Mock tides
    mock_manager.get_current_tide_info.return_value = sw_types.TideInfo(
        past=[
            sw_types.TideEntry(
                time=mock_dt - datetime.timedelta(hours=6),
                type=sw_types.TideCategory.LOW,
                prediction=-0.5,
            )
        ],
        next=[
            sw_types.TideEntry(
                time=mock_dt + datetime.timedelta(hours=6),
                type=sw_types.TideCategory.HIGH,
                prediction=1.2,
            )
        ],
    )
    mock_manager.predict_tide_at_time.return_value = None

    # Mock current with NaN magnitude_pct (the actual bug scenario)
    current_info_with_nan = sw_types.CurrentInfo(
        timestamp=mock_dt,
        source_type=sw_types.DataSourceType.PREDICTION,
        direction=sw_types.CurrentDirection.FLOODING,
        magnitude=1.5,
        magnitude_pct=np.nan,  # This caused the original 500 error
        phase=sw_types.CurrentPhase.FLOOD,
        strength=sw_types.CurrentStrength.STRONG,
        trend=sw_types.CurrentTrend.BUILDING,
        state_description="strong flood and building",
    )
    mock_manager.predict_flow_at_time.return_value = current_info_with_nan

    # Call endpoint
    response = test_client.get("/api/nyc/conditions")

    # Should succeed (not 500)
    assert response.status_code == status.HTTP_200_OK

    # Should be valid JSON
    data = response.json()
    assert_json_serializable(data)

    # NaN should be serialized as null
    assert data["current"]["magnitude_pct"] is None
    assert data["current"]["magnitude"] == 1.5

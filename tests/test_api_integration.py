"""Integration tests for the ShallWeSwim API endpoints.

These tests use FastAPI's TestClient to test only the API endpoints with real data.
They verify that the API endpoints return correctly structured data from the
real NOAA endpoints for the configured locations, without testing any HTML or UI components.

Run with: poetry run pytest tests/test_api_integration.py -v --run-integration
"""

# Standard library imports

# Third-party imports
import dateutil.parser
import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Local imports
import fastapi
from shallweswim import config, api

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

# Get all available locations from the config
TEST_LOCATIONS = list(config.CONFIGS.keys())


@pytest_asyncio.fixture(scope="module")
async def api_client() -> TestClient:
    """Create a test client for API testing.

    This creates a dedicated FastAPI app with only the API routes registered.
    """

    # Create a dedicated FastAPI app for API testing only
    app = fastapi.FastAPI(title="ShallWeSwim API Test App")

    # Initialize app.state.data_managers
    app.state.data_managers = {}

    # Initialize data for all test locations
    # We set wait_for_data=True to ensure data is loaded before tests run
    await api.initialize_location_data(
        location_codes=TEST_LOCATIONS,
        app=app,  # Pass the app instance
        wait_for_data=True,  # Wait for data to load before running tests
        timeout=30.0,  # Maximum time to wait for data to be ready
    )

    # Register only the API routes
    api.register_routes(app)

    # Create a test client for the API-only app
    client = TestClient(app)

    return client


def validate_conditions_response(response: httpx.Response, location_code: str) -> None:
    """Validate the response from the conditions API endpoint.

    This function dynamically validates the response based on the location's configuration.
    """
    # Get location config to determine what should be present
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    # Parse JSON response
    data = response.json()

    # Validate top-level structure
    assert "location" in data, "Missing location data"

    # Validate temperature data if the location has a temperature source with live_enabled=True
    has_temp_source = (
        hasattr(location_config, "temp_source")
        and location_config.temp_source is not None
    )

    # Only check live_enabled if temp_source exists
    has_live_temp = False
    if has_temp_source and location_config.temp_source is not None:
        has_live_temp = (
            hasattr(location_config.temp_source, "live_enabled")
            and location_config.temp_source.live_enabled is True
        )

    if has_live_temp:
        assert (
            "temperature" in data
        ), f"Missing temperature data for {location_code} which has live_enabled=True"
        temp = data["temperature"]
        if temp is not None:
            assert "timestamp" in temp, "Missing temperature timestamp"
            assert "water_temp" in temp, "Missing water temperature value"
            assert "units" in temp, "Missing temperature units"
            assert isinstance(
                temp["water_temp"], (int, float)
            ), "Water temperature is not a number"
    elif "temperature" in data:
        # If temperature data is present even though live_enabled is False, validate it
        temp = data["temperature"]
        if temp is not None:
            assert "timestamp" in temp, "Missing temperature timestamp"
            assert "water_temp" in temp, "Missing water temperature value"
            assert "units" in temp, "Missing temperature units"
            assert isinstance(
                temp["water_temp"], (int, float)
            ), "Water temperature is not a number"

    # Validate tides data if the location has a tide source
    has_tide_source = (
        hasattr(location_config, "tide_source")
        and location_config.tide_source is not None
    )
    if has_tide_source:
        assert "tides" in data, "Missing tides data"
        tides = data["tides"]
        if tides is not None:
            assert "past" in tides, "Missing past tides data"
            assert "next" in tides, "Missing next tides data"
            assert len(tides["past"]) > 0, "No past tides data"
            assert len(tides["next"]) > 0, "No next tides data"

        # Validate tide entry structure
        past_tide = tides["past"][0]
        assert "time" in past_tide, "Missing tide time"
        assert "type" in past_tide, "Missing tide type"
        assert "prediction" in past_tide, "Missing tide prediction"
        assert past_tide["type"] in [
            "high",
            "low",
            "unknown",
        ], f"Invalid tide type: {past_tide['type']}"

    # Validate location details
    location = data["location"]
    assert (
        location["code"] == location_code
    ), f"Expected location code {location_code}, got {location['code']}"
    assert "name" in location, "Missing location name"
    assert "swim_location" in location, "Missing swim location"

    # Additional validation for tides
    if has_tide_source and "tides" in data:
        # Get location config for timezone-aware comparisons
        assert location_config is not None, f"Config for {location_code} not found"

        # Get the current time in the location's timezone as a naive datetime
        # This matches how the NOAA API data is structured (local time, naive datetime)
        local_now = location_config.local_now()

        # Validate that past tides are actually in the past
        for tide in tides["past"]:
            tide_time = dateutil.parser.isoparse(tide["time"])
            # Ensure the tide_time is naive for comparison
            if tide_time.tzinfo is not None:
                tide_time = tide_time.replace(tzinfo=None)
            assert (
                tide_time <= local_now
            ), f"Past tide time {tide_time} is not in the past compared to local time {local_now}"

        # Validate that future tides are actually in the future
        for tide in tides["next"]:
            tide_time = dateutil.parser.isoparse(tide["time"])
            # Ensure the tide_time is naive for comparison
            if tide_time.tzinfo is not None:
                tide_time = tide_time.replace(tzinfo=None)
            assert (
                tide_time >= local_now
            ), f"Next tide time {tide_time} is not in the future compared to local time {local_now}"

        # Validate that all tide times are distinct
        all_tide_times = [tide["time"] for tide in tides["past"] + tides["next"]]
        assert len(all_tide_times) == len(
            set(all_tide_times)
        ), "Tide times are not unique"

    # Get location config for additional checks
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"


@pytest.mark.integration
@pytest.mark.parametrize("location_code", TEST_LOCATIONS)
def test_conditions_api(api_client: TestClient, location_code: str) -> None:
    """Test the conditions API endpoint for all configured locations.

    This test dynamically tests all locations in the configuration.
    """
    # Get location config to determine what should be present
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    response = api_client.get(f"/api/{location_code}/conditions")
    validate_conditions_response(response, location_code)

    # Verify location-specific details
    data = response.json()
    assert location_config.name in data["location"]["name"]
    assert location_config.swim_location in data["location"]["swim_location"]

    # Verify temperature data if the location has a temperature source with live_enabled=True
    has_temp_source = (
        hasattr(location_config, "temp_source")
        and location_config.temp_source is not None
    )

    # Only check live_enabled if temp_source exists
    has_live_temp = False
    if has_temp_source and location_config.temp_source is not None:
        has_live_temp = (
            hasattr(location_config.temp_source, "live_enabled")
            and location_config.temp_source.live_enabled is True
        )

    if has_live_temp:
        assert (
            "temperature" in data
        ), f"Temperature data missing for {location_code} which has live_enabled=True"
        assert (
            "water_temp" in data["temperature"]
        ), f"Water temperature missing for {location_code}"
    else:
        # If live_enabled is False, the API might still return temperature data if available,
        # but we shouldn't require it in our tests
        pass

    # Verify tides data if the location has a tide source
    has_tide_source = (
        hasattr(location_config, "tide_source")
        and location_config.tide_source is not None
    )
    if has_tide_source:
        assert "tides" in data
        assert data["tides"] is not None
    else:
        # Field might be absent or set to None
        if "tides" in data:
            assert (
                data["tides"] is None
            ), f"Expected tides to be None for {location_code} which has no tide source"


def validate_currents_response(response: httpx.Response, location_code: str) -> None:
    """Validate the response from the currents API endpoint.

    This function dynamically validates the response based on the location's configuration.
    """
    # Get location config to determine what should be present
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    # Check if this location supports currents
    has_currents_source = (
        hasattr(location_config, "currents_source")
        and location_config.currents_source is not None
    )

    # If the location doesn't support currents, we expect a 404 or 501 response
    if not has_currents_source:
        assert response.status_code in [
            404,
            501,
        ], f"Expected 404 or 501 for location without currents, got {response.status_code}"
        return

    # For locations with currents, validate the response
    assert (
        response.status_code == 200
    ), f"Expected 200 for location with currents, got {response.status_code}"
    assert "application/json" in response.headers["content-type"]

    # Parse JSON response
    data = response.json()

    # Validate top-level structure
    assert "location" in data, "Missing location data"
    assert "timestamp" in data, "Missing timestamp"
    assert "current" in data, "Missing current data"
    assert "legacy_chart" in data, "Missing legacy chart data"
    assert "current_chart_filename" in data, "Missing current chart filename"
    assert "navigation" in data, "Missing navigation data"

    # Validate location details
    location = data["location"]
    assert (
        location["code"] == location_code
    ), f"Expected location code {location_code}, got {location['code']}"
    assert "name" in location, "Missing location name"
    assert "swim_location" in location, "Missing swim location"

    # Validate current data
    current = data["current"]
    assert "timestamp" in current, "Missing current timestamp"
    assert "direction" in current, "Missing current direction"
    assert "magnitude" in current, "Missing current magnitude"
    assert "magnitude_pct" in current, "Missing current magnitude percentage"
    assert "state_description" in current, "Missing current state description"
    assert current["direction"] in [
        "flooding",
        "ebbing",
        "slack",
    ], f"Invalid current direction: {current['direction']}"
    assert isinstance(
        current["magnitude"], (int, float)
    ), "Current magnitude is not a number"
    assert isinstance(
        current["magnitude_pct"], (int, float)
    ), "Current magnitude percentage is not a number"

    # Validate legacy chart data
    chart = data["legacy_chart"]
    assert "hours_since_last_tide" in chart, "Missing hours since last tide"
    assert "last_tide_type" in chart, "Missing last tide type"
    assert "chart_filename" in chart, "Missing chart filename"
    assert "map_title" in chart, "Missing map title"
    assert chart["last_tide_type"] in [
        "high",
        "low",
        "unknown",
    ], f"Invalid last tide type: {chart['last_tide_type']}"

    # Validate navigation data
    nav = data["navigation"]
    assert "shift" in nav, "Missing shift value"
    assert "next_hour" in nav, "Missing next hour value"
    assert "prev_hour" in nav, "Missing previous hour value"
    assert "current_api_url" in nav, "Missing current API URL"
    assert "plot_url" in nav, "Missing plot URL"


@pytest.mark.integration
@pytest.mark.parametrize("location_code", TEST_LOCATIONS)
def test_currents_api(api_client: TestClient, location_code: str) -> None:
    """Test the currents API endpoint for all configured locations.

    This test dynamically tests all locations in the configuration.
    """
    # Get location config to determine what should be present
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    # Test the basic currents endpoint
    response = api_client.get(f"/api/{location_code}/currents")
    validate_currents_response(response, location_code)

    # Only continue with additional tests if the location supports currents
    has_currents_source = (
        hasattr(location_config, "currents_source")
        and location_config.currents_source is not None
    )
    if not has_currents_source:
        return

    # For locations with currents, verify the response details
    if response.status_code == 200:
        data = response.json()
        assert location_config.name in data["location"]["name"]
        assert location_config.swim_location in data["location"]["swim_location"]

        # Test with a shift parameter
        shift_response = api_client.get(f"/api/{location_code}/currents?shift=60")
        validate_currents_response(shift_response, location_code)
        if shift_response.status_code == 200:
            shift_data = shift_response.json()
            assert shift_data["navigation"]["shift"] == 60


@pytest.mark.integration
def test_invalid_api_location(api_client: TestClient) -> None:
    """Test that API requests for invalid locations return 404 errors."""
    # Test conditions endpoint
    response = api_client.get("/api/invalid_location/conditions")
    assert response.status_code == 404
    error_data = response.json()
    assert "detail" in error_data
    assert "not found" in error_data["detail"].lower()

    # Test currents endpoint
    response = api_client.get("/api/invalid_location/currents")
    assert response.status_code == 404
    error_data = response.json()
    assert "detail" in error_data
    assert "not found" in error_data["detail"].lower()


@pytest.mark.integration
def test_ready_endpoint(api_client: TestClient) -> None:
    """Test the ready API endpoint returns a boolean response.

    Since the data is loaded in the api_client fixture with wait_for_data=True,
    we expect the ready endpoint to return True.
    """
    response = api_client.get("/api/ready")

    # Verify response status code
    assert response.status_code == 200

    # Verify response is a boolean
    ready_status = response.json()
    assert isinstance(ready_status, bool)

    # Since the fixture waits for data to be ready, we expect True
    assert ready_status is True

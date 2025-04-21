"""Integration tests for the ShallWeSwim API endpoints.

These tests use FastAPI's TestClient to test the API endpoints with real data.
They verify that the API endpoints return correctly structured data from the
real NOAA endpoints for the configured locations.

Run with: poetry run pytest tests/test_api_integration.py -v --run-integration
"""

# Standard library imports
import datetime
from typing import Any

# Third-party imports
import dateutil.parser
import httpx
import pytest
from fastapi.testclient import TestClient

# Local imports
from shallweswim import config, main, api

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

# Test locations
NYC_LOCATION = "nyc"
SAN_LOCATION = "san"

# List of all locations to test
TEST_LOCATIONS = [NYC_LOCATION, SAN_LOCATION]


@pytest.fixture(scope="module")
def api_client(_: Any) -> TestClient:
    """
    Create a FastAPI test client that initializes data for all test locations.
    This directly mirrors what the lifespan function does in the main application,
    but waits for data to load before returning.
    """

    # Clear existing data to ensure a clean state
    api.data.clear()

    # Initialize data for all test locations
    # We set wait_for_data=True to ensure data is loaded before tests run
    api.initialize_location_data(
        location_codes=TEST_LOCATIONS,
        data_dict=api.data,
        wait_for_data=True,  # Wait for data to load before running tests
        max_wait_retries=45,
        retry_interval=1,
    )

    # Create a test client
    client = TestClient(main.app)

    # Let the test client use the initialized data
    return client


def validate_conditions_response(
    response: httpx.Response, location_code: str = NYC_LOCATION
) -> None:
    """Validate the response from the conditions API endpoint."""
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    # Parse JSON response
    data = response.json()

    # Validate top-level structure
    assert "location" in data, "Missing location data"
    assert "temperature" in data, "Missing temperature data"
    assert "tides" in data, "Missing tides data"

    # Validate location details
    location = data["location"]
    assert (
        location["code"] == location_code
    ), f"Expected location code {location_code}, got {location['code']}"
    assert "name" in location, "Missing location name"
    assert "swim_location" in location, "Missing swim location"

    # Validate temperature data
    temp = data["temperature"]
    assert "timestamp" in temp, "Missing temperature timestamp"
    assert "water_temp" in temp, "Missing water temperature value"
    assert "units" in temp, "Missing temperature units"
    assert isinstance(
        temp["water_temp"], (int, float)
    ), "Water temperature is not a number"

    # Validate tides data
    tides = data["tides"]
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

    # Validate that past tides are actually in the past
    now = datetime.datetime.now(datetime.timezone.utc)
    for tide in tides["past"]:
        tide_time = dateutil.parser.isoparse(tide["time"])
        assert (
            tide_time <= now
        ), f"Past tide time {tide_time} is not in the past compared to {now}"

    # Validate that future tides are actually in the future
    for tide in tides["next"]:
        tide_time = dateutil.parser.isoparse(tide["time"])
        assert (
            tide_time >= now
        ), f"Next tide time {tide_time} is not in the future compared to {now}"

    # Validate that all tide times are distinct
    all_tide_times = [tide["time"] for tide in tides["past"] + tides["next"]]
    assert len(all_tide_times) == len(set(all_tide_times)), "Tide times are not unique"

    # Get location config for additional checks
    location_config = config.Get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"


@pytest.mark.integration
def test_root_redirect(api_client: TestClient) -> None:
    """Test that the root path redirects to the default location."""
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code == 307  # Temporary redirect
    assert response.headers["location"] == "/nyc"


@pytest.mark.integration
def test_nyc_location(api_client: TestClient) -> None:
    """Test that the NYC location HTML endpoint returns a valid response."""
    response = api_client.get(f"/{NYC_LOCATION}")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Verify that the response contains HTML
    response_text = response.text
    assert "<html" in response_text.lower()
    assert "</html>" in response_text.lower()

    # Check for expected content elements specific to NYC
    assert "Battery" in response_text  # NYC Battery station should be mentioned


@pytest.mark.integration
def test_san_diego_location(api_client: TestClient) -> None:
    """Test that the San Diego location HTML endpoint returns a valid response."""
    response = api_client.get(f"/{SAN_LOCATION}")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Verify that the response contains HTML
    response_text = response.text
    assert "<html" in response_text.lower()
    assert "</html>" in response_text.lower()

    # Check for expected content elements specific to San Diego
    assert "La Jolla" in response_text
    assert "9410230" in response_text  # San Diego station ID should be mentioned


@pytest.mark.integration
def test_invalid_location(api_client: TestClient) -> None:
    """Test that requesting an invalid location returns a 404."""
    response = api_client.get("/invalid_location")
    assert response.status_code == 404
    assert "Bad location" in response.text


def validate_currents_response(
    response: httpx.Response, location_code: str = NYC_LOCATION
) -> None:
    """Validate the response from the currents API endpoint."""
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    # Parse JSON response
    data = response.json()

    # Validate top-level structure
    assert "location" in data, "Missing location data"
    assert "timestamp" in data, "Missing timestamp data"
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
def test_conditions_api_nyc(api_client: TestClient) -> None:
    """Test the conditions API endpoint for NYC location."""
    response = api_client.get(f"/api/{NYC_LOCATION}/conditions")
    validate_conditions_response(response, NYC_LOCATION)

    # Additional NYC-specific checks
    data = response.json()
    assert "Grimaldo's Chair" in data["location"]["swim_location"]
    assert "New York" in data["location"]["name"]


@pytest.mark.integration
def test_conditions_api_san_diego(api_client: TestClient) -> None:
    """Test the conditions API endpoint for San Diego location."""
    response = api_client.get(f"/api/{SAN_LOCATION}/conditions")
    validate_conditions_response(response, SAN_LOCATION)

    # Additional San Diego-specific checks
    data = response.json()
    assert "La Jolla" in data["location"]["swim_location"]
    assert "San Diego" in data["location"]["name"]


@pytest.mark.integration
def test_currents_api_nyc(api_client: TestClient) -> None:
    """Test the currents API endpoint for NYC location."""
    response = api_client.get(f"/api/{NYC_LOCATION}/currents")
    validate_currents_response(response, NYC_LOCATION)

    # Additional NYC-specific checks
    data = response.json()
    assert "Grimaldo's Chair" in data["location"]["swim_location"]
    assert "New York" in data["location"]["name"]

    # Test with a shift parameter
    response = api_client.get(f"/api/{NYC_LOCATION}/currents?shift=60")
    validate_currents_response(response, NYC_LOCATION)
    data = response.json()
    assert data["navigation"]["shift"] == 60


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

"""Integration tests for the main FastAPI application.

These tests use FastAPI's TestClient to test the API endpoints with real data.
They verify that configured locations (currently NYC and San Diego) can be
accessed without errors and that the API returns real data from NOAA endpoints.

Run with: poetry run pytest tests/test_api_integration.py -v --run-integration
"""

# Standard library imports
from typing import Any

# Third-party imports
import httpx
import pytest
from fastapi.testclient import TestClient

# Local imports
from shallweswim import config, main

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

# Test locations
NYC_LOCATION = "nyc"
SAN_LOCATION = "san"

# List of all locations to test
TEST_LOCATIONS = [NYC_LOCATION, SAN_LOCATION]


@pytest.fixture(scope="module")
def api_client(check_api_availability: Any) -> TestClient:
    """
    Create a FastAPI test client that initializes data for all test locations.
    This directly mirrors what the lifespan function does in the main application,
    but waits for data to load before returning.
    """

    # Clear existing data to ensure a clean state
    main.data.clear()

    # Initialize data for all test locations
    # We set wait_for_data=True to ensure data is loaded before tests run
    main.initialize_location_data(
        location_codes=TEST_LOCATIONS,
        data_dict=main.data,
        wait_for_data=True,  # Wait for data to load before running tests
        max_wait_retries=45,
        retry_interval=1,
    )

    # Create a test client
    client = TestClient(main.app)

    # Let the test client use the initialized data
    return client


def validate_location_response(
    response: httpx.Response, location_code: str = NYC_LOCATION
) -> None:
    """Validate the response from a location endpoint."""
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check for key elements in the response
    response_text = response.text
    assert "<html" in response_text.lower()
    assert "</html>" in response_text.lower()
    assert "<title>" in response_text.lower()

    # Verify that no exceptions occurred in the template
    assert "Internal Server Error" not in response_text
    assert "Exception:" not in response_text

    # Look for identifiable elements in the page
    assert "shall we swim today?" in response_text.lower()
    assert "water temperatures provided by" in response_text.lower()

    # Get location config and verify station information is present
    location_config = config.Get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    # Check that tide station information is in the response
    assert location_config.tide_station_name is not None, "Tide station name is None"
    assert location_config.tide_station is not None, "Tide station ID is None"
    assert (
        location_config.tide_station_name in response_text
    ), "Tide station name missing"
    assert str(location_config.tide_station) in response_text, "Tide station ID missing"

    # Check for NOAA reference
    assert "NOAA Tides and Currents" in response_text

    # If location has current predictions enabled, verify current-related content
    if location_config.current_predictions:
        assert (
            "currents" in response_text.lower() or "current" in response_text.lower()
        ), "No current data mentioned"


@pytest.mark.integration
def test_root_redirect(api_client: TestClient) -> None:
    """Test that the root path redirects to the default location."""
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code == 307  # Temporary redirect
    assert response.headers["location"] == "/nyc"


@pytest.mark.integration
def test_nyc_location(api_client: TestClient) -> None:
    """Test that the NYC location returns a valid response with real data."""
    response = api_client.get(f"/{NYC_LOCATION}")
    validate_location_response(response, NYC_LOCATION)

    # Additional checks for NYC-specific content
    response_text = response.text
    # Check for expected content elements specific to NYC
    assert "Coney Island" in response_text
    assert "Battery" in response_text  # NYC Battery station should be mentioned


@pytest.mark.integration
def test_san_diego_location(api_client: TestClient) -> None:
    """Test that the San Diego location returns a valid response with real data."""
    response = api_client.get(f"/{SAN_LOCATION}")
    validate_location_response(response, SAN_LOCATION)

    # Additional checks for San Diego-specific content
    response_text = response.text
    # Check for expected content elements specific to San Diego
    assert "La Jolla Cove" in response_text
    assert "9410230" in response_text  # San Diego station ID should be mentioned


@pytest.mark.integration
def test_invalid_location(api_client: TestClient) -> None:
    """Test that requesting an invalid location returns a 404."""
    response = api_client.get("/invalid_location")
    assert response.status_code == 404
    assert "Bad location" in response.text


@pytest.mark.integration
def test_freshness_endpoint(api_client: TestClient) -> None:
    """Test the freshness endpoint for data freshness information."""
    response = api_client.get("/freshness")
    assert response.status_code == 200

    # Check that the response is JSON
    data = response.json()

    # Verify the structure of the freshness data (real data)
    for dataset_name in ["tides_and_currents", "live_temps", "historic_temps"]:
        assert dataset_name in data, f"Missing dataset: {dataset_name}"
        dataset = data[dataset_name]

        # Check for expected fields in each dataset
        assert "fetch" in dataset
        assert "latest_value" in dataset

        # We only check for the presence of the keys, but don't check values
        # as they might be null during the initial data loading phase
        assert "time" in dataset["fetch"]
        assert "time" in dataset["latest_value"]

        # Check for age information which should be present
        assert "age" in dataset["fetch"]
        assert "age" in dataset["latest_value"]

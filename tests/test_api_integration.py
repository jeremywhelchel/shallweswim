# pylint: disable=duplicate-code
"""Integration tests for the ShallWeSwim API endpoints.

These tests use FastAPI's TestClient to test only the API endpoints with real data.
They verify that the API endpoints return correctly structured data from the
real NOAA endpoints for the configured locations, without testing any HTML or UI components.

Run with: poetry run pytest tests/test_api_integration.py -v --run-integration
"""

# Standard library imports
import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from typing import cast

# Third-party imports
import aiohttp
import dateutil.parser

# Local imports
import fastapi
import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from shallweswim import api, config
from shallweswim.types import CurrentDirection

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

# Get all available locations from the config
TEST_LOCATIONS = list(config.CONFIGS.keys())


@pytest_asyncio.fixture(scope="module")
async def api_client() -> AsyncGenerator[TestClient]:
    """Create a test client for API testing.

    This creates a dedicated FastAPI app with only the API routes registered.
    """

    # Create a dedicated FastAPI app for API testing only
    app = fastapi.FastAPI(title="ShallWeSwim API Test App")

    # Initialize app.state.data_managers
    app.state.data_managers = {}
    app.state.http_session = None  # Initialize state variable
    app.state.process_pool = None  # Initialize state variable

    # Create a process pool for CPU-bound tasks (e.g., plotting)
    pool = ProcessPoolExecutor()  # Create the pool
    app.state.process_pool = pool  # Assign to app state

    # Create and manage the HTTP session within the fixture's scope
    async with aiohttp.ClientSession() as session:
        app.state.http_session = session

        # Initialize data for all test locations
        # We set wait_for_data=False to allow individual locations to fail
        # without blocking other tests. Per-location readiness is checked in tests.
        await api.initialize_location_data(
            location_codes=TEST_LOCATIONS,
            app=app,  # Pass the app instance
            wait_for_data=False,  # Don't block - check readiness per-location in tests
        )

        # Wait for each location to become ready (with individual timeouts)
        # This allows some locations to fail without blocking others
        for location_code in TEST_LOCATIONS:
            data_manager = app.state.data_managers[location_code]
            try:
                await asyncio.wait_for(
                    data_manager.wait_until_ready(timeout=30.0),
                    timeout=35.0,  # Slightly longer outer timeout
                )
            except (TimeoutError, Exception):
                # Location failed to load - tests will skip/fail based on test_required
                pass

        # Register only the API routes
        api.register_routes(app)

        # Create a test client for the API-only app
        client = TestClient(app)

        # Yield the client to allow tests to run
        yield client

        # Clean up all data managers after tests are complete
        for data_manager in app.state.data_managers.values():
            await data_manager.stop()

        # Shut down the process pool
        pool.shutdown(wait=False)


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
        assert "temperature" in data, (
            f"Missing temperature data for {location_code} which has live_enabled=True"
        )
        temp = data["temperature"]
        if temp is not None:
            assert "timestamp" in temp, "Missing temperature timestamp"
            assert "water_temp" in temp, "Missing water temperature value"
            assert "units" in temp, "Missing temperature units"
            assert isinstance(temp["water_temp"], int | float), (
                "Water temperature is not a number"
            )
    elif "temperature" in data:
        # If temperature data is present even though live_enabled is False, validate it
        temp = data["temperature"]
        if temp is not None:
            assert "timestamp" in temp, "Missing temperature timestamp"
            assert "water_temp" in temp, "Missing water temperature value"
            assert "units" in temp, "Missing temperature units"
            assert isinstance(temp["water_temp"], int | float), (
                "Water temperature is not a number"
            )

            # Always verify the timestamp is recent (within the last 3 hours)
            # Parse the timestamp - should be naive
            temp_time = dateutil.parser.isoparse(temp["timestamp"])

            # Make sure it's naive for comparison
            if temp_time.tzinfo is not None:
                temp_time = temp_time.replace(tzinfo=None)

            # Get local now time (naive) from the location config
            local_now = location_config.local_now()

            # Calculate time difference
            time_diff = local_now - temp_time

            # Use a 3-hour window
            assert time_diff.total_seconds() <= 3 * 3600, (
                f"Temperature timestamp {temp_time} is more than 3 hours old "
                f"(current time: {local_now}, difference: {time_diff})"
            )

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
    assert location["code"] == location_code, (
        f"Expected location code {location_code}, got {location['code']}"
    )
    assert "name" in location, "Missing location name"
    assert "swim_location" in location, "Missing swim location"

    # Additional validation for tides
    if has_tide_source and "tides" in data:
        # Get location config for timezone-aware comparisons
        assert location_config is not None, f"Config for {location_code} not found"

        # Assign tides variable again to ensure it's bound in this scope
        tides = data["tides"]
        assert tides is not None, "Tides data is None"

        # Get the current time in the location's timezone as a naive datetime
        # This matches how the NOAA API data is structured (local time, naive datetime)
        local_now = location_config.local_now()

        # Validate that past tides are actually in the past
        for tide in tides["past"]:
            tide_time = dateutil.parser.isoparse(tide["time"])
            # Ensure the tide_time is naive for comparison
            if tide_time.tzinfo is not None:
                tide_time = tide_time.replace(tzinfo=None)
            assert tide_time <= local_now, (
                f"Past tide time {tide_time} is not in the past compared to local time {local_now}"
            )

        # Validate that future tides are actually in the future
        for tide in tides["next"]:
            tide_time = dateutil.parser.isoparse(tide["time"])
            # Ensure the tide_time is naive for comparison
            if tide_time.tzinfo is not None:
                tide_time = tide_time.replace(tzinfo=None)
            assert tide_time >= local_now, (
                f"Next tide time {tide_time} is not in the future compared to local time {local_now}"
            )

        # Validate that all tide times are distinct
        all_tide_times = [tide["time"] for tide in tides["past"] + tides["next"]]
        assert len(all_tide_times) == len(set(all_tide_times)), (
            "Tide times are not unique"
        )

    # === Validate current data ===
    has_current_source = (
        hasattr(location_config, "currents_source")
        and location_config.currents_source is not None
    )
    assert "current" in data, "Missing current data key"
    if has_current_source:
        current_data = data["current"]
        assert current_data is not None, "Current data should be present"
        # Validate CurrentInfo structure
        assert "timestamp" in current_data
        assert "magnitude" in current_data
        assert "source_type" in current_data
        assert isinstance(current_data["timestamp"], str)
        assert isinstance(current_data["magnitude"], int | float)
        assert isinstance(current_data["source_type"], str)
        # Optional fields can be None or the expected type
        assert isinstance(current_data.get("direction"), str | type(None)), (
            "Direction is not str or None"
        )
        if current_data.get("direction") is not None:
            # Check if it's a valid enum value if not None
            # We need to compare string values since the API returns strings
            valid_directions = [direction.value for direction in CurrentDirection]
            assert current_data["direction"] in valid_directions, (
                f"Invalid current direction: {current_data['direction']}"
            )
        assert isinstance(
            current_data.get("magnitude_pct"), int | float | type(None)
        ), "Magnitude pct is not number or None"
        assert isinstance(current_data.get("state_description"), str | type(None)), (
            "State description is not str or None"
        )
    else:
        assert data["current"] is None, "Current data should be null when no source"

    # TODO: Add validation for other fields if needed


@pytest.mark.integration
@pytest.mark.parametrize("location_code", TEST_LOCATIONS)
def test_conditions_api(api_client: TestClient, location_code: str) -> None:
    """Test the conditions API endpoint for all configured locations.

    This test dynamically tests all locations in the configuration.
    Locations with test_required=True must pass; others can skip on data unavailability.
    """
    # Get location config to determine what should be present
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    # Check if data is ready for this location
    app = cast(fastapi.FastAPI, api_client.app)
    data_manager = app.state.data_managers[location_code]
    if not data_manager.ready:
        if location_config.test_required:
            pytest.fail(
                f"Required location {location_code} has no data available (station outage?)"
            )
        else:
            pytest.skip(
                f"Location {location_code} data unavailable - skipping (station outage?)"
            )

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
        assert "temperature" in data, (
            f"Temperature data missing for {location_code} which has live_enabled=True"
        )
        assert "water_temp" in data["temperature"], (
            f"Water temperature missing for {location_code}"
        )
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
            assert data["tides"] is None, (
                f"Expected tides to be None for {location_code} which has no tide source"
            )


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

    # If the location doesn't support currents, we expect a 404 response
    if not has_currents_source:
        assert response.status_code == 404, (
            f"Expected 404 for location without currents, got {response.status_code}"
        )
        return

    # For locations with currents, validate the response
    assert response.status_code == 200, (
        f"Expected 200 for location with currents, got {response.status_code}"
    )
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
    assert location["code"] == location_code, (
        f"Expected location code {location_code}, got {location['code']}"
    )
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
        CurrentDirection.FLOODING.value,
        CurrentDirection.EBBING.value,
    ], f"Invalid current direction: {current['direction']}"
    assert isinstance(current["magnitude"], int | float), (
        "Current magnitude is not a number"
    )
    assert isinstance(current["magnitude_pct"], int | float), (
        "Current magnitude percentage is not a number"
    )

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
@pytest.mark.parametrize("location_code", ["nyc"])
def test_currents_api(api_client: TestClient, location_code: str) -> None:
    """Test the currents API endpoint for all configured locations.

    This test dynamically tests all locations in the configuration.
    """
    # Get location config to determine what should be present
    location_config = config.get(location_code)
    assert location_config is not None, f"Config for {location_code} not found"

    response = api_client.get(f"/api/{location_code}/currents")
    validate_currents_response(response, location_code)

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
def test_healthy_endpoint(api_client: TestClient) -> None:
    """Test the healthy API endpoint returns a valid response.

    The endpoint uses a lenient health check:
    - Returns 200 if at least 1 location has data (fresh or stale)
    - Returns 503 only if NO location can serve any data

    This ensures single station outages don't mark the entire service unhealthy.
    For detailed per-feed status, use /api/status instead.
    """
    response = api_client.get("/api/healthy")

    # Verify response status code is either 200 (all healthy) or 503 (some unhealthy)
    assert response.status_code in [
        200,
        503,
    ], f"Unexpected status: {response.status_code}"

    data = response.json()

    if response.status_code == 200:
        # Healthy: returns True
        assert isinstance(data, bool)
        assert data is True
    else:
        # Unhealthy (503): returns error detail
        assert isinstance(data, dict)
        assert "detail" in data


@pytest.mark.integration
def test_get_current_tide_plot_nyc(api_client: TestClient) -> None:
    """Test the GET /api/nyc/current_tide_plot endpoint."""
    response = api_client.get("/api/nyc/current_tide_plot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    svg_content = response.text
    assert svg_content is not None
    assert len(svg_content) > 100  # Check it's not trivially small
    assert "<svg" in svg_content
    assert "</svg>" in svg_content
    # TODO: Could add more specific checks on SVG content if needed

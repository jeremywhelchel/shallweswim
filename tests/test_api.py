"""Tests for the API status endpoints."""

# pylint: disable=duplicate-code,unused-argument

# Standard library imports
from unittest.mock import MagicMock, patch
from typing import Generator
import datetime

# Third-party imports
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Local imports
from shallweswim.api import register_routes
from shallweswim import config as config_lib
from shallweswim.data import LocationDataManager
from shallweswim.types import LocationStatus, FeedStatus
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
def mock_data_managers(app: FastAPI) -> Generator[None, None, None]:
    """Create mock data managers for testing."""
    # Create mock location configs
    nyc_config = MagicMock(spec=config_lib.LocationConfig)
    nyc_config.code = "nyc"
    nyc_config.name = "New York City"

    sf_config = MagicMock(spec=config_lib.LocationConfig)
    sf_config.code = "sf"
    sf_config.name = "San Francisco"

    # Create mock data managers
    nyc_data = MagicMock(spec=LocationDataManager)
    nyc_data.status = LocationStatus(
        feeds={
            "tides": FeedStatus(
                name="NoaaTidesFeed",
                location="nyc",
                timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
                age_seconds=3600,
                is_expired=False,
                is_ready=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            ),
            "currents": FeedStatus(
                name="MultiStationCurrentsFeed",
                location="nyc",
                timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
                age_seconds=3600,
                is_expired=False,
                is_ready=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            ),
        }
    )
    nyc_data.ready = True

    sf_data = MagicMock(spec=LocationDataManager)
    sf_data.status = LocationStatus(
        feeds={
            "tides": FeedStatus(
                name="NoaaTidesFeed",
                location="sf",
                timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
                age_seconds=3600,
                is_expired=False,
                is_ready=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            )
        }
    )
    sf_data.ready = True

    # Patch the config.get function to return our mock configs
    with patch("shallweswim.config.get") as mock_get:
        mock_get.side_effect = lambda code: {
            "nyc": nyc_config,
            "sf": sf_config,
        }.get(code)

        # Add data managers to the app state
        app.state.data_managers["nyc"] = nyc_data
        app.state.data_managers["sf"] = sf_data

        yield


def test_location_status_endpoint(
    test_client: TestClient, mock_data_managers: None
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
    assert status_data["feeds"]["tides"]["is_ready"] is True

    # Test the SF location status endpoint
    response = test_client.get("/api/sf/status")
    assert response.status_code == 200

    # Check that the response is a valid JSON object
    status_data = response.json()
    assert_json_serializable(status_data)

    # Check that the response contains the expected data
    assert "feeds" in status_data
    assert "tides" in status_data["feeds"]
    assert status_data["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["feeds"]["tides"]["location"] == "sf"

    # Test a non-existent location
    response = test_client.get("/api/nonexistent/status")
    assert response.status_code == 404


def test_all_locations_status_endpoint(
    test_client: TestClient, mock_data_managers: None
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
    assert "sf" in status_data

    # Check the NYC data
    assert "feeds" in status_data["nyc"]
    assert "tides" in status_data["nyc"]["feeds"]
    assert "currents" in status_data["nyc"]["feeds"]
    assert status_data["nyc"]["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["nyc"]["feeds"]["tides"]["location"] == "nyc"

    # Check the SF data
    assert "feeds" in status_data["sf"]
    assert "tides" in status_data["sf"]["feeds"]
    assert status_data["sf"]["feeds"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["sf"]["feeds"]["tides"]["location"] == "sf"

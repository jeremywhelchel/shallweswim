"""Tests for the API status endpoints."""

# pylint: disable=duplicate-code,unused-argument

# Standard library imports
from unittest.mock import MagicMock, patch
from typing import Generator

# Third-party imports
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Local imports
from shallweswim.api import register_routes, data
from shallweswim import config as config_lib
from shallweswim.data import DataManager
from tests.test_utils import assert_json_serializable


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    app = FastAPI()
    register_routes(app)
    return TestClient(app)


@pytest.fixture
def mock_data_managers() -> Generator[None, None, None]:
    """Create mock data managers for testing."""
    # Create mock location configs
    nyc_config = MagicMock(spec=config_lib.LocationConfig)
    nyc_config.code = "nyc"
    nyc_config.name = "New York City"

    sf_config = MagicMock(spec=config_lib.LocationConfig)
    sf_config.code = "sf"
    sf_config.name = "San Francisco"

    # Create mock data managers
    nyc_data = MagicMock(spec=DataManager)
    nyc_data.status = {
        "tides": {
            "name": "NoaaTidesFeed",
            "location": "nyc",
            "timestamp": "2025-04-27T12:00:00",
            "age_seconds": 3600,
            "is_expired": False,
            "is_ready": True,
            "data_shape": [24, 1],
            "expiration_seconds": 86400,
        },
        "currents": {
            "name": "MultiStationCurrentsFeed",
            "location": "nyc",
            "timestamp": "2025-04-27T12:00:00",
            "age_seconds": 3600,
            "is_expired": False,
            "is_ready": True,
            "data_shape": [24, 1],
            "expiration_seconds": 86400,
        },
    }
    nyc_data.ready = True

    sf_data = MagicMock(spec=DataManager)
    sf_data.status = {
        "tides": {
            "name": "NoaaTidesFeed",
            "location": "sf",
            "timestamp": "2025-04-27T12:00:00",
            "age_seconds": 3600,
            "is_expired": False,
            "is_ready": True,
            "data_shape": [24, 1],
            "expiration_seconds": 86400,
        },
    }
    sf_data.ready = True

    # Patch the config.get function to return our mock configs
    with patch("shallweswim.config.get") as mock_get:
        mock_get.side_effect = lambda code: {
            "nyc": nyc_config,
            "sf": sf_config,
        }.get(code)

        # Patch the global data dictionary
        original_data = data.copy()
        data.clear()
        data["nyc"] = nyc_data
        data["sf"] = sf_data

        yield

        # Restore the original data
        data.clear()
        data.update(original_data)


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
    assert "tides" in status_data
    assert "currents" in status_data
    assert status_data["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["tides"]["location"] == "nyc"
    assert status_data["tides"]["is_ready"] is True

    # Test the SF location status endpoint
    response = test_client.get("/api/sf/status")
    assert response.status_code == 200

    # Check that the response is a valid JSON object
    status_data = response.json()
    assert_json_serializable(status_data)

    # Check that the response contains the expected data
    assert "tides" in status_data
    assert status_data["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["tides"]["location"] == "sf"

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
    assert "tides" in status_data["nyc"]
    assert "currents" in status_data["nyc"]
    assert status_data["nyc"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["nyc"]["tides"]["location"] == "nyc"

    # Check the SF data
    assert "tides" in status_data["sf"]
    assert status_data["sf"]["tides"]["name"] == "NoaaTidesFeed"
    assert status_data["sf"]["tides"]["location"] == "sf"

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
from shallweswim.types import (
    LocationStatus,
    FeedStatus,
)
from shallweswim.dataframe_models import WaterTempDataModel
from tests.helpers import assert_json_serializable
import pandas as pd
import pandera as pa
from fastapi import status


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
    sf_data.status = LocationStatus(
        feeds={
            "tides": FeedStatus(
                name="NoaaTidesFeed",
                location="sf",
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
    with patch("shallweswim.config.get") as mock_get:
        mock_get.side_effect = lambda code: {
            "nyc": nyc_config,
            "sf": sf_config,
        }.get(code)

        # Initialize and add data managers to the app state
        app.state.data_managers = {}  # Ensure the dictionary exists
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


# --- Feed Data Endpoint Tests (Synchronous using TestClient) ---


@pytest.mark.filterwarnings(
    # Broaden message match for the specific UserWarning from pydantic.type_adapter
    "ignore:Pydantic serializer warnings.*:UserWarning:pydantic.type_adapter"
)
def test_get_feed_data_success(
    test_client: TestClient, mock_data_managers: None
) -> None:
    """Test successfully retrieving data for a valid feed."""
    # Mock the specific feed data expected
    # Create a sample TimeSeriesData DataFrame with only valid columns
    mock_time = pd.to_datetime("2025-05-04 12:00:00")
    mock_df = pd.DataFrame(
        {
            "water_temp": [15.0, 15.1],
        },
        index=pd.DatetimeIndex(
            [mock_time, mock_time + pd.Timedelta(hours=1)], name="time"
        ),
    )

    # Ensure the mock DataFrame is valid according to the model
    try:
        WaterTempDataModel.validate(mock_df)
    except pa.errors.SchemaError as e:
        pytest.fail(f"Mock DataFrame is invalid: {e}")

    # Mock the specific feed within the data manager
    mock_feed = MagicMock()
    mock_feed.values = mock_df
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    mock_data_managers_dict = test_client.app.state.data_managers
    # Ensure _feeds exists; create if necessary (it might not exist in the simple mock)
    if (
        not hasattr(mock_data_managers_dict["nyc"], "_feeds")
        or mock_data_managers_dict["nyc"]._feeds is None
    ):
        mock_data_managers_dict["nyc"]._feeds = {}
    mock_data_managers_dict["nyc"]._feeds[
        "live_temps"
    ] = mock_feed  # Use 'live_temps' as the example feed

    response = test_client.get("/api/nyc/data/live_temps")
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "application/json"
    data = response.json()

    # Expect dict with timestamps as keys due to orient='index'
    assert isinstance(data, dict)
    if not data:
        pytest.fail("Mocked data resulted in empty response, expected data.")

    # Reconstruct and validate DataFrame
    try:
        # Reconstruct from index-oriented dict
        df = pd.DataFrame.from_dict(data, orient="index")

        # Convert index back to datetime (FastAPI TestClient might return strings)
        df.index = pd.to_datetime(df.index)
        df.index.name = "time"  # Restore index name

        # Ensure necessary columns are present based on mock_df
        present_cols = set(mock_df.columns)
        assert present_cols == set(df.columns)  # Check for exact match now

        # Validate the reconstructed DataFrame
        WaterTempDataModel.validate(df, lazy=True)  # Use lazy for better errors

    except (ValueError, pa.errors.SchemaError, KeyError, TypeError) as e:
        pytest.fail(f"Response data failed validation or parsing: {e}\nData: {data}")


def test_get_feed_data_location_not_found(
    test_client: TestClient, mock_data_managers: None
) -> None:
    """Test requesting data for a non-existent location."""
    response = test_client.get("/api/nonexistent/data/tides")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {
        "detail": "Location 'nonexistent' not found or data not loaded"
    }


def test_get_feed_data_feed_not_found(
    test_client: TestClient, mock_data_managers: None
) -> None:
    """Test requesting a non-existent feed for a valid location."""
    # Ensure the mock setup for 'nyc' doesn't accidentally include 'badfeed'
    assert isinstance(test_client.app, FastAPI)  # Help mypy
    mock_data_managers_dict = test_client.app.state.data_managers
    if (
        hasattr(mock_data_managers_dict["nyc"], "_feeds")
        and "badfeed" in mock_data_managers_dict["nyc"]._feeds
    ):
        del mock_data_managers_dict["nyc"]._feeds["badfeed"]

    response = test_client.get("/api/nyc/data/badfeed")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    # Check detail message (could vary slightly based on implementation)
    assert "detail" in response.json()
    assert "badfeed" in response.json()["detail"]
    assert "nyc" in response.json()["detail"]

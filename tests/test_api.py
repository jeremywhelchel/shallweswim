"""Tests for the API status endpoints."""

# pylint: disable=duplicate-code,unused-argument

# Standard library imports
import concurrent.futures
import datetime
from collections.abc import Generator
from unittest.mock import MagicMock, call, patch

import pandas as pd

# Third-party imports
import pytest
import pytz
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from freezegun import freeze_time

from shallweswim import types as sw_types

# Local imports
from shallweswim.api import initialize_location_data, register_routes
from shallweswim.api.routes import app_source_citations
from shallweswim.api_types import (
    FeedStatus,
    HistoricalTempStatus,
    LocationConditions,
    LocationStatus,
)
from shallweswim.config import (
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    LocationConfig,
    NdbcTempFeedConfig,
)
from shallweswim.core.feeds import FEED_CURRENTS, FEED_TIDES
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
        default_temperature_unit="F",
        live_temp_source=CoopsTempFeedConfig(
            station=8518750, name="The Battery", live_enabled=True
        ),
        historic_temp_source=CoopsTempFeedConfig(
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
        default_temperature_unit="F",
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
                next_fetch_after=datetime.datetime.fromisoformat("2025-04-28T12:00:00"),
                age_seconds=3600,
                seconds_until_next_fetch=82800,
                consecutive_failures=0,
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
                next_fetch_after=datetime.datetime.fromisoformat("2025-04-28T12:00:00"),
                age_seconds=3600,
                seconds_until_next_fetch=82800,
                consecutive_failures=0,
                is_expired=False,
                is_healthy=True,
                expiration_seconds=86400,
                data_summary=None,
                error=None,
            ),
            "historic_temps": FeedStatus(
                name="HistoricalTempsFeed",
                location="nyc",
                fetch_timestamp=None,
                next_fetch_after=datetime.datetime.fromisoformat("2025-04-27T12:01:00"),
                age_seconds=None,
                seconds_until_next_fetch=60,
                consecutive_failures=1,
                is_expired=True,
                is_healthy=False,
                expiration_seconds=10800,
                data_summary=None,
                error="Historical temperature fetch incomplete for years: 2025",
                historical_temp_status=HistoricalTempStatus(
                    required_years=[2024, 2025],
                    available_years=[2024],
                    cached_years=[2024],
                    missing_years=[2025],
                    fetched_years=[],
                    failed_years={2025: "ValueError: test failure"},
                ),
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
                next_fetch_after=datetime.datetime.fromisoformat("2025-04-28T12:00:00"),
                age_seconds=3600,
                seconds_until_next_fetch=82800,
                consecutive_failures=0,
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
    assert status_data["feeds"]["historic_temps"]["consecutive_failures"] == 1
    assert status_data["feeds"]["historic_temps"]["seconds_until_next_fetch"] == 60
    assert status_data["feeds"]["historic_temps"]["historical_temp_status"] == {
        "required_years": [2024, 2025],
        "available_years": [2024],
        "cached_years": [2024],
        "missing_years": [2025],
        "fetched_years": [],
        "failed_years": {"2025": "ValueError: test failure"},
    }

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
    assert status_data["nyc"]["feeds"]["historic_temps"]["historical_temp_status"][
        "missing_years"
    ] == [2025]

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
    assert data["app_name"] == "shall we swim?"
    assert data["short_name"] == "shallweswim"
    assert data["default_location_code"] == "nyc"
    assert "nyc" in data["location_order"]
    assert "manifest" not in data

    nyc = data["locations"]["nyc"]
    assert nyc["metadata"]["code"] == "nyc"
    assert nyc["metadata"]["default_temperature_unit"] == "F"
    assert nyc["metadata"]["features"]["temperature"] is True
    assert nyc["metadata"]["temperature_plots"] == {"live": True, "historic": True}
    assert nyc["metadata"]["citations"]["temperature"] is not None
    assert nyc["metadata"]["citations"]["live_temperature"] is None
    assert nyc["metadata"]["citations"]["historical_temperature"] is None
    assert nyc["metadata"]["features"]["webcam"] is True
    assert nyc["metadata"]["features"]["transit"] is True
    assert nyc["metadata"]["features"]["windy"] is True
    assert nyc["metadata"]["features"]["water_movement_planning"] is True
    assert nyc["metadata"]["features"]["water_movement_detail"] is True
    assert data["source_code_link"]["url"].endswith("/shallweswim")
    assert nyc["integrations"]["webcam"]["provider"] == "youtube_live"
    assert nyc["integrations"]["webcam"]["channel_id"]
    assert nyc["integrations"]["webcam"]["alternative"]["label"] == (
        "Earth Cam Coney Island"
    )
    assert nyc["integrations"]["webcam"]["source"]["description"] == (
        "thanks to David K and Karol L"
    )
    assert "youtube_live" not in nyc["integrations"]
    assert "webcam_alternative" not in nyc["integrations"]
    assert "webcam_source" not in nyc["integrations"]
    assert nyc["integrations"]["transit_source"]["url"] == "https://goodservice.io"
    assert nyc["integrations"]["transit_routes"][0]["goodservice_route_id"] == "B"
    assert nyc["integrations"]["transit_routes"][0]["goodservice_direction"] == "south"
    assert nyc["integrations"]["windy"] == {
        "overlay": "waves",
        "product": "ecmwfWaves",
        "level": "surface",
        "zoom": 11,
        "metric_wind": "default",
        "metric_temp": "°F",
    }

    chi = data["locations"]["chi"]
    assert chi["metadata"]["features"]["webcam"] is True
    assert chi["metadata"]["features"]["transit"] is False
    assert chi["metadata"]["features"]["water_movement_planning"] is False
    assert chi["metadata"]["features"]["water_movement_detail"] is False
    assert chi["integrations"]["webcam"]["provider"] == "iframe"
    assert chi["integrations"]["webcam"]["embed_url"].startswith(
        "https://api.wetmet.net"
    )
    assert "youtube_live" not in chi["integrations"]

    sdf = data["locations"]["sdf"]
    assert sdf["metadata"]["temperature_plots"] == {"live": True, "historic": False}
    assert sdf["metadata"]["citations"]["temperature"] is None
    assert sdf["metadata"]["citations"]["live_temperature"] is not None
    assert sdf["metadata"]["citations"]["historical_temperature"] is None
    assert sdf["metadata"]["features"]["webcam"] is True
    assert sdf["metadata"]["features"]["water_movement_planning"] is False
    assert sdf["metadata"]["features"]["water_movement_detail"] is False
    assert sdf["integrations"]["webcam"]["provider"] == "earthcam_embed"
    assert sdf["integrations"]["webcam"]["embed_url"].startswith(
        "https://share.earthcam.net/"
    )
    assert sdf["integrations"]["webcam"]["script_url"] is None
    assert sdf["integrations"]["windy"]["overlay"] == "wind"
    assert sdf["integrations"]["windy"]["product"] == "ecmwf"

    dov = data["locations"]["dov"]
    assert dov["metadata"]["default_temperature_unit"] == "C"
    assert dov["metadata"]["temperature_plots"] == {"live": True, "historic": True}
    assert dov["metadata"]["citations"]["temperature"] is None
    assert dov["metadata"]["citations"]["live_temperature"] is not None
    assert dov["metadata"]["citations"]["historical_temperature"] is not None


def test_app_source_citations_split_different_temperature_sources() -> None:
    """Different live and historical temperature sources get scoped citations."""
    cfg = LocationConfig(
        code="mix",
        name="Mixed",
        swim_location="Mixed Beach",
        swim_location_link="https://example.com/mixed",
        description="Mixed-source location",
        latitude=40.0,
        longitude=-70.0,
        timezone=pytz.timezone("UTC"),
        default_temperature_unit="F",
        live_temp_source=CoopsTempFeedConfig(station=8518750, name="Live Station"),
        historic_temp_source=NdbcTempFeedConfig(
            station="44013", name="Historic Station"
        ),
    )

    citations = app_source_citations(cfg)

    assert citations.temperature is None
    assert citations.live_temperature is not None
    assert citations.live_temperature.startswith("Live temperature: ")
    assert "Live Station" in citations.live_temperature
    assert citations.historical_temperature is not None
    assert citations.historical_temperature.startswith("Historical temperature: ")
    assert "Historic Station" in citations.historical_temperature

    assert [row.label for row in cfg.temperature_source_citations] == [
        "Live temperature",
        "Historical temperature",
    ]


def test_app_source_citations_deduplicates_same_temperature_source() -> None:
    """Equivalent live and historical temperature sources get one citation."""
    temp_source = CoopsTempFeedConfig(station=8518750, name="Shared Station")
    cfg = LocationConfig(
        code="shr",
        name="Shared",
        swim_location="Shared Beach",
        swim_location_link="https://example.com/shared",
        description="Shared-source location",
        latitude=40.0,
        longitude=-70.0,
        timezone=pytz.timezone("UTC"),
        default_temperature_unit="F",
        live_temp_source=temp_source,
        historic_temp_source=temp_source,
    )

    citations = app_source_citations(cfg)

    assert citations.temperature is not None
    assert "Shared Station" in citations.temperature
    assert citations.live_temperature is None
    assert citations.historical_temperature is None

    assert [row.label for row in cfg.temperature_source_citations] == ["Temperature"]


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
        range=sw_types.CurrentRange(
            slack=sw_types.CurrentRangePoint(
                timestamp=mock_dt - datetime.timedelta(hours=2),
                magnitude=0.0,
            ),
            peak=sw_types.CurrentRangePoint(
                timestamp=mock_dt + datetime.timedelta(hours=1),
                magnitude=1.4,
                phase=sw_types.CurrentPhase.FLOOD,
            ),
        ),
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
    # Mock the return value for get_tide_info_at_time method
    mock_manager.get_tide_info_at_time.return_value = mock_tide_info
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
    assert "water_temp" not in data["temperature"]
    assert "units" not in data["temperature"]
    assert data["temperature"]["water_temp_f"] == mock_temp_value
    assert data["temperature"]["water_temp_c"] == pytest.approx(
        round((mock_temp_value - 32) * 5 / 9, 1)
    )
    nyc_config = mock_data_managers["nyc"]
    assert nyc_config.live_temp_source is not None  # Help mypy
    assert (
        data["temperature"]["station_name"] == nyc_config.live_temp_source.name
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
    assert data["current"]["range"] is not None
    assert (
        data["current"]["range"]["slack"]["timestamp"]
        == (mock_dt - datetime.timedelta(hours=2)).isoformat()
    )
    assert data["current"]["range"]["slack"]["magnitude"] == 0.0
    assert data["current"]["range"]["slack"]["units"] == "kt"
    assert data["current"]["range"]["slack"]["phase"] is None
    assert (
        data["current"]["range"]["peak"]["timestamp"]
        == (mock_dt + datetime.timedelta(hours=1)).isoformat()
    )
    assert data["current"]["range"]["peak"]["magnitude"] == 1.4
    assert data["current"]["range"]["peak"]["units"] == "kt"
    assert data["current"]["range"]["peak"]["phase"] == "flood"


@freeze_time("2026-05-18T18:00:00Z")
def test_conditions_endpoint_accepts_local_at_parameter(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """The conditions API shifts tide and prediction current state for planner time."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]
    mock_dt = datetime.datetime(2026, 5, 18, 15, 30, 0)
    mock_manager.get_current_temperature.return_value = sw_types.TemperatureReading(
        timestamp=datetime.datetime(2026, 5, 18, 13, 55, 0),
        temperature=65.0,
    )
    mock_manager.get_tide_info_at_time.return_value = sw_types.TideInfo(
        past=[
            sw_types.TideEntry(
                time=mock_dt - datetime.timedelta(hours=1),
                type=sw_types.TideCategory.LOW,
                prediction=0.2,
            )
        ],
        next=[
            sw_types.TideEntry(
                time=mock_dt + datetime.timedelta(hours=5),
                type=sw_types.TideCategory.HIGH,
                prediction=4.8,
            )
        ],
    )
    mock_manager.predict_tide_at_time.return_value = sw_types.TideState(
        timestamp=mock_dt,
        estimated_height=2.2,
        units="ft",
        trend=sw_types.TideTrend.RISING,
        height_pct=0.52,
    )
    mock_manager.predict_flow_at_time.return_value = sw_types.CurrentInfo(
        timestamp=mock_dt,
        magnitude=1.4,
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.75,
        direction=sw_types.CurrentDirection.EBBING,
        phase=sw_types.CurrentPhase.EBB,
        strength=sw_types.CurrentStrength.MODERATE,
        trend=sw_types.CurrentTrend.EASING,
        state_description="moderate ebb and easing",
    )

    response = test_client.get("/api/nyc/conditions?at=2026-05-18T15:30:00")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["tides"]["state"]["timestamp"] == "2026-05-18T15:30:00"
    assert data["tides"]["state"]["estimated_height"] == 2.2
    assert data["current"]["timestamp"] == "2026-05-18T15:30:00"
    assert data["current"]["magnitude"] == 1.4
    mock_manager.get_tide_info_at_time.assert_called_once_with(mock_dt)
    mock_manager.predict_tide_at_time.assert_called_once_with(mock_dt)
    mock_manager.predict_flow_at_time.assert_called_once_with(mock_dt)


@freeze_time("2026-05-18T18:00:00Z")
def test_planner_at_is_consistent_across_time_aware_endpoints(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Planner-aware endpoints resolve one location-local target time."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]
    mock_dt = datetime.datetime(2026, 5, 18, 15, 30, 0)
    mock_manager.has_data = True

    def has_feed_data_side_effect(feed_name: str) -> bool:
        return feed_name in {FEED_TIDES, FEED_CURRENTS}

    mock_manager.has_feed_data.side_effect = has_feed_data_side_effect
    mock_manager._feeds[FEED_TIDES] = MagicMock(
        values=pd.DataFrame({"prediction": [0.2, 4.8]})
    )
    mock_manager._feeds[FEED_CURRENTS] = MagicMock(
        values=pd.DataFrame({"magnitude": [0.0, 1.4]})
    )
    mock_manager.get_tide_info_at_time.return_value = sw_types.TideInfo(
        past=[
            sw_types.TideEntry(
                time=mock_dt - datetime.timedelta(hours=1),
                type=sw_types.TideCategory.LOW,
                prediction=0.2,
            )
        ],
        next=[
            sw_types.TideEntry(
                time=mock_dt + datetime.timedelta(hours=5),
                type=sw_types.TideCategory.HIGH,
                prediction=4.8,
            )
        ],
    )
    mock_manager.predict_tide_at_time.return_value = sw_types.TideState(
        timestamp=mock_dt,
        estimated_height=2.2,
        units="ft",
        trend=sw_types.TideTrend.RISING,
        height_pct=0.52,
    )
    mock_manager.predict_flow_at_time.return_value = sw_types.CurrentInfo(
        timestamp=mock_dt,
        magnitude=1.4,
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.75,
        direction=sw_types.CurrentDirection.EBBING,
        phase=sw_types.CurrentPhase.EBB,
        strength=sw_types.CurrentStrength.MODERATE,
        trend=sw_types.CurrentTrend.EASING,
        state_description="moderate ebb and easing",
    )

    class FakeFigure:
        def savefig(self, output, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            output.write("<svg></svg>")

    plot_call: dict[str, object] = {}

    def fake_create_tide_current_plot(
        _tides_data: pd.DataFrame,
        _currents_data: pd.DataFrame,
        timestamp: datetime.datetime,
        _cfg: LocationConfig,
    ) -> FakeFigure:
        plot_call["timestamp"] = timestamp
        return FakeFigure()

    test_client.app.state.process_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )
    try:
        with patch(
            "shallweswim.api.routes._create_tide_current_plot",
            side_effect=fake_create_tide_current_plot,
        ):
            conditions_response = test_client.get(
                "/api/nyc/conditions?at=2026-05-18T15:30:00"
            )
            currents_response = test_client.get(
                "/api/nyc/currents?at=2026-05-18T15:30:00"
            )
            plot_response = test_client.get(
                "/api/nyc/plots/current_tide?at=2026-05-18T15:30:00"
            )
    finally:
        test_client.app.state.process_pool.shutdown(wait=False)

    assert conditions_response.status_code == status.HTTP_200_OK
    assert currents_response.status_code == status.HTTP_200_OK
    assert plot_response.status_code == status.HTTP_200_OK

    conditions = conditions_response.json()
    currents = currents_response.json()
    assert conditions["tides"]["state"]["timestamp"] == "2026-05-18T15:30:00"
    assert conditions["current"]["timestamp"] == "2026-05-18T15:30:00"
    assert currents["timestamp"] == "2026-05-18T15:30:00"
    assert currents["navigation"]["at"] == "2026-05-18T15:30:00"
    assert plot_call["timestamp"] == mock_dt
    mock_manager.get_tide_info_at_time.assert_called_once_with(mock_dt)
    mock_manager.predict_tide_at_time.assert_called_once_with(mock_dt)
    assert mock_manager.predict_flow_at_time.mock_calls == [
        call(mock_dt),
        call(mock_dt),
    ]


@pytest.mark.parametrize(
    "at",
    [
        "2026-05-18T15:30:00-04:00",
        "2026-05-18T19:30:00Z",
    ],
)
def test_conditions_endpoint_rejects_at_with_timezone_offset(
    at: str, test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Planner condition times are location-local and reject explicit offsets."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]

    response = test_client.get("/api/nyc/conditions", params={"at": at})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "must not include a timezone offset" in response.json()["detail"]
    mock_manager.get_tide_info_at_time.assert_not_called()


@freeze_time("2026-05-18T18:00:00Z")
def test_conditions_endpoint_rejects_at_outside_prediction_window(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Planner condition times must stay within the supported prediction window."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]

    response = test_client.get("/api/nyc/conditions?at=2026-05-19T14:01:00")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "within 24 hours" in response.json()["detail"]
    mock_manager.get_tide_info_at_time.assert_not_called()


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
    assert "range" in schemas["CurrentInfo"]["properties"]
    timestamp_fields = [
        schemas["TemperatureInfo"]["properties"]["timestamp"],
        schemas["TideEntry"]["properties"]["time"],
        schemas["TideState"]["properties"]["timestamp"],
        schemas["CurrentInfo"]["properties"]["timestamp"],
        schemas["CurrentRangePoint"]["properties"]["timestamp"],
        schemas["CurrentsResponse"]["properties"]["timestamp"],
    ]
    for timestamp_field in timestamp_fields:
        assert timestamp_field["type"] == "string"
        assert timestamp_field["format"] == "date-time"

    navigation_at = schemas["NavigationInfo"]["properties"]["at"]
    assert {"type": "string", "format": "date-time"} in navigation_at["anyOf"]
    assert {"type": "null"} in navigation_at["anyOf"]

    range_point_timestamp = schemas["CurrentRangePoint"]["properties"]["timestamp"]
    assert range_point_timestamp["type"] == "string"
    assert range_point_timestamp["format"] == "date-time"
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
        default_temperature_unit="F",
        live_temp_source=CoopsTempFeedConfig(
            station=8518750, name="The Battery", live_enabled=True
        ),
        historic_temp_source=CoopsTempFeedConfig(
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


def test_currents_endpoint_no_currents_source_rejects_before_at_validation(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Unsupported current prediction routes should not parse planner params first."""
    response = test_client.get("/api/sfo/currents?at=not-a-time")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "does not support" in response.json()["detail"]


def test_currents_endpoint_serializes_current_range(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Current prediction responses include optional slack-to-peak range context."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]
    mock_dt = datetime.datetime(2025, 5, 4, 12, 0, 0)
    mock_manager.predict_flow_at_time.return_value = sw_types.CurrentInfo(
        timestamp=mock_dt,
        magnitude=0.753,
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.65,
        direction=sw_types.CurrentDirection.FLOODING,
        phase=sw_types.CurrentPhase.FLOOD,
        strength=sw_types.CurrentStrength.MODERATE,
        trend=sw_types.CurrentTrend.BUILDING,
        state_description="moderate flood and building",
        range=sw_types.CurrentRange(
            slack=sw_types.CurrentRangePoint(
                timestamp=mock_dt - datetime.timedelta(hours=2),
                magnitude=0.0,
            ),
            peak=sw_types.CurrentRangePoint(
                timestamp=mock_dt + datetime.timedelta(hours=1),
                magnitude=1.4,
                phase=sw_types.CurrentPhase.FLOOD,
            ),
        ),
    )

    response = test_client.get("/api/nyc/currents")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["current"]["range"] is not None
    assert (
        data["current"]["range"]["slack"]["timestamp"]
        == (mock_dt - datetime.timedelta(hours=2)).isoformat()
    )
    assert data["current"]["range"]["slack"]["magnitude"] == 0.0
    assert data["current"]["range"]["slack"]["phase"] is None
    assert (
        data["current"]["range"]["peak"]["timestamp"]
        == (mock_dt + datetime.timedelta(hours=1)).isoformat()
    )
    assert data["current"]["range"]["peak"]["magnitude"] == 1.4
    assert data["current"]["range"]["peak"]["phase"] == "flood"


@freeze_time("2026-05-18T18:00:00Z")
def test_currents_endpoint_accepts_local_at_parameter(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """The currents API accepts local planner time and computes shift metadata."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]
    mock_dt = datetime.datetime(2026, 5, 18, 15, 30, 0)
    mock_manager.predict_flow_at_time.return_value = sw_types.CurrentInfo(
        timestamp=mock_dt,
        magnitude=0.753,
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.65,
        direction=sw_types.CurrentDirection.FLOODING,
        phase=sw_types.CurrentPhase.FLOOD,
        strength=sw_types.CurrentStrength.MODERATE,
        trend=sw_types.CurrentTrend.BUILDING,
        state_description="moderate flood and building",
    )

    response = test_client.get("/api/nyc/currents?at=2026-05-18T15:30:00")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["timestamp"] == "2026-05-18T15:30:00"
    assert data["navigation"]["shift"] == 90
    assert data["navigation"]["at"] == "2026-05-18T15:30:00"
    assert (
        data["navigation"]["plot_url"]
        == "/api/nyc/plots/current_tide?at=2026-05-18T15%3A30%3A00"
    )
    mock_manager.predict_flow_at_time.assert_called_once_with(mock_dt)


@freeze_time("2026-05-18T18:00:00Z")
def test_currents_endpoint_prefers_at_over_shift(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """When both are present, at is the canonical effective time."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]
    mock_dt = datetime.datetime(2026, 5, 18, 15, 30, 0)
    mock_manager.predict_flow_at_time.return_value = sw_types.CurrentInfo(
        timestamp=mock_dt,
        magnitude=0.753,
        source_type=sw_types.DataSourceType.PREDICTION,
        magnitude_pct=0.65,
        direction=sw_types.CurrentDirection.FLOODING,
        phase=sw_types.CurrentPhase.FLOOD,
        strength=sw_types.CurrentStrength.MODERATE,
        trend=sw_types.CurrentTrend.BUILDING,
        state_description="moderate flood and building",
    )

    response = test_client.get("/api/nyc/currents?shift=-180&at=2026-05-18T15:30:00")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["navigation"]["shift"] == 90
    mock_manager.predict_flow_at_time.assert_called_once_with(mock_dt)


@pytest.mark.parametrize(
    "at",
    [
        "2026-05-18T15:30:00-04:00",
        "2026-05-18T19:30:00Z",
    ],
)
def test_currents_endpoint_rejects_at_with_timezone_offset(
    at: str, test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Planner times are location-local and reject explicit offsets."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]

    response = test_client.get("/api/nyc/currents", params={"at": at})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "must not include a timezone offset" in response.json()["detail"]
    mock_manager.predict_flow_at_time.assert_not_called()


@freeze_time("2026-05-18T18:00:00Z")
def test_currents_endpoint_rejects_at_outside_prediction_window(
    test_client: TestClient, mock_data_managers: dict[str, LocationConfig]
) -> None:
    """Planner times must stay within the supported prediction window."""
    assert isinstance(test_client.app, FastAPI)
    mock_manager = test_client.app.state.data_managers["nyc"]

    response = test_client.get("/api/nyc/currents?at=2026-05-19T14:01:00")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "within 24 hours" in response.json()["detail"]
    mock_manager.predict_flow_at_time.assert_not_called()


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
        default_temperature_unit="F",
        live_temp_source=CoopsTempFeedConfig(
            station=8518750, name="The Battery", live_enabled=True
        ),
        historic_temp_source=CoopsTempFeedConfig(
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
    mock_manager.get_tide_info_at_time.return_value = sw_types.TideInfo(
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
    assert data["current"]["range"] is None

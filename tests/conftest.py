"""Configuration for pytest.

This file contains setup for integration tests that hit real NOAA CO-OPS API endpoints,
as well as shared test fixtures including fake location configs for deterministic testing.
"""

# Third-party imports
import pytest
import pytz

from shallweswim.config import (
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    LocationConfig,
)

# =============================================================================
# Fake Test Configs
# =============================================================================
# These configs are used by unit and E2E tests to ensure deterministic behavior
# independent of production config changes. Integration tests should use real
# production configs instead.

# Full-featured location (like NYC) - has temp, tides, and currents
# Uses code="nyc" because the currents endpoint has NYC-specific logic
TEST_CONFIG_FULL = LocationConfig(
    code="nyc",
    name="Test Location",
    swim_location="Test Beach",
    swim_location_link="http://example.com/test",
    description="Test location with all features",
    latitude=40.0,
    longitude=-74.0,
    timezone=pytz.timezone("US/Eastern"),
    temp_source=CoopsTempFeedConfig(station=1234567, name="Test Temp Station"),
    tide_source=CoopsTideFeedConfig(station=1234567, name="Test Tide Station"),
    currents_source=CoopsCurrentsFeedConfig(stations=["TEST001"], name="Test Current"),
    enabled=True,
)

# Has currents_source but NOT NYC (for testing 404 path in /currents endpoint)
TEST_CONFIG_NON_NYC_WITH_CURRENTS = LocationConfig(
    code="xxx",  # Not "nyc"
    name="Non-NYC Test Location",
    swim_location="Other Beach",
    swim_location_link="http://example.com/other",
    description="Test location with currents but not NYC",
    latitude=35.0,
    longitude=-80.0,
    timezone=pytz.timezone("US/Eastern"),
    temp_source=CoopsTempFeedConfig(station=2345678, name="Other Temp"),
    tide_source=CoopsTideFeedConfig(station=2345678, name="Other Tide"),
    currents_source=CoopsCurrentsFeedConfig(stations=["OTHER01"], name="Other Current"),
    enabled=True,
)

# No currents source
TEST_CONFIG_NO_CURRENTS = LocationConfig(
    code="noc",
    name="No Currents Location",
    swim_location="Calm Beach",
    swim_location_link="http://example.com/calm",
    description="Test location without currents",
    latitude=30.0,
    longitude=-85.0,
    timezone=pytz.timezone("US/Eastern"),
    temp_source=CoopsTempFeedConfig(station=3456789, name="Calm Temp"),
    tide_source=CoopsTideFeedConfig(station=3456789, name="Calm Tide"),
    currents_source=None,  # No currents
    enabled=True,
)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit live NOAA CO-OPS APIs",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as hitting live NOAA CO-OPS API service"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection to skip integration tests unless requested."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(
            reason="Need --run-integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

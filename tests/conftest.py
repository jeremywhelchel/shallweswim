"""Configuration for pytest.

This file contains setup for integration tests that hit real NOAA CO-OPS API endpoints,
as well as shared test fixtures including fake location configs for deterministic testing.
"""

# Standard library imports
import os

# Third-party imports
import pytest
import pytz

from shallweswim.config import (
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    LocationConfig,
    NwisCurrentFeedConfig,
)

# =============================================================================
# Fake Test Configs
# =============================================================================
# These configs are used by unit and E2E tests to ensure deterministic behavior
# independent of production config changes. Integration tests should use real
# production configs instead.

# Full-featured location - has temp, tides, currents (PREDICTION type), and charts
# Uses code="nyc" to match existing test URLs (the NYC-specific logic has been removed)
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
    currents_source=CoopsCurrentsFeedConfig(
        stations=["TEST001"], name="Test Current", has_static_charts=True
    ),
    enabled=True,
)

# Has currents_source but OBSERVATION type (for testing 404 path in /currents endpoint)
# The /currents endpoint only works with PREDICTION-type sources
TEST_CONFIG_OBSERVATION_CURRENTS = LocationConfig(
    code="obs",
    name="Observation Currents Location",
    swim_location="River Beach",
    swim_location_link="http://example.com/river",
    description="Test location with observation-only currents (like Louisville)",
    latitude=35.0,
    longitude=-80.0,
    timezone=pytz.timezone("US/Eastern"),
    temp_source=CoopsTempFeedConfig(station=2345678, name="Other Temp"),
    tide_source=CoopsTideFeedConfig(station=2345678, name="Other Tide"),
    currents_source=NwisCurrentFeedConfig(
        site_no="12345678", parameter_cd="72255", name="River Current"
    ),
    enabled=True,
)

# PREDICTION-type currents but no static charts (for testing null chart fields)
TEST_CONFIG_PREDICTION_NO_CHARTS = LocationConfig(
    code="pnc",
    name="Prediction No Charts Location",
    swim_location="Chart-less Beach",
    swim_location_link="http://example.com/chartless",
    description="Test location with PREDICTION currents but no static chart assets",
    latitude=32.0,
    longitude=-82.0,
    timezone=pytz.timezone("US/Eastern"),
    temp_source=CoopsTempFeedConfig(station=4567890, name="Chartless Temp"),
    tide_source=CoopsTideFeedConfig(station=4567890, name="Chartless Tide"),
    currents_source=CoopsCurrentsFeedConfig(
        stations=["TEST002"], name="Chartless Current", has_static_charts=False
    ),
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
    parser.addoption(
        "--run-browser",
        action="store_true",
        default=False,
        help="Run browser smoke tests that require Playwright browser binaries",
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance regression tests for request-path latency",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as hitting live NOAA CO-OPS API service"
    )
    config.addinivalue_line(
        "markers", "browser: mark test as requiring a real browser via Playwright"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a request-path performance guardrail"
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

    if not config.getoption("--run-browser"):
        skip_browser = pytest.mark.skip(reason="Need --run-browser option to run")
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(skip_browser)

    if not config.getoption("--run-performance"):
        skip_performance = pytest.mark.skip(
            reason="Need --run-performance option to run"
        )
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Force exit on GitHub Actions to avoid pytest-asyncio cleanup hangs.

    GitHub Actions has known issues with pytest cleanup hanging indefinitely
    after tests complete. This only affects CI - local runs exit normally.
    See: https://github.com/pytest-dev/pytest-asyncio/issues/222
    """
    if os.environ.get("GITHUB_ACTIONS"):
        os._exit(exitstatus)

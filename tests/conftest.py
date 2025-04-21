"""Configuration for pytest.

This file contains setup for integration tests that hit real NOAA API endpoints.
"""

# Third-party imports
import pytest

# Local imports
from shallweswim.noaa import NoaaApi

# Real station to use for API availability check
TIDE_STATION = 8518750  # NYC Battery - comprehensive station with good data coverage


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit live NOAA APIs",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as hitting live NOAA API service"
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


@pytest.fixture(scope="session")
def check_api_availability() -> bool:
    """Check if the NOAA API is available before running integration tests.

    This prevents all tests from failing if the API is down or there's a network issue.
    """
    try:
        # Make a simple request to verify API is accessible
        NoaaApi.Tides(station=TIDE_STATION)
        return True
    except Exception as e:
        pytest.skip(f"NOAA API unavailable, skipping integration tests: {e}")

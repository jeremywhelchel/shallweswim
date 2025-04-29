"""Configuration for pytest.

This file contains setup for integration tests that hit real NOAA CO-OPS API endpoints.
"""

# Third-party imports
import pytest

# Real station to use for API availability check
TIDE_STATION = 8518750  # NYC Battery - comprehensive station with good data coverage


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

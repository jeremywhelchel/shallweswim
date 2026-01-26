"""Unit tests for configuration module."""

import pydantic
import pytest
import pytz

from shallweswim import config


def test_config_list_not_empty() -> None:
    """Test that the location configurations list contains at least one location."""
    assert len(config.get_all_configs()) > 0, (
        "Location configurations list should not be empty"
    )


def test_all_configs_valid() -> None:
    """Test that all configs are valid according to Pydantic validation."""
    # This test implicitly validates all configs since they're already instantiated
    # If any were invalid, we would have gotten an error during module import
    for cfg in config.get_all_configs():
        assert isinstance(cfg, config.LocationConfig)


def test_coordinates_property() -> None:
    """Test the coordinates property returns the correct tuple."""
    for cfg in config.get_all_configs():
        assert cfg.coordinates == (cfg.latitude, cfg.longitude)


def test_display_name_property() -> None:
    """Test the display_name property formats correctly."""
    for cfg in config.get_all_configs():
        expected = f"{cfg.name} ({cfg.swim_location})"
        assert cfg.display_name == expected


def test_timezone_objects() -> None:
    """Test that timezone objects are used correctly."""
    # Valid timezone object should work directly
    tz_obj = pytz.timezone("US/Pacific")
    loc = config.LocationConfig(
        code="abc",
        name="Test",
        swim_location="Test Beach",
        swim_location_link="https://example.com",
        description="Test description",
        latitude=0.0,
        longitude=0.0,
        timezone=tz_obj,  # Using timezone object
    )
    assert loc.timezone is tz_obj

    # String should fail since we now require tzinfo objects
    with pytest.raises(pydantic.ValidationError):
        # This deliberately passes an invalid string to verify it's rejected
        # mypy will flag this, but it's intentional for the test
        config.LocationConfig(
            code="def",
            name="Test",
            swim_location="Test Beach",
            swim_location_link="https://example.com",
            description="Test description",
            latitude=0.0,
            longitude=0.0,
            timezone="US/Eastern",  # type: ignore # String should fail - intentional invalid input
        )


def test_get_function() -> None:
    """Test that the get() function returns the correct config."""
    # Test that we can retrieve all enabled configs by their code
    for cfg in config.get_all_configs():
        if cfg.enabled:
            retrieved = config.get(cfg.code)
            assert retrieved is not None, (
                f"get() should return a config for '{cfg.code}'"
            )
            assert retrieved.code == cfg.code
        else:
            retrieved = config.get(cfg.code)
            assert retrieved is None, (
                f"get() should not return a disabled config for '{cfg.code}'"
            )

    # Test with a non-existent code
    assert config.get("zzz") is None, "get() should return None for non-existent codes"

    # Test case insensitivity
    test_configs = {c.code.lower(): c for c in config.get_all_configs() if c.enabled}
    test_nyc = test_configs.get("nyc")
    assert test_nyc is not None
    assert test_nyc.code == "nyc"


def test_config_unique_codes() -> None:
    """Test that all location codes are unique."""
    codes = [cfg.code for cfg in config.get_all_configs()]
    assert len(codes) == len(set(codes)), "Location codes must be unique"

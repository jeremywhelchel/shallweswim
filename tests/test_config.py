"""Unit tests for configuration module."""

import pytest

from shallweswim import config


def test_config_list_not_empty() -> None:
    """Test that CONFIG_LIST contains at least one location."""
    assert len(config.CONFIG_LIST) > 0, "CONFIG_LIST should not be empty"


def test_all_configs_valid() -> None:
    """Test that all configs in CONFIG_LIST are valid according to Pydantic validation."""
    # This test implicitly validates all configs since they're already instantiated
    # If any were invalid, we would have gotten an error during module import
    for cfg in config.CONFIG_LIST:
        assert isinstance(cfg, config.LocationConfig)


# No need for explicit test for code validation - Pydantic's pattern validation handles it


# No need for explicit test for currents validation - Pydantic's min_length constraint handles it


def test_get_function() -> None:
    """Test that Get() function retrieves configs correctly."""
    # Test retrieving all existing configs
    for cfg in config.CONFIG_LIST:
        retrieved = config.Get(cfg.code)
        assert retrieved is not None, f"Get() should return a config for '{cfg.code}'"
        assert retrieved.code == cfg.code
        assert retrieved.name == cfg.name

    # Test with a non-existent code
    assert config.Get("zzz") is None, "Get() should return None for non-existent codes"


def test_config_unique_codes() -> None:
    """Test that all location codes are unique."""
    codes = [cfg.code for cfg in config.CONFIG_LIST]
    assert len(codes) == len(set(codes)), "Location codes must be unique"

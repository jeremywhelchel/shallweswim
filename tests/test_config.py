"""Unit tests for configuration module."""

import datetime

import pandas as pd
import pydantic
import pytest
import pytz

from shallweswim import config
from shallweswim import types as sw_types
from shallweswim.core.feeds import (
    CompositeFeed,
    HistoricalTempsFeed,
    MultiStationCurrentsFeed,
)


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


def test_presentation_integrations_are_typed() -> None:
    """Test configured presentation integrations are explicit typed providers."""
    nyc = config.get("nyc")
    assert nyc is not None
    assert nyc.presentation.webcam is not None
    assert nyc.presentation.webcam.provider == sw_types.WebcamProvider.YOUTUBE_LIVE
    assert nyc.presentation.transit is not None
    assert [
        route.goodservice_route_id for route in nyc.presentation.transit.routes
    ] == [
        "B",
        "Q",
    ]
    assert [
        route.goodservice_direction for route in nyc.presentation.transit.routes
    ] == [
        sw_types.GoodServiceDirection.SOUTH,
        sw_types.GoodServiceDirection.SOUTH,
    ]

    chi = config.get("chi")
    assert chi is not None
    assert chi.presentation.webcam is not None
    assert chi.presentation.webcam.provider == sw_types.WebcamProvider.IFRAME
    assert chi.presentation.webcam.embed_url is not None

    sdf = config.get("sdf")
    assert sdf is not None
    assert sdf.live_temp_source is not None
    assert sdf.historic_temp_source is None
    assert sdf.presentation.webcam is not None
    assert sdf.presentation.webcam.provider == sw_types.WebcamProvider.EARTHCAM_EMBED
    assert sdf.presentation.webcam.embed_url is not None
    assert sdf.presentation.windy.overlay == "wind"
    assert sdf.presentation.windy.product == "ecmwf"

    aus = config.get("aus")
    assert aus is not None
    assert aus.presentation.windy.overlay == "wind"
    assert aus.presentation.windy.product == "ecmwf"


def configured_composite_feeds() -> list[CompositeFeed]:
    """Build configured composite feeds whose combine output owns schema shape."""
    feeds: list[CompositeFeed] = []

    for cfg in config.get_all_configs():
        current_config = cfg.currents_source
        if (
            isinstance(current_config, config.CoopsCurrentsFeedConfig)
            and len(current_config.stations) > 1
        ):
            feeds.append(
                MultiStationCurrentsFeed(
                    location_config=cfg,
                    feed_config=current_config,
                    expiration_interval=datetime.timedelta(hours=1),
                )
            )

        if cfg.historic_temp_source is not None:
            feeds.append(
                HistoricalTempsFeed(
                    location_config=cfg,
                    feed_config=cfg.historic_temp_source,
                    expiration_interval=datetime.timedelta(hours=1),
                    start_year=2024,
                    end_year=2025,
                )
            )

    return feeds


def sample_source_frames(feed: CompositeFeed) -> list[pd.DataFrame]:
    """Return minimal source frames that should combine into the feed schema."""
    if isinstance(feed, MultiStationCurrentsFeed):
        index1 = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 0, 0, 0),
            periods=3,
            freq="1h",
        )
        index2 = pd.date_range(
            start=datetime.datetime(2025, 4, 22, 1, 0, 0),
            periods=3,
            freq="1h",
        )
        return [
            pd.DataFrame({"velocity": [1.0, 2.0, 3.0]}, index=index1),
            pd.DataFrame({"velocity": [4.0, 5.0, 6.0]}, index=index2),
        ]

    if isinstance(feed, HistoricalTempsFeed):
        index1 = pd.date_range(
            start=datetime.datetime(2024, 7, 15, 12, 0, 0),
            periods=3,
            freq="1h",
        )
        index2 = pd.date_range(
            start=datetime.datetime(2025, 7, 15, 12, 0, 0),
            periods=3,
            freq="1h",
        )
        return [
            pd.DataFrame({"water_temp": [60.0, 61.0, 62.0]}, index=index1),
            pd.DataFrame({"water_temp": [63.0, 64.0, 65.0]}, index=index2),
        ]

    raise AssertionError(f"No sample frames defined for {type(feed).__name__}")


def test_configured_composite_feeds_preserve_declared_schema() -> None:
    """Test configured composite feeds combine source frames into valid feed data."""
    checked_feeds = []

    for feed in configured_composite_feeds():
        result = feed._combine_feeds(sample_source_frames(feed))

        feed.data_model.validate(result, lazy=True)
        checked_feeds.append((feed.location_config.code, type(feed).__name__))

    assert ("nyc", "MultiStationCurrentsFeed") in checked_feeds
    assert any(feed_type == "HistoricalTempsFeed" for _, feed_type in checked_feeds)


def test_webcam_provider_requires_rendering_url() -> None:
    """Test webcam provider configs fail when required provider fields are absent."""
    with pytest.raises(pydantic.ValidationError):
        config.WebcamConfig(provider="iframe")

    with pytest.raises(pydantic.ValidationError):
        config.WebcamConfig(provider="earthcam_embed")

    with pytest.raises(pydantic.ValidationError):
        config.WebcamConfig(
            provider="earthcam_embed",
            embed_url="https://share.earthcam.net/test.player",
            script_url="https://share.earthcam.net/embed/test",
        )

    earthcam = config.WebcamConfig(
        provider="earthcam_embed",
        embed_url="https://share.earthcam.net/test.player",
    )
    assert earthcam.embed_url == "https://share.earthcam.net/test.player"


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
        default_temperature_unit="F",
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
            default_temperature_unit="F",
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
    assert config.get("NYC") == test_nyc
    assert config.get("Nyc") == test_nyc


def test_config_unique_codes() -> None:
    """Test that all location codes are unique."""
    codes = [cfg.code for cfg in config.get_all_configs()]
    assert len(codes) == len(set(codes)), "Location codes must be unique"

"""Configuration package - re-exports from locations module."""

from shallweswim.config.locations import (
    CONFIGS,
    BaseFeedConfig,
    CoopsCurrentsFeedConfig,
    CoopsTempFeedConfig,
    CoopsTideFeedConfig,
    CurrentsFeedConfig,
    LocationConfig,
    NdbcTempFeedConfig,
    NwisCurrentFeedConfig,
    NwisTempFeedConfig,
    TempFeedConfig,
    get,
    get_all_configs,
)

__all__ = [
    "CONFIGS",
    "BaseFeedConfig",
    "CoopsCurrentsFeedConfig",
    "CoopsTempFeedConfig",
    "CoopsTideFeedConfig",
    "CurrentsFeedConfig",
    "LocationConfig",
    "NdbcTempFeedConfig",
    "NwisCurrentFeedConfig",
    "NwisTempFeedConfig",
    "TempFeedConfig",
    "get",
    "get_all_configs",
]

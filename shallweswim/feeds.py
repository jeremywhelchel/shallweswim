"""Backwards compatibility - re-exports from core.feeds."""

from shallweswim.core.feeds import (
    CompositeFeed,
    CoopsCurrentsFeed,
    CoopsTempFeed,
    CoopsTidesFeed,
    CurrentsFeed,
    Feed,
    HistoricalTempsFeed,
    MultiStationCurrentsFeed,
    NdbcTempFeed,
    NwisCurrentFeed,
    NwisTempFeed,
    TempFeed,
    create_current_feed,
    create_temp_feed,
    create_tide_feed,
)

__all__ = [
    "CompositeFeed",
    "CoopsCurrentsFeed",
    "CoopsTempFeed",
    "CoopsTidesFeed",
    "CurrentsFeed",
    "Feed",
    "HistoricalTempsFeed",
    "MultiStationCurrentsFeed",
    "NdbcTempFeed",
    "NwisCurrentFeed",
    "NwisTempFeed",
    "TempFeed",
    "create_current_feed",
    "create_temp_feed",
    "create_tide_feed",
]

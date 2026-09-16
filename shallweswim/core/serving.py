"""Structural contracts between serving state and the code that reads it.

Two Protocols, and nothing else: `FeedData` is what the query functions in
`core/queries.py` read from one feed, and `LocationServing` is the surface the
API routes call on one location's manager. `Feed` and `LocationDataManager`
satisfy them unchanged; the snapshot-backed serving state in
`shallweswim/snapshot/` is their second implementation, so a route does not
know whether the location it serves was fetched in-process or loaded from a
published generation.
"""

import datetime
from typing import Protocol

import pandas as pd

from shallweswim import config as config_lib
from shallweswim.api_types import LocationStatus
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.types import (
    CurrentInfo,
    LegacyChartInfo,
    TemperatureReading,
    TideInfo,
    TideState,
)


class FeedData(Protocol):
    """One feed's served frame, whether it was fetched or loaded."""

    @property
    def has_data(self) -> bool:
        """Whether a served frame exists."""
        ...

    @property
    def values(self) -> pd.DataFrame:
        """The served frame; raises when `has_data` is False."""
        ...


class LocationServing(Protocol):
    """Everything the API routes ask of one location's serving state."""

    config: config_lib.LocationConfig

    @property
    def has_data(self) -> bool:
        """Whether any feed holds data, fresh or stale."""
        ...

    def has_feed(self, feed_name: FeedName | str) -> bool: ...

    def has_feed_data(self, feed_name: FeedName) -> bool: ...

    def get_feed_values(self, feed_name: FeedName | str) -> pd.DataFrame: ...

    def get_plot(self, plot_type: PlotName) -> bytes | None: ...

    @property
    def status(self) -> LocationStatus:
        """Per-feed status for `/api/status`."""
        ...

    def get_current_temperature(self) -> TemperatureReading: ...

    def get_tide_info_at_time(
        self, t: datetime.datetime | None = None
    ) -> TideInfo | None: ...

    def predict_tide_at_time(
        self, t: datetime.datetime | None = None
    ) -> TideState | None: ...

    def get_chart_info(
        self, t: datetime.datetime | None = None
    ) -> LegacyChartInfo | None: ...

    def get_current_flow_info(self) -> CurrentInfo: ...

    def predict_flow_at_time(
        self, t: datetime.datetime | None = None
    ) -> CurrentInfo | None: ...

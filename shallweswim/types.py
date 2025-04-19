"""Type definitions for shallweswim."""

from typing import TypedDict, Optional
import datetime


class TimeInfo(TypedDict):
    """Information about a timestamp with age details."""
    time: Optional[datetime.datetime]
    age: Optional[str]  # formatted as string duration
    age_seconds: Optional[float]


class DatasetInfo(TypedDict):
    """Information about a dataset's freshness."""
    fetch: TimeInfo
    latest_value: TimeInfo


class FreshnessInfo(TypedDict):
    """Complete freshness information for all datasets."""
    tides_and_currents: DatasetInfo
    live_temps: DatasetInfo
    historic_temps: DatasetInfo

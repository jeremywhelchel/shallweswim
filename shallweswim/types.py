"""Type definitions for shallweswim."""

from typing import TypedDict, Optional, Literal, Any, List, Dict
import datetime
from dataclasses import dataclass


# Dataset names for data freshness tracking
DatasetName = Literal["tides_and_currents", "live_temps", "historic_temps"]

# Tide types
TideType = Literal["high", "low", "unknown"]

# Current direction
CurrentDirection = Literal["flooding", "ebbing"]


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


@dataclass
class LegacyChartInfo:
    """Structured information about a tide chart."""

    hours_since_last_tide: float
    last_tide_type: TideType
    chart_filename: str
    map_title: str


@dataclass
class TideEntry:
    """Information about a single tide event."""

    time: datetime.datetime  # Time of the tide
    type: TideType  # 'high' or 'low'
    prediction: float  # Height of the tide in feet
    # Additional fields from the NOAA API can be added as needed


@dataclass
class TideInfo:
    """Structured information about previous and next tides."""

    past_tides: List[TideEntry]  # The most recent tide
    next_tides: List[TideEntry]  # The next two upcoming tides


@dataclass
class CurrentInfo:
    """Structured information about water current prediction."""

    direction: CurrentDirection  # 'flooding' or 'ebbing'
    magnitude: float  # Current strength in knots
    magnitude_pct: float  # Relative magnitude percentage (0.0-1.0)
    state_description: str  # Human-readable description of current state

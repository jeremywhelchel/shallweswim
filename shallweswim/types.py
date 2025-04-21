"""Type definitions for shallweswim."""

# Standard library imports
import datetime
from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict

# Third-party imports
from pydantic import BaseModel, Field


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


# Pydantic models for API responses
class LocationInfo(BaseModel):
    """Location information."""

    code: str = Field(..., description="Location code (e.g., 'nyc')")
    name: str = Field(..., description="Display name of the location")
    swim_location: str = Field(..., description="Specific swimming location")


class TemperatureInfo(BaseModel):
    """Water temperature information."""

    timestamp: str = Field(
        ..., description="ISO 8601 formatted timestamp of the reading"
    )
    water_temp: float = Field(..., description="Water temperature in degrees")
    units: str = Field("F", description="Temperature units (F for Fahrenheit)")


class ApiTideEntry(BaseModel):
    """Individual tide information for API responses."""

    time: str = Field(..., description="ISO 8601 formatted timestamp of the tide event")
    type: str = Field(..., description="Type of tide ('high', 'low', or 'unknown')")
    prediction: float = Field(..., description="Height of tide in feet")


class TidesInfo(BaseModel):
    """Collection of tide information."""

    past: List[ApiTideEntry] = Field(..., description="Recently occurred tides")
    next: List[ApiTideEntry] = Field(..., description="Upcoming tides")


class CurrentPredictionInfo(BaseModel):
    """Current prediction information."""

    timestamp: str = Field(
        ..., description="ISO 8601 formatted timestamp of the prediction"
    )
    direction: str = Field(
        ..., description="Direction of current ('flooding' or 'ebbing')"
    )
    magnitude: float = Field(..., description="Current strength in knots")
    magnitude_pct: float = Field(
        ..., description="Relative magnitude percentage (0.0-1.0)"
    )
    state_description: str = Field(
        ..., description="Human-readable description of current state"
    )


class LegacyChartDetails(BaseModel):
    """Information about legacy tide charts."""

    hours_since_last_tide: float = Field(
        ..., description="Hours since the last tide event"
    )
    last_tide_type: str = Field(..., description="Type of last tide ('high' or 'low')")
    chart_filename: str = Field(..., description="Filename of the legacy chart")
    map_title: str = Field(..., description="Title for the legacy map")


class CurrentsResponse(BaseModel):
    """Complete response for currents API endpoint."""

    location: LocationInfo
    timestamp: str = Field(
        ..., description="ISO 8601 formatted timestamp of the prediction"
    )
    current: CurrentPredictionInfo
    legacy_chart: LegacyChartDetails
    current_chart_filename: str = Field(
        ..., description="Filename of the current chart image"
    )
    navigation: dict[str, object] = Field(
        ..., description="Navigation parameters for time shifting"
    )


class LocationConditions(BaseModel):
    """Complete response for location conditions endpoint."""

    location: LocationInfo
    temperature: TemperatureInfo
    tides: TidesInfo

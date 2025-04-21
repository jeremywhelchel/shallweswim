"""Type definitions for shallweswim.

This module contains type definitions used throughout the application, separated into
two main categories:
1. Internal types - Used for internal data processing and storage
2. API types - Used for API request/response handling via Pydantic models
"""

# Standard library imports
import datetime
from dataclasses import dataclass
from typing import List, Literal

# Third-party imports
from pydantic import BaseModel, Field

#############################################################
# INTERNAL TYPES - Used for internal data processing         #
#############################################################

# Common type literals used across the application
DatasetName = Literal["tides_and_currents", "live_temps", "historic_temps"]
TideType = Literal["high", "low", "unknown"]
CurrentDirection = Literal["flooding", "ebbing"]


@dataclass
class TideEntry:
    """Information about a single tide event (internal representation)."""

    time: (
        datetime.datetime
    )  # Time of the tide (timezone-aware, in location's local timezone)
    type: TideType  # 'high' or 'low'
    prediction: float  # Height of the tide in feet
    # Additional fields from the NOAA API can be added as needed


@dataclass
class TideInfo:
    """Structured information about previous and next tides (internal)."""

    past_tides: List[TideEntry]  # The most recent tide
    next_tides: List[TideEntry]  # The next two upcoming tides


@dataclass
class CurrentInfo:
    """Structured information about water current prediction (internal)."""

    direction: CurrentDirection  # 'flooding' or 'ebbing'
    magnitude: float  # Current strength in knots
    magnitude_pct: float  # Relative magnitude percentage (0.0-1.0)
    state_description: str  # Human-readable description of current state


@dataclass
class LegacyChartInfo:
    """Structured information about a tide chart (internal)."""

    hours_since_last_tide: float
    last_tide_type: TideType
    chart_filename: str
    map_title: str


#############################################################
# API TYPES - Used for external API request/response models  #
#############################################################


class LocationInfo(BaseModel):
    """Location information for API responses."""

    code: str = Field(..., description="Location code (e.g., 'nyc')")
    name: str = Field(..., description="Display name of the location")
    swim_location: str = Field(..., description="Specific swimming location")


class TemperatureInfo(BaseModel):
    """Water temperature information for API responses."""

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the reading (in location's local timezone)",
    )
    water_temp: float = Field(..., description="Water temperature in degrees")
    units: str = Field("F", description="Temperature units (F for Fahrenheit)")


class ApiTideEntry(BaseModel):
    """Individual tide information for API responses."""

    time: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the tide event (in location's local timezone)",
    )
    type: str = Field(..., description="Type of tide ('high', 'low', or 'unknown')")
    prediction: float = Field(..., description="Height of tide in feet")


class TidesInfo(BaseModel):
    """Collection of tide information for API responses."""

    past: List[ApiTideEntry] = Field(..., description="Recently occurred tides")
    next: List[ApiTideEntry] = Field(..., description="Upcoming tides")


class CurrentPredictionInfo(BaseModel):
    """Current prediction information for API responses."""

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the prediction (in location's local timezone)",
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
    """Information about legacy tide charts for API responses."""

    hours_since_last_tide: float = Field(
        ..., description="Hours since the last tide event"
    )
    last_tide_type: str = Field(..., description="Type of last tide ('high' or 'low')")
    chart_filename: str = Field(..., description="Filename of the legacy chart")
    map_title: str = Field(..., description="Title for the legacy map")


#############################################################
# API RESPONSE MODELS - Complete response objects            #
#############################################################


class CurrentsResponse(BaseModel):
    """Complete response for currents API endpoint."""

    location: LocationInfo
    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the prediction (in location's local timezone)",
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

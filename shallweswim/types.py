"""Type definitions for shallweswim.

This module contains type definitions used throughout the application, separated into
two main categories:
1. Internal types - Used for internal data processing and storage
2. API types - Used for API request/response handling via Pydantic models
"""

# Standard library imports
import datetime
import enum
from typing import List, Optional, Dict

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict


#############################################################
# INTERNAL TYPES - Used for internal data processing         #
#############################################################


class CurrentDirection(enum.Enum):
    FLOODING = "flooding"
    EBBING = "ebbing"


class TideCategory(enum.Enum):
    LOW = "low"
    HIGH = "high"


class CurrentSystemType(enum.Enum):
    """Type of current system (reversing tidal or unidirectional river)."""

    TIDAL = "tidal"
    RIVER = "river"


class DataSourceType(enum.Enum):
    """Indicates whether data is based on a prediction or an observation."""

    PREDICTION = "prediction"
    OBSERVATION = "observation"


# Derive list for Pandera compatibility
TIDE_TYPE_CATEGORIES = [member.value for member in TideCategory]


class TideEntry(BaseModel):
    """Information about a single tide event (internal representation)."""

    time: (
        datetime.datetime
    )  # Time of the tide (timezone-aware, in location's local timezone)
    type: TideCategory  # TideCategory.LOW or TideCategory.HIGH
    prediction: float  # Height of the tide in feet
    # Additional fields from the NOAA API can be added as needed


class TideInfo(BaseModel):
    """Structured information about previous and next tides (internal)."""

    past_tides: List[TideEntry]  # The most recent tide
    next_tides: List[TideEntry]  # The next two upcoming tides


class CurrentInfo(BaseModel):
    """Structured information about the current water conditions (prediction or observation)."""

    # Indicates if the data is from a prediction or observation
    source_type: DataSourceType

    # Current direction (e.g., flooding, ebbing) or None for unidirectional systems
    direction: Optional[CurrentDirection]

    # Magnitude of the current in knots (adjusted for direction)
    magnitude: float
    magnitude_pct: float  # Relative magnitude percentage (0.0-1.0)
    state_description: str  # Human-readable description of current state
    timestamp: Optional[datetime.datetime] = (
        None  # Naive datetime for which the prediction was made
    )


class LegacyChartInfo(BaseModel):
    """Structured information about a tide chart (internal)."""

    hours_since_last_tide: float
    last_tide_type: Optional[TideCategory]
    chart_filename: str
    map_title: str


#############################################################
# API TYPES - Used for external API request/response models  #
#############################################################


class LocationInfo(BaseModel):
    """Location information for API responses."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Location code (e.g., 'nyc')")
    name: str = Field(..., description="Display name of the location")
    swim_location: str = Field(..., description="Specific swimming location")


class DataFrameSummary(BaseModel):
    """Provides summary statistics for a time-series DataFrame."""

    model_config = ConfigDict(extra="forbid")

    length: int = Field(..., description="Number of rows in the DataFrame.")
    width: int = Field(..., description="Number of columns in the DataFrame.")
    column_names: List[str] = Field(..., description="List of column names.")
    index_oldest: Optional[datetime.datetime] = Field(
        None,
        description="Oldest timestamp in the DataFrame index (timezone-naive). None if empty/non-datetime index.",
    )
    index_newest: Optional[datetime.datetime] = Field(
        None,
        description="Newest timestamp in the DataFrame index (timezone-naive). None if empty/non-datetime index.",
    )
    missing_values: Dict[str, int] = Field(
        ...,
        description="Dictionary mapping column names to the count of missing values.",
    )
    index_frequency: Optional[str] = Field(
        None,
        description="Inferred frequency of the DataFrame index (e.g., 'H', 'D', 'T'). None if irregular.",
    )
    memory_usage_bytes: int = Field(
        ...,
        description="Total memory usage of the DataFrame in bytes (including index, deep=True).",
    )


class TemperatureInfo(BaseModel):
    """Water temperature information for API responses."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the reading (in location's local timezone)",
    )
    water_temp: float = Field(..., description="Water temperature in degrees")
    units: str = Field("F", description="Temperature units (F for Fahrenheit)")
    station_name: Optional[str] = Field(
        None, description="Human-readable name of the temperature station"
    )


class ApiTideEntry(BaseModel):
    """Individual tide information for API responses."""

    model_config = ConfigDict(extra="forbid")

    time: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the tide event (in location's local timezone)",
    )
    type: str = Field(..., description="Type of tide ('high', 'low', or 'unknown')")
    prediction: float = Field(..., description="Height of tide in feet")


class TidesInfo(BaseModel):
    """Collection of tide information for API responses."""

    model_config = ConfigDict(extra="forbid")

    past: List[ApiTideEntry] = Field(..., description="Recently occurred tides")
    next: List[ApiTideEntry] = Field(..., description="Upcoming tides")


class CurrentPredictionInfo(BaseModel):
    """Current prediction information for API responses."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the prediction (in location's local timezone)",
    )
    direction: str = Field(
        ..., description="Direction of current (Enum: CurrentDirection)"
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

    model_config = ConfigDict(extra="forbid")

    hours_since_last_tide: float = Field(
        ..., description="Hours since the last tide event"
    )
    last_tide_type: str = Field(..., description="Type of last tide ('high' or 'low')")
    chart_filename: str = Field(..., description="Filename of the legacy chart")
    map_title: str = Field(..., description="Title for the legacy map")


class FeedStatus(BaseModel):
    """Represents the status of a single data feed."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Class name of the Feed")
    location: str = Field(..., description="Location code associated with the feed")
    fetch_timestamp: Optional[datetime.datetime] = Field(
        None, description="Timestamp of the last successful data fetch (naive UTC)"
    )
    age_seconds: Optional[float] = Field(
        None, description="Data age in seconds at the time of status check"
    )
    is_expired: bool = Field(..., description="Whether the data is considered expired")
    expiration_seconds: Optional[float] = Field(
        None, description="Configured expiration time in seconds for the feed"
    )
    is_healthy: bool = Field(
        ..., description="Whether the data is recent enough to be considered healthy"
    )
    data_summary: Optional[DataFrameSummary] = Field(
        None, description="Summary statistics of the feed's DataFrame, if available"
    )
    error: Optional[str] = Field(
        None, description="Last error message encountered by the feed, if any"
    )


class LocationStatus(BaseModel):
    """Represents the status of all feeds for a specific location."""

    feeds: Dict[str, FeedStatus] = Field(
        ..., description="Dictionary mapping feed names to their FeedStatus objects."
    )


#############################################################
# API RESPONSE MODELS - Complete response objects            #
#############################################################


class CurrentsResponse(BaseModel):
    """Complete response for currents API endpoint."""

    model_config = ConfigDict(extra="forbid")

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
    """Complete response for location conditions endpoint.

    Fields are conditionally included based on the location's configuration:
    - temperature: Only included if the location has a temperature source with live_enabled=True
    - tides: Only included if the location has a tide source
    """

    model_config = ConfigDict(extra="forbid")

    location: LocationInfo
    temperature: Optional[TemperatureInfo] = Field(
        None, description="Water temperature information (if available)"
    )
    tides: Optional[TidesInfo] = Field(
        None, description="Tide information (if available)"
    )

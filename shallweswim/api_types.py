"""API Type definitions for shallweswim.

This module contains type definitions used for API request/response handling
via Pydantic models. Internal types are defined in types.py.
"""

# Standard library imports
import datetime

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field

# Local imports
from shallweswim.types import (
    CurrentDirection,
    CurrentPhase,
    CurrentStrength,
    CurrentTrend,
    DataSourceType,
    TideCategory,
)

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
    column_names: list[str] = Field(..., description="List of column names.")
    index_oldest: datetime.datetime | None = Field(
        None,
        description="Oldest timestamp in the DataFrame index (timezone-naive). None if empty/non-datetime index.",
    )
    index_newest: datetime.datetime | None = Field(
        None,
        description="Newest timestamp in the DataFrame index (timezone-naive). None if empty/non-datetime index.",
    )
    missing_values: dict[str, int] = Field(
        ...,
        description="Dictionary mapping column names to the count of missing values.",
    )
    index_frequency: str | None = Field(
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
    station_name: str | None = Field(
        None, description="Human-readable name of the temperature station"
    )


class TideEntry(BaseModel):
    """Individual tide information for API responses."""

    model_config = ConfigDict(extra="forbid")

    time: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the tide event (in location's local timezone)",
    )
    type: TideCategory = Field(..., description="Type of tide (Enum: TideCategory)")
    prediction: float = Field(..., description="Height of tide in feet")


class TideInfo(BaseModel):
    """Collection of tide information for API responses."""

    model_config = ConfigDict(extra="forbid")

    past: list[TideEntry] = Field(..., description="Recently occurred tides")
    next: list[TideEntry] = Field(..., description="Upcoming tides")


class CurrentInfo(BaseModel):
    """Current prediction/observation information for API responses."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of the prediction (in location's local timezone)",
    )
    direction: CurrentDirection | None = Field(
        None,
        description="Direction of current (Enum: CurrentDirection, null if non-tidal)",
    )
    phase: CurrentPhase | None = Field(
        None,
        description=(
            "Compact current phase for displays. Tidal prediction values are "
            "flood, ebb, slack_before_flood, slack_before_ebb, or slack. Slack "
            "phases are assigned when absolute current magnitude is below 0.2 knots; "
            "slack_before_* indicates the next non-slack direction predicted by the curve."
        ),
    )
    strength: CurrentStrength | None = Field(
        None,
        description=(
            "Cycle-relative current strength for non-slack tidal predictions "
            "(light, moderate, or strong). Null for slack or non-tidal currents."
        ),
    )
    trend: CurrentTrend | None = Field(
        None,
        description=(
            "Current trend for non-slack tidal predictions "
            "(building, easing, or steady). Null for slack or non-tidal currents."
        ),
    )
    magnitude: float = Field(..., description="Current strength in knots")
    magnitude_pct: float | None = Field(
        None,
        description=(
            "Relative current strength (0.0-1.0, null if non-tidal), normalized "
            "against the nearest local flood/ebb peak in the available prediction curve."
        ),
    )
    state_description: str | None = Field(
        None,
        description=(
            "Display-ready current state phrase, such as 'strong ebb and building' "
            "or 'slack before flood' (null if non-tidal)."
        ),
    )
    source_type: DataSourceType = Field(
        ...,
        description="Indicates if data is prediction or observation (Enum: DataSourceType)",
    )


class LegacyChartInfo(BaseModel):
    """Information about legacy tide charts for API responses."""

    model_config = ConfigDict(extra="forbid")

    hours_since_last_tide: float = Field(
        ..., description="Hours since the last tide event"
    )
    last_tide_type: TideCategory = Field(
        ..., description="Type of last tide (Enum: TideCategory)"
    )
    chart_filename: str = Field(..., description="Filename of the legacy chart")
    map_title: str = Field(..., description="Title for the legacy map")


class FeedStatus(BaseModel):
    """Represents the status of a single data feed."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Class name of the Feed")
    location: str = Field(..., description="Location code associated with the feed")
    fetch_timestamp: datetime.datetime | None = Field(
        None, description="Timestamp of the last successful data fetch (naive UTC)"
    )
    age_seconds: float | None = Field(
        None, description="Data age in seconds at the time of status check"
    )
    is_expired: bool = Field(..., description="Whether the data is considered expired")
    expiration_seconds: float | None = Field(
        None, description="Configured expiration time in seconds for the feed"
    )
    is_healthy: bool = Field(
        ..., description="Whether the data is recent enough to be considered healthy"
    )
    data_summary: DataFrameSummary | None = Field(
        None, description="Summary statistics of the feed's DataFrame, if available"
    )
    error: str | None = Field(
        None, description="Last error message encountered by the feed, if any"
    )


class LocationStatus(BaseModel):
    """Represents the status of all feeds for a specific location."""

    feeds: dict[str, FeedStatus] = Field(
        ..., description="Dictionary mapping feed names to their FeedStatus objects."
    )


class LocationSummary(BaseModel):
    """Summary information for a swimming location."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="3-letter location code")
    name: str = Field(..., description="City or region name")
    swim_location: str = Field(..., description="Specific swimming spot name")
    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    has_data: bool = Field(..., description="Whether the location can serve data")


class AppFeatureFlags(BaseModel):
    """Presentation feature flags for a location in the React app."""

    model_config = ConfigDict(extra="forbid")

    temperature: bool = Field(..., description="Whether to show temperature UI")
    tides: bool = Field(..., description="Whether to show tide UI")
    currents: bool = Field(..., description="Whether to show current UI")
    webcam: bool = Field(..., description="Whether to show webcam UI")
    transit: bool = Field(..., description="Whether to show transit UI")
    windy: bool = Field(..., description="Whether to show Windy forecast UI")


class AppSourceCitations(BaseModel):
    """Trusted HTML source citations for a location."""

    model_config = ConfigDict(extra="forbid")

    temperature: str | None = Field(
        None, description="Trusted HTML citation for temperature data"
    )
    tides: str | None = Field(None, description="Trusted HTML citation for tide data")
    currents: str | None = Field(
        None, description="Trusted HTML citation for current data"
    )


class AppLocationMetadata(BaseModel):
    """Presentation metadata for a swimming location."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="3-letter location code")
    name: str = Field(..., description="City or region name")
    nav_label: str = Field(..., description="Short label for location navigation")
    swim_location: str = Field(..., description="Specific swimming spot name")
    swim_location_link: str = Field(..., description="URL for the swimming spot")
    description: str = Field(..., description="Description of the swimming location")
    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    timezone: str = Field(..., description="IANA timezone name")
    features: AppFeatureFlags = Field(..., description="Enabled presentation features")
    citations: AppSourceCitations = Field(
        ..., description="Trusted HTML citations for configured data sources"
    )


class AppManifestMetadata(BaseModel):
    """Installable web app manifest metadata exposed to the frontend."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Full web app name")
    short_name: str = Field(..., description="Short web app name")
    start_url: str = Field(..., description="Manifest start URL")
    scope: str = Field(..., description="Manifest scope")
    display: str = Field(..., description="Manifest display mode")
    theme_color: str = Field(..., description="Theme color")
    background_color: str = Field(..., description="Background color")


class YouTubeLiveConfig(BaseModel):
    """YouTube live embed configuration for a frontend integration."""

    model_config = ConfigDict(extra="forbid")

    channel_id: str = Field(..., description="YouTube channel ID")
    embed_url: str = Field(..., description="YouTube live embed URL")
    watch_url: str = Field(..., description="YouTube live watch URL")


class TransitRouteConfig(BaseModel):
    """Transit route metadata for frontend status cards."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="User-facing route label")
    goodservice_route_id: str = Field(..., description="GoodService route ID")
    icon_url: str | None = Field(None, description="Optional route icon URL")


class AppExternalIntegrations(BaseModel):
    """External presentation integrations used by the React app."""

    model_config = ConfigDict(extra="forbid")

    youtube_live: YouTubeLiveConfig | None = Field(
        None, description="YouTube live embed configuration"
    )
    transit_routes: list[TransitRouteConfig] = Field(
        default_factory=list, description="Transit routes to show for a location"
    )


class AppBootstrapLocation(BaseModel):
    """Complete frontend bootstrap data for one location."""

    model_config = ConfigDict(extra="forbid")

    metadata: AppLocationMetadata
    integrations: AppExternalIntegrations


class AppBootstrapResponse(BaseModel):
    """Frontend bootstrap payload for app presentation configuration."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = Field(..., description="Full application name")
    short_name: str = Field(..., description="Short application name")
    default_location_code: str = Field(..., description="Default location code")
    location_order: list[str] = Field(..., description="Ordered location codes")
    manifest: AppManifestMetadata = Field(
        ..., description="Installable app manifest metadata"
    )
    locations: dict[str, AppBootstrapLocation] = Field(
        ..., description="Per-location frontend bootstrap metadata"
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
    current: CurrentInfo
    legacy_chart: LegacyChartInfo | None = Field(
        default=None,
        description="Legacy chart info (only for locations with chart assets)",
    )
    current_chart_filename: str | None = Field(
        default=None,
        description="Filename of the current chart image (only for locations with chart assets)",
    )
    navigation: dict[str, object] = Field(
        ..., description="Navigation parameters for time shifting"
    )


class LocationConditions(BaseModel):
    """Complete response for location conditions endpoint.

    Fields are conditionally included based on the location's configuration:
    - temperature: Only included if the location has a temperature source with live_enabled=True
    - tides: Only included if the location has a tide source
    - current: Only included if the location has a current source
    """

    model_config = ConfigDict(extra="forbid")

    location: LocationInfo
    temperature: TemperatureInfo | None = Field(
        None, description="Water temperature information (if available)"
    )
    tides: TideInfo | None = Field(None, description="Tide information (if available)")
    current: CurrentInfo | None = Field(
        None, description="Current information (if available)"
    )

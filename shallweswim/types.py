"""Internal Type definitions for shallweswim.

This module contains type definitions used for internal data processing and storage.
API-related types are defined in api_types.py.
"""

# Standard library imports
import datetime
import enum
from dataclasses import dataclass

#############################################################
# INTERNAL TYPES - Used for internal data processing         #
#############################################################


class CurrentDirection(enum.Enum):
    FLOODING = "flooding"
    EBBING = "ebbing"


class CurrentPhase(enum.Enum):
    """Compact current phase for API consumers and displays."""

    FLOOD = "flood"
    EBB = "ebb"
    SLACK_BEFORE_FLOOD = "slack_before_flood"
    SLACK_BEFORE_EBB = "slack_before_ebb"
    SLACK = "slack"


class CurrentStrength(enum.Enum):
    """Cycle-relative current strength for displays."""

    LIGHT = "light"
    MODERATE = "moderate"
    STRONG = "strong"


class CurrentTrend(enum.Enum):
    """Whether the current is building or easing."""

    BUILDING = "building"
    EASING = "easing"
    STEADY = "steady"


class TideCategory(enum.Enum):
    LOW = "low"
    HIGH = "high"


class TideTrend(enum.Enum):
    """Whether the tide height is rising or falling."""

    RISING = "rising"
    FALLING = "falling"
    STEADY = "steady"


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


@dataclass
class TemperatureReading:
    """Structured information about a water temperature reading."""

    timestamp: (
        datetime.datetime
    )  # Time of the reading (timezone-naive, in location's local timezone)
    temperature: float  # Water temperature in degrees Celsius


@dataclass
class TideEntry:
    """Information about a single tide event (internal representation)."""

    time: (
        datetime.datetime
    )  # Time of the tide (timezone-aware, in location's local timezone)
    type: TideCategory  # TideCategory.LOW or TideCategory.HIGH
    prediction: float  # Height of the tide in feet
    # Additional fields from the NOAA API can be added as needed


@dataclass
class TideState:
    """Point-in-time estimated tide state."""

    timestamp: datetime.datetime
    estimated_height: float
    trend: TideTrend
    units: str = "ft"
    height_pct: float | None = None


@dataclass
class TideInfo:
    """Structured information about past, future, and current tide state."""

    past: list[TideEntry]  # The most recent tide
    next: list[TideEntry]  # The next two upcoming tides
    state: TideState | None = None  # Point-in-time estimated tide state


@dataclass
class CurrentInfo:
    """Structured information about the current water conditions (prediction or observation)."""

    # Naive datetime for which the prediction was made or observation was recorded.
    timestamp: datetime.datetime

    # Indicates if the data is from a prediction or observation
    source_type: DataSourceType

    # Magnitude of the current in knots (adjusted for direction)
    magnitude: float

    # Current direction (e.g., flooding, ebbing) or None for unidirectional systems
    direction: CurrentDirection | None = None

    # Compact phase for displays (e.g., flood, ebb, slack_before_flood)
    phase: CurrentPhase | None = None

    # Cycle-relative strength and trend for tidal predictions
    strength: CurrentStrength | None = None
    trend: CurrentTrend | None = None

    magnitude_pct: float | None = None
    state_description: str | None = None


@dataclass
class LegacyChartInfo:
    """Structured information about a tide chart (internal)."""

    hours_since_last_tide: float
    last_tide_type: TideCategory | None
    chart_filename: str
    map_title: str

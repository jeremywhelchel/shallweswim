"""Internal Type definitions for shallweswim.

This module contains type definitions used for internal data processing and storage.
API-related types are defined in api_types.py.
"""

# Standard library imports
import datetime
import enum
from typing import List, Optional
from dataclasses import dataclass

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
class TideInfo:
    """Structured information about past and future tides."""

    past: List[TideEntry]  # The most recent tide
    next: List[TideEntry]  # The next two upcoming tides


@dataclass
class CurrentInfo:
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


@dataclass
class LegacyChartInfo:
    """Structured information about a tide chart (internal)."""

    hours_since_last_tide: float
    last_tide_type: Optional[TideCategory]
    chart_filename: str
    map_title: str

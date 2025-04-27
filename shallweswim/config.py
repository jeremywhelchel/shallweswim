"""Application configuration.

This module defines the configuration for all supported swimming locations,
including geographic coordinates, NOAA CO-OPS station IDs, and descriptive information.

The configuration system is built around the LocationConfig class, which stores all
relevant information about swimming locations. The most important elements include:

1. Location identification: 3-letter code, name, and description
2. Geographic coordinates: latitude and longitude for weather services
3. NOAA CO-OPS Data Sources: Station IDs for fetching tides, temperature, and currents data
4. Timezone: For correct time-based display of conditions

The configuration uses specialized Pydantic models (TempSource/CoopsTempSource, CoopsTideSource, CoopsCurrentsSource)
to define data sources for each type of measurement, providing more flexibility and abstraction.

Configurations for all supported locations are stored in CONFIG_LIST and can be
accessed via the Get() function using the location's 3-letter code.
"""

# Standard library imports
import datetime
from datetime import tzinfo
from typing import Annotated, Dict, List, Optional, Tuple

# Third-party imports
import pytz
from pydantic import BaseModel, Field

# Local imports
from shallweswim import util


class TempSource(BaseModel, frozen=True):
    """Base configuration for temperature data source.

    Abstract base class for all temperature data sources.
    """

    name: Annotated[
        Optional[str],
        Field(
            description="Human-readable name of the temperature source (e.g., 'The Battery, NY')"
        ),
    ] = None

    outliers: Annotated[
        List[str],
        Field(
            description="List of timestamps (YYYY-MM-DD HH:MM:SS format) with erroneous temperature data to remove"
        ),
    ] = []


class CoopsTempSource(TempSource, frozen=True):
    """NOAA CO-OPS specific temperature data source configuration.

    Defines the NOAA CO-OPS station for fetching water/air temperature data.
    """

    station: Annotated[
        int,
        Field(
            ge=1000000,
            le=9999999,
            description="NOAA temperature station ID (7 digits) for fetching water/air temperature (e.g., 8518750 for NYC Battery)",
        ),
    ]


class CoopsTideSource(BaseModel, frozen=True):
    """Configuration for tide data source.

    Defines the NOAA CO-OPS station for fetching tide predictions.
    """

    station: Annotated[
        int,
        Field(
            ge=1000000,
            le=9999999,
            description="NOAA tide station ID (7 digits) for fetching tide predictions (e.g., 8517741 for Coney Island)",
        ),
    ]

    station_name: Annotated[
        Optional[str],
        Field(
            description="Human-readable name of the tide station (e.g., 'Coney Island, NY')"
        ),
    ] = None


class CoopsCurrentsSource(BaseModel, frozen=True):
    """Configuration for currents data source.

    Defines the NOAA CO-OPS station(s) for fetching water current predictions.
    """

    stations: Annotated[
        List[Annotated[str, Field(min_length=1)]],
        Field(
            description="Current stations for water current speed and direction (strings like 'ACT3876', unlike temp/tide stations)"
        ),
    ]


class LocationConfig(BaseModel, frozen=True):
    """Configuration for a swimming location.

    This model defines all the information needed to display conditions for
    a specific open water swimming location. LocationConfig objects are immutable
    (frozen=True) to prevent accidental modification after creation.

    Each location is identified by a unique 3-letter code and includes
    geographic coordinates, descriptive information, and references to
    NOAA data stations that provide tide, temperature, and current information.

    The NOAA CO-OPS station IDs are used by the CO-OPS client to fetch real-time and
    historical data about water temperature, tides, and currents. Not all locations
    have all types of data available.

    Typical usage:
        # Get a location config by its 3-letter code
        nyc_config = config.Get("nyc")

        # Access properties
        lat, lon = nyc_config.coordinates
        temp_station = nyc_config.temp_station  # For NOAA API calls
    """

    model_config = {"arbitrary_types_allowed": True}

    code: Annotated[
        str,
        Field(
            min_length=3,
            max_length=3,
            pattern=r"^[a-z]{3}$",
            description="3-letter lowercase location code that uniquely identifies this location",
        ),
    ]
    name: Annotated[
        str,
        Field(description="City or region name (e.g., 'New York', 'San Francisco')"),
    ]

    swim_location: Annotated[
        str,
        Field(
            description="Specific swimming spot name (e.g., 'Grimaldo's Chair', 'La Jolla Cove')"
        ),
    ]
    swim_location_link: Annotated[
        str, Field(description="URL with information about the swim location")
    ]

    description: Annotated[
        str,
        Field(description="Detailed description of the swimming conditions/location"),
    ]
    latitude: Annotated[
        float,
        Field(ge=-90, le=90, description="Latitude in decimal degrees (-90 to 90)"),
    ]
    longitude: Annotated[
        float,
        Field(
            ge=-180, le=180, description="Longitude in decimal degrees (-180 to 180)"
        ),
    ]
    timezone: Annotated[
        tzinfo,
        Field(
            description="Timezone object for correctly displaying time-based data such as tide charts (e.g., pytz.timezone('US/Eastern'))"
        ),
    ]

    # Data source configurations
    temp_source: Annotated[
        Optional[TempSource],
        Field(description="Configuration for temperature data source"),
    ] = None

    tide_source: Annotated[
        Optional[CoopsTideSource],
        Field(description="Configuration for tide data source"),
    ] = None

    currents_source: Annotated[
        Optional[CoopsCurrentsSource],
        Field(description="Configuration for currents data source"),
    ] = None

    @property
    def coordinates(self) -> Tuple[float, float]:
        """Return the location's coordinates as a (latitude, longitude) tuple.

        This is convenient for passing to mapping APIs or weather services that
        require coordinates as a tuple.

        Returns:
            A tuple of (latitude, longitude) as floats
        """
        return (self.latitude, self.longitude)

    @property
    def display_name(self) -> str:
        """Return a formatted display name for the location.

        Creates a user-friendly display name that combines the city/region name
        with the specific swimming location.

        Returns:
            A string in the format "City (Swim Location)"
        """
        return f"{self.name} ({self.swim_location})"

    def local_now(self) -> datetime.datetime:
        """Return the current time in the location's timezone as a naive datetime.

        This method converts the current UTC time to the location's timezone and
        strips the timezone information to provide a naive datetime that represents
        the current local time at this location.

        Returns:
            A naive datetime object representing the current local time
        """
        # Get current UTC time without timezone info
        now_utc = util.utc_now()

        # Add UTC timezone info, convert to location timezone, then strip tzinfo
        now_utc_with_tz = now_utc.replace(tzinfo=datetime.timezone.utc)
        local_now = now_utc_with_tz.astimezone(self.timezone)

        # Return naive datetime in local time
        return local_now.replace(tzinfo=None)


CONFIG_LIST = [
    LocationConfig(
        code="nyc",
        name="New York",
        swim_location="Grimaldo's Chair",
        swim_location_link="https://cibbows.org/about/essentials/",
        latitude=40.573,
        longitude=-73.954,
        timezone=pytz.timezone("US/Eastern"),
        temp_source=CoopsTempSource(
            station=8518750,
            name="The Battery, NY",
            outliers=[
                "2017-05-23 12:00:00",
                "2020-05-22 13:00:00",
            ],
        ),
        tide_source=CoopsTideSource(
            station=8517741,
            station_name="Coney Island, NY",
        ),
        currents_source=CoopsCurrentsSource(
            stations=[
                "ACT3876",  # Coney Island Channel
                "NYH1905",  # Rockaway Inslet
            ],
        ),
        description="Coney Island Brighton Beach open water swimming conditions",
    ),
    LocationConfig(
        code="san",
        name="San Diego",
        swim_location="La Jolla Cove",
        swim_location_link="https://www.lajollacoveswimclub.com/page-1518748",
        latitude=32.850,
        longitude=-117.272,
        timezone=pytz.timezone("US/Pacific"),
        temp_source=CoopsTempSource(
            station=9410230,
            name="La Jolla, CA",
        ),
        tide_source=CoopsTideSource(
            station=9410230,
            station_name="La Jolla, CA",
        ),
        description="La Jolla Cove open water swimming conditions",
    ),
    # LocationConfig(
    #     code="chi",
    #     name="Chicago",
    #     swim_location="TBD",
    #     swim_location_link="TBD",
    #     latitude=41.894,
    #     longitude=-87.613,
    #     timezone=pytz.timezone("US/Central"),
    #     # temp_source=CoopsTempSource(
    #     #     station=45198,
    #     #     name="TBD",
    #     # ),
    #     # tide_source=CoopsTideSource(
    #     #     station=None,
    #     #     station_name="",
    #     # ),
    #     description="Chicago TBD open water swimming conditions",
    # ),
    # LocationConfig(
    #    # TODO more SF stuff can be added. see here: https://dolphinclub.org/weather/
    #    code="sfo",
    #    name="San Francisco",
    #    swim_location="San Francisco Aquatic Park",
    #    swim_location_link="https://serc.com/swimming/swimming-in-aquatic-park/",
    #    latitude=37.808,
    #    longitude=-122.426,
    #    timezone=pytz.timezone("US/Pacific"),
    #    # Note that North Point Pier temp (stn 9414305) is a operational forecast (OFS).
    #    # It is not a live reading (and not available via the same API), so we don't use it.
    #    # The San Francisco, CA - Station ID: 9414290 has been disabled with this message
    #    # Disabled - 2025-01-17 02:01:00, Suspect Data - Data failed to meet QC standards - under review.
    #    temp_source=CoopsTempSource(
    #        station=9414769,
    #        name="San Francisco, CA",
    #    ),
    #    tide_source=CoopsTideSource(
    #        station=9414305,
    #        station_name="North Point Pier",
    #    ),
    #    description="San Francisco Aquatic Park open water swimming conditions",
    #    # webcam https://dolphinclub.org/weather/ (code in page JS source...)
    # ),
    # LocationConfig(
    #    code="sdf",
    #    name="Louisville",
    #    swim_location="Louisville Community Boathouse",
    #    swim_location_link="https://www.kylmsc.org/rats",
    #    latitude=38.2647556,
    #    longitude=-85.7323204,
    #    # XXX windy "waves" mode inapplicable here
    #    timezone=pytz.timezone("US/Eastern"),
    #    description="Louisville Kentucky open water swimming conditions",
    #    # TODO: add this
    #    # - temp data
    #    #    water velocity (mph)
    #    #    https://colab.research.google.com/drive/17-oyc95BBUUI3g1GBR5QpACAm-4G5vBk#scrollTo=vMXIJh8CCQCF
    #    # - webcam:
    #    #    https://www.earthcam.com/usa/kentucky/louisville/?cam=ohioriver
    #    #    https://ohiorivercam.com/
    # ),
]
# Build lookup dictionaries - use lowercase keys for case-insensitive lookup
CONFIGS: Dict[str, LocationConfig] = {c.code.lower(): c for c in CONFIG_LIST}


def get(code: str) -> Optional[LocationConfig]:
    """Get location config by 3-letter code.

    The lookup is case-insensitive, so "NYC", "nyc", and "Nyc" all retrieve
    the same configuration. This function is the primary way to access
    location configurations throughout the application.

    Args:
        code: 3-letter location code (e.g., "nyc", "sfo", "san")

    Returns:
        LocationConfig if found, None otherwise
    """
    return CONFIGS.get(code)

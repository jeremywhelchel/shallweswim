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

Configurations for all supported locations are stored internally and can be
accessed via the Get() function using the location's 3-letter code.
"""

# Standard library imports
import abc
import datetime
from datetime import tzinfo
from types import MappingProxyType
from typing import Annotated, List, Optional, Tuple

# Third-party imports
import pytz
from pydantic import BaseModel, Field, ConfigDict

# Local imports
from shallweswim import types
from shallweswim import util


class TempSource(BaseModel, abc.ABC, frozen=True):
    """Base configuration for temperature data source.

    Abstract base class for all temperature data sources.
    """

    model_config = ConfigDict(extra="forbid")

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

    live_enabled: Annotated[
        bool,
        Field(
            description="Whether to enable live temperature data fetching for this source"
        ),
    ] = True

    historic_enabled: Annotated[
        bool,
        Field(
            description="Whether to enable historical temperature data fetching for this source"
        ),
    ] = True

    start_year: Annotated[
        Optional[int],
        Field(description="Starting year for historical temperature data."),
    ] = None

    end_year: Annotated[
        Optional[int],
        Field(
            description="Ending year for historical temperature data. If not provided, defaults to current year."
        ),
    ] = None

    @property
    @abc.abstractmethod
    def citation(self) -> str:
        """Return an HTML snippet with source citation information.

        This property should be implemented by all subclasses to provide
        appropriate citation information for the data source. The returned
        string should be a valid HTML snippet that can be included in a webpage.

        Returns:
            HTML string with citation information
        """
        pass


class CoopsTempSource(TempSource, frozen=True):
    """NOAA CO-OPS specific temperature data source configuration.

    Defines the NOAA CO-OPS station for fetching water/air temperature data.
    """

    model_config = ConfigDict(extra="forbid")

    station: Annotated[
        int,
        Field(
            ge=1000000,
            le=9999999,
            description="NOAA temperature station ID (7 digits) for fetching water/air temperature (e.g., 8518750 for NYC Battery)",
        ),
    ]

    @property
    def citation(self) -> str:
        """Return an HTML snippet with NOAA CO-OPS citation information.

        Returns:
            HTML string with citation information for NOAA CO-OPS data
        """
        station_url = (
            f"https://tidesandcurrents.noaa.gov/stationhome.html?id={self.station}"
        )
        return f"Water temperature data provided by <a href=\"{station_url}\" target=\"_blank\">NOAA CO-OPS Station {self.station}</a> ({self.name or 'Unknown'})"


class CoopsTideSource(BaseModel, frozen=True):
    """Configuration for tide data source.

    Defines the NOAA CO-OPS station for fetching tide predictions.
    """

    model_config = ConfigDict(extra="forbid")

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

    @property
    def citation(self) -> str:
        """Return an HTML snippet with NOAA CO-OPS tide citation information.

        Returns:
            HTML string with citation information for NOAA CO-OPS tide data
        """
        station_url = (
            f"https://tidesandcurrents.noaa.gov/stationhome.html?id={self.station}"
        )
        return f"Tide data provided by <a href=\"{station_url}\" target=\"_blank\">NOAA CO-OPS Station {self.station}</a> ({self.station_name or 'Unknown'})"


class CurrentsSourceConfigBase(BaseModel, abc.ABC, frozen=True):
    """Base configuration for currents data sources.

    Abstract base class for all currents data sources.
    """

    model_config = ConfigDict(extra="forbid")

    system_type: Annotated[
        types.CurrentSystemType,
        Field(
            description="Type of current system (tidal with reversals or unidirectional river)"
        ),
    ] = types.CurrentSystemType.TIDAL

    @property
    @abc.abstractmethod
    def citation(self) -> str:
        """Return an HTML snippet with source citation information."""
        pass


class CoopsCurrentsSource(CurrentsSourceConfigBase, frozen=True):
    """Configuration for currents data source.

    Defines the NOAA CO-OPS stations for fetching current predictions.
    """

    model_config = ConfigDict(extra="forbid")

    stations: Annotated[
        List[Annotated[str, Field(min_length=1)]],
        Field(
            description="Current stations for water current speed and direction (strings like 'ACT3876', unlike temp/tide stations)"
        ),
    ]

    @property
    def citation(self) -> str:
        """Return an HTML snippet with NOAA CO-OPS currents citation information.

        Returns:
            HTML string with citation information for NOAA CO-OPS currents data
        """
        if len(self.stations) == 1:
            station_id = self.stations[0]
            station_url = f"https://prod.tidesandcurrents.noaa.gov/noaacurrents/predictions.html?id={station_id}_1"
            return f'Current data provided by <a href="{station_url}" target="_blank">NOAA CO-OPS Station {station_id}</a>'
        else:
            station_links = []
            for station_id in self.stations:
                station_url = f"https://prod.tidesandcurrents.noaa.gov/noaacurrents/predictions.html?id={station_id}_1"
                station_links.append(
                    f'<a href="{station_url}" target="_blank">{station_id}</a>'
                )

            return f"Current data provided by NOAA CO-OPS Stations: {', '.join(station_links)}"


class NdbcTempSource(TempSource, frozen=True):
    """NDBC specific temperature data source configuration.

    Defines the NDBC buoy or station for fetching sea temperature and
    water temperature data from buoys and coastal stations.
    """

    model_config = ConfigDict(extra="forbid")

    station: Annotated[
        str,
        Field(
            min_length=4,
            max_length=5,
            description="NDBC station ID, either 5 digits (e.g., '41001' for offshore buoys) or 4 characters (e.g., 'brhc3' for C-MAN stations)",
        ),
    ]

    @property
    def citation(self) -> str:
        """Return an HTML snippet with NDBC citation information.

        Returns:
            HTML string with citation information for NDBC data
        """
        station_url = (
            f"https://www.ndbc.noaa.gov/station_page.php?station={self.station}"
        )
        return f"Water temperature data provided by <a href=\"{station_url}\" target=\"_blank\">NDBC Station {self.station}</a> ({self.name or 'Unknown'})"


class NwisTempSource(TempSource, frozen=True):
    """USGS NWIS specific temperature data source configuration.

    Defines the USGS National Water Information System (NWIS) site for fetching
    water temperature data from rivers, lakes, and other water bodies.
    """

    model_config = ConfigDict(extra="forbid")

    site_no: Annotated[
        str,
        Field(
            min_length=8,
            max_length=15,
            description="USGS site number (e.g., '01646500' for Potomac River at Little Falls near Washington, DC)",
        ),
    ]

    parameter_cd: Annotated[
        str,
        Field(
            default="00010",
            description="USGS parameter code for water temperature. Default is '00010', but some stations may use '00011'.",
        ),
    ] = "00010"

    @property
    def citation(self) -> str:
        """Return an HTML snippet with USGS NWIS citation information.

        Returns:
            HTML string with citation information for USGS NWIS data
        """
        site_url = f"https://waterdata.usgs.gov/monitoring-location/{self.site_no}/"
        return f"Water temperature data provided by <a href=\"{site_url}\" target=\"_blank\">USGS NWIS Site {self.site_no}</a> ({self.name or 'Unknown'})"


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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
        Optional[CurrentsSourceConfigBase],
        Field(description="Configuration for currents data source"),
    ] = None

    enabled: Annotated[
        bool,
        Field(
            description="Whether this location is enabled and available in the application"
        ),
    ] = True

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


_CONFIG_LIST = [
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
    LocationConfig(
        code="chi",
        name="Chicago",
        swim_location="Ohio Street Beach",
        swim_location_link="https://www.chicagoparkdistrict.com/parks-facilities/ohio-street-beach",
        latitude=41.894,
        longitude=-87.613,
        timezone=pytz.timezone("US/Central"),
        temp_source=NdbcTempSource(
            station="45198",
            name="Chicago Buoy",
            # Currently the only data is available for this range.
            live_enabled=False,
            start_year=2021,
            end_year=2024,
        ),
        # temp_source=NdbcTempSource(
        #    station="45198",
        #    name="Ohio Street Beach",
        #    live_enabled=False,
        # ),
        # temp_source=NdbcTempSource(
        #     station="45007",
        #     name="South Michigan",
        # ),
        # XXX Tides?
        description="Chicago Ohio Street Beach open water swimming conditions",
    ),
    LocationConfig(
        # TODO more SF stuff can be added. see here: https://dolphinclub.org/weather/
        code="sfo",
        name="San Francisco",
        swim_location="Aquatic Park",
        swim_location_link="https://serc.com/swimming/swimming-in-aquatic-park/",
        latitude=37.808,
        longitude=-122.426,
        timezone=pytz.timezone("US/Pacific"),
        # The San Francisco, CA - Station ID: 9414290 is not currently available.
        # Disabled - 2025-01-17 02:01:00, Suspect Data - Data failed to meet QC standards - under review.
        # temp_source=CoopsTempSource(
        #     station=9414290,
        #     name="San Francisco, CA",
        #     live_enabled=False,
        # ),
        # Alternative buoy that's further away. Note that this could be as much
        # as 5 degrees off the Bay location!
        temp_source=NdbcTempSource(
            station="46237",
            name="San Francisco Bar Buoy",
        ),
        tide_source=CoopsTideSource(
            station=9414305,
            station_name="North Point Pier",
        ),
        description="San Francisco Aquatic Park open water swimming conditions",
        # webcam https://dolphinclub.org/weather/ (code in page JS source...)
    ),
    LocationConfig(
        code="sdf",
        name="Louisville",
        swim_location="Community Boathouse",
        swim_location_link="https://www.kylmsc.org/rats",
        latitude=38.264,
        longitude=-85.732,
        timezone=pytz.timezone("US/Eastern"),
        # Ohio River Water Tower
        # https://waterdata.usgs.gov/monitoring-location/03292494/#dataTypeId=continuous-00011-0&period=P365D
        temp_source=NwisTempSource(
            site_no="03292494",
            parameter_cd="00011",  # Water temperature
            name="Ohio River at Water Tower",
            # It appears that this source may only provide ~120 days worth of data.
            start_year=2025,
        ),
        description="Louisville Kentucky open water swimming conditions",
        # TODO:
        # - Add webcam:
        #   https://www.earthcam.com/usa/kentucky/louisville/?cam=ohioriver
        #   https://ohiorivercam.com/
        # - Fix Windy embed mode. "waves" isnt relevant here.
    ),
    LocationConfig(
        code="aus",
        name="Austin",
        swim_location="Barton Springs",
        swim_location_link="https://www.austintexas.gov/department/barton-springs-pool",
        latitude=30.2639,
        longitude=-97.77,
        timezone=pytz.timezone("US/Central"),
        temp_source=NwisTempSource(
            site_no="08155500",
            parameter_cd="00010",
            name="Barton Springs",
        ),
        description="Austin, TX open water swimming conditions",
    ),
    LocationConfig(
        code="bos",
        name="Boston",
        swim_location="L Street Beach",
        swim_location_link="https://www.openwaterpedia.com/wiki/L_Street_Bathhouse",
        latitude=42.329,
        longitude=-71.036,
        timezone=pytz.timezone("US/Eastern"),
        temp_source=NdbcTempSource(
            station="44013",
            name="Boston Approach Lighted Buoy (16 NM East)",
        ),
        tide_source=CoopsTideSource(
            station=8443970,
            station_name="Boston, MA",
        ),
        description="Boston, MA open water swimming conditions",
    ),
    LocationConfig(
        code="sea",
        name="Seattle",
        swim_location="Alki Beach",
        swim_location_link="https://www.seattle.gov/parks/rentals-and-permits/indoor-event-rentals/alki-beach-bathhouse",
        latitude=47.580,
        longitude=-122.410,
        timezone=pytz.timezone("US/Pacific"),
        temp_source=CoopsTempSource(
            station=9446484,
            name="Station TCNW1 - Tacoma, WA",
        ),
        tide_source=CoopsTideSource(
            station=9447130,
            station_name="Seattle, WA",
        ),
        description="Seattle, WA open water swimming conditions",
    ),
    LocationConfig(
        enabled=False,
        code="tst",
        name="Test",
        swim_location="TBD",
        swim_location_link="TBD",
        latitude=25.9,
        longitude=-89.7,
        timezone=pytz.timezone("US/Central"),
        # https://www.ndbc.noaa.gov/station_page.php?station=42001
        temp_source=NdbcTempSource(
            station="42001",  # 44025",
            name="180 nm South of Southwest Pass, LA",
        ),
        description="Test location. MID GULF",
    ),
]

# Build lookup dictionaries - use lowercase keys for case-insensitive lookup
# Only include enabled locations in the CONFIGS dictionary
# Use MappingProxyType to create an immutable view of the dictionary
CONFIGS = MappingProxyType({c.code.lower(): c for c in _CONFIG_LIST if c.enabled})


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


def get_all_configs() -> List[LocationConfig]:
    """Get all location configurations, including disabled ones.

    This function is primarily intended for testing purposes.

    Returns:
        List of all LocationConfig objects
    """
    return _CONFIG_LIST.copy()

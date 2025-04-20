"""Application configuration."""

from typing import Annotated, List, Optional

from pydantic import BaseModel, Field


class LocationConfig(BaseModel):
    """Configuration for a swimming location."""

    code: Annotated[
        str,
        Field(
            min_length=3,
            max_length=3,
            pattern=r"^[a-z]{3}$",
            description="3-letter lowercase location code",
        ),
    ]
    name: str

    swim_location: str
    swim_location_link: str

    description: str
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
    # TODO: may need to add more windy params (e.g. zoom, aspect ratio)
    timezone: str

    current_predictions: bool = False

    # NOAA parameters
    # Water/met stations are 7 digit ints. Currents station ids are strings.
    temp_station: Annotated[
        Optional[int],
        Field(
            ge=1000000, le=9999999, description="NOAA temperature station ID (7 digits)"
        ),
    ] = None
    tide_station: Annotated[
        Optional[int],
        Field(ge=1000000, le=9999999, description="NOAA tide station ID (7 digits)"),
    ] = None
    # XXX still need to generalize currents
    currents_stations: Optional[List[Annotated[str, Field(min_length=1)]]] = None

    temp_station_name: Optional[str] = None
    tide_station_name: Optional[str] = None

    # No custom validators needed - using Pydantic's built-in constraints


CONFIG_LIST = [
    LocationConfig(
        code="nyc",
        name="New York",
        swim_location="Grimaldo's Chair",
        swim_location_link="https://cibbows.org/about/essentials/",
        latitude=40.573,
        longitude=-73.954,
        timezone="US/Eastern",
        current_predictions=True,
        temp_station=8518750,
        temp_station_name="The Battery, NY",
        tide_station=8517741,
        tide_station_name="Coney Island, NY",
        currents_stations=[
            "ACT3876",  # Coney Island Channel
            "NYH1905",  # Rockaway Inslet
        ],
        description="Coney Island Brighton Beach open water swimming conditions",
    ),
    LocationConfig(
        code="san",
        name="San Diego",
        swim_location="La Jolla Cove",
        swim_location_link="https://www.lajollacoveswimclub.com/page-1518748",
        latitude=32.850,
        longitude=-117.272,
        timezone="US/Pacific",
        temp_station=9410230,
        temp_station_name="La Jolla, CA",
        tide_station=9410230,
        tide_station_name="La Jolla, CA",
        description="La Jolla Cove open water swimming conditions",
    ),
    LocationConfig(
        # TODO more SF stuff can be added. see here: https://dolphinclub.org/weather/
        code="sfo",
        name="San Francisco",
        swim_location="San Francisco Aquatic Park",
        swim_location_link="https://serc.com/swimming/swimming-in-aquatic-park/",
        latitude=37.808,
        longitude=-122.426,
        timezone="US/Pacific",
        # Note that North Point Pier temp (stn 9414305) is a operational forecast (OFS).
        # It is not a live reading (and not available via the same API), so we don't use it.
        temp_station=9414290,
        temp_station_name="San Francisco, CA",
        tide_station=9414305,
        tide_station_name="North Point Pier",
        description="San Francisco Aquatic Park open water swimming conditions",
        # webcam https://dolphinclub.org/weather/ (code in page JS source...)
    ),
    LocationConfig(
        code="sdf",
        name="Louisville",
        swim_location="Louisville Community Boathouse",
        swim_location_link="https://www.kylmsc.org/rats",
        latitude=38.2647556,
        longitude=-85.7323204,
        # XXX windy "waves" mode inapplicable here
        timezone="US/Eastern",
        description="Louisville Kentucky open water swimming conditions",
        # TODO: add this
        # - temp data
        #    water velocity (mph)
        #    https://colab.research.google.com/drive/17-oyc95BBUUI3g1GBR5QpACAm-4G5vBk#scrollTo=vMXIJh8CCQCF
        # - webcam:
        #    https://www.earthcam.com/usa/kentucky/louisville/?cam=ohioriver
        #    https://ohiorivercam.com/
    ),
]
CONFIGS = {c.code: c for c in CONFIG_LIST}


def Get(code: str) -> Optional[LocationConfig]:
    """Get location config by 3-letter code.

    Args:
        code: 3-letter location code

    Returns:
        LocationConfig if found, None otherwise
    """
    return CONFIGS.get(code)

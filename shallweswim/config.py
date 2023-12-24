"""Application configuration."""

import pydantic


class LocationConfig(pydantic.BaseModel):
    code: str
    name: str

    swim_location: str
    swim_location_link: str

    description: str
    latitude: float
    longitude: float
    # TODO: may need to add more windy params (e.g. zoom, aspect ratio)
    timezone: str

    current_predictions: bool = False

    # NOAA parameters
    # Water/met stations are 7 digit ints. Currents station ids are strings.
    temp_station: int | None = None
    tide_station: int | None = None
    # XXX still need to generalize currents
    currents_stations: list[str] | None = None

    temp_station_name: str | None = None
    tide_station_name: str | None = None


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
        tida_station_name="La Jolla, CA",
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
        swim_location="XXX",
        swim_location_link="XXX",
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


def Get(code) -> LocationConfig:
    assert code in CONFIGS, "bad location: " + code
    return CONFIGS[code]

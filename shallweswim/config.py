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
    # XXX add windy params: zoom. ratio (height/width)
    timezone: str

    current_predictions: bool = False

    # NOAA parameters
    # Water/met stations are 7 digit ints. Currents station ids are strings.
    temp_station: int | None = None
    tide_station: int | None = None
    # XXX This may need to be a list to take avg velocity
    # e.g. mean of coney_channel and rockaway_inlet
    currents_stations: list[str] | None = None

    # XXX add station names
    temp_station_name: str | None = None
    # XXX add tide station name


NYC = LocationConfig(
    code="nyc",
    name="New York",
    swim_location="Grimaldo's Chair",
    swim_location_link="https://cibbows.org/about/essentials/",
    latitude=40.573,
    longitude=-73.954,
    timezone="US/Eastern",
    current_predictions=True,
    temp_station=8518750,  # battery NYC
    temp_station_name="The Battery, NY",
    tide_station=8517741,  # coney island
    tide_station_name="Coney Island, NY",
    currents_stations=[
        "ACT3876",  # Coney Island Channel
        "NYH1905",  # Rockaway Inslet
    ],
    description="Coney Island Brighton Beach open water swimming conditions",
)


def Get(code) -> LocationConfig:
    assert code == "nyc", code
    return NYC

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


NYC = LocationConfig(
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
)


def Get(code) -> LocationConfig:
    assert code == "nyc", code
    return NYC

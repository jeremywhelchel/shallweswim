"""Type definitions for NOAA API client."""

from typing import TypedDict, Literal, Optional
import datetime

ProductType = Literal[
    "predictions", "currents_predictions", "air_temperature", "water_temperature"
]
TimeInterval = Literal["hilo", "MAX_SLACK", "h", None]
DateFormat = "%Y%m%d"

# Temperature product types
air_temperature = "air_temperature"
water_temperature = "water_temperature"


class NoaaRequestParams(TypedDict, total=False):
    """Parameters for NOAA API requests."""

    product: ProductType
    datum: str
    begin_date: str
    end_date: str
    station: int | str
    interval: TimeInterval
    application: str
    time_zone: str
    units: str
    format: str


class TideData(TypedDict):
    """Tide prediction data."""

    prediction: float
    type: Literal["low", "high"]


class CurrentData(TypedDict):
    """Current prediction data."""

    velocity: float
    depth: Optional[float]
    type: Optional[str]
    mean_flood_dir: Optional[float]
    bin: Optional[int]


class TemperatureData(TypedDict):
    """Temperature data."""

    water_temp: Optional[float]
    air_temp: Optional[float]

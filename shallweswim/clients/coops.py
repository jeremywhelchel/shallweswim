"""NOAA CO-OPS (Center for Operational Oceanographic Products and Services) API client."""

# Standard library imports
import asyncio
import datetime
import io
import logging
import urllib.parse
from typing import Literal, Optional, TypedDict, cast

# Third-party imports
import aiohttp
import pandas as pd

# Local imports
from shallweswim.clients.base import BaseApiClient

# Type definitions for NOAA CO-OPS API client
ProductType = Literal[
    "predictions", "currents_predictions", "air_temperature", "water_temperature"
]
TimeInterval = Literal["hilo", "MAX_SLACK", "h", None]
DateFormat = "%Y%m%d"

# Temperature product types
air_temperature = "air_temperature"
water_temperature = "water_temperature"


class CoopsRequestParams(TypedDict, total=False):
    """Parameters for NOAA CO-OPS API requests."""

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


class CoopsApiError(Exception):
    """Base error for NOAA CO-OPS API calls."""


class CoopsConnectionError(CoopsApiError):
    """Error connecting to NOAA CO-OPS API."""


class CoopsDataError(CoopsApiError):
    """Error in data returned by NOAA CO-OPS API."""


class CoopsApi(BaseApiClient):
    """Client for the NOAA CO-OPS Tides and Currents API.

    This class provides methods to fetch tide predictions, current predictions,
    and temperature data from NOAA's CO-OPS API.

    API documentation: https://api.tidesandcurrents.noaa.gov/api/prod/

    All methods return pandas DataFrames with timestamps localized to the station's
    local time (either standard or daylight time).
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    BASE_PARAMS: CoopsRequestParams = {
        "application": "shallweswim",
        "time_zone": "lst_ldt",
        "units": "english",
        "format": "csv",
    }

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    @classmethod
    def _format_date(cls, date: datetime.date | datetime.datetime) -> str:
        """Format a date for NOAA CO-OPS API requests."""
        if isinstance(date, datetime.datetime):
            date = date.date()
        return date.strftime(DateFormat)

    @classmethod
    async def _Request(
        cls, params: CoopsRequestParams, location_code: str = "unknown"
    ) -> pd.DataFrame:
        """Make a request to the NOAA CO-OPS API with retries.

        Args:
            params: API request parameters
            location_code: Optional location code for logging

        Returns:
            DataFrame containing the API response

        Raises:
            CoopsConnectionError: If connection to API fails
            CoopsDataError: If API returns error response
        """
        url_params = dict(cls.BASE_PARAMS, **params)
        url = cls.BASE_URL + "?" + urllib.parse.urlencode(url_params)

        async with aiohttp.ClientSession() as session:
            for attempt in range(cls.MAX_RETRIES):
                try:
                    logging.info(
                        f"[{location_code}][coops] NOAA CO-OPS API request (attempt {attempt + 1}): {url}"
                    )
                    async with session.get(url) as response:
                        if response.status != 200:
                            error_msg = f"HTTP error: {response.status}"
                            logging.error(f"[{location_code}][coops] {error_msg}")
                            raise CoopsConnectionError(error_msg)

                        # Read CSV data
                        csv_data = await response.text()
                        df = pd.read_csv(io.StringIO(csv_data))

                        if len(df) == 1:
                            error_msg = df.iloc[0].values[0]
                            logging.error(
                                f"[{location_code}][coops] NOAA CO-OPS API data error: {error_msg}"
                            )
                            raise CoopsDataError(error_msg)
                        return df
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == cls.MAX_RETRIES - 1:
                        error_msg = f"Failed to connect to NOAA CO-OPS API: {e}"
                        logging.error(f"[{location_code}][coops] {error_msg}")
                        raise CoopsConnectionError(error_msg)
                    await asyncio.sleep(cls.RETRY_DELAY * (attempt + 1))

        error_msg = "Unexpected error in NOAA CO-OPS API request"
        logging.error(f"[{location_code}][coops] {error_msg}")
        raise CoopsConnectionError(error_msg)

    @classmethod
    async def tides(
        cls,
        station: int,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return tide predictions from yesterday to two days from now.

        Args:
            station: NOAA station ID

        Returns:
            DataFrame with index=timestamp and columns:
                prediction: float - Water level in feet relative to MLLW
                type: str - Either 'low' or 'high'
        """
        today = datetime.date.today()
        begin_date = today - datetime.timedelta(days=1)
        end_date = today + datetime.timedelta(days=2)
        params: CoopsRequestParams = {
            "product": "predictions",
            "datum": "MLLW",
            "begin_date": cls._format_date(begin_date),
            "end_date": cls._format_date(end_date),
            "station": station,
            "interval": "hilo",
        }

        logging.info(
            f"[{location_code}][coops] Fetching tide predictions for station {station} from {cls._format_date(begin_date)} to {cls._format_date(end_date)}"
        )
        df = await cls._Request(params, location_code)
        df = (
            df.pipe(cls._FixTime)
            .rename(columns={" Prediction": "prediction", " Type": "type"})
            .assign(type=lambda x: x["type"].map({"L": "low", "H": "high"}))[
                ["prediction", "type"]
            ]
        )
        return cast("pd.DataFrame[TideData]", df)

    @classmethod
    async def currents(
        cls,
        station: str,
        interpolate: bool = True,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return current predictions from yesterday to two days from now.

        Args:
            station: NOAA current station ID (string format)
            interpolate: If True, interpolate between flood/slack/ebb points

        Returns:
            DataFrame with index=timestamp and columns:
                velocity: float - Current velocity in knots (positive=flood, negative=ebb)
                depth: Optional[float] - Depth in feet (if available)
                type: Optional[str] - Current type (flood/slack/ebb)
                mean_flood_dir: Optional[float] - Mean flood direction in degrees
                bin: Optional[int] - Bin number
        """
        today = datetime.date.today()
        begin_date = today - datetime.timedelta(days=1)
        end_date = today + datetime.timedelta(days=2)
        params: CoopsRequestParams = {
            "product": "currents_predictions",
            "datum": "MLLW",
            "begin_date": cls._format_date(begin_date),
            "end_date": cls._format_date(end_date),
            "station": station,
            "interval": "MAX_SLACK",
        }

        logging.info(
            f"[{location_code}][coops] Fetching current predictions for station {station} from {cls._format_date(begin_date)} to {cls._format_date(end_date)}"
        )
        df = await cls._Request(params, location_code)
        currents = (
            df.pipe(cls._FixTime, time_col="Time").rename(
                columns={
                    " Depth": "depth",
                    " Type": "type",
                    " Velocity_Major": "velocity",
                    " meanFloodDir": "mean_flood_dir",
                    " Bin": "bin",
                }
            )
            # only return velocity for now to avoid some issues with other columns
            [["velocity"]]
        )

        if interpolate:
            # Data is just flood/slack/ebb datapoints. Create a smooth curve
            # using polynomial interpolation if we have enough points, otherwise linear
            resampled = currents.resample("60s")
            if len(currents) >= 3:
                # With 3+ points, use quadratic interpolation for smoother transitions
                currents = resampled.interpolate("polynomial", order=2)
            else:
                # With sparse data, fall back to linear interpolation
                currents = resampled.interpolate(method="linear")

        return cast("pd.DataFrame[CurrentData]", currents)

    @classmethod
    async def temperature(
        cls,
        station: int,
        product: Literal["air_temperature", "water_temperature"],
        begin_date: datetime.date,
        end_date: datetime.date,
        interval: TimeInterval = None,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Fetch buoy temperature dataset.

        Args:
            station: NOAA station ID
            product: Type of temperature data to fetch
            begin_date: Start date for data fetch
            end_date: End date for data fetch
            interval: Optional time interval (if None, returns 6-minute intervals)

        Returns:
            DataFrame with index=timestamp and columns:
                water_temp: Optional[float] - Water temperature in °F
                air_temp: Optional[float] - Air temperature in °F

        Raises:
            ValueError: If product is invalid or date range is invalid
        """
        if begin_date > end_date:
            raise ValueError("begin_date must be <= end_date")

        if product not in ["air_temperature", "water_temperature"]:
            raise ValueError(f"Invalid product: {product}")

        params: CoopsRequestParams = {
            "product": product,
            "begin_date": cls._format_date(begin_date),
            "end_date": cls._format_date(end_date),
            "station": station,
            "interval": interval,
        }

        logging.info(
            f"[{location_code}][coops] Fetching temperature data for station {station} from {cls._format_date(begin_date)} to {cls._format_date(end_date)}"
        )
        df = await cls._Request(params, location_code)
        df = (
            df.pipe(cls._FixTime)
            .rename(
                columns={
                    " Water Temperature": "water_temp",
                    " Air Temperature": "air_temp",
                }
            )
            .drop(
                columns=[" X", " N", " R "], errors="ignore"
            )  # Metadata columns we don't use
        )

        return cast("pd.DataFrame[TemperatureData]", df)

    @classmethod
    def _FixTime(cls, df: pd.DataFrame, time_col: str = "Date Time") -> pd.DataFrame:
        """Fix timestamp column in NOAA CO-OPS API response.

        Args:
            df: DataFrame from NOAA CO-OPS API
            time_col: Name of the timestamp column

        Returns:
            DataFrame with:
            - Timestamp column converted to datetime and set as index
            - Timezone info removed (already in local time from API)
        """
        return (
            df.assign(time=lambda x: pd.to_datetime(x[time_col], utc=True))
            .drop(columns=time_col)
            .set_index("time")
            # Drop timezone info. Already in local time (LST/LDT in request)
            .tz_localize(None)
        )

"""NOAA tides and current API client."""

# Standard library imports
import datetime
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal, cast

# Third-party imports
import pandas as pd

# Local imports
from shallweswim.noaa_types import (
    DateFormat,
    NoaaRequestParams,
    TimeInterval,
)


class NoaaApiError(Exception):
    """Base error for NOAA API calls."""


class NoaaConnectionError(NoaaApiError):
    """Error connecting to NOAA API."""


class NoaaDataError(NoaaApiError):
    """Error in data returned by NOAA API."""


class NoaaApi:
    """Client for the NOAA Tides and Currents API.

    This class provides methods to fetch tide predictions, current predictions,
    and temperature data from NOAA's CO-OPS API.

    API documentation: https://api.tidesandcurrents.noaa.gov/api/prod/

    All methods return pandas DataFrames with timestamps localized to the station's
    local time (either standard or daylight time).
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    BASE_PARAMS: NoaaRequestParams = {
        "application": "shallweswim",
        "time_zone": "lst_ldt",
        "units": "english",
        "format": "csv",
    }

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    @classmethod
    def _format_date(cls, date: datetime.date | datetime.datetime) -> str:
        """Format a date for NOAA API requests."""
        if isinstance(date, datetime.datetime):
            date = date.date()
        return date.strftime(DateFormat)

    @classmethod
    def _Request(cls, params: NoaaRequestParams) -> pd.DataFrame:
        """Make a request to the NOAA API with retries.

        Args:
            params: API request parameters

        Returns:
            DataFrame containing the API response

        Raises:
            NoaaConnectionError: If connection to API fails
            NoaaDataError: If API returns error response
        """
        url_params = dict(cls.BASE_PARAMS, **params)
        url = cls.BASE_URL + "?" + urllib.parse.urlencode(url_params)

        for attempt in range(cls.MAX_RETRIES):
            try:
                logging.info("NOAA API request (attempt %d): %s", attempt + 1, url)
                df = pd.read_csv(url)
                if len(df) == 1:
                    raise NoaaDataError(df.iloc[0].values[0])
                return df
            except urllib.error.URLError as e:
                if attempt == cls.MAX_RETRIES - 1:
                    raise NoaaConnectionError("Failed to connect to NOAA API: %s" % e)
                time.sleep(cls.RETRY_DELAY * (attempt + 1))

        raise NoaaConnectionError("Unexpected error in NOAA API request")

    @classmethod
    def Tides(
        cls,
        station: int,
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
        params: NoaaRequestParams = {
            "product": "predictions",
            "datum": "MLLW",
            "begin_date": cls._format_date(today - datetime.timedelta(days=1)),
            "end_date": cls._format_date(today + datetime.timedelta(days=2)),
            "station": station,
            "interval": "hilo",
        }

        df = (
            cls._Request(params)
            .pipe(cls._FixTime)
            .rename(columns={" Prediction": "prediction", " Type": "type"})
            .assign(type=lambda x: x["type"].map({"L": "low", "H": "high"}))[
                ["prediction", "type"]
            ]
        )
        return cast("pd.DataFrame[TideData]", df)

    @classmethod
    def Currents(
        cls,
        station: str,
        interpolate: bool = True,
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
        params: NoaaRequestParams = {
            "product": "currents_predictions",
            "datum": "MLLW",
            "begin_date": cls._format_date(today - datetime.timedelta(days=1)),
            "end_date": cls._format_date(today + datetime.timedelta(days=2)),
            "station": station,
            "interval": "MAX_SLACK",
        }

        currents = (
            cls._Request(params)
            .pipe(cls._FixTime, time_col="Time")
            .rename(
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
    def Temperature(
        cls,
        station: int,
        product: Literal["air_temperature", "water_temperature"],
        begin_date: datetime.date,
        end_date: datetime.date,
        interval: TimeInterval = None,
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

        params: NoaaRequestParams = {
            "product": product,
            "begin_date": cls._format_date(begin_date),
            "end_date": cls._format_date(end_date),
            "station": station,
            "interval": interval,
        }

        df = (
            cls._Request(params)
            .pipe(cls._FixTime)
            .rename(
                columns={
                    " Water Temperature": "water_temp",
                    " Air Temperature": "air_temp",
                }
            )
            .drop(columns=[" X", " N", " R "])  # Metadata columns we don't use
        )

        return cast("pd.DataFrame[TemperatureData]", df)

    @classmethod
    def _FixTime(cls, df: pd.DataFrame, time_col: str = "Date Time") -> pd.DataFrame:
        """Fix timestamp column in NOAA API response.

        Args:
            df: DataFrame from NOAA API
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

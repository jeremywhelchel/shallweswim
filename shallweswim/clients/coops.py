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
from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,
    RetryableClientError,
    StationUnavailableError,
)
from shallweswim.types import (
    TIDE_TYPE_CATEGORIES,
)

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


class CoopsApiError(BaseClientError):
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

    @property
    def client_type(self) -> str:
        return "coops"

    def __init__(self, session: aiohttp.ClientSession):
        """Initialize CoopsApi with an aiohttp client session."""
        super().__init__(session=session)

    def _format_date(self, date: datetime.date | datetime.datetime) -> str:
        """Format a date for NOAA CO-OPS API requests."""
        if isinstance(date, datetime.datetime):
            date = date.date()
        return date.strftime(DateFormat)

    async def _execute_request(self, url: str, location_code: str) -> pd.DataFrame:
        """Performs the CO-OPS API request, handles errors, and parses the response.

        This method is intended to be called by the `request_with_retry` logic
        in the base class.

        Args:
            url: The fully constructed URL for the CO-OPS API request.
            location_code: The location code for logging purposes.

        Returns:
            A pandas DataFrame containing the successfully parsed data.

        Raises:
            RetryableClientError: For transient network errors (connection, timeout).
            CoopsConnectionError: For non-retryable HTTP errors (e.g., status 404, 500).
            CoopsDataError: For errors parsing the response or API-level errors in data.
        """
        self.log(
            f"Executing CO-OPS request: {url}",
            location_code=location_code,
        )
        csv_data: str
        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    # Treat non-200 responses as non-retryable connection errors
                    error_msg = f"HTTP error {response.status} for {url}"
                    self.log(
                        error_msg, level=logging.ERROR, location_code=location_code
                    )
                    # Raise specific connection error, not retryable
                    raise CoopsConnectionError(error_msg)

                # Read CSV data if status is OK
                csv_data = await response.text()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Convert specific connection/timeout errors into our standard retryable error
            error_msg = f"Network error during CO-OPS request to {url}: {e.__class__.__name__}: {e}"
            # Log is handled by the tenacity retry logger in the base class
            raise RetryableClientError(error_msg) from e

        # --- Parsing logic (outside the network try/except) ---
        try:
            df = pd.read_csv(io.StringIO(csv_data))
        except Exception as e:
            # Catch broad exceptions during parsing
            error_msg = f"Failed to parse CO-OPS CSV response from {url}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise CoopsDataError(error_msg) from e  # Parsing error is a data error

        # Check for API-level errors reported in the CSV data
        # NOAA returns errors in different formats:
        # 1. Single-column: {'Error': 'No data was found...'}
        # 2. Multi-column: 'Date Time, Water Temperature, X, N, R \n Error: No data was found...'
        # Check if any cell in the first row contains an error message
        if len(df) >= 1:
            # Check first column of first row for error message
            first_cell = str(df.iloc[0, 0])
            if "error" in first_cell.lower():
                error_msg = first_cell
                # Distinguish between "no data" (expected) and other errors (unexpected)
                if "no data" in first_cell.lower():
                    # Station has no data - expected operational condition
                    self.log(
                        f"NOAA CO-OPS station has no data for {url}: {error_msg}",
                        level=logging.WARNING,
                        location_code=location_code,
                    )
                    raise StationUnavailableError(error_msg)
                else:
                    # Other API error - unexpected, needs investigation
                    self.log(
                        f"NOAA CO-OPS API error for {url}: {error_msg}",
                        level=logging.ERROR,
                        location_code=location_code,
                    )
                    raise CoopsDataError(error_msg)

        self.log(
            f"Successfully parsed {len(df)} records from {url}",
            location_code=location_code,
        )
        return df  # Return the raw DataFrame, no processing here

    async def tides(
        self,
        station: int,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return tide predictions from yesterday to two days from now.

        Args:
            station: NOAA station ID

        Returns:
            DataFrame with index=time and columns:
                prediction: float - Water level in feet relative to MLLW
                type: str - Either 'low' or 'high'
        """
        today = datetime.date.today()
        begin_date = today - datetime.timedelta(days=1)
        end_date = today + datetime.timedelta(days=2)
        params: CoopsRequestParams = {
            "product": "predictions",
            "datum": "MLLW",
            "begin_date": self._format_date(begin_date),
            "end_date": self._format_date(end_date),
            "station": station,
            "interval": "hilo",
        }

        self.log(
            f"Fetching tide predictions for station {station} from {self._format_date(begin_date)} to {self._format_date(end_date)}",
            location_code=location_code,
        )

        # Construct URL and call using request_with_retry
        url_params = dict(self.BASE_PARAMS, **params)
        url = self.BASE_URL + "?" + urllib.parse.urlencode(url_params)
        raw_df: pd.DataFrame = await self.request_with_retry(location_code, url)

        # Existing processing logic
        df = (
            raw_df.pipe(self._FixTime)
            .rename(columns={" Prediction": "prediction", " Type": "type"})
            .assign(type=lambda x: x["type"].map({"L": "low", "H": "high"}))
            .astype({"type": pd.CategoricalDtype(TIDE_TYPE_CATEGORIES)})[
                ["prediction", "type"]
            ]
        )
        return cast("pd.DataFrame[TideData]", df)

    async def currents(
        self,
        station: str,
        interpolate: bool = True,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return current predictions from yesterday to two days from now.

        Args:
            station: NOAA current station ID (string format)
            interpolate: If True, interpolate between flood/slack/ebb points

        Returns:
            DataFrame with index=time and columns:
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
            "begin_date": self._format_date(begin_date),
            "end_date": self._format_date(end_date),
            "station": station,
            "interval": "MAX_SLACK",
        }

        self.log(
            f"Fetching current predictions for station {station} from {self._format_date(begin_date)} to {self._format_date(end_date)}",
            location_code=location_code,
        )

        # Construct URL and call using request_with_retry
        url_params = dict(self.BASE_PARAMS, **params)
        url = self.BASE_URL + "?" + urllib.parse.urlencode(url_params)
        raw_df: pd.DataFrame = await self.request_with_retry(location_code, url)

        # Existing processing logic (using raw_df as input)
        currents = (
            raw_df.pipe(self._FixTime, time_col="Time").rename(
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

    async def temperature(
        self,
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
            DataFrame with index=time and columns:
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
            "begin_date": self._format_date(begin_date),
            "end_date": self._format_date(end_date),
            "station": station,
            "interval": interval,
        }

        self.log(
            f"Fetching temperature data for station {station} from {self._format_date(begin_date)} to {self._format_date(end_date)}",
            location_code=location_code,
        )

        # Construct URL and call using request_with_retry
        url_params = dict(self.BASE_PARAMS, **params)
        url = self.BASE_URL + "?" + urllib.parse.urlencode(url_params)
        raw_df: pd.DataFrame = await self.request_with_retry(location_code, url)

        # Existing processing logic
        df = (
            raw_df.pipe(self._FixTime)
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

    def _FixTime(self, df: pd.DataFrame, time_col: str = "Date Time") -> pd.DataFrame:
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

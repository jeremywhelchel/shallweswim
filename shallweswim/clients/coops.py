"""NOAA CO-OPS (Center for Operational Oceanographic Products and Services) API client."""

# Standard library imports
import datetime
import io
import json
import logging
import urllib.parse
from typing import ClassVar, Literal, TypedDict

# Third-party imports
import aiohttp
import pandas as pd

# Local imports
from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,
    RetryableClientError,
    StationUnavailableError,
    provider_request_slot,
    raise_if_retryable_http_status,
    request_timeout,
    retryable_network_error,
)
from shallweswim.types import (
    TIDE_TYPE_CATEGORIES,
)

# Type definitions for NOAA CO-OPS API client
ProductType = Literal[
    "predictions", "currents_predictions", "air_temperature", "water_temperature"
]
TimeInterval = Literal["hilo", "MAX_SLACK", "h", "6-min", None]
# CO-OPS accepts no date format without a time component, and requests are made
# in GMT, so every window edge is sent as an explicit UTC minute.
RequestTimeFormat = "%Y%m%d %H:%M"
COOPS_PROVIDER = "coops"
COOPS_MAX_CONCURRENT_REQUESTS = 4
# CO-OPS caps the explicit range of one request by product cadence: the hourly
# product serves 365 days at a time, which a window over a leap year already
# exceeds, while the six-minute product serves only 31. A request without an
# interval returns six-minute readings, so it takes the six-minute limit.
# Sub-windows stay strictly under whichever limit applies.
COOPS_MAX_REQUEST_RANGE: dict[TimeInterval, datetime.timedelta] = {
    "h": datetime.timedelta(days=365),
    "6-min": datetime.timedelta(days=31),
    None: datetime.timedelta(days=31),
}
# How much of an error response body to quote back in an exception message.
RESPONSE_DETAIL_CHARS = 200
# Phrase NOAA uses for the stable answer "this station has no data for this
# product", which is an expected operational condition rather than a transient
# failure. Other error bodies are transient often enough to be worth retrying.
STATION_NO_DATA_MARKER = "no data"
# Phrase shared by NOAA's no-data answers, including the ones that never say
# "error" ("No Predictions data was found. Please make sure the Datum input is
# valid."). Only the stable "no data" wording above is an expected condition;
# the rest are transient rejections.
NO_DATA_PROSE_MARKER = "data was found"

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
    depth: float | None
    type: str | None
    mean_flood_dir: float | None
    bin: int | None


class TemperatureData(TypedDict):
    """Temperature data."""

    water_temp: float | None
    air_temp: float | None


def response_detail(body: str) -> str:
    """Collapse a response body into a short single-line message detail.

    NOAA reports the reason a request was rejected in the response body rather
    than the status line, so an error message that omits it is undiagnosable.
    """
    return " ".join(body.split())[:RESPONSE_DETAIL_CHARS]


def error_body_detail(body: str) -> str | None:
    """Return a collapsed detail when a 200 body is an error instead of CSV.

    CO-OPS answers some valid requests with HTTP 200 and a prose or JSON error
    message ("No Predictions data was found. Please make sure the Datum input
    is valid", `{"error": {"message": ...}}`) where CSV was requested. Parsing
    such a body as CSV yields a nonsense frame, so it has to be recognized
    before parsing.

    The check is deliberately narrow: every CSV product this client requests
    has at least two columns, so a first line without a comma is never a CSV
    header, and a JSON object is never CSV at all.

    Args:
        body: The decoded response body of a 200 response.

    Returns:
        The collapsed error detail, or None when the body looks like CSV.
    """
    stripped = body.strip()
    if not stripped:
        # An empty body is not an error message; CSV parsing reports it.
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            return response_detail(stripped)
        if not isinstance(payload, dict) or "error" not in payload:
            return None
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else error
        return response_detail(str(message) if message else stripped)
    first_line = stripped.split("\n", 1)[0]
    if "," in first_line:
        return None
    return response_detail(stripped)


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

    All methods return pandas DataFrames indexed by timezone-aware UTC
    timestamps. Requests use ``time_zone=gmt`` so both folds of a daylight
    saving fall-back hour and the skipped spring-forward hour are exact.
    Request windows are still expressed in station-local days: callers pass a
    naive local date and a ``timezone``, which this client converts to UTC for
    the request.

    CO-OPS reports failures in the response body, sometimes under HTTP 200. A
    200 body that is a NOAA error message rather than CSV is retried as a
    transient rejection, except for the stable "no data" answer, which raises
    ``StationUnavailableError`` as an expected operational condition.
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    BASE_PARAMS: ClassVar[CoopsRequestParams] = {
        "application": "shallweswim",
        "time_zone": "gmt",
        "units": "english",
        "format": "csv",
    }

    @property
    def client_type(self) -> str:
        return "coops"

    def __init__(self, session: aiohttp.ClientSession):
        """Initialize CoopsApi with an aiohttp client session."""
        super().__init__(session=session)

    def _window_edge(
        self,
        date: datetime.date | datetime.datetime,
        timezone: str,
        *,
        end_of_day: bool = False,
    ) -> pd.Timestamp:
        """Resolve one edge of a station-local request window to a UTC instant.

        A window covers whole local days, from local midnight through the last
        local minute, which is what date-only edges meant while requests were
        made in local time. A local edge that a daylight saving transition
        makes nonexistent or ambiguous is resolved deterministically rather
        than raising: window edges select data, they do not describe an
        observation.

        Args:
            date: Local window edge; a datetime contributes only its date.
            timezone: Station timezone the naive edge is expressed in.
            end_of_day: Whether the edge closes the window rather than opens it.

        Returns:
            The edge as a timezone-aware UTC timestamp.
        """
        if isinstance(date, datetime.datetime):
            date = date.date()
        wall = datetime.datetime.combine(
            date, datetime.time(23, 59) if end_of_day else datetime.time.min
        )
        local = pd.Timestamp(wall).tz_localize(
            timezone, nonexistent="shift_forward", ambiguous=False
        )
        return local.tz_convert("UTC")

    def _format_request_time(
        self,
        date: datetime.date | datetime.datetime,
        timezone: str,
        *,
        end_of_day: bool = False,
    ) -> str:
        """Format one edge of a station-local request window for a request."""
        return self._window_edge(date, timezone, end_of_day=end_of_day).strftime(
            RequestTimeFormat
        )

    def _split_request_window(
        self, begin: pd.Timestamp, end: pd.Timestamp, interval: TimeInterval
    ) -> list[tuple[str, str]]:
        """Split a UTC window into request windows within the CO-OPS range limit.

        The limit depends on the requested cadence: a single hourly request may
        span at most 365 days, which a window over a leap year exceeds, and a
        single six-minute request at most 31, so a year at either cadence has
        to be fetched in parts. Consecutive parts abut to the minute: each ends
        one minute before the next begins, so every reading falls in exactly
        one part and none falls between two.

        Args:
            begin: First instant of the window, timezone-aware UTC.
            end: Last instant of the window, timezone-aware UTC.
            interval: Requested cadence, which selects the range limit.

        Returns:
            Formatted (begin, end) request strings in ascending instant order.
        """
        max_range = COOPS_MAX_REQUEST_RANGE[interval]
        minute = datetime.timedelta(minutes=1)
        windows: list[tuple[str, str]] = []
        start = begin
        while start <= end:
            stop = min(start + max_range - minute, end)
            windows.append(
                (
                    start.strftime(RequestTimeFormat),
                    stop.strftime(RequestTimeFormat),
                )
            )
            start = stop + minute
        return windows

    def _build_url(self, params: CoopsRequestParams) -> str:
        """Build a CO-OPS datagetter URL with default and request params."""
        url_params = dict(self.BASE_PARAMS, **params)
        # Percent-encode rather than plus-encode: request times contain a space
        # separator, and "%20" decodes to a space under every query parsing rule.
        return (
            self.BASE_URL
            + "?"
            + urllib.parse.urlencode(url_params, quote_via=urllib.parse.quote)
        )

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
            RetryableClientError: For transient network errors (connection,
                timeout) and for a 200 response that carries a NOAA error
                message rather than data, whether as the whole body or as a
                prose row under a CSV header.
            CoopsConnectionError: For non-retryable HTTP errors (e.g., status 404, 500).
            StationUnavailableError: When NOAA reports the station has no data.
            CoopsDataError: For errors parsing the response or API-level errors in data.
        """
        self.log(
            f"Executing CO-OPS request: {url}",
            level=logging.DEBUG,
            location_code=location_code,
        )
        csv_data: str
        try:
            timeout = request_timeout(self.REQUEST_TIMEOUT)
            async with provider_request_slot(
                COOPS_PROVIDER, COOPS_MAX_CONCURRENT_REQUESTS
            ):
                async with self._session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        error_msg = f"HTTP error {response.status} for {url}"
                        try:
                            detail = response_detail(await response.text())
                        except (TimeoutError, aiohttp.ClientError):
                            # The body is diagnostic detail only; failing to read
                            # it must not reclassify the HTTP error itself.
                            detail = ""
                        if detail:
                            error_msg = f"{error_msg}: {detail}"
                        raise_if_retryable_http_status(response.status, error_msg)

                        self.log(
                            error_msg, level=logging.ERROR, location_code=location_code
                        )
                        raise CoopsConnectionError(error_msg, status=response.status)

                    # Read CSV data if status is OK
                    csv_data = await response.text()

        except (TimeoutError, aiohttp.ClientError) as e:
            # Convert specific connection/timeout errors into our standard retryable error
            # Log is handled by the tenacity retry logger in the base class
            raise retryable_network_error(
                provider="CO-OPS",
                action=f"to {url}",
                error=e,
            ) from e

        # --- VALIDATION PHASE (outside the network try/except) ---
        # NOAA sometimes answers a valid request with HTTP 200 and an error
        # message instead of CSV. Parsing that as CSV would raise a terminal
        # data error, so classify the body first.
        error_detail = error_body_detail(csv_data)
        if error_detail is not None:
            if STATION_NO_DATA_MARKER in error_detail.lower():
                # Stable answer: the station offers no data for this product.
                self.log(
                    f"NOAA CO-OPS station has no data for {url}: {error_detail}",
                    level=logging.WARNING,
                    location_code=location_code,
                )
                raise StationUnavailableError(error_detail)
            # Anything else is a transient rejection that the same request
            # usually survives moments later (seen for tide predictions during
            # bursts of concurrent requests). The retry logger records it.
            raise RetryableClientError(
                f"NOAA CO-OPS returned an error body for {url}: {error_detail}"
            )

        # --- Parsing logic ---
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
        # 3. A real CSV header followed by prose that never says "error":
        #    'Date Time, Prediction, Type \n No Predictions data was found. ...'
        # Form 3 is classified here rather than in `error_body_detail` because
        # that check only asks whether a body is CSV-shaped at all, and this
        # body is: its first line is a genuine header. Only the parsed row
        # shows it carries prose instead of data, and `read_csv` pads the short
        # prose line to the header's width, so the message lands in the first
        # cell of a one-row frame.
        if len(df) >= 1:
            # Check the first row for an error message. A one-row frame is
            # short enough to read whole, so a message that an embedded comma
            # split across cells is still seen; the padding cells read_csv adds
            # to a short row are empty and drop out.
            first_row = (
                " ".join(str(cell) for cell in df.iloc[0] if pd.notna(cell))
                if len(df) == 1
                else str(df.iloc[0, 0])
            )
            lowered = first_row.lower()
            if "error" in lowered or NO_DATA_PROSE_MARKER in lowered:
                error_msg = first_row
                # Distinguish between "no data" (expected) and other errors (unexpected)
                if STATION_NO_DATA_MARKER in lowered:
                    # Station has no data - expected operational condition
                    self.log(
                        f"NOAA CO-OPS station has no data for {url}: {error_msg}",
                        level=logging.WARNING,
                        location_code=location_code,
                    )
                    raise StationUnavailableError(error_msg)
                elif NO_DATA_PROSE_MARKER in lowered:
                    # A "<product> data was found" answer that is not the stable
                    # "no data" one, classified like the same prose arriving as
                    # an error body: a transient rejection worth retrying.
                    raise RetryableClientError(
                        f"NOAA CO-OPS returned an error row for {url}: {error_msg}"
                    )
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
            level=logging.DEBUG,
            location_code=location_code,
        )
        return df  # Return the raw DataFrame, no processing here

    async def tides(
        self,
        station: int,
        timezone: str,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return tide predictions from yesterday to two days from now.

        Args:
            station: NOAA station ID
            timezone: Station timezone the request window is expressed in
            location_code: Location code for logging purposes

        Returns:
            DataFrame indexed by timezone-aware UTC time, with columns:
                prediction: float - Water level in feet relative to MLLW
                type: str - Either 'low' or 'high'
        """
        today = datetime.date.today()
        begin = self._format_request_time(today - datetime.timedelta(days=1), timezone)
        end = self._format_request_time(
            today + datetime.timedelta(days=2), timezone, end_of_day=True
        )
        params: CoopsRequestParams = {
            "product": "predictions",
            "datum": "MLLW",
            "begin_date": begin,
            "end_date": end,
            "station": station,
            "interval": "hilo",
        }

        self.log(
            f"Fetching tide predictions for station {station} from {begin} to {end} UTC",
            level=logging.DEBUG,
            location_code=location_code,
        )

        url = self._build_url(params)
        raw_df = await self.request_with_retry(
            location_code, self._execute_request, url
        )

        # Existing processing logic
        df = (
            raw_df.pipe(self._FixTime)
            .rename(columns={" Prediction": "prediction", " Type": "type"})
            .assign(type=lambda x: x["type"].map({"L": "low", "H": "high"}))
            .astype({"type": pd.CategoricalDtype(TIDE_TYPE_CATEGORIES)})[
                ["prediction", "type"]
            ]
        )
        return df

    async def currents(
        self,
        station: str,
        timezone: str,
        interpolate: bool = True,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return current predictions from yesterday to two days from now.

        Args:
            station: NOAA current station ID (string format)
            timezone: Station timezone the request window is expressed in
            interpolate: If True, interpolate between flood/slack/ebb points
            location_code: Location code for logging purposes

        Returns:
            DataFrame indexed by timezone-aware UTC time, with columns:
                velocity: float - Current velocity in knots (positive=flood, negative=ebb)
                depth: Optional[float] - Depth in feet (if available)
                type: Optional[str] - Current type (flood/slack/ebb)
                mean_flood_dir: Optional[float] - Mean flood direction in degrees
                bin: Optional[int] - Bin number
        """
        today = datetime.date.today()
        begin = self._format_request_time(today - datetime.timedelta(days=1), timezone)
        end = self._format_request_time(
            today + datetime.timedelta(days=2), timezone, end_of_day=True
        )
        params: CoopsRequestParams = {
            "product": "currents_predictions",
            "datum": "MLLW",
            "begin_date": begin,
            "end_date": end,
            "station": station,
            "interval": "MAX_SLACK",
        }

        self.log(
            f"Fetching current predictions for station {station} from {begin} to {end} UTC",
            level=logging.DEBUG,
            location_code=location_code,
        )

        url = self._build_url(params)
        raw_df = await self.request_with_retry(
            location_code, self._execute_request, url
        )

        # Existing processing logic (using raw_df as input)
        currents = raw_df.pipe(self._FixTime, time_col="Time").rename(
            columns={
                " Depth": "depth",
                " Type": "type",
                " Velocity_Major": "velocity",
                " meanFloodDir": "mean_flood_dir",
                " Bin": "bin",
            }
        )[
            # only return velocity for now to avoid some issues with other columns
            ["velocity"]
        ]

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

        return currents

    async def temperature(
        self,
        station: int,
        product: Literal["air_temperature", "water_temperature"],
        begin_date: datetime.date,
        end_date: datetime.date,
        timezone: str,
        interval: TimeInterval = None,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Fetch buoy temperature dataset.

        Args:
            station: NOAA station ID
            product: Type of temperature data to fetch
            begin_date: Local start date for data fetch
            end_date: Local end date for data fetch
            timezone: Station timezone the request window is expressed in
            interval: Optional time interval (if None, returns 6-minute intervals)
            location_code: Location code for logging purposes

        A window longer than the CO-OPS range limit for the requested interval
        is fetched as consecutive requests and stitched back together, so
        callers pass the window they want regardless of its length or cadence.

        Returns:
            DataFrame indexed by timezone-aware UTC time, with columns:
                water_temp: Optional[float] - Water temperature in °F
                air_temp: Optional[float] - Air temperature in °F

        Raises:
            ValueError: If product is invalid or date range is invalid
        """
        if begin_date > end_date:
            raise ValueError("begin_date must be <= end_date")

        if product not in ["air_temperature", "water_temperature"]:
            raise ValueError(f"Invalid product: {product}")

        windows = self._split_request_window(
            self._window_edge(begin_date, timezone),
            self._window_edge(end_date, timezone, end_of_day=True),
            interval,
        )

        self.log(
            f"Fetching temperature data for station {station} from "
            f"{windows[0][0]} to {windows[-1][1]} UTC in {len(windows)} request(s)",
            level=logging.DEBUG,
            location_code=location_code,
        )

        raw_frames: list[pd.DataFrame] = []
        for begin, end in windows:
            params: CoopsRequestParams = {
                "product": product,
                "begin_date": begin,
                "end_date": end,
                "station": station,
                "interval": interval,
            }
            url = self._build_url(params)
            raw_frames.append(
                await self.request_with_retry(location_code, self._execute_request, url)
            )

        # Existing processing logic
        df = (
            pd.concat(raw_frames, ignore_index=True)
            .pipe(self._FixTime)
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

        if len(raw_frames) > 1:
            # Stitched windows are ours to make coherent: order by instant and
            # keep the first reading should a boundary ever be returned twice.
            # A single response is passed through as the provider sent it, so a
            # genuine provider repeat still reaches the archive's conflict rule.
            df = df.sort_index(kind="stable")
            df = df[~df.index.duplicated(keep="first")]

        return df

    def _FixTime(self, df: pd.DataFrame, time_col: str = "Date Time") -> pd.DataFrame:
        """Fix timestamp column in NOAA CO-OPS API response.

        Args:
            df: DataFrame from NOAA CO-OPS API
            time_col: Name of the timestamp column

        Returns:
            DataFrame whose timestamp column is parsed into a timezone-aware
            UTC index named "time". Responses are requested in GMT, so the
            index carries the absolute instant of every reading, including both
            folds of a daylight saving fall-back hour.
        """
        return (
            df.assign(time=lambda x: pd.to_datetime(x[time_col], utc=True))
            .drop(columns=time_col)
            .set_index("time")
        )

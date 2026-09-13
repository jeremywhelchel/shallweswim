"""Marine Institute Ireland ERDDAP API client."""

import datetime
import io
import logging
import urllib.parse
from typing import ClassVar

import aiohttp
import pandas as pd

from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,
    StationUnavailableError,
    provider_request_slot,
    raise_if_retryable_http_status,
    request_timeout,
    retryable_network_error,
)
from shallweswim.types import TIDE_TYPE_CATEGORIES

MARINE_INSTITUTE_PROVIDER = "marine-institute"
MARINE_INSTITUTE_MAX_CONCURRENT_REQUESTS = 2
MARINE_INSTITUTE_TIDE_HIGH_LOW_URL = (
    "https://erddap.marine.ie/erddap/tabledap/IMI_TidePrediction_HighLow.csv"
)
METERS_TO_FEET = 3.280839895013123


class MarineInstituteApiError(BaseClientError):
    """Base error for Marine Institute API calls."""


class MarineInstituteConnectionError(MarineInstituteApiError):
    """Error connecting to Marine Institute APIs."""


class MarineInstituteDataError(MarineInstituteApiError):
    """Error in Marine Institute data returned by ERDDAP."""


class MarineInstituteApi(BaseApiClient):
    """Client for Marine Institute Ireland ERDDAP datasets.

    ERDDAP stamps every row with a UTC ISO instant, so all methods return
    pandas DataFrames indexed by timezone-aware UTC timestamps. Request windows
    are UTC days as well, so no station timezone is needed.
    """

    TIDE_HIGH_LOW_COLUMNS: ClassVar[tuple[str, ...]] = (
        "time",
        "stationID",
        "longitude",
        "latitude",
        "tide_time_category",
        "Water_Level_ODMalin",
    )

    @property
    def client_type(self) -> str:
        return "marine-institute"

    def __init__(self, session: aiohttp.ClientSession):
        """Initialize MarineInstituteApi with an aiohttp client session."""
        super().__init__(session=session)

    def _build_tide_high_low_url(
        self,
        *,
        station_id: str,
        begin_utc: datetime.datetime,
        end_utc: datetime.datetime,
    ) -> str:
        """Build a Marine Institute high/low tide tabledap CSV URL."""
        query = ",".join(self.TIDE_HIGH_LOW_COLUMNS)
        constraints = (
            ("stationID=", f'"{station_id}"'),
            ("time>=", _format_erddap_time(begin_utc)),
            ("time<=", _format_erddap_time(end_utc)),
        )
        for operator, value in constraints:
            query += f"&{operator}{value}"
        return (
            MARINE_INSTITUTE_TIDE_HIGH_LOW_URL
            + "?"
            + urllib.parse.quote(
                query,
                safe=",&=:",
            )
        )

    async def _execute_csv_request(self, url: str, location_code: str) -> pd.DataFrame:
        """Fetch and parse a Marine Institute ERDDAP CSV response."""
        self.log(
            f"Executing Marine Institute request: {url}",
            level=logging.DEBUG,
            location_code=location_code,
        )

        try:
            timeout = request_timeout(self.REQUEST_TIMEOUT)
            async with provider_request_slot(
                MARINE_INSTITUTE_PROVIDER,
                MARINE_INSTITUTE_MAX_CONCURRENT_REQUESTS,
            ):
                async with self._session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        error_msg = f"HTTP error {response.status} for {url}"
                        raise_if_retryable_http_status(response.status, error_msg)
                        self.log(
                            error_msg,
                            level=logging.ERROR,
                            location_code=location_code,
                        )
                        raise MarineInstituteConnectionError(error_msg)
                    csv_data = await response.text()
        except (TimeoutError, aiohttp.ClientError) as e:
            raise retryable_network_error(
                provider="Marine Institute",
                action=f"to {url}",
                error=e,
            ) from e

        try:
            df = pd.read_csv(io.StringIO(csv_data), skiprows=[1])
        except Exception as e:
            error_msg = f"Failed to parse Marine Institute CSV response from {url}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise MarineInstituteDataError(error_msg) from e

        if df.empty:
            error_msg = f"Marine Institute returned no data for {url}"
            self.log(error_msg, level=logging.WARNING, location_code=location_code)
            raise StationUnavailableError(error_msg)

        return df

    async def tides(
        self,
        *,
        station_id: str,
        height_offset_m: float = 0.0,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return high/low tide predictions from yesterday through two days ahead.

        Marine Institute's high/low summary gives official event timing and
        metre heights relative to Ordnance Datum Malin. `height_offset_m`
        converts those heights into the desired local tide-height datum before
        the app's standard feet conversion.

        Args:
            station_id: Marine Institute tide station identifier
            height_offset_m: Metres added to each OD Malin height before the
                conversion to feet
            location_code: Location code for logging purposes

        Returns:
            DataFrame indexed by timezone-aware UTC time, with columns:
                prediction: float - Water level in feet
                type: str - Either 'low' or 'high'

        Raises:
            MarineInstituteDataError: If the response schema is unexpected
            StationUnavailableError: If the station published no usable events
        """
        today = datetime.datetime.now(datetime.UTC).date()
        begin_utc = datetime.datetime.combine(
            today - datetime.timedelta(days=1),
            datetime.time.min,
            tzinfo=datetime.UTC,
        )
        end_utc = datetime.datetime.combine(
            today + datetime.timedelta(days=2),
            datetime.time.max,
            tzinfo=datetime.UTC,
        )

        self.log(
            (
                f"Fetching high/low tide predictions for station {station_id} from "
                f"{_format_erddap_time(begin_utc)} to {_format_erddap_time(end_utc)}"
            ),
            level=logging.DEBUG,
            location_code=location_code,
        )

        url = self._build_tide_high_low_url(
            station_id=station_id,
            begin_utc=begin_utc,
            end_utc=end_utc,
        )
        raw_df = await self.request_with_retry(
            location_code,
            self._execute_csv_request,
            url,
        )

        return _high_low_tide_predictions_to_feed(
            raw_df=raw_df,
            height_offset_m=height_offset_m,
        )


def _format_erddap_time(timestamp: datetime.datetime) -> str:
    """Format a timezone-aware datetime for ERDDAP ISO constraints."""
    if timestamp.tzinfo is None:
        raise ValueError("ERDDAP query timestamps must be timezone-aware")
    return timestamp.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _high_low_tide_predictions_to_feed(
    *,
    raw_df: pd.DataFrame,
    height_offset_m: float,
) -> pd.DataFrame:
    """Convert Marine Institute high/low rows into app-native tide events.

    The parsed index is the absolute instant ERDDAP published, timezone-aware in
    UTC, so a repeated event is identified by instant rather than by wall time.
    """
    required_columns = {"time", "Water_Level_ODMalin", "tide_time_category"}
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        raise MarineInstituteDataError(
            "Marine Institute tide response missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = (
        raw_df.assign(
            time=lambda frame: pd.to_datetime(frame["time"], utc=True, errors="coerce"),
            prediction=lambda frame: (
                (
                    pd.to_numeric(frame["Water_Level_ODMalin"], errors="coerce")
                    + height_offset_m
                )
                * METERS_TO_FEET
            ),
            type=lambda frame: (
                frame["tide_time_category"]
                .astype(str)
                .str.lower()
                .map({"high": "high", "low": "low"})
            ),
        )
        .dropna(subset=["time", "prediction", "type"])
        # Order by the absolute instant with a stable sort, so a repeated event
        # deterministically keeps the first row ERDDAP reported for it.
        .sort_values("time", kind="stable")
        .drop_duplicates(subset=["time", "type"])
        .set_index("time")[["prediction", "type"]]
    )
    if df.empty:
        raise StationUnavailableError(
            "Marine Institute tide response had no usable high/low events"
        )

    result = pd.DataFrame(
        {
            "prediction": df["prediction"].to_numpy(dtype=float),
            "type": pd.Categorical(
                df["type"],
                categories=TIDE_TYPE_CATEGORIES,
            ),
        },
        index=df.index,
    )
    result.index.name = "time"
    return result

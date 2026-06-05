"""NOAA NDBC (National Data Buoy Center) API client.

References:
    - NDBC Measurement Descriptions: https://www.ndbc.noaa.gov/faq/measdes.shtml
"""

import asyncio
import datetime
import logging
from calendar import month_abbr
from io import StringIO
from typing import Literal

import aiohttp
import pandas as pd

from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,
    RetryableClientError,
    StationUnavailableError,
    is_retryable_http_status,
)
from shallweswim.util import c_to_f


class NdbcApiError(BaseClientError):
    """Base error for NOAA NDBC API calls."""


class NdbcDataError(NdbcApiError):
    """Error in data returned by NOAA NDBC API."""


NDBC_BASE_URL = "https://www.ndbc.noaa.gov/"
NDBC_REALTIME_PATH = "data/realtime2/"
NDBC_HISTORICAL_VIEW_PATH = "view_text_file.php"
NDBC_DATA_PATH = "data/"
NDBC_HISTORICAL_DATA_PATH = "data/historical/"
NDBC_CURRENT_MONTH_FILE_EXTENSION = ".txt"
NDBC_MAX_CONCURRENT_REQUESTS = 3
NDBC_HISTORICAL_THRESHOLD = datetime.timedelta(days=44)
NDBC_NAN_VALUES = ["MM", 99.0, 999, 9999, 9999.0]

_ndbc_request_semaphore: asyncio.Semaphore | None = None
_ndbc_request_loop: asyncio.AbstractEventLoop | None = None


def _get_ndbc_request_semaphore() -> asyncio.Semaphore:
    """Return a process-local semaphore for NDBC HTTP concurrency."""
    global _ndbc_request_loop, _ndbc_request_semaphore

    loop = asyncio.get_running_loop()
    if _ndbc_request_semaphore is None or _ndbc_request_loop is not loop:
        _ndbc_request_loop = loop
        _ndbc_request_semaphore = asyncio.Semaphore(NDBC_MAX_CONCURRENT_REQUESTS)
    return _ndbc_request_semaphore


class NdbcApi(BaseApiClient):
    """NOAA NDBC API client using direct async HTTP requests."""

    @property
    def client_type(self) -> str:
        return "ndbc"

    def __init__(self, session: aiohttp.ClientSession | None) -> None:
        """Initialize the NDBC client."""
        super().__init__(session)

    async def _execute_request(
        self,
        station_id: str,
        mode: Literal["stdmet", "ocean"],
        start_time: str,
        end_time: str,
        location_code: str,
    ) -> pd.DataFrame:
        """Fetch and parse raw NDBC data for a station/date range."""
        start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%d")
        station_id = station_id.lower()

        urls = self._build_request_urls(
            station_id=station_id,
            mode=mode,
            start_time=start_dt,
            end_time=end_dt,
            now=datetime.datetime.now(),
        )
        self.log(
            (
                f"Executing NDBC request for station {station_id}, mode={mode}, "
                f"start={start_time}, end={end_time}, urls={len(urls)}"
            ),
            location_code=location_code,
        )

        components: list[pd.DataFrame] = []
        unavailable_urls: list[str] = []

        for url in urls:
            response = await self._fetch_url(
                url=url,
                station_id=station_id,
                location_code=location_code,
            )
            if response.status == 404 or "Unable to access data file" in response.body:
                unavailable_urls.append(url)
                continue
            if response.status != 200:
                raise NdbcApiError(
                    f"NDBC request failed with HTTP {response.status} for {url}"
                )

            parsed = self._parse_response_body(
                body=response.body,
                station_id=station_id,
                mode=mode,
            )
            if not parsed.empty:
                components.append(parsed)

        if not components:
            error_msg = f"NDBC station {station_id} returned no data"
            if unavailable_urls:
                error_msg += f" ({len(unavailable_urls)} unavailable URL(s))"
            self.log(error_msg, level=logging.WARNING, location_code=location_code)
            raise StationUnavailableError(error_msg)

        result = pd.concat(components)
        result = (
            result.reset_index()
            .drop_duplicates(subset="timestamp", keep="first")
            .set_index("timestamp")
            .sort_index()
        )
        result = result.loc[(result.index >= start_dt) & (result.index <= end_dt)]
        if result.empty:
            raise StationUnavailableError(
                f"NDBC station {station_id} returned no data in requested range"
            )

        result["station_id"] = station_id
        return result.set_index("station_id", append=True)

    async def _fetch_url(
        self,
        *,
        url: str,
        station_id: str,
        location_code: str,
    ) -> "_NdbcHttpResponse":
        """Fetch one NDBC URL with real async timeout and process-wide throttling."""
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)

        try:
            if self._session is None:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    return await self._fetch_url_with_session(
                        session=session,
                        url=url,
                        station_id=station_id,
                        location_code=location_code,
                        timeout=timeout,
                    )
            return await self._fetch_url_with_session(
                session=self._session,
                url=url,
                station_id=station_id,
                location_code=location_code,
                timeout=timeout,
            )
        except TimeoutError as e:
            raise RetryableClientError(
                f"Request timed out after {self.REQUEST_TIMEOUT}s for NDBC station {station_id}"
            ) from e
        except aiohttp.ClientError as e:
            raise RetryableClientError(
                f"Network error during NDBC request for station {station_id}: "
                f"{e.__class__.__name__}: {e}"
            ) from e

    async def _fetch_url_with_session(
        self,
        *,
        session: aiohttp.ClientSession,
        url: str,
        station_id: str,
        location_code: str,
        timeout: aiohttp.ClientTimeout,
    ) -> "_NdbcHttpResponse":
        semaphore = _get_ndbc_request_semaphore()
        async with semaphore:
            self.log(f"GET {url}", level=logging.DEBUG, location_code=location_code)
            async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
                body = await resp.text()
                if is_retryable_http_status(resp.status):
                    raise RetryableClientError(
                        f"NDBC request for station {station_id} returned HTTP {resp.status}"
                    )
                return _NdbcHttpResponse(status=resp.status, body=body)

    @classmethod
    def _build_request_urls(
        cls,
        *,
        station_id: str,
        mode: Literal["stdmet", "ocean"],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        now: datetime.datetime,
    ) -> list[str]:
        """Build NDBC URLs using the realtime/monthly/yearly text-file split."""
        if mode not in {"stdmet", "ocean"}:
            raise NdbcDataError(f"Unsupported NDBC mode '{mode}'")

        if now - start_time < NDBC_HISTORICAL_THRESHOLD:
            return [cls._realtime_url(station_id=station_id, mode=mode)]

        current_year = now.year
        has_realtime = now - end_time < NDBC_HISTORICAL_THRESHOLD
        months_req_date = now - NDBC_HISTORICAL_THRESHOLD
        months_req_year = months_req_date.year
        last_available_month = months_req_date.month

        urls: list[str] = []
        for year in range(start_time.year, min(current_year, end_time.year + 1)):
            urls.append(cls._historical_year_url(station_id, mode, year))

        if end_time.year == months_req_year:
            for month in range(
                start_time.month,
                min(end_time.month, last_available_month) + 1,
            ):
                urls.append(
                    cls._historical_month_url(
                        station_id, mode, month, year=end_time.year
                    )
                )
            if last_available_month <= end_time.month:
                urls.append(
                    cls._current_month_url(station_id, mode, last_available_month)
                )

        if has_realtime:
            urls.append(cls._realtime_url(station_id=station_id, mode=mode))

        return urls

    @staticmethod
    def _mode_file_parts(mode: Literal["stdmet", "ocean"]) -> tuple[str, str, str]:
        if mode == "stdmet":
            return "stdmet", ".txt", "h"
        return "ocean", ".ocean", "o"

    @classmethod
    def _realtime_url(cls, *, station_id: str, mode: Literal["stdmet", "ocean"]) -> str:
        _, file_format, _ = cls._mode_file_parts(mode)
        return f"{NDBC_BASE_URL}{NDBC_REALTIME_PATH}{station_id.upper()}{file_format}"

    @classmethod
    def _historical_year_url(
        cls, station_id: str, mode: Literal["stdmet", "ocean"], year: int
    ) -> str:
        data_format, _, historical_identifier = cls._mode_file_parts(mode)
        filename = f"{station_id}{historical_identifier}{year}.txt.gz"
        data_dir = f"{NDBC_HISTORICAL_DATA_PATH}{data_format}/"
        return cls._historical_view_url(filename=filename, data_dir=data_dir)

    @classmethod
    def _historical_month_url(
        cls,
        station_id: str,
        mode: Literal["stdmet", "ocean"],
        month: int,
        *,
        year: int,
    ) -> str:
        data_format, _, _ = cls._mode_file_parts(mode)
        month_name = month_abbr[month].capitalize()
        filename = f"{station_id}{month}{year}.txt.gz"
        data_dir = f"{NDBC_DATA_PATH}{data_format}/{month_name}/"
        return cls._historical_view_url(filename=filename, data_dir=data_dir)

    @classmethod
    def _current_month_url(
        cls, station_id: str, mode: Literal["stdmet", "ocean"], month: int
    ) -> str:
        data_format, _, _ = cls._mode_file_parts(mode)
        month_name = month_abbr[month].capitalize()
        return (
            f"{NDBC_BASE_URL}{NDBC_DATA_PATH}{data_format}/{month_name}/"
            f"{station_id}{NDBC_CURRENT_MONTH_FILE_EXTENSION}"
        )

    @staticmethod
    def _historical_view_url(*, filename: str, data_dir: str) -> str:
        return (
            f"{NDBC_BASE_URL}{NDBC_HISTORICAL_VIEW_PATH}"
            f"?filename={filename}&dir={data_dir}"
        )

    @classmethod
    def _parse_response_body(
        cls,
        *,
        body: str,
        station_id: str,
        mode: Literal["stdmet", "ocean"],
    ) -> pd.DataFrame:
        header: list[str] = []
        data: list[str] = []
        for line in StringIO(body):
            if line.startswith("#"):
                header.append(line)
            elif line.strip():
                data.append(line)

        if not data:
            return pd.DataFrame()
        if not header:
            raise NdbcDataError(f"NDBC response for station {station_id} has no header")

        names = [name for name in header[0].strip("#").strip().split(" ") if name]
        if not names:
            raise NdbcDataError(
                f"NDBC response for station {station_id} has no column names"
            )

        first_data_width = len([value for value in data[0].strip().split(" ") if value])
        if first_data_width != len(names):
            raise NdbcDataError(
                f"NDBC {mode} response for station {station_id} has {first_data_width} "
                f"values but {len(names)} columns"
            )

        try:
            df = pd.read_csv(
                StringIO("".join(data)),
                names=names,
                sep=r"\s+",
                na_values=NDBC_NAN_VALUES,
            )
        except (NotImplementedError, TypeError, ValueError) as e:
            raise NdbcDataError(
                f"Failed to parse NDBC {mode} response for station {station_id}: {e}"
            ) from e

        date_col_names = names[:5]
        date_strings = df[date_col_names].astype(str).agg(" ".join, axis=1)
        try:
            df["timestamp"] = pd.to_datetime(date_strings, format="%Y %m %d %H %M")
        except ValueError as e:
            raise NdbcDataError(
                f"Failed to parse NDBC timestamps for station {station_id}: {e}"
            ) from e

        return df.drop(columns=date_col_names).set_index("timestamp")

    async def temperature(
        self,
        station_id: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        timezone: str,
        location_code: str = "unknown",
        mode: Literal["stdmet", "ocean"] = "stdmet",
    ) -> pd.DataFrame:
        """Fetch water temperature data from NDBC station with retries."""
        location_code = location_code or station_id
        begin_date_str = begin_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        raw_df = await self.request_with_retry(
            location_code=location_code,
            execute_request=self._execute_request,
            station_id=station_id,
            mode=mode,
            start_time=begin_date_str,
            end_time=end_date_str,
        )

        try:
            temp_column = "WTMP" if mode == "stdmet" else "OTMP"
            if temp_column not in raw_df.columns:
                error_msg = (
                    f"No water temperature data ('{temp_column}') available for "
                    f"NDBC station {station_id} in mode '{mode}'. "
                    f"Columns: {raw_df.columns.tolist()}"
                )
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

            temp_df = (
                raw_df[[temp_column]].copy().rename(columns={temp_column: "water_temp"})
            )
            temp_df["water_temp"] = pd.to_numeric(
                temp_df["water_temp"], errors="coerce"
            )
            temp_df["water_temp"] = temp_df["water_temp"].apply(c_to_f)

            if not isinstance(temp_df.index, pd.MultiIndex):
                error_msg = (
                    f"Expected MultiIndex from NDBC API, got {type(temp_df.index)}"
                )
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

            temp_df.index = temp_df.index.droplevel("station_id")
            if not pd.api.types.is_datetime64_any_dtype(temp_df.index):
                error_msg = (
                    "Index after dropping level is not DatetimeIndex, "
                    f"got {type(temp_df.index)}"
                )
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

            temp_df = self._fix_time(temp_df, timezone)
            temp_df.sort_index(inplace=True)

            self.log(
                (
                    f"Successfully processed {len(temp_df)} temperature readings "
                    f"for NDBC station {station_id} from {begin_date_str} to {end_date_str}"
                ),
                location_code=location_code,
            )
            return temp_df

        except Exception as e:
            error_msg = (
                f"Error processing NDBC data for station {station_id} after "
                f"successful fetch: {e.__class__.__name__}: {e}"
            )
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise NdbcDataError(error_msg) from e

    def _fix_time(self, df: pd.DataFrame, timezone: str) -> pd.DataFrame:
        """Convert UTC timestamps to local naive timestamps."""
        df.index = (
            pd.DatetimeIndex(df.index)
            .tz_localize("UTC")
            .tz_convert(timezone)
            .tz_localize(None)
        )
        df.index.name = "time"
        return df


class _NdbcHttpResponse:
    """Small HTTP response container for NDBC text requests."""

    def __init__(self, *, status: int, body: str) -> None:
        self.status = status
        self.body = body

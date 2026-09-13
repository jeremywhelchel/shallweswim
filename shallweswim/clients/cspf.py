"""Channel Swimming & Piloting Federation Sandettie data client."""

import datetime
import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

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
    retryable_timeout_error,
)
from shallweswim.util import c_to_f


class CspfApiError(BaseClientError):
    """Base error for CSPF API calls."""


class CspfDataError(CspfApiError):
    """Error in data returned by CSPF pages."""


CSPF_BASE_URL = "https://cspf.co.uk/"
CSPF_PROVIDER = "cspf"
CSPF_MAX_CONCURRENT_REQUESTS = 2
CSPF_USER_AGENT = "shallweswim/0.1 (+https://shallweswim.today)"
CSPF_TEMP_ARRAY_RE = re.compile(
    r"var\s+a_[A-Za-z0-9]+\s*=\s*\[(?P<body>.*?)\];",
    re.DOTALL,
)
CSPF_TEMP_POINT_RE = re.compile(
    r"\['(?P<timestamp_ms>\d+)'\s*,\s*(?P<temperature_c>[-+]?\d+(?:\.\d+)?)\]"
)


@dataclass(frozen=True)
class _CspfPage:
    url: str
    body: str


class CspfApi(BaseApiClient):
    """Client for CSPF Sandettie historical water temperature pages.

    CSPF embeds each reading as epoch milliseconds, an absolute instant, so all
    methods return pandas DataFrames indexed by timezone-aware UTC timestamps
    and de-duplicate on that instant. Request windows stay station-local: the
    caller's naive edges name the local calendar year whose pages are read and
    trim the result, which is why the methods still take a ``timezone``.
    """

    @property
    def client_type(self) -> str:
        return "cspf"

    async def sandettie_temperature(
        self,
        *,
        begin_date: datetime.datetime,
        end_date: datetime.datetime,
        location_code: str,
        timezone: datetime.tzinfo,
        station_slug: str = "sandettie-data",
    ) -> pd.DataFrame:
        """Fetch Sandettie historical sea temperatures for a date range.

        Args:
            begin_date: Naive station-local instant that opens the window
            end_date: Naive station-local instant that closes the window
            location_code: Location code for logging purposes
            timezone: Station timezone the request window is expressed in
            station_slug: CSPF page slug for the station

        Returns:
            DataFrame indexed by timezone-aware UTC time, with columns:
                water_temp: float - Water temperature in °F

        Raises:
            CspfDataError: If the window spans more than one year
            StationUnavailableError: If CSPF published no data in the window
        """
        return await self.request_with_retry(
            location_code,
            self._execute_request,
            begin_date=begin_date,
            end_date=end_date,
            station_slug=station_slug,
            timezone=timezone,
        )

    async def _execute_request(
        self,
        *,
        begin_date: datetime.datetime,
        end_date: datetime.datetime,
        location_code: str,
        station_slug: str,
        timezone: datetime.tzinfo,
    ) -> pd.DataFrame:
        if begin_date.year != end_date.year:
            raise CspfDataError("CSPF temperature fetch expects a single year range")

        year = begin_date.year
        begin_utc = _window_edge_utc(begin_date, timezone)
        end_utc = _window_edge_utc(end_date, timezone)

        try:
            temperature_frame = await self._fetch_monthly_temperature(
                year=year,
                station_slug=station_slug,
                location_code=location_code,
            )
            if temperature_frame.empty:
                annual_page = await self._fetch_page(
                    path=f"{station_slug}/{year}",
                    location_code=location_code,
                )
                temperature_frame = self._parse_temperature_page(annual_page)
        except TimeoutError as e:
            raise retryable_timeout_error(
                timeout_seconds=self.REQUEST_TIMEOUT,
                provider="CSPF",
                resource=f"{station_slug}/{year}",
            ) from e
        except aiohttp.ClientError as e:
            raise retryable_network_error(
                provider="CSPF",
                action=f"for {station_slug}/{year}",
                error=e,
            ) from e

        if temperature_frame.empty:
            message = f"CSPF Sandettie returned no temperature data for {year}"
            self.log(message, level=logging.WARNING, location_code=location_code)
            raise StationUnavailableError(message)

        # Order by the absolute instant with a stable sort, so a repeated
        # instant keeps the value of whichever page supplied it last.
        result = temperature_frame.sort_index(kind="stable")
        result = result[~result.index.duplicated(keep="last")]
        result = result.loc[(result.index >= begin_utc) & (result.index <= end_utc)]
        if result.empty:
            message = (
                f"CSPF Sandettie returned no temperature data in requested range "
                f"{begin_date.date()} to {end_date.date()}"
            )
            self.log(message, level=logging.WARNING, location_code=location_code)
            raise StationUnavailableError(message)

        return result

    async def _fetch_monthly_temperature(
        self,
        *,
        year: int,
        station_slug: str,
        location_code: str,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for month in range(1, 13):
            page = await self._fetch_page(
                path=f"{station_slug}/{year}/{month}",
                location_code=location_code,
            )
            frame = self._parse_temperature_page(page)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()
        # Neighbouring month pages overlap at their edges; a stable sort by the
        # absolute instant keeps the later month's value for a shared instant.
        result = pd.concat(frames).sort_index(kind="stable")
        return result[~result.index.duplicated(keep="last")]

    async def _fetch_page(self, *, path: str, location_code: str) -> _CspfPage:
        url = urljoin(CSPF_BASE_URL, path)
        timeout = request_timeout(self.REQUEST_TIMEOUT)
        try:
            if self._session is None:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    return await self._fetch_page_with_session(
                        session=session,
                        url=url,
                        location_code=location_code,
                        timeout=timeout,
                    )
            return await self._fetch_page_with_session(
                session=self._session,
                url=url,
                location_code=location_code,
                timeout=timeout,
            )
        except TimeoutError as e:
            raise retryable_timeout_error(
                timeout_seconds=self.REQUEST_TIMEOUT,
                provider="CSPF",
                resource=url,
            ) from e
        except aiohttp.ClientError as e:
            raise retryable_network_error(
                provider="CSPF",
                action=f"for {url}",
                error=e,
            ) from e

    async def _fetch_page_with_session(
        self,
        *,
        session: aiohttp.ClientSession,
        url: str,
        location_code: str,
        timeout: aiohttp.ClientTimeout,
    ) -> _CspfPage:
        async with provider_request_slot(CSPF_PROVIDER, CSPF_MAX_CONCURRENT_REQUESTS):
            self.log(f"GET {url}", level=logging.DEBUG, location_code=location_code)
            async with session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": CSPF_USER_AGENT},
            ) as response:
                body = await response.text()
                raise_if_retryable_http_status(
                    response.status,
                    f"CSPF request returned HTTP {response.status} for {url}",
                )
                if response.status == 404:
                    return _CspfPage(url=url, body="")
                if response.status != 200:
                    raise CspfApiError(
                        f"CSPF request failed with HTTP {response.status} for {url}"
                    )
                return _CspfPage(url=url, body=body)

    @classmethod
    def _parse_temperature_page(cls, page: _CspfPage) -> pd.DataFrame:
        """Parse CSPF's embedded sea-temperature JavaScript array.

        Each point carries epoch milliseconds, so the parsed index is the
        absolute instant CSPF published, timezone-aware in UTC.
        """
        match = CSPF_TEMP_ARRAY_RE.search(page.body)
        if not match:
            return pd.DataFrame()

        rows = [
            (
                int(point.group("timestamp_ms")),
                c_to_f(float(point.group("temperature_c"))),
            )
            for point in CSPF_TEMP_POINT_RE.finditer(match.group("body"))
        ]
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=["timestamp", "water_temp"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp")
        frame.index.name = "time"
        return frame.sort_index(kind="stable")


def _window_edge_utc(
    edge: datetime.datetime, timezone: datetime.tzinfo
) -> pd.Timestamp:
    """Return one naive station-local window edge as a UTC instant.

    CSPF windows are station-local: the edge names the local calendar year whose
    pages are read and trims the parsed instants. An edge that a daylight saving
    transition makes nonexistent or ambiguous is resolved deterministically
    rather than raising: window edges select data, they do not describe an
    observation.
    """
    return (
        pd.Timestamp(edge)
        .tz_localize(timezone, nonexistent="shift_forward", ambiguous=False)
        .tz_convert("UTC")
    )

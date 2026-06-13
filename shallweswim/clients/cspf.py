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
    """Client for CSPF Sandettie historical water temperature pages."""

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
        """Fetch Sandettie historical sea temperatures for a date range."""
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

        try:
            temperature_frame = await self._fetch_monthly_temperature(
                year=year,
                station_slug=station_slug,
                location_code=location_code,
                timezone=timezone,
            )
            if temperature_frame.empty:
                annual_page = await self._fetch_page(
                    path=f"{station_slug}/{year}",
                    location_code=location_code,
                )
                temperature_frame = self._parse_temperature_page(
                    annual_page,
                    timezone=timezone,
                )
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

        result = temperature_frame.sort_index()
        result = result[~result.index.duplicated(keep="last")]
        result = result.loc[(result.index >= begin_date) & (result.index <= end_date)]
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
        timezone: datetime.tzinfo,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for month in range(1, 13):
            page = await self._fetch_page(
                path=f"{station_slug}/{year}/{month}",
                location_code=location_code,
            )
            frame = self._parse_temperature_page(page, timezone=timezone)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames).sort_index()
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
    def _parse_temperature_page(
        cls, page: _CspfPage, *, timezone: datetime.tzinfo
    ) -> pd.DataFrame:
        """Parse CSPF's embedded sea-temperature JavaScript array."""
        match = CSPF_TEMP_ARRAY_RE.search(page.body)
        if not match:
            return pd.DataFrame()

        rows = [
            (
                _local_naive_datetime(
                    int(point.group("timestamp_ms")),
                    timezone=timezone,
                ).replace(tzinfo=None),
                c_to_f(float(point.group("temperature_c"))),
            )
            for point in CSPF_TEMP_POINT_RE.finditer(match.group("body"))
        ]
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=["timestamp", "water_temp"]).set_index(
            "timestamp"
        )
        frame.index.name = "time"
        return frame.sort_index()


def _local_naive_datetime(
    timestamp_ms: int, *, timezone: datetime.tzinfo
) -> datetime.datetime:
    """Convert CSPF epoch milliseconds to station-local naive time."""
    utc_timestamp = datetime.datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=datetime.UTC,
    )
    return utc_timestamp.astimezone(timezone).replace(tzinfo=None)

"""Irish Lights MetOcean API client."""

import datetime
import json
import logging
import urllib.parse
from typing import Any

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

IRISH_LIGHTS_METOCEAN_URL = "https://cilpublic.cil.ie/MetOcean/MetOcean.ashx"
IRISH_LIGHTS_METOCEAN_ACCESS_TOKEN = "B9EF21E2-C563-4C07-94E9-198AF132C447"
IRISH_LIGHTS_PROVIDER = "irish-lights"
IRISH_LIGHTS_MAX_CONCURRENT_REQUESTS = 2
IRISH_LIGHTS_MIN_WATER_TEMP_C = 0.0
IRISH_LIGHTS_MAX_WATER_TEMP_C = 25.0


class IrishLightsApiError(BaseClientError):
    """Base error for Irish Lights API calls."""


class IrishLightsDataError(IrishLightsApiError):
    """Error in data returned by Irish Lights MetOcean."""


class IrishLightsApi(BaseApiClient):
    """Client for Irish Lights MetOcean buoy observations."""

    @property
    def client_type(self) -> str:
        return "irish-lights"

    def _build_metocean_url(
        self,
        *,
        mmsi: str,
        begin_utc: datetime.datetime,
        end_utc: datetime.datetime,
    ) -> str:
        """Build an Irish Lights MetOcean request URL."""
        query = urllib.parse.urlencode(
            {
                "accesstoken": IRISH_LIGHTS_METOCEAN_ACCESS_TOKEN,
                "MMSI": mmsi,
                "FromDate": _format_metocean_time(begin_utc),
                "ToDate": _format_metocean_time(end_utc),
            }
        )
        return f"{IRISH_LIGHTS_METOCEAN_URL}?{query}"

    async def water_temperature(
        self,
        *,
        mmsi: str,
        begin_date: datetime.datetime,
        end_date: datetime.datetime,
        timezone: datetime.tzinfo,
        location_code: str,
        min_valid_temp_c: float = IRISH_LIGHTS_MIN_WATER_TEMP_C,
        max_valid_temp_c: float = IRISH_LIGHTS_MAX_WATER_TEMP_C,
    ) -> pd.DataFrame:
        """Return buoy water-temperature observations in app-native schema."""
        begin_utc = _to_utc_aware(begin_date)
        end_utc = _to_utc_aware(end_date)
        if end_utc < begin_utc:
            raise IrishLightsDataError("end_date must be greater than begin_date")

        return await self.request_with_retry(
            location_code,
            self._execute_temperature_request,
            mmsi=mmsi,
            begin_utc=begin_utc,
            end_utc=end_utc,
            timezone=timezone,
            min_valid_temp_c=min_valid_temp_c,
            max_valid_temp_c=max_valid_temp_c,
        )

    async def _execute_temperature_request(
        self,
        *,
        mmsi: str,
        begin_utc: datetime.datetime,
        end_utc: datetime.datetime,
        timezone: datetime.tzinfo,
        min_valid_temp_c: float,
        max_valid_temp_c: float,
        location_code: str,
    ) -> pd.DataFrame:
        """Fetch and parse one Irish Lights MetOcean temperature response."""
        url = self._build_metocean_url(
            mmsi=mmsi,
            begin_utc=begin_utc,
            end_utc=end_utc,
        )
        self.log(f"GET {url}", level=logging.DEBUG, location_code=location_code)

        try:
            payload = await self._fetch_json(url=url, location_code=location_code)
        except TimeoutError as e:
            raise retryable_timeout_error(
                timeout_seconds=self.REQUEST_TIMEOUT,
                provider="Irish Lights",
                resource=f"MMSI {mmsi}",
            ) from e
        except aiohttp.ClientError as e:
            raise retryable_network_error(
                provider="Irish Lights",
                action=f"for MMSI {mmsi}",
                error=e,
            ) from e

        return _metocean_temperature_to_feed(
            payload=payload,
            timezone=timezone,
            min_valid_temp_c=min_valid_temp_c,
            max_valid_temp_c=max_valid_temp_c,
            mmsi=mmsi,
        )

    async def _fetch_json(self, *, url: str, location_code: str) -> dict[str, Any]:
        """Fetch a MetOcean JSON payload."""
        timeout = request_timeout(self.REQUEST_TIMEOUT)
        async with provider_request_slot(
            IRISH_LIGHTS_PROVIDER,
            IRISH_LIGHTS_MAX_CONCURRENT_REQUESTS,
        ):
            async with self._session.get(url, timeout=timeout) as response:
                body = await response.text()
                raise_if_retryable_http_status(
                    response.status,
                    f"Irish Lights request returned HTTP {response.status} for {url}",
                )
                if response.status != 200:
                    raise IrishLightsApiError(
                        f"Irish Lights request failed with HTTP {response.status} for {url}"
                    )

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            self.log(
                f"Failed to parse Irish Lights JSON response from {url}: {e}",
                level=logging.ERROR,
                location_code=location_code,
            )
            raise IrishLightsDataError(
                "Irish Lights response was not valid JSON"
            ) from e

        if not isinstance(payload, dict):
            raise IrishLightsDataError("Irish Lights response must be a JSON object")
        return payload


def _format_metocean_time(timestamp: datetime.datetime) -> str:
    """Format a timezone-aware datetime for Irish Lights MetOcean."""
    if timestamp.tzinfo is None:
        raise ValueError("Irish Lights query timestamps must be timezone-aware")
    return timestamp.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _to_utc_aware(timestamp: datetime.datetime) -> datetime.datetime:
    """Treat naive app timestamps as UTC and return aware UTC."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=datetime.UTC)
    return timestamp.astimezone(datetime.UTC)


def _metocean_temperature_to_feed(
    *,
    payload: dict[str, Any],
    timezone: datetime.tzinfo,
    min_valid_temp_c: float,
    max_valid_temp_c: float,
    mmsi: str,
) -> pd.DataFrame:
    """Convert Irish Lights MetOcean rows into app-native temperature rows."""
    rows = payload.get("MetOceanData")
    if not isinstance(rows, list):
        raise IrishLightsDataError("Irish Lights response missing MetOceanData list")

    error_messages = [
        str(row["Error"]) for row in rows if isinstance(row, dict) and row.get("Error")
    ]
    if error_messages:
        raise IrishLightsDataError(
            "Irish Lights returned an error row: " + "; ".join(error_messages)
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise StationUnavailableError(
            f"Irish Lights MMSI {mmsi} returned no water temperature data"
        )
    required_columns = {"hour", "WaterTemperature"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise IrishLightsDataError(
            "Irish Lights response missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.assign(
        time=pd.to_datetime(frame["hour"], utc=True, errors="coerce"),
        temperature_c=pd.to_numeric(frame["WaterTemperature"], errors="coerce"),
    )
    frame = frame.dropna(subset=["time", "temperature_c"])
    frame = frame[
        frame["temperature_c"].between(
            min_valid_temp_c,
            max_valid_temp_c,
            inclusive="both",
        )
    ]
    if frame.empty:
        raise StationUnavailableError(
            f"Irish Lights MMSI {mmsi} returned no usable water temperature data"
        )

    time_index = frame["time"].dt.tz_convert(timezone).dt.tz_localize(None)
    result = pd.DataFrame(
        {"water_temp": frame["temperature_c"].map(c_to_f).to_numpy(dtype=float)},
        index=pd.DatetimeIndex(time_index, name="time"),
    )
    result = result.sort_index()
    return result[~result.index.duplicated(keep="last")]

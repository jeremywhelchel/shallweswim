"""USGS NWIS (National Water Information System) API client.

The USGS Water Data APIs provide access to water data from rivers, lakes,
wells, and other water bodies across the United States. This module focuses on
retrieving continuous water temperature and current observations from USGS
monitoring locations.

References:
    - USGS Water Data APIs: https://api.waterdata.usgs.gov/
    - Continuous values: https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous
    - USGS Water Data for the Nation: https://waterdata.usgs.gov/nwis
"""

import datetime
import logging
import os
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

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

NWIS_BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/"
NWIS_CONTINUOUS_ITEMS_PATH = "collections/continuous/items"
NWIS_PAGE_LIMIT = 50000
NWIS_MAX_PAGES = 100
NWIS_INSTANTANEOUS_STATISTIC_ID = "00011"
USGS_WATERDATA_API_KEY_ENV = "USGS_WATERDATA_API_KEY"
NWIS_REQUEST_HEADERS = {
    "Accept": "application/geo+json, application/json",
    "User-Agent": "shallweswim/0.1 (+https://shallweswim.today)",
}


class NwisApiError(BaseClientError):
    """Base error for USGS NWIS API calls."""


class NwisConnectionError(NwisApiError):
    """Error connecting to USGS NWIS API."""


class NwisDataError(NwisApiError):
    """Error in data returned by USGS NWIS API."""


class NwisApi(BaseApiClient):
    """Client for USGS continuous water observations using direct async HTTP.

    The public methods intentionally preserve the old NWIS client interface so
    feed code can migrate without changing call sites.
    """

    @property
    def client_type(self) -> str:
        return "nwis"

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the NWIS client.

        Args:
            session: The aiohttp client session.
        """
        super().__init__(session)

    @staticmethod
    def _request_headers() -> dict[str, str]:
        """Return USGS request headers, adding an API key when configured."""
        headers = dict(NWIS_REQUEST_HEADERS)
        api_key = os.getenv(USGS_WATERDATA_API_KEY_ENV, "").strip()
        if api_key:
            headers["X-Api-Key"] = api_key
        return headers

    async def _execute_request(
        self,
        sites: str,
        service: str,
        parameterCd: list[str],
        start: str,
        end: str,
        location_code: str,
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """Fetch raw continuous observations and return legacy-compatible rows."""
        if service != "iv":
            raise NwisDataError(f"Unsupported NWIS service '{service}'")
        if len(parameterCd) != 1:
            raise NwisDataError("NWIS client expects exactly one parameter code")

        parameter_cd = parameterCd[0]
        url = urljoin(NWIS_BASE_URL, NWIS_CONTINUOUS_ITEMS_PATH)
        params = self._continuous_request_params(
            site_no=sites,
            parameter_cd=parameter_cd,
            start=start,
            end=end,
            timezone=timezone,
        )

        self.log(
            f"Executing NWIS request for site {sites}, params={parameterCd}, start={start}, end={end}",
            location_code=location_code,
        )

        # --- Fetch phase (network errors caught here) ---
        try:
            payloads = await self._fetch_json_pages(
                url=url,
                params=params,
                site_no=sites,
                location_code=location_code,
            )
        except TimeoutError as e:
            raise RetryableClientError(
                f"Request timed out after {self.REQUEST_TIMEOUT}s for NWIS site {sites}"
            ) from e
        except aiohttp.ClientError as e:
            raise RetryableClientError(
                f"Network error during NWIS request for site {sites}: "
                f"{e.__class__.__name__}: {e}"
            ) from e

        # --- Validation phase (outside try - exceptions propagate naturally) ---
        raw_result = self._parse_continuous_payloads(
            payloads=payloads,
            site_no=sites,
            parameter_cd=parameter_cd,
        )
        if raw_result.empty:
            error_msg = f"NWIS site {sites} returned no data for params {parameterCd}"
            self.log(error_msg, level=logging.WARNING, location_code=location_code)
            raise StationUnavailableError(error_msg)

        return raw_result

    @classmethod
    def _continuous_request_params(
        cls,
        *,
        site_no: str,
        parameter_cd: str,
        start: str,
        end: str,
        timezone: str,
    ) -> dict[str, str | int]:
        """Build query params for the USGS continuous values endpoint."""
        return {
            "f": "json",
            "lang": "en-US",
            "monitoring_location_id": f"USGS-{site_no}",
            "parameter_code": parameter_cd,
            "time": f"{cls._local_date_to_utc_rfc3339(start, timezone=timezone)}/{cls._local_date_to_utc_rfc3339(end, timezone=timezone, end_of_day=True)}",
            "skipGeometry": "true",
            "limit": NWIS_PAGE_LIMIT,
        }

    @staticmethod
    def _local_date_to_utc_rfc3339(
        date_string: str,
        *,
        timezone: str,
        end_of_day: bool = False,
    ) -> str:
        """Convert a local YYYY-MM-DD boundary to a UTC RFC3339 timestamp."""
        date = datetime.datetime.strptime(date_string, "%Y-%m-%d").date()
        time = datetime.time.max if end_of_day else datetime.time.min
        return (
            datetime.datetime.combine(date, time, tzinfo=ZoneInfo(timezone))
            .astimezone(datetime.UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    async def _fetch_json_pages(
        self,
        *,
        url: str,
        params: dict[str, str | int],
        site_no: str,
        location_code: str,
    ) -> list[dict[str, Any]]:
        """Fetch all pages for a USGS continuous-values request."""
        payloads: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params: dict[str, str | int] | None = params
        seen_urls: set[str] = set()

        while next_url:
            if next_url in seen_urls:
                raise NwisDataError(
                    f"NWIS pagination loop detected for site {site_no}: {next_url}"
                )
            if len(seen_urls) >= NWIS_MAX_PAGES:
                raise NwisDataError(
                    f"NWIS pagination exceeded {NWIS_MAX_PAGES} pages for site {site_no}"
                )
            seen_urls.add(next_url)

            payload = await self._fetch_json_page(
                url=next_url,
                params=next_params,
                site_no=site_no,
                location_code=location_code,
            )
            payloads.append(payload)
            next_url = self._next_page_url(payload)
            next_params = None

        return payloads

    async def _fetch_json_page(
        self,
        *,
        url: str,
        params: dict[str, str | int] | None,
        site_no: str,
        location_code: str,
    ) -> dict[str, Any]:
        """Fetch one USGS JSON page."""
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        self.log(f"GET {url}", level=logging.DEBUG, location_code=location_code)

        async with self._session.get(
            url,
            params=params,
            timeout=timeout,
            headers=self._request_headers(),
        ) as response:
            if response.status != 200:
                error_msg = (
                    f"NWIS request for site {site_no} returned HTTP {response.status}"
                )
                if is_retryable_http_status(response.status):
                    raise RetryableClientError(error_msg)
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NwisConnectionError(error_msg)

            try:
                payload = await response.json()
            except Exception as e:
                raise NwisDataError(
                    f"Failed to parse NWIS JSON response for site {site_no}: {e}"
                ) from e

        if not isinstance(payload, dict):
            raise NwisDataError(f"NWIS response for site {site_no} was not an object")
        return payload

    @staticmethod
    def _next_page_url(payload: dict[str, Any]) -> str | None:
        """Return the next page URL from a USGS FeatureCollection payload."""
        links = payload.get("links", [])
        if not isinstance(links, list):
            return None

        for link in links:
            if (
                isinstance(link, dict)
                and link.get("rel") == "next"
                and isinstance(link.get("href"), str)
            ):
                return link["href"]
        return None

    @staticmethod
    def _parse_continuous_payloads(
        *,
        payloads: list[dict[str, Any]],
        site_no: str,
        parameter_cd: str,
    ) -> pd.DataFrame:
        """Parse USGS FeatureCollection pages into the legacy raw NWIS shape."""
        records: list[tuple[pd.Timestamp, float]] = []
        matched_observations = 0
        invalid_values = 0

        for payload in payloads:
            features = payload.get("features")
            if not isinstance(features, list):
                raise NwisDataError(
                    f"NWIS response for site {site_no} missing features list"
                )

            for feature in features:
                if not isinstance(feature, dict):
                    raise NwisDataError(
                        f"NWIS response for site {site_no} contained invalid feature"
                    )
                properties = feature.get("properties")
                if not isinstance(properties, dict):
                    raise NwisDataError(
                        f"NWIS response for site {site_no} contained feature without properties"
                    )

                if properties.get("parameter_code") != parameter_cd:
                    continue
                if properties.get("statistic_id") != NWIS_INSTANTANEOUS_STATISTIC_ID:
                    continue

                matched_observations += 1
                timestamp_value = properties.get("time")
                if not isinstance(timestamp_value, str):
                    raise NwisDataError(
                        f"NWIS observation for site {site_no} missing timestamp"
                    )

                try:
                    timestamp = pd.Timestamp(timestamp_value)
                except Exception as e:
                    raise NwisDataError(
                        f"NWIS observation for site {site_no} had invalid timestamp"
                    ) from e

                value = pd.to_numeric(properties.get("value"), errors="coerce")
                if pd.isna(value):
                    invalid_values += 1
                    continue
                if timestamp.tz is None:
                    raise NwisDataError(
                        f"NWIS timestamp for site {site_no} was not timezone-aware"
                    )

                records.append((timestamp, float(value)))

        if not records:
            if matched_observations and invalid_values:
                raise NwisDataError(
                    f"NWIS response for site {site_no} had observations but no parseable values"
                )
            return pd.DataFrame()

        index = pd.DatetimeIndex([record[0] for record in records], name="datetime")
        column_name = f"{parameter_cd}_{NWIS_INSTANTANEOUS_STATISTIC_ID}"
        return pd.DataFrame(
            {
                "site_no": site_no,
                column_name: [record[1] for record in records],
            },
            index=index,
        )

    async def temperature(
        self,
        site_no: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        timezone: str,
        location_code: str = "unknown",
        parameter_cd: str = "00010",  # Default is water temperature (in Celsius)
    ) -> pd.DataFrame:
        """Fetch water temperature data from USGS NWIS station with retries.

        Args:
            site_no: USGS site number (e.g., '01646500')
            begin_date: Start date for data fetch
            end_date: End date for data fetch
            timezone: Timezone to convert timestamps to
            location_code: Optional location code for logging
            parameter_cd: USGS parameter code, default '00010' for water temperature
                          Note: Some stations may use '00011' instead

        Returns:
            DataFrame with index=time and columns:
                water_temp: float - Water temperature in °F

        Raises:
            NwisConnectionError: If connection to API fails
            NwisDataError: If API returns error or invalid data
        """
        # Format dates as strings for the NWIS API
        location_code = (
            location_code or site_no
        )  # Use site_no if location_code is empty
        begin_date_str = begin_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Convert parameter_cd to list if it's a string
        param_list = [parameter_cd] if isinstance(parameter_cd, str) else parameter_cd

        # Call the execution logic via the retry wrapper
        raw_result = await self.request_with_retry(
            location_code,
            self._execute_request,
            site_no,
            "iv",
            param_list,
            begin_date_str,
            end_date_str,
            timezone=timezone,
        )

        # --- Post-processing starts (UNCHANGED from original logic) ---
        # Wrap unexpected errors during post-processing
        try:
            # Find the temperature column in the result
            # The exact column name format can vary, so we need to search for the parameter code
            temp_columns = [col for col in raw_result.columns if parameter_cd in col]

            if not temp_columns:
                error_msg = f"No water temperature data (parameter {parameter_cd}) available for NWIS site {site_no}"
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NwisDataError(error_msg)

            # Extract the first temperature column found
            temp_column = temp_columns[0]

            # Create a new DataFrame with just the temperature data
            temp_df = pd.DataFrame(
                {"water_temp": raw_result[temp_column]}, index=raw_result.index
            )

            # Convert temperature from Celsius to Fahrenheit if parameter is 00010
            if parameter_cd == "00010":
                self.log(
                    "Converting temperature from Celsius to Fahrenheit for parameter 00010",
                    location_code=location_code,
                )
                temp_df["water_temp"] = temp_df["water_temp"].map(c_to_f)

            # Convert timestamps to local timezone
            temp_df = self._fix_time(temp_df, timezone)

            # Sort by time to ensure chronological order
            temp_df.sort_index(inplace=True)

            self.log(
                f"Successfully processed {len(temp_df)} temperature readings for NWIS site {site_no} from {begin_date_str} to {end_date_str}",
                location_code=location_code,
            )

            return temp_df

        except NwisDataError:  # Let specific NwisDataError raised above propagate
            raise
        except Exception as e:
            # Catch *unexpected* errors during post-processing (column finding, conversion, time fixing)
            error_msg = f"Unexpected error processing NWIS data for site {site_no} after successful fetch: {e.__class__.__name__}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            # Raise as NwisDataError as it indicates an issue with the retrieved data format/content
            raise NwisDataError(error_msg) from e
        # --- Post-processing ends ---

    async def currents(
        self,
        site_no: str,
        parameter_cd: str,
        timezone: str,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Fetch current velocity/direction data from USGS NWIS station.

        Uses the 'iv' (instantaneous values) service.
        Currently focuses on discharge (00060, 00061) or velocity (00055) parameters.

        Args:
            site_no: USGS site number.
            parameter_cd: USGS parameter code(s) for velocity/discharge (e.g., '00060').
            timezone: Timezone to convert timestamps to.
            location_code: Optional location code for logging.

        Returns:
            DataFrame with index=time and columns like:
                discharge_cfs: float - Discharge in cubic feet per second (if param 00060)
                discharge_cms: float - Discharge in cubic meters per second (if param 00061)
                velocity_fps: float - Velocity in feet per second (if param 00055)

        Raises:
            NwisConnectionError: If connection to API fails.
            NwisDataError: If API returns no data or expected columns are missing.
            NwisApiError: For other unexpected errors during data processing.
        """
        self.log(
            f"Fetching NWIS current data for site {site_no}, param {parameter_cd}",
            location_code=location_code,
        )

        # Define time range for 'iv' service (e.g., last 24 hours)
        # NWIS often returns only the most recent value for 'iv' regardless of range,
        # but providing a range is good practice.
        end_date = datetime.datetime.now(datetime.UTC).date()
        start_date = end_date - datetime.timedelta(days=1)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        try:
            raw_result = await self.request_with_retry(
                location_code=location_code,
                execute_request=self._execute_request,
                sites=site_no,
                service="iv",  # Instantaneous values
                parameterCd=[parameter_cd],
                start=start_date_str,
                end=end_date_str,
                timezone=timezone,
            )

            # Post-processing similar to temperature method
            # Find the data column based on parameter code
            data_columns = [col for col in raw_result.columns if parameter_cd in col]

            if not data_columns:
                error_msg = f"No current data (parameter {parameter_cd}) available for NWIS site {site_no}"
                self.log(error_msg, level=logging.WARNING, location_code=location_code)
                raise NwisDataError(error_msg)

            # Select the first matching column
            data_column_name = data_columns[0]

            # Determine standard column name based on parameter code
            if parameter_cd == "00060":
                output_col_name = "discharge_cfs"
            elif parameter_cd == "00061":
                output_col_name = "discharge_cms"
            elif (
                parameter_cd == "00055" or parameter_cd == "72255"
            ):  # Mean velocity, ft/sec
                output_col_name = "velocity_fps"
            else:
                # Fallback for unknown but present parameter codes
                output_col_name = f"value_{parameter_cd}"
                self.log(
                    f"Using fallback column name '{output_col_name}' for unknown parameter {parameter_cd}",
                    level=logging.WARNING,
                    location_code=location_code,
                )

            # Create DataFrame with standard column name
            current_df = pd.DataFrame(
                {output_col_name: raw_result[data_column_name]}, index=raw_result.index
            )

            # Convert timestamps to local timezone
            current_df = self._fix_time(current_df, timezone)

            # Sort by time
            current_df.sort_index(inplace=True)

            self.log(
                f"Successfully processed {len(current_df)} current readings for NWIS site {site_no} (param {parameter_cd})",
                location_code=location_code,
            )
            return current_df

        except (NwisDataError, RetryableClientError, NwisApiError):
            # Let specific handled errors propagate
            raise
        except Exception as e:
            # Catch *unexpected* errors during processing
            error_msg = f"Unexpected error processing NWIS current data for site {site_no}: {e.__class__.__name__}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise NwisApiError(error_msg) from e

    def _fix_time(self, df: pd.DataFrame, timezone: str) -> pd.DataFrame:
        """Convert timestamps to local timezone.

        Args:
            df: DataFrame with timestamps in the index (either naive or tz-aware)
            timezone: Timezone to convert timestamps to

        Returns:
            DataFrame with local timezone timestamps (naive datetimes)
        """

        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise NwisApiError(
                "NWIS timestamps must be timezone-aware before conversion"
            )

        # Convert to the location's timezone and then make the timestamps naive again.
        datetime_index = df.index
        local_index = datetime_index.tz_convert(timezone)
        df.index = local_index.tz_localize(None)

        # Rename the index to 'time' to match our convention
        df.index.name = "time"

        return df

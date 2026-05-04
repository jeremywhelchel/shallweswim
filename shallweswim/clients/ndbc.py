"""NOAA NDBC (National Data Buoy Center) API client.

References:
    - NDBC Measurement Descriptions: https://www.ndbc.noaa.gov/faq/measdes.shtml
"""

# Standard library imports
import asyncio
import datetime
import logging
from functools import partial
from typing import Any

# Third-party imports
import pandas as pd
from ndbc_api import NdbcApi as NdbcApiClient  # Alias to avoid name clash

# Local imports
from shallweswim.clients.base import (
    RETRYABLE_REQUESTS_EXCEPTIONS,
    BaseApiClient,
    BaseClientError,
    RetryableClientError,
    StationUnavailableError,
    get_blocking_executor,
)
from shallweswim.util import c_to_f


class NdbcApiError(BaseClientError):
    """Base error for NOAA NDBC API calls."""


class NdbcDataError(NdbcApiError):
    """Error in data returned by NOAA NDBC API (e.g., missing data, wrong format)."""


class NdbcApi(BaseApiClient):
    """NOAA NDBC API client using the ndbc-api library.

    The third-party ndbc-api client performs its own synchronous HTTP requests,
    so this wrapper does not use the shared aiohttp session directly. It still
    follows the BaseApiClient fetch/validation split and runs blocking calls in
    the shared executor so retries, logging, and shutdown behavior stay
    consistent with the other clients.
    """

    @property
    def client_type(self) -> str:
        return "ndbc"

    def __init__(self, session: Any) -> None:
        """Initialize the NDBC client.

        Args:
            session: The aiohttp client session (required by base class, not used here).
        """
        super().__init__(session)

    async def _execute_request(
        self,
        station_id: str,
        mode: str,
        start_time: str,
        end_time: str,
        location_code: str,
    ) -> pd.DataFrame:
        """Executes the ndbc_api.get_data call within asyncio.to_thread and handles errors.

        This method is called by the `request_with_retry` logic in the base class.

        Args:
            station_id: NDBC station ID.
            mode: Data mode ('stdmet' or 'ocean').
            start_time: Start date string (YYYY-MM-DD).
            end_time: End date string (YYYY-MM-DD).
            location_code: Location code for logging.

        Returns:
            The raw DataFrame returned by ndbc_api.get_data.

        Raises:
            RetryableClientError: If a transient connection/timeout error occurs.
            NdbcDataError: If the API returns unexpected data type (e.g., dict instead of DataFrame).
            NdbcApiError: For other unexpected errors during the ndbc_api call.
        """
        self.log(
            f"Executing NDBC request for station {station_id}, mode={mode}, start={start_time}, end={end_time}",
            location_code=location_code,
        )

        # --- Fetch phase (network errors caught here) ---
        try:
            # Initialize the synchronous NDBC API client within the method
            # because it doesn't seem thread-safe or designed for reuse across awaits.
            api = NdbcApiClient()

            # Fetch the data using our shared executor to avoid blocking.
            # Using our own executor (vs asyncio.to_thread's default) allows us
            # to control shutdown - important for tests where threads may be stuck.
            loop = asyncio.get_running_loop()
            raw_result = await asyncio.wait_for(
                loop.run_in_executor(
                    get_blocking_executor(),
                    partial(
                        api.get_data,
                        station_id=station_id,
                        mode=mode,
                        start_time=start_time,
                        end_time=end_time,
                    ),
                ),
                timeout=self.REQUEST_TIMEOUT,
            )
        except TimeoutError as e:
            # Convert timeout to our retryable error type
            error_msg = f"Request timed out after {self.REQUEST_TIMEOUT}s for NDBC station {station_id}"
            raise RetryableClientError(error_msg) from e
        except RETRYABLE_REQUESTS_EXCEPTIONS as e:
            # Convert known transient requests errors (assuming ndbc-api uses requests)
            error_msg = f"Network error during NDBC request for station {station_id}: {e.__class__.__name__}: {e}"
            raise RetryableClientError(error_msg) from e
        except Exception as e:
            # Catch other potential errors from ndbc-api within the thread
            error_msg = f"Unexpected error during NDBC fetch for station {station_id}: {e.__class__.__name__}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise NdbcApiError(error_msg) from e

        # --- Validation phase (outside try - exceptions propagate naturally) ---
        # Check if the result is a dictionary (often empty or error indication)
        if isinstance(raw_result, dict):
            if raw_result == {}:
                # Empty dict = station has no data (expected operational condition)
                error_msg = f"NDBC station {station_id} returned no data"
                self.log(error_msg, level=logging.WARNING, location_code=location_code)
                raise StationUnavailableError(error_msg)
            else:
                # Non-empty dict = unexpected format (potential API change)
                error_msg = f"NDBC API returned unexpected dict: {raw_result}"
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

        # Explicitly check if it's a DataFrame
        if not isinstance(raw_result, pd.DataFrame):
            error_msg = f"NDBC API returned unexpected type {type(raw_result)}, expected DataFrame."
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise NdbcDataError(error_msg)

        return raw_result

    async def temperature(
        self,
        station_id: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        timezone: str,
        location_code: str = "unknown",
        mode: str = "stdmet",
    ) -> pd.DataFrame:
        """Fetch water temperature data from NDBC station with retries.

        Args:
            station_id: NDBC station ID
            begin_date: Start date for data fetch
            end_date: End date for data fetch
            timezone: Timezone to convert timestamps to
            location_code: Optional location code for logging
            mode: Data mode to fetch ('stdmet' or 'ocean')

        Returns:
            DataFrame with index=time and columns:
                water_temp: float - Water temperature in °F

        Raises:
            RetryableClientError: If connection fails after retries.
            NdbcDataError: If API returns invalid data or data is missing expected columns/structure.
            NdbcApiError: For other unexpected API errors.
        """
        location_code = location_code or station_id
        # Format dates as strings for the NDBC API
        begin_date_str = begin_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Call the execution logic via the retry wrapper
        raw_df = await self.request_with_retry(
            location_code=location_code,
            execute_request=self._execute_request,
            station_id=station_id,
            mode=mode,
            start_time=begin_date_str,
            end_time=end_date_str,
        )

        # --- Post-processing after successful retrieval ---
        # Wrap post-processing in a try-except block to catch format/data issues
        try:
            # Determine which temperature column to use based on mode
            temp_column = "WTMP" if mode == "stdmet" else "OTMP"

            # Check if water temperature data is available
            if temp_column not in raw_df.columns:
                error_msg = f"No water temperature data ('{temp_column}') available for NDBC station {station_id} in mode '{mode}'. Columns: {raw_df.columns.tolist()}"
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

            # Extract water temperature data and rename
            temp_df = (
                raw_df[[temp_column]].copy().rename(columns={temp_column: "water_temp"})
            )

            # Convert Celsius to Fahrenheit
            self.log(
                f"Converting {temp_column} from Celsius to Fahrenheit",
                location_code=location_code,
            )
            temp_df["water_temp"] = pd.to_numeric(
                temp_df["water_temp"], errors="coerce"
            )
            temp_df["water_temp"] = temp_df["water_temp"].apply(c_to_f)

            # Process index: NDBC returns MultiIndex (timestamp, station_id)
            # Assert that the index is a MultiIndex before proceeding
            if not isinstance(temp_df.index, pd.MultiIndex):
                error_msg = (
                    f"Expected MultiIndex from NDBC API, got {type(temp_df.index)}"
                )
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

            # Convert MultiIndex to DatetimeIndex by dropping the station ID level
            self.log(
                f"Converting MultiIndex to DatetimeIndex for NDBC station {station_id}",
                location_code=location_code,
                level=logging.DEBUG,  # More appropriate level for standard processing step
            )
            temp_df.index = temp_df.index.droplevel("station_id")

            # Assert that the resulting index is a DatetimeIndex
            if not pd.api.types.is_datetime64_any_dtype(temp_df.index):
                error_msg = f"Index after dropping level is not DatetimeIndex, got {type(temp_df.index)}"
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NdbcDataError(error_msg)

            # Convert UTC timestamps to local timezone
            temp_df = self._fix_time(temp_df, timezone)

            # Sort by time to ensure chronological order
            temp_df.sort_index(inplace=True)

            self.log(
                f"Successfully processed {len(temp_df)} temperature readings for NDBC station {station_id} from {begin_date_str} to {end_date_str}",
                location_code=location_code,
            )
            return temp_df

        except Exception as e:
            # Catch errors during post-processing
            error_msg = f"Error processing NDBC data for station {station_id} after successful fetch: {e.__class__.__name__}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            # Raise as NdbcDataError as it indicates an issue with the data format/content
            raise NdbcDataError(error_msg) from e

    def _fix_time(self, df: pd.DataFrame, timezone: str) -> pd.DataFrame:
        """Convert UTC timestamps to local timezone.

        Args:
            df: DataFrame with UTC timestamps in the index
            timezone: Timezone to convert timestamps to

        Returns:
            DataFrame with local timezone timestamps (naive datetimes)
        """
        # First, make the timestamps timezone-aware (UTC)
        df.index = df.index.tz_localize("UTC")

        # Then convert to the location's timezone
        df.index = df.index.tz_convert(timezone)

        # Finally, make the timestamps naive again (remove timezone info)
        df.index = df.index.tz_localize(None)

        # Rename the index to 'time' to match our convention
        df.index.name = "time"

        return df

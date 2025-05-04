"""USGS NWIS (National Water Information System) API client.

The USGS NWIS API provides access to water data from rivers, lakes, wells, and other
water bodies across the United States. This module focuses on retrieving water temperature
data from NWIS stations.

References:
    - USGS NWIS Web Services: https://waterservices.usgs.gov/
    - USGS Water Data for the Nation: https://waterdata.usgs.gov/nwis
"""

# Standard library imports
import asyncio
import datetime
import logging

# Third-party imports
import aiohttp
import dataretrieval.nwis as nwis
import pandas as pd

# Local imports
from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,  # Import BaseClientError
    RetryableClientError,
)  # noqa
from shallweswim.util import c_to_f


class NwisApiError(BaseClientError):  # Inherit from BaseClientError
    """Base error for USGS NWIS API calls."""


class NwisConnectionError(NwisApiError):
    """Error connecting to USGS NWIS API."""


class NwisDataError(NwisApiError):
    """Error in data returned by USGS NWIS API."""


class NwisApi(BaseApiClient):
    """Client for the USGS NWIS API using the dataretrieval library.

    This class provides methods to fetch hydrological and water quality data
    from USGS's National Water Information System (NWIS) stations.
    """

    @property
    def client_type(self) -> str:
        return "nwis"

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the NWIS client.

        Args:
            session: The aiohttp client session (required by base class, but not used directly by dataretrieval).
        """
        super().__init__(session)

    async def _execute_request(
        self,
        sites: str,
        service: str,
        parameterCd: list[str],
        start: str,
        end: str,
        location_code: str,
    ) -> pd.DataFrame:
        """Executes the nwis.get_record call within asyncio.to_thread and handles errors.

        This method is called by the `request_with_retry` logic in the base class.

        Args:
            sites: Site number.
            service: NWIS service (e.g., 'iv').
            parameterCd: List of parameter codes.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).
            location_code: Location code for logging.

        Returns:
            The raw DataFrame returned by nwis.get_record.

        Raises:
            RetryableClientError: If a transient connection/timeout error occurs.
            NwisDataError: If no data is returned or essential columns are missing.
            NwisApiError: For other unexpected errors during the nwis call.
        """
        self.log(
            f"Executing NWIS request for site {sites}, params={parameterCd}, start={start}, end={end}",
            location_code=location_code,
        )
        try:
            # Fetch the data using asyncio.to_thread to avoid blocking
            # dataretrieval uses requests internally
            raw_result = await asyncio.to_thread(
                nwis.get_record,
                sites=sites,
                service=service,
                parameterCd=parameterCd,
                start=start,
                end=end,
            )
            # Check if the result is empty or not a DataFrame after successful execution
            if (
                raw_result is None
                or not isinstance(raw_result, pd.DataFrame)
                or raw_result.empty
            ):
                error_msg = f"No data returned from NWIS API for site {sites}, params {parameterCd}"
                # This is treated as a data error, not necessarily retryable
                self.log(error_msg, level=logging.WARNING, location_code=location_code)
                raise NwisDataError(error_msg)

            return raw_result

        except (
            # Assuming dataretrieval might raise these if it uses requests
            # Need to confirm actual exceptions raised by dataretrieval on network errors
            aiohttp.ClientConnectionError,  # Example: if underlying requests uses aiohttp? Unlikely.
            asyncio.TimeoutError,
            # Add specific exceptions from dataretrieval/requests if known
        ) as e:
            # Convert known transient network errors to our retryable type
            error_msg = f"Network error during NWIS request for site {sites}: {e.__class__.__name__}: {e}"
            # Log is handled by tenacity, just raise the correct error type
            raise RetryableClientError(error_msg) from e
        except Exception as e:
            # Catch other potential errors from dataretrieval or pandas within the thread
            error_msg = f"Unexpected error during NWIS request for site {sites}: {e.__class__.__name__}: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            # Re-raise as a general NwisApiError
            raise NwisApiError(error_msg) from e

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
        raw_result: pd.DataFrame = await self.request_with_retry(
            location_code,  # First arg for request_with_retry
            # Remaining positional args for the implicit self._execute_request:
            site_no,
            "iv",
            param_list,
            begin_date_str,
            end_date_str,
        )

        # --- Post-processing starts (UNCHANGED from original logic) ---
        # Wrap unexpected errors during post-processing
        try:
            # Find the temperature column in the result
            # The exact column name format can vary, so we need to search for the parameter code
            temp_columns = [
                col for col in raw_result.columns if str(parameter_cd) in str(col)
            ]

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
                    f"Converting temperature from Celsius to Fahrenheit for parameter 00010",
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
        end_date = datetime.datetime.now(datetime.timezone.utc).date()
        start_date = end_date - datetime.timedelta(days=1)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        try:
            # Use request_with_retry which calls _execute_request internally
            raw_result: pd.DataFrame = await self.request_with_retry(
                sites=site_no,
                service="iv",  # Instantaneous values
                parameterCd=[parameter_cd],
                start=start_date_str,
                end=end_date_str,
                location_code=location_code,
            )

            # Post-processing similar to temperature method
            # Find the data column based on parameter code
            data_columns = [
                col for col in raw_result.columns if str(parameter_cd) in str(col)
            ]

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
            elif parameter_cd == "00055":  # Mean velocity, ft/sec
                output_col_name = "velocity_fps"
            elif parameter_cd == "72255":  # Stream velocity, ft/sec
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

        assert df.index.tzinfo is not None

        # Convert to the location's timezone
        df.index = df.index.tz_convert(timezone)

        # Finally, make the timestamps naive again (remove timezone info)
        df.index = df.index.tz_localize(None)

        # Rename the index to 'time' to match our convention
        df.index.name = "time"

        return df

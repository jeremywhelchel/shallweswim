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
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.base import BaseClientError
from shallweswim.util import c_to_f


class NwisApiError(BaseClientError):
    """Base error for USGS NWIS API calls."""


class NwisConnectionError(NwisApiError):
    """Error connecting to USGS NWIS API."""


class NwisDataError(NwisApiError):
    """Error in data returned by USGS NWIS API."""


class NwisApi(BaseApiClient):
    """Client for the USGS NWIS API.

    This class provides methods to fetch hydrological and water quality data
    from USGS's National Water Information System (NWIS) stations.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    @property
    def client_type(self) -> str:
        return "nwis"

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the NWIS client with an aiohttp session.

        Args:
            session: The aiohttp client session (not used directly by dataretrieval).
        """
        super().__init__(session)

    async def temperature(
        self,
        site_no: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        timezone: str,
        location_code: str = "unknown",
        parameter_cd: str = "00010",  # Default is water temperature (in Celsius)
    ) -> pd.DataFrame:
        """Fetch water temperature data from USGS NWIS station.

        Args:
            site_no: USGS site number (e.g., '01646500')
            begin_date: Start date for data fetch
            end_date: End date for data fetch
            timezone: Timezone to convert timestamps to
            location_code: Optional location code for logging
            parameter_cd: USGS parameter code, default '00010' for water temperature
                          Note: Some stations may use '00011' instead

        Returns:
            DataFrame with index=timestamp and columns:
                water_temp: float - Water temperature in °F

        Raises:
            NwisConnectionError: If connection to API fails
            NwisDataError: If API returns error or invalid data
        """
        # Format dates as strings for the NWIS API
        begin_date_str = begin_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        try:
            # Fetch the data using asyncio.to_thread to avoid blocking
            self.log(
                f"Fetching NWIS data for site {site_no} with parameter {parameter_cd} from {begin_date_str} to {end_date_str}",
                location_code=location_code,
            )

            # Convert parameter_cd to list if it's a string
            param_list = (
                [parameter_cd] if isinstance(parameter_cd, str) else parameter_cd
            )

            raw_result = await asyncio.to_thread(
                nwis.get_record,
                sites=site_no,
                service="iv",  # Instantaneous values
                parameterCd=param_list,
                start=begin_date_str,
                end=end_date_str,
            )

            # Check if the result is empty or not a DataFrame
            if (
                raw_result is None
                or not isinstance(raw_result, pd.DataFrame)
                or raw_result.empty
            ):
                error_msg = f"No data returned from NWIS API for site {site_no}"
                self.log(error_msg, level=logging.ERROR, location_code=location_code)
                raise NwisDataError(error_msg)

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
                f"Successfully fetched {len(temp_df)} temperature readings for NWIS site {site_no} with parameter {parameter_cd} from {begin_date_str} to {end_date_str}",
                location_code=location_code,
            )

            return temp_df

        except Exception as e:
            error_msg = f"Error fetching NWIS data: {e}"
            self.log(error_msg, level=logging.ERROR, location_code=location_code)
            raise NwisApiError(error_msg)

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

        # Rename the index to 'timestamp' to match our convention
        df.index.name = "timestamp"

        return df

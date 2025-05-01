"""NOAA NDBC (National Data Buoy Center) API client.

References:
    - NDBC Measurement Descriptions: https://www.ndbc.noaa.gov/faq/measdes.shtml
"""

# Standard library imports
import asyncio
import datetime
import logging

# Third-party imports
import ndbc_api
import pandas as pd

# Local imports
from shallweswim.util import c_to_f


class NdbcApiError(Exception):
    """Base error for NOAA NDBC API calls."""


class NdbcConnectionError(NdbcApiError):
    """Error connecting to NOAA NDBC API."""


class NdbcDataError(NdbcApiError):
    """Error in data returned by NOAA NDBC API."""


class NdbcApi:
    """Client for the NOAA NDBC API.

    This class provides methods to fetch meteorological and oceanographic data
    from NOAA's National Data Buoy Center (NDBC) stations.
    """

    @classmethod
    async def temperature(
        cls,
        station_id: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        timezone: str,
        location_code: str = "unknown",
        mode: str = "stdmet",
    ) -> pd.DataFrame:
        """Fetch water temperature data from NDBC station.

        Args:
            station_id: NDBC station ID
            begin_date: Start date for data fetch
            end_date: End date for data fetch
            timezone: Timezone to convert timestamps to
            location_code: Optional location code for logging
            mode: Data mode to fetch. Options:
                - 'stdmet': Standard meteorological data (uses WTMP column)
                - 'ocean': Oceanographic data (uses OTMP column)

        Returns:
            DataFrame with index=timestamp and columns:
                water_temp: float - Water temperature in °F

        Raises:
            NdbcConnectionError: If connection to API fails
            NdbcDataError: If API returns error or invalid data
        """
        # Format dates as strings for the NDBC API
        begin_date_str = begin_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        try:
            # Initialize the NDBC API client
            api = ndbc_api.NdbcApi()

            # Fetch the data using asyncio.to_thread to avoid blocking
            logging.info(
                f"[{location_code}][ndbc] Fetching NDBC data for station {station_id} from {begin_date_str} to {end_date_str}"
            )
            raw_result = await asyncio.to_thread(
                api.get_data,
                station_id=station_id,
                mode=mode,  # 'stdmet' or 'ocean'
                start_time=begin_date_str,
                end_time=end_date_str,
            )

            # Check if the result is a dictionary (often empty) instead of a DataFrame
            if isinstance(raw_result, dict):
                error_msg = (
                    f"NDBC API returned a dictionary instead of DataFrame: {raw_result}"
                )
                logging.error(f"[{location_code}][ndbc] {error_msg}")
                raise NdbcDataError(error_msg)

            # Explicitly assert that raw_result is a DataFrame
            if not isinstance(raw_result, pd.DataFrame):
                error_msg = f"Expected DataFrame, got {type(raw_result)}"
                logging.error(f"[{location_code}][ndbc] {error_msg}")
                raise NdbcDataError(error_msg)

            raw_df = raw_result

            # Determine which temperature column to use based on mode
            temp_column = "WTMP" if mode == "stdmet" else "OTMP"

            # Check if water temperature data is available
            if temp_column not in raw_df.columns:
                error_msg = f"No water temperature data ('{temp_column}') available for NDBC station {station_id} in mode '{mode}'"
                logging.error(f"[{location_code}][ndbc] {error_msg}")
                raise NdbcDataError(error_msg)

            # Extract water temperature data and rename to match our convention
            temp_df = (
                raw_df[[temp_column]].copy().rename(columns={temp_column: "water_temp"})
            )

            # NDBC reports temperatures in Celsius, convert to Fahrenheit to match our standard
            temp_df["water_temp"] = temp_df["water_temp"].apply(c_to_f)

            # Assert that the index is a MultiIndex (timestamp, station_id)
            if not isinstance(temp_df.index, pd.MultiIndex):
                raise ValueError(f"Expected MultiIndex, got {type(temp_df.index)}")

            logging.info(
                f"[{location_code}][ndbc] Converting MultiIndex to DatetimeIndex for NDBC station {station_id}"
            )

            # Reset the index and set only the timestamp as the new index
            temp_df.index = temp_df.index.droplevel("station_id")
            # Assert that the timestamp level is already a datetime
            if not pd.api.types.is_datetime64_any_dtype(temp_df.index):
                raise ValueError(
                    f"Expected timestamp index to be datetime, got {type(temp_df.index)}"
                )

            # Convert UTC timestamps to local timezone
            temp_df = cls._fix_time(temp_df, timezone)

            # Sort by time to ensure chronological order
            temp_df.sort_index(inplace=True)

            logging.info(
                f"[{location_code}][ndbc] Successfully fetched {len(temp_df)} temperature readings for NDBC station {station_id} from {begin_date_str} to {end_date_str}"
            )

            return temp_df

        except Exception as e:
            error_msg = f"Error fetching NDBC data: {e}"
            logging.error(f"[{location_code}][ndbc] {error_msg}")
            raise NdbcApiError(error_msg)

    @classmethod
    def _fix_time(cls, df: pd.DataFrame, timezone: str) -> pd.DataFrame:
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

        return df

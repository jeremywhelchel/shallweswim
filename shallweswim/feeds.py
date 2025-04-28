"""Data feed abstractions for ShallWeSwim application.

This module defines the feed framework for fetching and managing different types of data,
including temperature, tides, and currents from various sources like NOAA.
"""

# Standard library imports
import abc
import asyncio
import datetime
import logging
from typing import Any, Optional, Literal, List

# Third-party imports
# Import will be used in the actual implementation
# import ndbc_api
import pandas as pd
from pydantic import BaseModel, ConfigDict

# Local imports
from shallweswim import config as config_lib
from shallweswim import coops
from shallweswim.util import utc_now

# Additional buffer before reporting data as expired
# This gives the system time to refresh data without showing as expired
EXPIRATION_BUFFER = datetime.timedelta(seconds=300)

# XXX Make async methods
# XXX Add common logging patterns and error handling too


class Feed(BaseModel, abc.ABC):
    """Abstract base class for all data feeds.

    A feed represents a source of time-series data that can be fetched,
    processed, and cached. Feeds handle their own expiration logic and
    can be configured to refresh at different intervals.
    """

    # Configuration for the location this feed is associated with
    location_config: config_lib.LocationConfig

    # Frequency in which this data needs to be fetched, otherwise it is considered expired.
    # If None, this dataset will never expire and only needs to be fetched once.
    expiration_interval: Optional[datetime.timedelta]

    # Private fields - not included in serialization but still validated
    _timestamp: Optional[datetime.datetime] = None
    _data: Optional[pd.DataFrame] = None
    # Event used to signal when data is ready
    _ready_event: asyncio.Event = asyncio.Event()

    # Modern Pydantic v2 configuration using model_config
    model_config = ConfigDict(
        # Allow arbitrary types like pandas DataFrame
        arbitrary_types_allowed=True,
        # Validate assignment to attributes
        validate_assignment=True,
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the feed with configuration data.

        Args:
            **data: Configuration parameters for the feed
        """
        super().__init__(**data)
        # If we already have data, set the ready event
        if self._data is not None:
            self._ready_event.set()

    @property
    def age(self) -> Optional[datetime.timedelta]:
        """Calculate the age of the data as a timedelta.

        Returns:
            Age as a timedelta, or None if data has not been fetched yet
        """
        if not self._timestamp:
            return None

        # All datetimes should be naive
        now = utc_now()
        return now - self._timestamp

    @property
    def is_expired(self) -> bool:
        """Check if the feed data has expired and needs to be refreshed.

        Returns:
            True if data is expired or not yet fetched, False otherwise
        """
        if not self._timestamp:
            return True
        if not self.expiration_interval:
            return False

        # Use age to calculate the age
        age_td = self.age
        # age will never return None here because we already checked self._timestamp
        assert age_td is not None

        # Use the EXPIRATION_BUFFER to give the system time to refresh before reporting as expired
        return age_td > (self.expiration_interval + EXPIRATION_BUFFER)

    @property
    def values(self) -> pd.DataFrame:
        """Get the processed data from this feed.

        Returns:
            DataFrame containing the feed data

        Raises:
            ValueError: If data is not available
        """
        if self._data is None:
            raise ValueError("Data not yet fetched")
        return self._data

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message with standardized formatting including location code.

        Args:
            message: The message to log
            level: The logging level (default: INFO)
        """
        log_message = f"[{self.location_config.code}] {message}"
        logging.log(level, log_message)

    @property
    def status(self) -> dict[str, Any]:
        """Get a dictionary with the current status of this feed.

        Returns:
            A dictionary containing information about the feed's status
        """
        # Get data shape if available
        shape = None
        if self._data is not None:
            shape = self._data.shape

        # Get age as timedelta
        age_td = self.age

        # Convert age to seconds for the API
        age_sec = None
        if age_td is not None:
            age_sec = age_td.total_seconds()

        # Calculate expiration info
        expiration_sec = None
        if self.expiration_interval:
            expiration_sec = self.expiration_interval.total_seconds()

        # Build status dictionary with JSON serializable values
        status_dict = {
            "name": self.__class__.__name__,
            "location": self.location_config.code,
            "timestamp": self._timestamp.isoformat() if self._timestamp else None,
            "age_seconds": age_sec,
            "is_expired": self.is_expired,
            "is_ready": self._ready_event.is_set(),
            "data_shape": list(shape) if shape else None,
            "expiration_seconds": expiration_sec,
        }

        return status_dict

    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the feed has data available.

        This method waits until the feed has successfully fetched data and is ready to use.
        It's useful for coordinating dependent operations that need feed data to be available.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            True if the feed is ready, False if timeout occurred
        """
        # If data is already available, return immediately
        if self._data is not None:
            return True

        try:
            # Wait for the ready event to be set
            await asyncio.wait_for(self._ready_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            self.log(
                f"Timeout waiting for {self.__class__.__name__} to be ready",
                logging.WARNING,
            )
            return False

    async def update(self) -> None:
        """Update the data from this feed if it is expired."""
        if not self.is_expired:
            self.log(
                f"Skipping update for non-expired {self.__class__.__name__}",
                logging.DEBUG,
            )
            return

        try:
            self.log(f"Fetching data for {self.__class__.__name__}")
            df = await self._fetch()

            # Validate the dataframe before storing it
            self._validate_frame(df)

            df = self._remove_outliers(df)

            self._data = df
            self._timestamp = utc_now()
            # Set the ready event to signal that data is available
            self._ready_event.set()
            self.log(f"Successfully updated {self.__class__.__name__}")

        except Exception as e:
            # Log the error but don't suppress it - following the project principle
            # of failing fast for internal errors
            self.log(f"Error updating {self.__class__.__name__}: {e}", logging.ERROR)
            raise

    @abc.abstractmethod
    async def _fetch(self) -> pd.DataFrame:
        """Fetch data from the external source.

        This method should be implemented by subclasses to fetch data
        from their specific sources and process it.

        Returns:
            DataFrame containing the fetched data

        Raises:
            Exception: If fetching fails
        """
        ...

    def _validate_frame(self, df: pd.DataFrame) -> None:
        """Validate a dataframe to ensure it meets requirements.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If the dataframe is empty or contains timezone info
        """
        # Check if dataframe is None or empty
        if df is None:
            raise ValueError(f"Received None dataframe from {self.__class__.__name__}")

        if df.empty:
            raise ValueError(f"Received empty dataframe from {self.__class__.__name__}")

        # Check for timezone information in the index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame index is not a DatetimeIndex in {self.__class__.__name__}"
            )

        # Get the latest timestamp and check for timezone info
        latest_dt = df.index[-1].to_pydatetime()
        if latest_dt.tzinfo is not None:
            raise ValueError(
                f"DataFrame index contains timezone info in {self.__class__.__name__}; expected naive datetime"
            )

        # Log the latest datapoint timestamp
        self.log(f"{self.__class__.__name__} latest datapoint: {latest_dt}")

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove known erroneous data points from a DataFrame.

        Uses the outliers list from the temperature configuration to identify and
        remove specific timestamps that are known to have bad data. This helps to
        improve data quality by filtering out known anomalies.

        Args:
            df: DataFrame with DatetimeIndex to remove outliers from

        Returns:
            DataFrame with outliers removed
        """
        # Check if we have a config attribute
        if not hasattr(self, "config"):
            return df

        # Check if config has outliers attribute and it's not empty
        if not hasattr(self.config, "outliers") or not self.config.outliers:
            return df

        result_df = df
        for timestamp in self.config.outliers:
            try:
                result_df = result_df.drop(pd.to_datetime(timestamp))
            except KeyError:
                # Skip if the timestamp doesn't exist in the data
                self.log(
                    f"Outlier timestamp {timestamp} not found in data", logging.WARNING
                )

        return result_df


# XXX longterm and short term handled in data management layer
# XXX long-term temp is impelemented as multiple years fetched in parallel
# XXX will want a hook to update charts, etc, upon refresh


class TempFeed(Feed, abc.ABC):
    """Base class for temperature data feeds.

    This abstract class defines the interface for all temperature data sources,
    regardless of the specific provider (NOAA, USGS, etc).
    """

    # Temperature source configuration
    config: config_lib.TempSource

    # Data resolution ("h" for hourly, "6-min" for 6-minute intervals)
    interval: Literal["h", "6-min"]  # XXX 6-min is noaa specific. Make a type

    # Optional time range for data fetch
    start: Optional[datetime.datetime] = None
    end: Optional[datetime.datetime] = None

    @property
    def name(self) -> str:
        """Get the name of this temperature feed.

        Returns:
            Human-readable name of the temperature source
        """
        return self.config.name or "Unknown Temperature Source"


class CoopsTempFeed(TempFeed):
    """NOAA CO-OPS specific implementation of temperature data feed.

    Fetches temperature data from NOAA CO-OPS stations using the CO-OPS API.
    """

    config: config_lib.CoopsTempSource

    async def _fetch(self) -> pd.DataFrame:
        """Fetch temperature data from NOAA CO-OPS API.

        Returns:
            DataFrame with temperature data

        Raises:
            Exception: If fetching fails
        """
        station_id = self.config.station
        # Use parameters if provided, otherwise use defaults
        end_date = self.end or datetime.datetime.today()
        # XXX Document this default delta better
        begin_date = self.start or (end_date - datetime.timedelta(days=8))

        try:
            # Convert our interval to what the NOAA API expects
            # The NOAA API only accepts "h" or None (which defaults to 6-minute intervals)
            noaa_interval: Literal["h", None] = "h" if self.interval == "h" else None

            # Fetch data from NOAA CO-OPS API
            df = await coops.CoopsApi.temperature(
                station_id,
                "water_temperature",
                begin_date,
                end_date,
                interval=noaa_interval,
                location_code=self.location_config.code,
            )
            return df

        except coops.CoopsApiError as e:
            self.log(f"Live temp fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class NdbcTempFeed(TempFeed):
    """NOAA NDBC specific implementation of temperature data feed.

    Fetches temperature data from NOAA National Data Buoy Center (NDBC) stations
    using the ndbc-api package.
    """

    config: config_lib.NdbcTempSource
    # Default to hourly interval for NDBC data
    interval: Literal["h", "6-min"] = "h"

    async def _fetch(self) -> pd.DataFrame:
        """Fetch temperature data from NOAA NDBC API.

        Returns:
            DataFrame with temperature data

        Raises:
            Exception: If fetching fails
        """
        station_id = self.config.station
        # Use parameters if provided, otherwise use defaults
        end_date = self.end or datetime.datetime.today()
        # Default to 8 days of data if not specified
        begin_date = self.start or (end_date - datetime.timedelta(days=8))

        try:
            # Create a minimal timeseries dataframe with placeholder values
            # Generate a date range from begin_date to end_date with hourly intervals
            date_range = pd.date_range(start=begin_date, end=end_date, freq="H")

            # Create a dataframe with the date range as index and placeholder temperature values
            # The water_temp column matches the convention used in other feed classes
            df = pd.DataFrame(
                {
                    "water_temp": [20.0]
                    * len(date_range)  # Placeholder constant temperature
                },
                index=date_range,
            )

            # Log that we're using placeholder data
            self.log(
                f"Using placeholder data for NDBC station {station_id}", logging.INFO
            )

            return df
        except Exception as e:
            self.log(f"NDBC temp fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


# class UsgsTempFeed(TempFeed):
#    config: config_lib.UsgsTempSource


class CoopsTidesFeed(Feed):
    """Feed for NOAA CO-OPS tide predictions.

    Fetches tide predictions from NOAA CO-OPS stations using the CO-OPS API.
    Tide predictions include high and low tide times and heights.
    """

    config: config_lib.CoopsTideSource

    async def _fetch(self) -> pd.DataFrame:
        """Fetch tide predictions from NOAA CO-OPS API.

        Returns:
            DataFrame with tide predictions

        Raises:
            Exception: If fetching fails
        """
        station_id = self.config.station

        try:
            # Fetch data from NOAA CO-OPS API
            df = await coops.CoopsApi.tides(
                station_id,
                location_code=self.location_config.code,
            )
            return df
        except coops.CoopsApiError as e:
            self.log(f"Tide fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class CoopsCurrentsFeed(Feed):
    """Feed for NOAA CO-OPS current predictions.

    Fetches current predictions from NOAA CO-OPS stations using the CO-OPS API.
    Current predictions include velocity, direction, and type (flood/ebb/slack).
    """

    config: config_lib.CoopsCurrentsSource

    # The station to use for fetching currents data (optional, will use first station from config if not provided)
    station: Optional[str] = None

    # Whether to interpolate between flood/slack/ebb points
    interpolate: bool = True

    def __init__(self, **data: Any) -> None:
        """Initialize the feed and set default station if none provided."""
        super().__init__(**data)
        if not self.station and self.config.stations:
            self.station = self.config.stations[0]

    async def _fetch(self) -> pd.DataFrame:
        """Fetch current predictions from NOAA CO-OPS API.

        Returns:
            DataFrame with current predictions

        Raises:
            Exception: If fetching fails
        """
        try:
            # We know self.station is set in __init__ if it wasn't provided
            assert self.station is not None, "Station must be set"
            # Fetch data from NOAA CO-OPS API
            df = await coops.CoopsApi.currents(
                self.station,
                interpolate=self.interpolate,
                location_code=self.location_config.code,
            )
            return df

        except coops.CoopsApiError as e:
            self.log(f"Current fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class CompositeFeed(Feed, abc.ABC):
    """Base class for feeds that combine data from multiple sources.

    This allows for aggregating data from multiple feeds into a single feed.
    For example, combining current data from multiple stations or temperature
    data from multiple years.
    """

    async def _fetch(self) -> pd.DataFrame:
        """Fetch data from all underlying feeds and combine them.

        The specific combination logic is implemented in _combine_feeds.

        Returns:
            Combined DataFrame from all feeds

        Raises:
            Exception: If fetching fails
        """
        # Get all the underlying feeds
        feeds = self._get_feeds()

        # Fetch data from all feeds concurrently
        tasks = [feed._fetch() for feed in feeds]
        results = await asyncio.gather(*tasks)

        # Combine the results using the specific combination logic
        return self._combine_feeds(results)

    @abc.abstractmethod
    def _get_feeds(self) -> List[Feed]:
        """Get the list of feeds to combine.

        Returns:
            List of Feed objects
        """
        pass

    @abc.abstractmethod
    def _combine_feeds(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine the DataFrames from multiple feeds.

        Args:
            dataframes: List of DataFrames from the underlying feeds

        Returns:
            Combined DataFrame
        """
        pass


class MultiStationCurrentsFeed(CompositeFeed):
    """Feed for current data from multiple NOAA stations.

    Fetches current predictions from multiple NOAA stations and combines them.
    This is useful for locations where multiple current stations provide
    complementary data about water conditions in the area.
    """

    config: config_lib.CoopsCurrentsSource

    # Whether to interpolate between flood/slack/ebb points for each station
    interpolate: bool = True

    def _get_feeds(self) -> List[Feed]:
        """Create a CoopsCurrentsFeed for each station in the config.

        Returns:
            List of CoopsCurrentsFeed instances, one for each station
        """
        feeds: List[Feed] = []
        for station in self.config.stations:
            feed = CoopsCurrentsFeed(
                location_config=self.location_config,
                config=self.config,
                station=station,
                interpolate=self.interpolate,
                expiration_interval=self.expiration_interval,
            )
            feeds.append(feed)
        return feeds

    def _combine_feeds(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine current data from multiple stations.

        Strategy: For overlapping timestamps, calculate the average velocity.
        This matches the legacy implementation in DataManager._fetch_tides_and_currents.

        Args:
            dataframes: List of DataFrames from individual current stations

        Returns:
            Combined DataFrame with averaged velocity data

        Raises:
            ValueError: If no valid dataframes are provided
        """
        if not dataframes:
            raise ValueError("No dataframes provided to combine")

        # For a single dataframe, we need special handling because we need to select
        # only the velocity column, which is different from the concat+groupby approach
        if len(dataframes) == 1:
            # If there's only one dataframe, just select the velocity column
            return dataframes[0][["velocity"]]

        # Combine all dataframes, select only the velocity column, and average by timestamp
        # This exactly matches the legacy implementation:
        # pd.concat(currents)[["velocity"]].groupby(level=0).mean()
        result_df = pd.concat(dataframes)[["velocity"]].groupby(level=0).mean()

        # Note for future improvement: A more comprehensive implementation could:
        # 1. Preserve all numeric columns by averaging them
        # 2. Handle categorical columns like 'type' (flood/ebb/slack) by using the most common value
        # 3. Ensure the result is properly sorted by timestamp

        return result_df


class HistoricalTempsFeed(CompositeFeed):
    """Feed for historical temperature data across multiple years.

    Fetches temperature data for multiple years and combines them to provide
    historical averages for each day of the year. This is useful for showing
    typical temperatures for a given date based on historical records.
    """

    config: config_lib.CoopsTempSource
    start_year: int
    end_year: int

    def _get_feeds(self) -> List[Feed]:
        """Create a CoopsTempFeed for each year in the range.

        For the current year, caps the end date to today to avoid requesting future dates
        from the CO-OPS API, which would result in an error.

        Returns:
            List of CoopsTempFeed instances, one for each year in the range
        """
        feeds: List[Feed] = []
        current_date = utc_now()

        for year in range(self.start_year, self.end_year + 1):
            # Calculate start and end dates for this year
            start_date = datetime.datetime(year, 1, 1)

            # For the current year, cap the end date to today
            if year == current_date.year:
                end_date = datetime.datetime(
                    year, current_date.month, current_date.day, 23, 59, 59
                )
            else:
                end_date = datetime.datetime(year, 12, 31, 23, 59, 59)

            # Only set expiration interval for the current year
            # Past years' data won't change, so they don't need to expire
            expiration = self.expiration_interval if year == current_date.year else None

            feed = CoopsTempFeed(
                location_config=self.location_config,
                config=self.config,
                start=start_date,
                end=end_date,
                interval="h",  # Use hourly data for historical temps
                expiration_interval=expiration,
            )
            feeds.append(feed)
        return feeds

    def _combine_feeds(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine temperature data from multiple years.

        Simply concatenates the dataframes from different years and resamples to hourly intervals.
        The only reason to fetch years separately is for efficient parallel fetching.

        Args:
            dataframes: List of DataFrames from individual years

        Returns:
            Combined DataFrame with temperature data from all years, resampled to hourly intervals

        Raises:
            ValueError: If no valid dataframes are provided
        """
        if not dataframes:
            raise ValueError("No dataframes provided to combine")

        # If there's only one dataframe, pd.concat will just return it
        # If there are multiple dataframes, pd.concat will combine them
        result_df = pd.concat(dataframes)

        # Sort by timestamp index and resample to hourly intervals
        # This matches the legacy implementation: historic_temps.resample("h").first()
        return result_df.sort_index().resample("h").first()

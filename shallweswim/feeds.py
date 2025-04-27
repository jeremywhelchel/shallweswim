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
import pandas as pd
from pydantic import BaseModel, ConfigDict

# Local imports
from shallweswim import config as config_lib
from shallweswim import noaa
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

    # Modern Pydantic v2 configuration using model_config
    model_config = ConfigDict(
        # Allow arbitrary types like pandas DataFrame
        arbitrary_types_allowed=True,
        # Validate assignment to attributes
        validate_assignment=True,
    )

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
        now = utc_now()
        # Use the EXPIRATION_BUFFER to give the system time to refresh before reporting as expired
        age = now - self._timestamp
        return age > (self.expiration_interval + EXPIRATION_BUFFER)

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


class NoaaTempFeed(TempFeed):
    """NOAA-specific implementation of temperature data feed.

    Fetches temperature data from NOAA stations using the NOAA API.
    """

    config: config_lib.NoaaTempSource

    async def _fetch(self) -> pd.DataFrame:
        """Fetch temperature data from NOAA API.

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
            df = await noaa.NoaaApi.temperature(
                station_id,
                "water_temperature",
                begin_date,
                end_date,
                location_code=self.location_config.code,
                # XXX add interval too
            )
            return df

        except noaa.NoaaApiError as e:
            self.log(f"Live temp fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


# class UsgsTempFeed(TempFeed):
#    config: config_lib.UsgsTempSource


class NoaaTidesFeed(Feed):
    """Feed for tide data from NOAA.

    Fetches tide predictions from NOAA stations using the NOAA API.
    Tide predictions include high and low tide times and heights.
    """

    config: config_lib.NoaaTideSource

    async def _fetch(self) -> pd.DataFrame:
        """Fetch tide predictions from NOAA API.

        Returns:
            DataFrame with tide prediction data

        Raises:
            Exception: If fetching fails
        """
        station_id = self.config.station

        try:
            df = await noaa.NoaaApi.tides(
                station_id,
                location_code=self.location_config.code,
            )
            return df

        except noaa.NoaaApiError as e:
            self.log(f"Tide fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class NoaaCurrentsFeed(Feed):
    """Feed for current data from NOAA.

    Fetches current predictions from NOAA stations using the NOAA API.
    Current predictions include velocity, direction, and type (flood/ebb/slack).
    """

    config: config_lib.NoaaCurrentsSource

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
        """Fetch current predictions from NOAA API.

        Returns:
            DataFrame with current prediction data

        Raises:
            Exception: If fetching fails
        """
        try:
            # We know self.station is set in __init__ if it wasn't provided
            assert self.station is not None, "Station must be set"
            df = await noaa.NoaaApi.currents(
                self.station,
                interpolate=self.interpolate,
                location_code=self.location_config.code,
            )
            return df

        except noaa.NoaaApiError as e:
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

    config: config_lib.NoaaCurrentsSource

    # Whether to interpolate between flood/slack/ebb points for each station
    interpolate: bool = True

    def _get_feeds(self) -> List[Feed]:
        """Create a NoaaCurrentsFeed for each station in the config.

        Returns:
            List of NoaaCurrentsFeed instances, one for each station
        """
        feeds: List[Feed] = []
        for station in self.config.stations:
            feed = NoaaCurrentsFeed(
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
        This provides a more comprehensive view of currents in the area by
        considering multiple measurement points.

        Args:
            dataframes: List of DataFrames from individual current stations

        Returns:
            Combined DataFrame with averaged current data

        Raises:
            ValueError: If no valid dataframes are provided
        """
        if not dataframes:
            raise ValueError("No dataframes provided to combine")

        if len(dataframes) == 1:
            # If there's only one dataframe, just return it
            return dataframes[0]

        # Combine all dataframes
        combined_df = pd.concat(dataframes)

        # Group by timestamp and calculate mean for numeric columns
        # For non-numeric columns like 'type' (flood/ebb/slack), use the most common value
        grouped = combined_df.groupby(combined_df.index)

        result_df = pd.DataFrame()

        # Process numeric columns (like velocity, direction)
        numeric_cols = combined_df.select_dtypes(include=["number"]).columns
        if not numeric_cols.empty:
            result_df[numeric_cols] = grouped[numeric_cols].mean()

        # Process categorical columns (like type: flood/ebb/slack)
        cat_cols = combined_df.select_dtypes(exclude=["number"]).columns
        for col in cat_cols:
            # Use the most common value for each timestamp
            result_df[col] = grouped[col].agg(
                lambda x: x.mode()[0] if not x.mode().empty else None
            )

        # Sort by timestamp
        result_df = result_df.sort_index()

        return result_df

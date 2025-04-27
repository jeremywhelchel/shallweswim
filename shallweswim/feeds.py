"""Data feed abstractions for ShallWeSwim application.

This module defines the feed framework for fetching and managing different types of data,
including temperature, tides, and currents from various sources like NOAA.
"""

# Standard library imports
import abc
import datetime
import logging
from typing import Optional, Literal

# Third-party imports
import pandas as pd
from pydantic import BaseModel

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

    class Config:
        # Allow arbitrary types like pandas DataFrame
        arbitrary_types_allowed = True
        # Validate assignment to attributes
        validate_assignment = True

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
    @abc.abstractmethod
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
        # XXX Implement
        return df


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


class TidesFeed(Feed, abc.ABC): ...


# XXX this one will return a feed that includes future projections. But the prev/next split will
# be done elsewhere.


class CurrentsFeed(Feed, abc.ABC):
    # XXX this one I believe is primarily predictions
    ...

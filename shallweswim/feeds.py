"""Data feed abstractions for ShallWeSwim application.

This module defines the feed framework for fetching and managing different types of data,
including temperature, tides, and currents from various sources like NOAA.
"""

# Standard library imports
import abc
import asyncio
import datetime
import logging
from typing import Any, Optional, Literal, List, Dict, Type

# Third-party imports
import pandas as pd
from pandera import DataFrameModel
from pydantic import BaseModel, ConfigDict

# Local imports
from shallweswim import config as config_lib
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients import coops
from shallweswim.clients import ndbc
from shallweswim.clients import nwis
from shallweswim import dataframe_models as df_models
from shallweswim.util import utc_now, summarize_dataframe, fps_to_knots
from shallweswim.api_types import FeedStatus, DataFrameSummary

# Additional buffer before reporting data as expired
# This gives the system time to refresh data without showing as expired
HEALTH_CHECK_BUFFER = datetime.timedelta(
    minutes=15
)  # Longer buffer for service health checks


class Feed(BaseModel, abc.ABC):
    """Abstract base class for all data feeds.

    A feed represents a source of time-series data that can be fetched,
    processed, and cached. Feeds handle their own expiration logic and
    can be configured to refresh at different intervals.
    """

    # Configuration for the location this feed is associated with
    location_config: config_lib.LocationConfig

    # Feed configuration
    feed_config: config_lib.BaseFeedConfig

    # Frequency in which this data needs to be fetched, otherwise it is considered expired.
    # If None, this dataset will never expire and only needs to be fetched once.
    expiration_interval: Optional[datetime.timedelta]

    # Private fields - not included in serialization but still validated
    _fetch_timestamp: Optional[datetime.datetime] = None
    _data: Optional[pd.DataFrame] = None
    # Event used to signal when data is ready
    _ready_event: asyncio.Event = asyncio.Event()
    _last_error: Optional[Exception] = None

    # Modern Pydantic v2 configuration using model_config
    model_config = ConfigDict(
        # Allow arbitrary types like pandas DataFrame
        arbitrary_types_allowed=True,
        # Validate assignment to attributes
        validate_assignment=True,
        # Forbid extra fields not defined in the model
        extra="forbid",
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
        if not self._fetch_timestamp:
            return None

        # All datetimes should be naive
        now = utc_now()
        return now - self._fetch_timestamp

    @property
    def is_expired(self) -> bool:
        """Check if the feed data has expired and needs to be refreshed.

        Returns:
            True if data is expired or not yet fetched, False otherwise
        """
        if not self._fetch_timestamp:
            return True
        if not self.expiration_interval:
            return False

        # Use age to calculate the age
        age_td = self.age
        # age will never return None here because we already checked self._fetch_timestamp
        assert age_td is not None

        # Check age directly against the configured interval
        assert self.expiration_interval is not None  # Help mypy
        return age_td > self.expiration_interval

    @property
    def is_healthy(self) -> bool:
        """Check if the feed data is recent enough to be considered healthy for service status.

        This uses a longer buffer (HEALTH_CHECK_BUFFER) than is_expired to avoid flapping
        the service health status due to minor delays in data fetching.

        Returns:
            bool: True if the data is considered healthy, False otherwise.
        """
        # No data yet, definitely not healthy
        if self._fetch_timestamp is None:
            return False

        # No interval means it's always considered healthy (as it doesn't expire)
        if self.expiration_interval is None:
            return True

        age_td = self.age
        # Healthy if age is within the expiration interval plus the health check buffer
        assert self.expiration_interval is not None  # Help mypy
        assert age_td is not None  # Help mypy
        return age_td <= (self.expiration_interval + HEALTH_CHECK_BUFFER)

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
    def status(self) -> FeedStatus:
        """Get a Pydantic model with the current status of this feed.

        Returns:
            A FeedStatus object containing information about the feed's status
        """
        data_summary: Optional[DataFrameSummary] = None
        if self._data is not None:
            data_summary = summarize_dataframe(self._data)

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

        # Build status object
        status_obj = FeedStatus(
            name=self.__class__.__name__,
            location=self.location_config.code,
            fetch_timestamp=self._fetch_timestamp,  # Pass datetime object directly
            age_seconds=age_sec,
            is_expired=self.is_expired,
            is_healthy=self.is_healthy,
            expiration_seconds=expiration_sec,
            data_summary=data_summary,  # Pass the summary object directly
            error=str(self._last_error) if self._last_error else None,
        )

        return status_obj

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

    async def update(self, clients: Dict[str, BaseApiClient]) -> None:
        """Update the data from this feed if it is expired."""
        if not self.is_expired:
            self.log(
                f"Skipping update for non-expired {self.__class__.__name__}",
                logging.DEBUG,
            )
            return

        try:
            self.log(f"Fetching data for {self.__class__.__name__}")
            # Pass clients to _fetch
            df = await self._fetch(clients=clients)

            # Validate the dataframe before storing it
            self._validate_frame(df)

            df = self._remove_outliers(df)

            self._data = df
            self._fetch_timestamp = utc_now()
            # Set the ready event to signal that data is available
            self._ready_event.set()
            self.log(f"Successfully updated {self.__class__.__name__}")

        except Exception as e:
            # Log the error but don't suppress it - following the project principle
            # of failing fast for internal errors
            self.log(f"Error updating {self.__class__.__name__}: {e}", logging.ERROR)
            self._last_error = e
            raise

    @abc.abstractmethod
    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch data from the external source.

        This method should be implemented by subclasses to fetch data
        from their specific sources and process it.

        Returns:
            DataFrame containing the fetched data

        Raises:
            Exception: If fetching fails
        """
        ...

    @abc.abstractproperty
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        ...

    def _validate_frame(self, df: pd.DataFrame) -> None:
        """Validate a dataframe to ensure it meets requirements.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If the dataframe is empty or contains timezone info
        """
        # Check if dataframe is None
        if df is None:
            raise ValueError(f"Received None dataframe from {self.__class__.__name__}")

        # lazy=True collects all validation errors.
        self.data_model.validate(df, lazy=True)

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
        # Check if we have a feed_config attribute
        if not hasattr(self, "feed_config"):
            return df

        # Check if feed_config has outliers attribute and it's not empty
        if not hasattr(self.feed_config, "outliers") or not self.feed_config.outliers:
            return df

        result_df = df
        for timestamp in self.feed_config.outliers:
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

    # Feed configuration
    feed_config: config_lib.TempFeedConfig

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
        return self.feed_config.name or "Unknown Temperature Source"


class CurrentsFeed(Feed, abc.ABC):
    """Abstract base class for all currents data feeds.

    A feed represents a source of current data (velocity, direction) that can be fetched,
    processed, and cached.
    """

    # Feed configuration
    feed_config: config_lib.CurrentsFeedConfig

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        # All current feeds should conform to the CurrentDataModel
        return df_models.CurrentDataModel  # type: ignore[return-value]


class CoopsTempFeed(TempFeed):
    """NOAA CO-OPS specific implementation of temperature data feed.

    Fetches temperature data from NOAA CO-OPS stations using the CO-OPS API.
    """

    feed_config: config_lib.CoopsTempFeedConfig
    product: Literal["air_temperature", "water_temperature"] = "water_temperature"

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.WaterTempDataModel  # type: ignore[return-value]

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch temperature data from NOAA CO-OPS API.

        Returns:
            DataFrame with temperature data

        Raises:
            Exception: If fetching fails
        """
        station_id = self.feed_config.station
        # Use parameters if provided, otherwise use defaults
        begin_date = self.start or (datetime.date.today() - datetime.timedelta(days=8))
        end_date = self.end or datetime.date.today()

        try:
            # Get Coops client instance
            coops_client: CoopsApi = clients["coops"]  # type: ignore

            # Fetch data from NOAA CO-OPS API using the client instance
            df = await coops_client.temperature(
                station=station_id,
                begin_date=begin_date,
                end_date=end_date,
                product=self.product,  # Use the feed's product attribute
                interval=self.interval,
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
    using the ndbc module.
    """

    feed_config: config_lib.NdbcTempFeedConfig
    interval: Literal["h", "6-min"] = "h"
    mode: Literal["stdmet", "ocean"] = "stdmet"
    client: ndbc.NdbcApi  # Add type hint for mypy

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.WaterTempDataModel  # type: ignore[return-value]

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch temperature data from NOAA NDBC API.

        Returns:
            DataFrame with temperature data

        Raises:
            Exception: If fetching fails
        """
        station_id = self.feed_config.station
        # Use parameters if provided, otherwise use defaults
        begin_date = self.start or (
            datetime.datetime.today() - datetime.timedelta(days=8)
        )
        end_date = self.end or datetime.datetime.today()

        try:
            # Fetch the data using the NDBC API client
            self.log(f"Fetching NDBC data for station {station_id}", logging.INFO)

            temp_df = await self.client.temperature(  # Corrected call
                station_id=station_id,
                begin_date=begin_date,
                end_date=end_date,
                timezone=str(self.location_config.timezone),
                location_code=self.location_config.code,
                mode=self.mode,
            )

            self.log(
                f"Successfully fetched {len(temp_df)} temperature readings for NDBC station {station_id}",
                logging.INFO,
            )

            return temp_df
        except ndbc.NdbcApiError as e:
            self.log(f"Error fetching NDBC data: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class NwisTempFeed(TempFeed):
    """USGS NWIS specific implementation of temperature data feed.

    Fetches temperature data from USGS National Water Information System (NWIS) sites
    using the nwis module.
    """

    feed_config: config_lib.NwisTempFeedConfig
    interval: Literal["h", "6-min"] = "h"  # NWIS typically provides hourly data

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.WaterTempDataModel  # type: ignore[return-value]

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch temperature data from USGS NWIS API.

        Returns:
            DataFrame with temperature data

        Raises:
            Exception: If fetching fails
        """
        site_no = self.feed_config.site_no
        parameter_cd = self.feed_config.parameter_cd
        # Use parameters if provided, otherwise use defaults
        begin_date = self.start or (
            datetime.datetime.today() - datetime.timedelta(days=8)
        )
        end_date = self.end or datetime.datetime.today()

        try:
            # Get NWIS instance from the passed clients dict
            nwis_client: NwisApi = clients["nwis"]  # type: ignore
            temp_df = await nwis_client.temperature(
                site_no=site_no,
                begin_date=begin_date,
                end_date=end_date,
                timezone=str(self.location_config.timezone),
                location_code=self.location_config.code,
                parameter_cd=parameter_cd,
            )

            self.log(
                f"Successfully fetched {len(temp_df)} temperature readings for NWIS site {site_no}",
                logging.INFO,
            )

            return temp_df
        except nwis.NwisApiError as e:
            self.log(f"Error fetching NWIS data: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class CoopsTidesFeed(Feed):
    """Feed for NOAA CO-OPS tide predictions.

    Fetches tide predictions from NOAA CO-OPS stations using the CO-OPS API.
    Tide predictions include high and low tide times and heights.
    """

    feed_config: config_lib.CoopsTideFeedConfig
    interval: Literal["h", "6-min"] = "h"  # Add default interval
    start: Optional[datetime.date] = None

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.TidePredictionDataModel  # type: ignore[return-value]

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch tide predictions from NOAA CO-OPS API.

        Returns:
            DataFrame with tide predictions

        Raises:
            Exception: If fetching fails
        """
        station_id = self.feed_config.station

        try:
            # Get Coops client instance
            coops_client: CoopsApi = clients["coops"]  # type: ignore

            # Fetch data from NOAA CO-OPS API using the client instance
            df = await coops_client.tides(
                station=station_id,
                location_code=self.location_config.code,
            )
            return df
        except coops.CoopsApiError as e:
            self.log(f"Tide fetch error: {e}", logging.WARNING)
            # Following the project principle of failing fast for internal errors
            raise


class CoopsCurrentsFeed(CurrentsFeed):
    """Feed for NOAA CO-OPS current predictions.

    Fetches current predictions from NOAA CO-OPS stations using the CO-OPS API.
    Current predictions include velocity, direction, and type (flood/ebb/slack).
    """

    feed_config: config_lib.CoopsCurrentsFeedConfig
    interval: Literal["h", "6-min"] = "h"  # Add default interval
    start: Optional[datetime.date] = None

    # The station to use for fetching currents data (optional, will use first station from config if not provided)
    station: Optional[str] = None

    # Whether to interpolate between flood/slack/ebb points
    interpolate: bool = True

    def __init__(self, **data: Any) -> None:
        """Initialize the feed and set default station if none provided."""
        super().__init__(**data)
        if not self.station and self.feed_config.stations:
            self.station = self.feed_config.stations[0]

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch current predictions from NOAA CO-OPS API.

        Returns:
            DataFrame with current predictions

        Raises:
            Exception: If fetching fails
        """
        try:
            # Get Coops client instance
            coops_client: CoopsApi = clients["coops"]  # type: ignore

            # We know self.station is set in __init__ if it wasn't provided
            assert self.station is not None, "Station must be set"

            # Fetch data from NOAA CO-OPS API using the client instance
            df = await coops_client.currents(
                station=self.station,
                location_code=self.location_config.code,
                interpolate=self.interpolate,
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

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch data from all underlying feeds and combine them.

        The specific combination logic is implemented in _combine_feeds.

        Returns:
            Combined DataFrame from all feeds

        Raises:
            Exception: If fetching fails
        """
        # Get all the underlying feeds
        feeds = self._get_feeds(clients=clients)

        # Fetch data from all feeds concurrently
        tasks = [feed._fetch(clients=clients) for feed in feeds]
        results = await asyncio.gather(*tasks)

        # Combine the results using the specific combination logic
        return self._combine_feeds(results)

    @abc.abstractmethod
    def _get_feeds(self, clients: Dict[str, BaseApiClient]) -> List[Feed]:
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

    feed_config: config_lib.CoopsCurrentsFeedConfig

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.CurrentDataModel  # type: ignore[return-value]

    def _get_feeds(self, clients: Dict[str, BaseApiClient]) -> List[Feed]:
        """Create a CoopsCurrentsFeed for each station in the config.

        Returns:
            List of CoopsCurrentsFeed instances, one for each station
        """
        feeds: List[Feed] = []
        for station in self.feed_config.stations:
            feed = create_current_feed(
                location_config=self.location_config,
                current_config=self.feed_config,
                station=station,
                expiration_interval=self.expiration_interval,
                clients=clients,
            )
            feeds.append(feed)
        return feeds

    def _combine_feeds(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine current data from multiple stations.

        Strategy: For overlapping timestamps, calculate the average velocity.
        This matches the legacy implementation in LocationDataManager._fetch_tides_and_currents.

        Args:
            dataframes: List of DataFrames from individual current stations

        Returns:
            Combined DataFrame with averaged velocity data

        Raises:
            ValueError: If no valid dataframes are provided
        """
        if not dataframes:
            raise ValueError("No dataframes provided to combine")

        # If there's only one dataframe, we need special handling because we need to select
        # only the velocity column, which is different from the concat+groupby approach
        if len(dataframes) == 1:
            # If there's only one dataframe, just select the velocity column
            return dataframes[0][["velocity"]]

        # Combine all dataframes, select only the velocity column, and average by timestamp
        # This exactly matches the legacy implementation: pd.concat(currents)[["velocity"]].groupby(level=0).mean()
        result_df = pd.concat(dataframes)[["velocity"]].groupby(level=0).mean()

        # Note for future improvement: A more comprehensive implementation could:
        # 1. Preserve all numeric columns by averaging them
        # 2. Handle categorical columns like 'type' (flood/ebb/slack) by using the most common value
        # 3. Ensure the result is properly sorted by timestamp

        return result_df


def create_temp_feed(
    location_config: config_lib.LocationConfig,
    temp_config: config_lib.TempFeedConfig,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    interval: Literal["h", "6-min"] = "h",
    expiration_interval: Optional[datetime.timedelta] = None,
    clients: Dict[str, BaseApiClient] = {},  # Add clients dict parameter
    **kwargs: Any,
) -> TempFeed:
    """Create a temperature feed based on the configuration type.

    This factory function creates the appropriate temperature feed based on the
    type of the temperature source configuration.

    Args:
        location_config: Location configuration
        temp_config: Temperature source configuration
        start: Start date for data fetching (optional)
        end: End date for data fetching (optional)
        interval: Data interval ('h' for hourly, '6-min' for 6-minute)
        expiration_interval: Custom expiration interval (optional)
        clients: Dict of clients to use for fetching data (optional)
        **kwargs: Additional keyword arguments for specific feed types

    Returns:
        Configured temperature feed

    Raises:
        TypeError: If an unsupported temperature source type is provided
    """
    # Create the appropriate feed based on the config type
    if isinstance(temp_config, config_lib.CoopsTempFeedConfig):
        return CoopsTempFeed(
            location_config=location_config,
            feed_config=temp_config,
            start=start,
            end=end,
            interval=interval,
            expiration_interval=expiration_interval,
        )
    elif isinstance(temp_config, config_lib.NdbcTempFeedConfig):
        ndbc_client = clients.get("ndbc")
        if not isinstance(ndbc_client, ndbc.NdbcApi):
            raise TypeError(
                "NDBC client not found or incorrect type in provided clients dict"
            )

        # Get the mode parameter if provided
        mode = kwargs.get("mode", "stdmet")

        return NdbcTempFeed(
            location_config=location_config,
            feed_config=temp_config,
            start=start,
            end=end,
            mode=mode,
            client=ndbc_client,  # Pass the client instance
            expiration_interval=expiration_interval,
        )
    elif isinstance(temp_config, config_lib.NwisTempFeedConfig):
        # Get the parameter_cd if provided
        parameter_cd = kwargs.get("parameter_cd", None)
        feed_kwargs = {}
        if parameter_cd is not None:
            feed_kwargs["parameter_cd"] = parameter_cd

        # Retrieve the specific NWIS client
        nwis_client = clients.get("nwis")
        if not isinstance(nwis_client, nwis.NwisApi):
            raise TypeError(
                "NWIS client not found or incorrect type in provided clients dict"
            )

        return NwisTempFeed(
            location_config=location_config,
            feed_config=temp_config,
            start=start,
            end=end,
            expiration_interval=expiration_interval,
            **feed_kwargs,
        )
    else:
        # Unsupported temperature source type - fail fast and loud
        raise TypeError(
            f"Unsupported temperature source type: {type(temp_config).__name__}"
        )


def create_current_feed(
    location_config: config_lib.LocationConfig,
    current_config: config_lib.CurrentsFeedConfig,
    station: Optional[str] = None,
    expiration_interval: Optional[datetime.timedelta] = None,
    clients: Optional[Dict[str, BaseApiClient]] = None,  # Add clients param
) -> Feed:
    """Create a current feed based on the configuration type.

    This factory function creates the appropriate current feed based on the
    configuration type, handling the specific initialization requirements for each.

    Args:
        location_config: Location configuration
        current_config: Currents source configuration (subclass of CurrentsFeedConfig)
        station: Station ID to use (optional, for multi-station CO-OPS configs)
        expiration_interval: Custom expiration interval (optional)
        clients: Dict of clients to use for fetching data (optional)

    Returns:
        Configured current feed

    Raises:
        TypeError: If the current_config type is unsupported.
        NotImplementedError: If the current_config type is recognized but not yet implemented.
    """
    # --- Dispatch based on config type ---
    if isinstance(current_config, config_lib.CoopsCurrentsFeedConfig):
        # Add check for coops client
        if clients is None or "coops" not in clients:
            raise ValueError("CO-OPS client required for CO-OPS current feeds")
        # For multi-station configs without a specific station, create a composite feed
        if hasattr(current_config, "stations") and not station:
            return MultiStationCurrentsFeed(
                location_config=location_config,
                feed_config=current_config,
                expiration_interval=expiration_interval,
            )
        # For single station or when a specific station is provided
        else:
            return CoopsCurrentsFeed(
                location_config=location_config,
                feed_config=current_config,
                station=station,
                expiration_interval=expiration_interval,
            )
    elif isinstance(current_config, config_lib.NwisCurrentFeedConfig):
        if clients is None or "nwis" not in clients:
            raise ValueError("NWIS client required for NwisCurrentFeed")
        return NwisCurrentFeed(
            location_config=location_config,
            feed_config=current_config,
            expiration_interval=expiration_interval,
        )

    else:
        # Unsupported config type
        raise TypeError(
            f"Unsupported current configuration type: {type(current_config).__name__}"
        )


def create_tide_feed(
    location_config: config_lib.LocationConfig,
    tide_config: config_lib.CoopsTideFeedConfig,
    expiration_interval: Optional[datetime.timedelta] = None,
) -> Feed:
    """Create a tide feed based on the configuration type.

    This factory function creates the appropriate tide feed based on the
    tide source configuration. Currently only supports NOAA CO-OPS tide sources.

    Args:
        location_config: Location configuration
        tide_config: NOAA CO-OPS tide source configuration
        expiration_interval: Custom expiration interval (optional)

    Returns:
        Configured tide feed
    """
    # Create the tide feed
    return CoopsTidesFeed(
        location_config=location_config,
        feed_config=tide_config,
        expiration_interval=expiration_interval,
    )


class NwisCurrentFeed(CurrentsFeed):
    """Feed for fetching current data from USGS NWIS.

    Uses the NwisClient to retrieve current velocity/discharge data based on
    site number and parameter code specified in the configuration.
    """

    feed_config: config_lib.NwisCurrentFeedConfig

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.CurrentDataModel  # type: ignore[return-value]

    async def _fetch(self, clients: Dict[str, BaseApiClient]) -> pd.DataFrame:
        """Fetch currents data from NWIS.

        Returns:
            DataFrame with currents data

        Raises:
            Exception: If fetching fails
        """
        self.log("Fetching NWIS current data")

        nwis_client = clients.get("nwis")
        if not isinstance(nwis_client, nwis.NwisApi):  # Use correct class name NwisApi
            raise TypeError("NWIS client not configured or is incorrect type")

        # Check for site configuration *after* verifying client
        if not self.feed_config.site_no:
            raise ValueError("NWIS site number not configured for currents feed")

        # Fetch raw data using site_no and parameter_cd
        # Call the new 'currents' method on the client
        df = await nwis_client.currents(
            site_no=self.feed_config.site_no,
            parameter_cd=self.feed_config.parameter_cd,
            timezone=str(
                self.location_config.timezone
            ),  # Use str() for timezone string
            location_code=self.location_config.code,
        )

        # --- DataFrame Processing --- #
        if df is None or df.empty:
            raise ValueError(
                "NWIS client returned no current data for the requested site/parameter."
            )

        # Guard clause: Check if the expected column is missing
        if "velocity_fps" not in df.columns:
            error_msg = f"Expected 'velocity_fps' column not found in NWIS data. Columns: {df.columns.tolist()}"
            self.log(error_msg, level=logging.ERROR)
            raise ValueError(error_msg)

        # Convert fps to knots and assign to 'velocity' column
        df["velocity"] = df["velocity_fps"].apply(fps_to_knots)
        processed_df = df[["velocity"]]

        self.log(
            f"Successfully processed {len(processed_df)} rows of NWIS current data."
        )
        return processed_df
        # --- End Processing --- #


class HistoricalTempsFeed(CompositeFeed):
    """Feed for historical temperature data across multiple years.

    Fetches temperature data for multiple years and combines them to provide
    historical averages for each day of the year. This is useful for showing
    typical temperatures for a given date based on historical records.
    """

    feed_config: config_lib.TempFeedConfig
    start_year: int
    end_year: int
    interval: Literal["h", "6-min"] = "h"
    clients: Dict[str, BaseApiClient] = {}  # Add clients dict parameter

    @property
    def data_model(self) -> Type[DataFrameModel]:
        """The Pandera data model class used to validate the fetched data."""
        return df_models.WaterTempDataModel  # type: ignore[return-value]

    def _get_feeds(self, clients: Dict[str, BaseApiClient]) -> List[Feed]:
        """Create temperature feeds for each year in the range.

        For the current year, caps the end date to today to avoid requesting future dates
        from the API, which would result in an error.

        Returns:
            List of TempFeed instances, one for each year in the range
        """
        feeds: List[Feed] = []
        current_date = utc_now()

        for year in range(self.start_year, self.end_year + 1):
            # For each year, create a feed with start/end dates for that year
            start_date = datetime.datetime(year, 1, 1)
            # For the current year, cap the end date to today
            if year == current_date.year:
                end_date = current_date
            else:
                end_date = datetime.datetime(year, 12, 31, 23, 59, 59)

            # Set expiration based on whether it's historical or current data
            # Historical data won't change, so set expiration to None (never expire)
            # Current year data should use the configured interval for the historical feed
            expiration_interval = (
                None  # Never expire for past years
                if year < current_date.year
                else self.expiration_interval  # Use configured interval for current year
            )

            # Create the appropriate feed using the factory function
            feed = create_temp_feed(
                location_config=self.location_config,
                temp_config=self.feed_config,
                start=start_date,
                end=end_date,
                interval="h",  # Use hourly data for historical feeds
                expiration_interval=expiration_interval,
                clients=clients,
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

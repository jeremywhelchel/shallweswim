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

# Local imports
from shallweswim import config as config_lib
from shallweswim import noaa
from shallweswim.util import utc_now

# Additional buffer before reporting data as expired
# This gives the system time to refresh data without showing as expired
EXPIRATION_BUFFER = datetime.timedelta(seconds=300)

# XXX Make async methods
# XXX Add common logging patterns and error handling too


class Feed(abc.ABC):
    # Frequency in which this data needs to be fetched, otherwise it is considered expired.
    # If None, this dataset will never expire and only needs to be fetched once.
    expiration_interval: Optional[datetime.timedelta]
    # Timestamp at which the data was last successfully fetched. None means it has never been fetched.
    _timestamp: Optional[datetime.datetime] = None
    # Internal data that will be updated
    _data: Optional[pd.DataFrame] = None

    location_config: config_lib.LocationConfig

    @property
    def is_expired(self) -> bool:
        if not self._timestamp:
            return False
        if not self.expiration_interval:
            return True
        now = utc_now()
        # Use the EXPIRATION_BUFFER to give the system time to refresh before reporting as expired
        age = now - self._timestamp
        return age > (self.expiration_interval + EXPIRATION_BUFFER)

    @property
    @abc.abstractmethod
    def values(self) -> pd.DataFrame:
        return self._data

    async def update(self) -> None:
        """Update the data from this feed if it is not expired."""
        if not self.is_expired:
            return
        df = await self._fetch()
        # XXX Perform some basic validation here
        # XXX assert no tzinfo. log latest datapoint, etc...
        # XXX remove outliers
        self._data = df
        self._timestamp = utc_now()

    @abc.abstractmethod
    async def _fetch(self) -> pd.DataFrame: ...

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # XXX implement this
        return df


# XXX longterm and short term handled in data management layer
# XXX long-term temp is impelemented as multiple years fetched in parallel
# XXX will want a hook to update charts, etc, upon refresh


class TempFeed(Feed, abc.ABC):
    name: str
    config: config_lib.TempSource

    interval: Literal["h", "6-min"]  # XXX 6-min is noaa specific. Make a type

    # XXX if unspecified, end should be now, and we should read the most recent data (how much? XXX)
    start: Optional[datetime.datetime]
    end: Optional[datetime.datetime]


class NoaaTempFeed(TempFeed):
    config: config_lib.NoaaTempSource

    async def _fetch(self) -> pd.DataFrame:
        station_id = self.config.station
        # XXX Get properly from parameters
        begin_date = datetime.datetime.today() - datetime.timedelta(days=8)
        end_date = datetime.datetime.today()
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
            logging.warning(f"[{self.location_config.code}] Live temp fetch error: {e}")


# class UsgsTempFeed(TempFeed):
#    config: config_lib.UsgsTempSource


class TidesFeed(Feed, abc.ABC): ...


# XXX this one will return a feed that includes future projections. But the prev/next split will
# be done elsewhere.


class CurrentsFeed(Feed, abc.ABC):
    # XXX this one I believe is primarily predictions
    ...

"""Data fetching and management for ShallWeSwim application.

This module handles data retrieval from NOAA APIs, data processing,
and provides the necessary data for plotting and presentation.
"""

# Standard library imports
import datetime
import logging
import threading
import time
from concurrent import futures
from typing import Optional, Tuple, cast

# Third-party imports
import pandas as pd

# Local imports
from shallweswim import config as config_lib
from shallweswim import noaa
from shallweswim import plot
from shallweswim import util
from shallweswim.types import (
    DatasetName,
    LegacyChartInfo,
    TideInfo,
    TideEntry,
    CurrentInfo,
)


# Use the utility function for consistent time handling
Now = util.Now


def LatestTimeValue(df: Optional[pd.DataFrame]) -> Optional[datetime.datetime]:
    """Extract the timestamp of the most recent data point from a DataFrame.

    Args:
        df: DataFrame with DatetimeIndex, or None

    Returns:
        Timezone-aware datetime object (in UTC) of the last index value,
        or None if DataFrame is None
    """
    if df is None:
        return None
    # Get the datetime and ensure it's timezone-aware
    dt = df.index[-1].to_pydatetime()
    # Add UTC timezone if datetime is naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return cast(datetime.datetime, dt)


class Data(object):
    """Data management for ShallWeSwim application.

    This class handles fetching, processing, and storing data from various NOAA sources,
    including tides, currents, and temperature readings. It maintains data freshness
    and provides methods to access processed data for the web application.
    """

    def __init__(self, config: config_lib.LocationConfig):
        """Initialize the Data object with configuration settings.

        Args:
            config: Location-specific configuration settings
        """
        self.config = config

        # Data storage
        self.tides = None  # Tide predictions
        self.currents = None  # Current predictions
        self.historic_temps = None  # Historical temperature data
        self.live_temps = None  # Recent temperature readings

        # Timestamps for tracking when data was last fetched
        self._tides_timestamp: datetime.datetime | None = None
        self._live_temps_timestamp: datetime.datetime | None = None
        self._historic_temps_timestamp: datetime.datetime | None = None

        # Data expiration periods
        self.expirations = {
            # Tidal predictions already cover a wide past/present window
            "tides_and_currents": datetime.timedelta(hours=24),
            # Live temperature readings occur every 6 minutes, and are
            # generally already 5 minutes old when a new reading first appears
            "live_temps": datetime.timedelta(minutes=10),
            # Hourly fetch historic temps + generate charts
            "historic_temps": datetime.timedelta(hours=3),
        }

    def _Expired(self, dataset: DatasetName) -> bool:
        """Check if a dataset has expired and needs to be refreshed.

        Args:
            dataset: The name of the dataset to check

        Returns:
            True if the dataset is expired or missing, False otherwise
        """
        timestamp = None
        if dataset == "tides_and_currents":
            timestamp = self._tides_timestamp
        elif dataset == "live_temps":
            timestamp = self._live_temps_timestamp
        elif dataset == "historic_temps":
            timestamp = self._historic_temps_timestamp

        # If no timestamp, data is missing
        if timestamp is None:
            return True

        # Check if data is older than the expiration period
        # Use timezone-aware datetime with the consistent Now() function
        now = Now()  # Now returns a timezone-aware datetime in UTC

        # Make sure timestamp is timezone-aware for proper comparison
        if timestamp.tzinfo is None:
            # Convert naive timestamp to UTC for consistent comparison
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

        age = now - timestamp
        return age > self.expirations[dataset]

    def _Update(self) -> None:
        """Background thread that continuously updates data.

        This runs as a daemon thread and periodically checks if datasets
        have expired. If so, it fetches new data and generates updated plots.
        """
        while True:
            if self._Expired("tides_and_currents"):
                self._FetchTidesAndCurrents()

            if self._Expired("live_temps"):
                self._FetchLiveTemps()
                plot.GenerateLiveTempPlot(
                    self.live_temps, self.config.code, self.config.temp_station_name
                )

            if self._Expired("historic_temps"):
                self._FetchHistoricTemps()
                plot.GenerateHistoricPlots(
                    self.historic_temps, self.config.code, self.config.temp_station_name
                )

            # TODO: Can probably be increased to 1s even... but would need to add API spam buffer
            time.sleep(60)

    def Start(self) -> None:
        """Start the background data fetching process.

        This creates and starts a daemon thread that periodically fetches
        and processes new data. The thread name is unique per location.

        Raises:
            AssertionError: If a thread for this location is already running
        """
        thread_name = f"DataUpdateThread_{self.config.code}"
        for thread in threading.enumerate():
            assert thread.name != thread_name, "Data update thread already running"

        logging.info("Starting data fetch thread")
        self._update_thread = threading.Thread(
            target=self._Update, name=thread_name, daemon=True
        )
        self._update_thread.start()

    def PrevNextTide(self) -> TideInfo:
        """Return the previous tide and next two tides.

        Retrieves the most recent tide before current time and the next two
        upcoming tides from the tide predictions data.

        Returns:
            A TideInfo object containing:
                - past_tides: List of TideEntry objects with the most recent tide information
                - next_tides: List of TideEntry objects with the next two upcoming tides
            If no tide data is available, returns placeholder values.
        """
        if self.tides is None:
            # Create placeholder entries for missing data
            placeholder_time = datetime.datetime.combine(
                datetime.date.today(), datetime.time(0)
            )
            placeholder_entry = TideEntry(
                time=placeholder_time, type="unknown", prediction=0.0
            )
            past_tides = [placeholder_entry]
            next_tides = [placeholder_entry, placeholder_entry]
        else:
            # Convert the DataFrame records to TideEntry objects
            # Get the current time with proper timezone handling
            now = Now()

            # Handle the timezone compatibility for DataFrame slicing
            # Check if the tides DataFrame has a timezone-naive index
            if self.tides.index.tz is None:
                # If index is naive, we need a naive datetime for slicing
                now_for_slicing = now.replace(tzinfo=None)
            else:
                # If index has timezone info, ensure it's in the same timezone
                now_for_slicing = now

            # When we reset_index, the DatetimeIndex becomes a column named 'time'
            past_tide_dicts = (
                self.tides[:now_for_slicing]
                .tail(1)
                .reset_index()
                .to_dict(orient="records")
            )
            next_tide_dicts = (
                self.tides[now_for_slicing:]
                .head(2)
                .reset_index()
                .to_dict(orient="records")
            )

            # Get the location's timezone for converting times
            location_tz = self.config.timezone

            # Convert DataFrame records directly to TideEntry objects with timezone conversion
            past_tides = []
            for record in past_tide_dicts:
                # Get the time and ensure it has timezone info
                tide_time = record.get("time", Now())
                if tide_time.tzinfo is None:
                    tide_time = tide_time.replace(tzinfo=datetime.timezone.utc)

                # Convert to location timezone
                local_time = tide_time.astimezone(location_tz)

                past_tides.append(
                    TideEntry(
                        time=local_time,
                        type=record.get("type", "unknown"),
                        prediction=record.get("prediction", 0.0),
                    )
                )

            next_tides = []
            for record in next_tide_dicts:
                # Get the time and ensure it has timezone info
                tide_time = record.get("time", Now())
                if tide_time.tzinfo is None:
                    tide_time = tide_time.replace(tzinfo=datetime.timezone.utc)

                # Convert to location timezone
                local_time = tide_time.astimezone(location_tz)

                next_tides.append(
                    TideEntry(
                        time=local_time,
                        type=record.get("type", "unknown"),
                        prediction=record.get("prediction", 0.0),
                    )
                )

        return TideInfo(past_tides=past_tides, next_tides=next_tides)

    def LegacyChartInfo(self, t: Optional[datetime.datetime] = None) -> LegacyChartInfo:
        """Generate legacy chart information based on tide data.

        Args:
            t: The time to generate chart info for, defaults to current time

        Returns:
            A LegacyChartInfo object containing:
                - hours_since_last_tide: Number of hours since last tide
                - last_tide_type: Type of last tide (high/low)
                - chart_filename: Filename for the chart image
                - map_title: Title for the legacy map

        Raises:
            AssertionError: If tide data is not available
        """
        if not t:
            t = Now()
        assert self.tides is not None

        # Handle the timezone compatibility for DataFrame index lookup
        # Check if the DataFrame has a timezone-naive index
        if self.tides.index.tz is None:
            # If index is naive, we need a naive datetime for slicing
            t_for_lookup = t.replace(tzinfo=None) if t.tzinfo is not None else t
        else:
            # If index has timezone info, ensure it's in the same timezone
            t_for_lookup = t

        row = self.tides.loc[self.tides.index.asof(t_for_lookup)]
        # TODO: Break this into a function
        tide_type = row["type"]

        # Ensure row.name has the same timezone awareness as t before subtraction
        row_time = row.name
        if t.tzinfo is not None and row_time.tzinfo is None:
            # If t is timezone-aware but row_time is naive, add the same timezone to row_time
            row_time = row_time.replace(tzinfo=t.tzinfo)
        elif t.tzinfo is None and row_time.tzinfo is not None:
            # If t is naive but row_time is timezone-aware, make t timezone-aware
            t = t.replace(tzinfo=row_time.tzinfo)

        offset = t - row_time
        offset_hrs = offset.seconds / (60 * 60)
        if offset_hrs > 5.5:
            chart_num = 0
            INVERT = {"high": "low", "low": "high"}
            chart_type = INVERT[tide_type]
            # TODO: last tide type will be wrong here in the chart
        else:
            chart_num = round(offset_hrs)
            chart_type = tide_type

        legacy_map_title = "%s Water at New York" % chart_type.capitalize()
        if chart_num:
            legacy_map_title = (
                "%i Hour%s after " % (chart_num, "s" if chart_num > 1 else "")
                + legacy_map_title
            )
        filename = "%s+%s.png" % (chart_type, chart_num)

        return LegacyChartInfo(
            hours_since_last_tide=offset_hrs,
            last_tide_type=tide_type,
            chart_filename=filename,
            map_title=legacy_map_title,
        )

    def CurrentPrediction(self, t: Optional[datetime.datetime] = None) -> CurrentInfo:
        """Predict current conditions for a specific time.

        Args:
            t: Time to predict current for, defaults to current time

        Returns:
        A CurrentInfo object containing:
            - direction: Direction of current ("flooding" or "ebbing")
            - magnitude: Magnitude of current in knots
            - magnitude_pct: Relative magnitude percentage (0.0-1.0)
            - state_description: Human-readable message describing the current's state

        Raises:
            AssertionError: If current data is not available
            ValueError: If the current state cannot be determined
        """
        if not t:
            t = Now()

        assert self.currents is not None
        v = self.currents["velocity"]
        df = pd.DataFrame(
            {
                "v": v,
                "magnitude": v.abs(),
                "slope": v.abs().shift(-1) - v.abs().shift(1),
            }
        )  # TODO: Calculate trend more accurately
        df["ef"] = (df["v"] > 0).map({True: "flooding", False: "ebbing"})

        # TODO: This doesn't work when df has a stronger current in a different
        # tidal cycle. Try to only look at +/- til we get to 0 or something
        df["mag_pct"] = df.groupby("ef")["magnitude"].rank(pct=True)

        # Handle the timezone compatibility for DataFrame slicing
        # Check if the DataFrame has a timezone-naive index
        if df.index.tz is None:
            # If index is naive, we need a naive datetime for slicing
            t_for_slicing = t.replace(tzinfo=None) if t.tzinfo is not None else t
        else:
            # If index has timezone info, ensure it's in the same timezone
            t_for_slicing = t

        row = df[t_for_slicing:].iloc[0]

        STRONG_THRESHOLD = 0.85  # 30% on either side of peak
        WEAK_THRESHOLD = 0.15  # 30% on either side of bottom
        magnitude = row["magnitude"]

        # TODO: Move to template for better separation of concerns
        if row["mag_pct"] < WEAK_THRESHOLD:
            msg = "at its weakest (slack)"
        elif row["mag_pct"] > STRONG_THRESHOLD:
            msg = "at its strongest"
        elif row["slope"] < 0:
            msg = "getting weaker"
        elif row["slope"] > 0:
            msg = "getting stronger"
        else:
            raise ValueError(row)

        ef = row["ef"]

        # Return a structured object with current information
        return CurrentInfo(
            direction=ef,
            magnitude=magnitude,
            magnitude_pct=row["mag_pct"],
            state_description=msg,
        )

    def LiveTempReading(self) -> Tuple[pd.Timestamp, float]:
        """Get the most recent water temperature reading.

        Returns:
            A tuple containing (timestamp, temperature in °F)
            If no data is available, returns default values
        """
        if self.live_temps is None:
            return datetime.time(0), 0.0
        ((time, temp),) = self.live_temps.tail(1)["water_temp"].items()
        return time, temp

    def _FetchTidesAndCurrents(self) -> None:
        """Fetch tide and current data from NOAA API.

        Updates the tides and currents instance variables with fresh data from NOAA.
        Sets the _tides_timestamp to track when data was last retrieved.

        If multiple current stations are configured, their data is averaged.
        Logs warnings if the NOAA API returns an error.
        """
        logging.info("Fetching tides and currents")
        try:
            if self.config.tide_station:
                self.tides = noaa.NoaaApi.Tides(station=self.config.tide_station)

            if self.config.currents_stations:
                currents = [
                    noaa.NoaaApi.Currents(stn) for stn in self.config.currents_stations
                ]
                self.currents = (
                    pd.concat(currents)[["velocity"]].groupby(level=0).mean()
                )

            self._tides_timestamp = Now()
        except noaa.NoaaApiError as e:
            logging.warning(f"Tide fetch error: {e}")

    def _FetchHistoricTempYear(self, year: int) -> pd.DataFrame:
        """Fetch temperature data for a specific year.

        Args:
            year: The year to fetch data for

        Returns:
            DataFrame containing both air and water temperature data for the year

        Raises:
            AssertionError: If temperature station is not configured
        """
        begin_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        assert self.config.temp_station is not None
        return pd.concat(
            [
                noaa.NoaaApi.Temperature(
                    self.config.temp_station,
                    "air_temperature",
                    begin_date,
                    end_date,
                    interval="h",
                ),
                noaa.NoaaApi.Temperature(
                    self.config.temp_station,
                    "water_temperature",
                    begin_date,
                    end_date,
                    interval="h",
                ),
            ],
            axis=1,
        )

    def _FetchHistoricTemps(self) -> None:
        """Fetch hourly temperature data since 2011.

        Uses multiple threads to fetch data by year in parallel, then concatenates
        the results. Removes known erroneous data points and resamples to hourly intervals.
        Sets the _historic_temps_timestamp to track when data was last retrieved.

        Skips fetching if no temperature station is configured.
        Logs warnings if the NOAA API returns an error.
        """
        if not self.config.temp_station:
            return
        logging.info("Fetching historic temps")
        try:
            years = range(2011, Now().year + 1)
            threadpool = futures.ThreadPoolExecutor(len(years))
            year_frames = threadpool.map(self._FetchHistoricTempYear, years)
            # Concat all the yearly data frames
            historic_temps = pd.concat(year_frames)

            # Remove outliers using the helper method
            historic_temps = self._RemoveOutliers(historic_temps)

            # Resample to hourly intervals and assign to instance variable
            self.historic_temps = historic_temps.resample("h").first()
            self._historic_temps_timestamp = Now()
        except noaa.NoaaApiError as e:
            logging.warning(f"Historic temp fetch error: {e}")

        # TODO: Test by disabling local wifi briefly to ensure error handling works

    def _RemoveOutliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove known erroneous data points from a DataFrame.

        Uses the temp_outliers list from the location configuration to identify and
        remove specific timestamps that are known to have bad data. This helps to
        improve data quality by filtering out known anomalies.

        Args:
            df: DataFrame with DatetimeIndex to remove outliers from

        Returns:
            DataFrame with outliers removed
        """
        if not self.config.temp_outliers:
            return df

        result_df = df.copy()
        for timestamp in self.config.temp_outliers:
            try:
                result_df = result_df.drop(pd.to_datetime(timestamp))
            except KeyError:
                # Skip if the timestamp doesn't exist in the data
                logging.debug(f"Outlier timestamp {timestamp} not found in data")

        return result_df

    def _FetchLiveTemps(self) -> None:
        """Fetch recent air and water temperatures (last 8 days).

        Updates the live_temps instance variable with the most recent temperature
        data from NOAA. Sets the _live_temps_timestamp to track when data was last retrieved.

        Skips fetching if no temperature station is configured.
        Logs the age of the most recent data point.
        Logs warnings if the NOAA API returns an error.
        """
        if not self.config.temp_station:
            return
        logging.info("Fetching live temps")
        begin_date = datetime.datetime.today() - datetime.timedelta(days=8)
        end_date = datetime.datetime.today()
        # TODO: Resample to 6min for more consistent data intervals
        try:
            # Concatenate air and water temperature data
            temp_data = pd.concat(
                [
                    noaa.NoaaApi.Temperature(
                        self.config.temp_station,
                        "air_temperature",
                        begin_date,
                        end_date,
                    ),
                    noaa.NoaaApi.Temperature(
                        self.config.temp_station,
                        "water_temperature",
                        begin_date,
                        end_date,
                    ),
                ],
                axis=1,
            )

            # Remove outliers using the helper method
            self.live_temps = self._RemoveOutliers(temp_data)
            self._live_temps_timestamp = Now()

            # Log info about the most recent data point
            last_time = LatestTimeValue(self.live_temps)
            age_str = "unknown"
            if last_time is not None:
                now = Now()
                # Ensure last_time has timezone info before subtraction
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=datetime.timezone.utc)
                age = now - last_time
                age_str = str(datetime.timedelta(seconds=int(age.total_seconds())))
            logging.info(f"Fetched live temps. Last datapoint age: {age_str}")
        except noaa.NoaaApiError as e:
            logging.warning(f"Live temp fetch error: {e}")

"""Data fetching and management for ShallWeSwim application.

This module handles data retrieval from NOAA APIs, data processing,
and provides the necessary data for plotting and presentation.
"""

# Standard library imports
import asyncio
import datetime
import logging
from typing import Any, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Local imports
from shallweswim import config as config_lib
from shallweswim import noaa
from shallweswim import plot
from shallweswim import util
from shallweswim.util import latest_time_value
from shallweswim.config import NoaaTempSource
from shallweswim.types import (
    DatasetName,
    LegacyChartInfo,
    TideInfo,
    TideEntry,
    CurrentInfo,
)

# Additional buffer before reporting data as expired
# This gives the system time to refresh data without showing as expired
EXPIRATION_BUFFER = datetime.timedelta(seconds=300)


# Use the utility function for consistent time handling
utc_now = util.utc_now


def _process_local_magnitude_pct(
    df: pd.DataFrame,
    current_df: pd.DataFrame,
    direction_type: str,
    invert: bool = False,
) -> pd.DataFrame:
    """Calculate magnitude percentages relative to local peaks.

    Args:
        df: Main DataFrame containing all current data
        current_df: DataFrame containing current data for one direction
        direction_type: String identifier ("flooding" or "ebbing")
        invert: Whether to invert values for finding peaks (for ebb currents)

    Returns:
        DataFrame with added local_mag_pct column for the specified direction
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()

    if current_df.empty:
        return result_df

    # Get magnitude values, invert for ebb
    values = current_df["magnitude"].values
    if invert:
        values = -values

    # Find peaks with optimized parameters
    # For flooding, peaks are maxima; for ebbing, peaks are minima
    # Since find_peaks finds maxima, we negate values for ebbing to find minima
    search_values = -values if direction_type == "ebbing" else values
    peaks, _ = find_peaks(search_values, prominence=0.1, distance=3)

    # No peaks found - use global approach as fallback
    if len(peaks) == 0:
        # Set global percentage for all points of this direction
        for idx in result_df.index[result_df["ef"] == direction_type]:
            result_df.at[idx, "local_mag_pct"] = result_df.at[idx, "mag_pct"]
        return result_df

    # For each peak, find its "influence zone" (range of time where it's the closest peak)
    peak_times = current_df.index[peaks]
    peak_values = [current_df.at[pt, "magnitude"] for pt in peak_times]

    # For each time point, find the nearest peak and calculate percentage relative to it
    for idx in result_df.index[result_df["ef"] == direction_type]:
        # Find closest peak in time
        time_diffs = [
            (idx - peak_time).total_seconds() / 3600 for peak_time in peak_times
        ]
        abs_diffs = [abs(diff) for diff in time_diffs]
        nearest_peak_idx = abs_diffs.index(min(abs_diffs))
        # We only need the peak value, not the time
        nearest_peak_value = peak_values[nearest_peak_idx]

        # Calculate percentage of current magnitude relative to nearest peak
        current_value = result_df.at[idx, "magnitude"]

        # Both current_value and nearest_peak_value are already positive magnitudes
        # We always want the percentage to be: current magnitude / peak magnitude
        # This works for both ebbing and flooding currents since we're dealing with
        # absolute magnitude values, not raw velocities
        local_pct = current_value / nearest_peak_value if nearest_peak_value > 0 else 0

        # Store the local percentage
        result_df.at[idx, "local_mag_pct"] = local_pct

    return result_df


class DataManager(object):
    """DataManager for ShallWeSwim application.

    This class manages all the data feeds for one location. It handles fetching, processing,
    and storing data from various NOAA sources, including tides, currents, and temperature
    readings. It maintains data freshness and provides methods to access processed data
    for the web application.
    """

    def __init__(self, config: config_lib.LocationConfig) -> None:
        """Initialize the Data object with configuration settings.

        Args:
            config: Location-specific configuration settings
        """
        self.config = config

        # Data caches
        self.tides: Optional[pd.DataFrame] = None
        self.currents: Optional[pd.DataFrame] = None
        self.live_temps: Optional[pd.DataFrame] = None
        self.historic_temps: Optional[pd.DataFrame] = None

        # Timestamps for last data retrieval
        self._tides_timestamp: Optional[datetime.datetime] = None
        self._live_temps_timestamp: Optional[datetime.datetime] = None
        self._historic_temps_timestamp: Optional[datetime.datetime] = None

        # Background update task
        self._update_task: Optional[asyncio.Task[None]] = None

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

    @property
    def ready(self) -> bool:
        """Check if all datasets have been fetched and are not expired.

        Returns:
            True if all datasets have been fetched and are not expired, False otherwise
        """
        # Check each dataset for expiration
        datasets: list[DatasetName] = [
            "tides_and_currents",
            "live_temps",
            "historic_temps",
        ]
        return all(not self._expired(dataset) for dataset in datasets)

    def _expired(self, dataset: DatasetName) -> bool:
        """Check if a dataset has expired and needs to be refreshed.

        Args:
            dataset: The dataset to check

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
        # Use naive datetime from utc_now() for consistent handling
        now = utc_now()  # utc_now returns a naive datetime in UTC
        assert now.tzinfo is None, "utc_now() should return naive datetime"

        # No timezone information allowed - strict assertion
        assert (
            timestamp.tzinfo is None
        ), f"Timestamp for {dataset} has timezone info - all timestamps must be naive"

        # Use the EXPIRATION_BUFFER to give the system time to refresh before reporting as expired
        age = now - timestamp
        return age > (self.expirations[dataset] + EXPIRATION_BUFFER)

    async def _update_dataset(self, dataset: DatasetName) -> None:
        """Update a specific dataset if it has expired.

        Args:
            dataset: The dataset to update
        """
        if dataset == "tides_and_currents":
            await self._fetch_tides_and_currents()
        elif dataset == "live_temps":
            await self._fetch_live_temps()
        elif dataset == "historic_temps":
            await self._fetch_historic_temps()

    async def __update_loop(self) -> None:
        """Background asyncio task that continuously updates data.

        This runs as an asyncio task and periodically checks if datasets
        have expired. If so, it fetches new data and generates updated plots.

        This method catches and logs exceptions to prevent silent failures in the event loop,
        but will re-raise them to ensure they're not silently ignored.
        """
        try:
            while True:
                try:
                    if self._expired("tides_and_currents"):
                        await self._update_dataset("tides_and_currents")

                    if self._expired("live_temps"):
                        await self._update_dataset("live_temps")
                        # Only generate plot if we have valid data
                        if self.live_temps is not None and len(self.live_temps) >= 2:
                            plot.generate_and_save_live_temp_plot(
                                self.live_temps,
                                self.config.code,
                                (
                                    self.config.temp_source.name
                                    if self.config.temp_source
                                    else None
                                ),
                            )

                    if self._expired("historic_temps"):
                        await self._update_dataset("historic_temps")
                        # Only generate plots if we have valid historical data
                        if (
                            self.historic_temps is not None
                            and len(self.historic_temps) >= 10
                        ):
                            plot.generate_and_save_historic_plots(
                                self.historic_temps,
                                self.config.code,
                                (
                                    self.config.temp_source.name
                                    if self.config.temp_source
                                    else None
                                ),
                            )
                except Exception as e:
                    # Log the exception but don't swallow it - let it propagate
                    logging.exception(
                        f"Error in data update loop for {self.config.code}: {e}"
                    )
                    # Re-raise to ensure error is not silently ignored
                    raise

                # Sleep for 1 second before the next update check
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # This is expected when the task is cancelled, so we let it propagate
            raise
        except Exception as e:
            # Log any unexpected exceptions at the outer level as well
            logging.exception(
                f"Fatal error in data update loop for {self.config.code}: {e}"
            )
            raise  # Re-raise to ensure error is not silently ignored

    def start(self) -> None:
        """Start the background data fetching process.

        This creates and starts an asyncio task that periodically fetches
        and processes new data.

        Raises:
            AssertionError: If a task for this location is already running
        """
        task_name = f"DataUpdateTask_{self.config.code}"
        # Check if task already exists
        for task in asyncio.all_tasks():
            if task.get_name() == task_name and not task.done():
                raise AssertionError("Data update task already running")

        logging.info(f"[{self.config.code}] Starting data fetch task")
        self._update_task = asyncio.create_task(self.__update_loop())
        self._update_task.set_name(task_name)

        # Add exception handling to the task
        self._update_task.add_done_callback(self._handle_task_exception)

    def prev_next_tide(self) -> TideInfo:
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
            # Get current time in the location's timezone as a naive datetime
            now = self.config.local_now()

            # Ensure DataFrame has no timezone info for consistent comparison
            assert (
                self.tides.index.tz is None
            ), "Tide DataFrame should use naive datetimes"

            # When we reset_index, the DatetimeIndex becomes a column named 'time'
            # Convert to pandas Timestamp for proper slicing (to satisfy type checking)
            now_ts = pd.Timestamp(now)
            past_tide_dicts = (
                self.tides[:now_ts].tail(1).reset_index().to_dict(orient="records")
            )
            next_tide_dicts = (
                self.tides[now_ts:].head(2).reset_index().to_dict(orient="records")
            )

            # Get the location's timezone for converting times
            location_tz = self.config.timezone

            # Convert DataFrame records directly to TideEntry objects with timezone conversion
            past_tides = []
            for record in past_tide_dicts:
                # Get the time from the record
                tide_time = record.get("time", utc_now())

                # If it has timezone info, we need to convert to local time and remove timezone
                # All times must be naive datetimes in their appropriate timezone
                if tide_time.tzinfo is not None:
                    # Convert to location timezone
                    local_time = tide_time.astimezone(location_tz).replace(tzinfo=None)
                else:
                    # Already naive, assume it's in the correct timezone
                    local_time = tide_time

                past_tides.append(
                    TideEntry(
                        time=local_time,
                        type=record.get("type", "unknown"),
                        prediction=record.get("prediction", 0.0),
                    )
                )

            next_tides = []
            for record in next_tide_dicts:
                # Get the time from the record
                tide_time = record.get("time", utc_now())

                # If it has timezone info, we need to convert to local time and remove timezone
                # All times must be naive datetimes in their appropriate timezone
                if tide_time.tzinfo is not None:
                    # Convert to location timezone
                    local_time = tide_time.astimezone(location_tz).replace(tzinfo=None)
                else:
                    # Already naive, assume it's in the correct timezone
                    local_time = tide_time

                next_tides.append(
                    TideEntry(
                        time=local_time,
                        type=record.get("type", "unknown"),
                        prediction=record.get("prediction", 0.0),
                    )
                )

        return TideInfo(past_tides=past_tides, next_tides=next_tides)

    def legacy_chart_info(
        self, t: Optional[datetime.datetime] = None
    ) -> LegacyChartInfo:
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
            t = self.config.local_now()
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

    def current_prediction(self, t: Optional[datetime.datetime] = None) -> CurrentInfo:
        """Predict current conditions for a specific time.

        Args:
            t: Time to predict current for, defaults to current time

        Returns:
            A CurrentInfo object containing:
                - direction: Direction of current ("flooding" or "ebbing")
                - magnitude: Magnitude of current in knots
                - magnitude_pct: Relative magnitude percentage (0.0-1.0)
                - state_description: Text description of current state
        """
        if not t:
            t = self.config.local_now()

        assert self.currents is not None, "Current data must be loaded first"

        # Create a working copy of the current data for analysis
        df = self.currents.copy()

        # Convert raw velocity to magnitude and direction
        df["v"] = df["velocity"]
        df["magnitude"] = df["v"].abs()

        # Add a slope column to track if current is strengthening or weakening
        # For ebb currents (negative values), we need to negate the slope
        # so that strengthening (becoming more negative) is represented by positive slope
        df["raw_slope"] = df["v"].diff()

        # Create a directionally correct slope column
        # For flooding: positive slope = strengthening, negative slope = weakening
        # For ebbing: negative slope = strengthening (more negative), positive slope = weakening (less negative)
        conditions = [
            (df["v"] > 0),  # flooding
            (df["v"] < 0),  # ebbing
        ]
        choices = [
            df["raw_slope"],  # for flooding, use raw slope
            -df["raw_slope"],  # for ebbing, negate the slope
        ]
        df["slope"] = np.select(conditions, choices, default=0)

        # Mark direction as flood or ebb
        conditions = [df["v"] > 0, df["v"] < 0]
        choices = ["flooding", "ebbing"]
        df["ef"] = pd.Series(
            np.select(conditions, choices, default="unknown"), index=df.index
        )

        # Initialize column for local magnitude percentage
        df["local_mag_pct"] = 0.0  # Default to 0% of local peak

        # Process flood and ebb separately
        flood_df = df[df["v"] > 0].copy()
        ebb_df = df[df["v"] < 0].copy()

        # Calculate mag_pct for global magnitude ranking (needed for fallback)
        df["mag_pct"] = df.groupby("ef")["magnitude"].rank(pct=True)

        # Now calculate magnitude percentages relative to local peaks
        # Process both directions sequentially, passing the result of one to the next
        df = _process_local_magnitude_pct(df, flood_df, "flooding")
        df = _process_local_magnitude_pct(df, ebb_df, "ebbing", invert=True)

        # Ensure we're using a naive datetime for DataFrame slicing
        # All datetimes must be naive in the appropriate timezone
        assert t.tzinfo is None, "Input datetime must be naive"

        # Ensure DataFrame has no timezone info for consistent comparison
        assert df.index.tz is None, "DataFrame should use naive datetimes"

        # Convert to pandas Timestamp for proper slicing (to satisfy type checking)
        t_ts = pd.Timestamp(t)
        row = df[t_ts:].iloc[0]

        # Constants for determining current state based on local magnitude percentage
        STRONG_THRESHOLD = 0.85  # 85% of local peak is considered "strong"
        SLACK_THRESHOLD = (
            0.2  # Absolute threshold for considering current as slack water
        )
        magnitude = row["magnitude"]
        local_pct = row["local_mag_pct"]

        # First check if we're near slack water (zero crossing)
        # This takes precedence over magnitude percentage
        if magnitude < SLACK_THRESHOLD:
            msg = "at its weakest (slack)"
        # Check if we're at/near a peak based on local percentage
        elif local_pct > STRONG_THRESHOLD:
            msg = "at its strongest"  # Near a local peak
        # Otherwise use slope to determine if strengthening or weakening
        # The slope has been adjusted already to account for current direction
        # so we can use a consistent interpretation: positive = strengthening, negative = weakening
        elif row["slope"] < 0:
            msg = "getting weaker"
        elif row["slope"] > 0:
            msg = "getting stronger"
        else:
            msg = "stable"

        ef = row["ef"]

        # Return a structured object with current information
        return CurrentInfo(
            direction=ef,
            magnitude=magnitude,
            magnitude_pct=row["mag_pct"],
            state_description=msg,
        )

    def live_temp_reading(self) -> Tuple[pd.Timestamp, float]:
        """Get the most recent water temperature reading.

        Returns:
            A tuple containing (timestamp, temperature in °F)
            If no data is available, returns default values
        """
        if self.live_temps is None:
            return datetime.time(0), 0.0
        ((time, temp),) = self.live_temps.tail(1)["water_temp"].items()
        return time, temp

    async def _fetch_tides_and_currents(self) -> None:
        """Fetch tide and current data from NOAA API.

        Updates the tides and currents instance variables with fresh data from NOAA.
        Sets the _tides_timestamp to track when data was last retrieved.

        If multiple current stations are configured, their data is averaged.
        Logs warnings if the NOAA API returns an error.
        """
        logging.info(f"[{self.config.code}] Fetching tides and currents")
        try:
            if self.config.tide_source and self.config.tide_source.station:
                self.tides = await noaa.NoaaApi.tides(
                    station=self.config.tide_source.station,
                    location_code=self.config.code,
                )

            if self.config.currents_source and self.config.currents_source.stations:
                # Use asyncio.gather to fetch all current stations concurrently
                current_tasks = [
                    noaa.NoaaApi.currents(stn, location_code=self.config.code)
                    for stn in self.config.currents_source.stations
                ]
                currents = await asyncio.gather(*current_tasks)
                self.currents = (
                    pd.concat(currents)[["velocity"]].groupby(level=0).mean()
                )

            # Ensure we always store naive timestamps
            now = utc_now()
            assert now.tzinfo is None, "utc_now() must return naive datetime"
            self._tides_timestamp = now
        except noaa.NoaaApiError as e:
            logging.warning(f"[{self.config.code}] Tide fetch error: {e}")

    async def _fetch_historic_temp_year(self, year: int) -> pd.DataFrame:
        """Fetch historical temperature data for a specified year.

        This method retrieves both air and water temperature from the
        NOAA API for the specified year, creating a complete dataset
        with hourly readings.

        Args:
            year: The year to fetch data for

        Returns:
            DataFrame containing both air and water temperature data for the year

        Raises:
            AssertionError: If temperature station is not configured
            TypeError: If temperature source is not a supported type
        """
        assert self.config.temp_source, "Temperature source not configured"

        temp_config = self.config.temp_source
        station_id = None

        if isinstance(temp_config, NoaaTempSource):
            station_id = temp_config.station
            assert station_id, "NOAA temperature station not configured"
        else:
            raise TypeError(f"Unsupported temperature source type: {type(temp_config)}")

        logging.info(f"[{self.config.code}] Fetching historic temps for year {year}")

        begin_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        try:
            # Get both air and water temperatures concurrently
            air_temp_task = noaa.NoaaApi.temperature(
                station_id,
                "air_temperature",
                begin_date,
                end_date,
                interval="h",
                location_code=self.config.code,
            )
            water_temp_task = noaa.NoaaApi.temperature(
                station_id,
                "water_temperature",
                begin_date,
                end_date,
                interval="h",
                location_code=self.config.code,
            )

            air_temp, water_temp = await asyncio.gather(air_temp_task, water_temp_task)
            # Merge the results
            df = pd.concat([air_temp, water_temp], axis=1)
            return df
        except noaa.NoaaApiError as e:
            logging.warning(
                f"[{self.config.code}] Historic temp fetch error for {year}: {e}"
            )
            return pd.DataFrame()

    async def _fetch_historic_temps(self) -> None:
        """Fetch hourly temperature data since 2011.

        Uses asyncio.gather to fetch data by year in parallel, then concatenates
        the results. Removes known erroneous data points and resamples to hourly intervals.
        Sets the _historic_temps_timestamp to track when data was last retrieved.

        Skips fetching if no temperature station is configured.
        Logs warnings if the NOAA API returns an error.
        """
        if not self.config.temp_source:
            return

        temp_config = self.config.temp_source

        if isinstance(temp_config, NoaaTempSource):
            if not temp_config.station:
                return
        else:
            raise TypeError(f"Unsupported temperature source type: {type(temp_config)}")
        logging.info(f"[{self.config.code}] Fetching historic temps")
        try:
            years = range(2011, utc_now().year + 1)
            # Create tasks for each year and gather them
            tasks = [self._fetch_historic_temp_year(year) for year in years]
            year_frames = await asyncio.gather(*tasks)
            # Concat all the yearly data frames
            historic_temps = pd.concat(year_frames)

            # Remove outliers using the helper method
            historic_temps = self._remove_outliers(historic_temps)

            # Resample to hourly intervals and assign to instance variable
            self.historic_temps = historic_temps.resample("h").first()
            # Ensure we always store naive timestamps
            now = utc_now()
            assert now.tzinfo is None, "utc_now() must return naive datetime"
            self._historic_temps_timestamp = now
        except noaa.NoaaApiError as e:
            logging.warning(f"[{self.config.code}] Historic temp fetch error: {e}")

    def _handle_task_exception(self, task: asyncio.Task[Any]) -> None:
        """Handle exceptions from asyncio tasks to prevent them from being silently ignored.

        This callback is attached to asyncio tasks and will log any exceptions that occur,
        ensuring they're not silently swallowed by the event loop.

        Args:
            task: The asyncio task that completed (successfully or with an exception)
        """
        try:
            # If the task raised an exception, this will re-raise it
            task.result()
        except asyncio.CancelledError:
            # Task was cancelled, which is normal during shutdown
            logging.debug(f"[{self.config.code}] Task {task.get_name()} was cancelled")
        except Exception as e:
            # Log the exception that was raised by the task
            logging.exception(
                f"[{self.config.code}] Unhandled exception in task {task.get_name()}: {e}"
            )
            # You could potentially restart the task here or notify an admin
            # For now, we'll just make sure it's logged properly

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove known erroneous data points from a DataFrame.

        Uses the temp_outliers list from the location configuration to identify and
        remove specific timestamps that are known to have bad data. This helps to
        improve data quality by filtering out known anomalies.

        Args:
            df: DataFrame with DatetimeIndex to remove outliers from

        Returns:
            DataFrame with outliers removed
        """
        if not self.config.temp_source or not self.config.temp_source.outliers:
            return df

        result_df = df.copy()
        for timestamp in self.config.temp_source.outliers:
            try:
                result_df = result_df.drop(pd.to_datetime(timestamp))
            except KeyError:
                # Skip if the timestamp doesn't exist in the data
                logging.debug(
                    f"[{self.config.code}] Outlier timestamp {timestamp} not found in data"
                )

        return result_df

    async def _fetch_live_temps(self) -> None:
        """Fetch recent air and water temperatures (last 8 days).

        Updates the live_temps instance variable with the most recent temperature
        data from NOAA. Sets the _live_temps_timestamp to track when data was last retrieved.

        Skips fetching if no temperature station is configured.
        Logs the age of the most recent data point.
        Logs warnings if the NOAA API returns an error.
        """
        if not self.config.temp_source:
            logging.info(f"[{self.config.code}] No temperature source configured")
            return

        temp_config = self.config.temp_source

        if isinstance(temp_config, NoaaTempSource):
            if not temp_config.station:
                logging.info(
                    f"[{self.config.code}] No NOAA station configured for temperature"
                )
                return
        else:
            raise TypeError(f"Unsupported temperature source type: {type(temp_config)}")

        logging.info(f"[{self.config.code}] Fetching live temps")
        begin_date = datetime.datetime.today() - datetime.timedelta(days=8)
        end_date = datetime.datetime.today()
        # TODO: Resample to 6min for more consistent data intervals
        try:
            # We already know it's a NoaaTempSource with a valid station at this point
            station_id = self.config.temp_source.station  # type: ignore

            # Fetch air and water temperature data concurrently
            air_temp_task = noaa.NoaaApi.temperature(
                station_id,
                "air_temperature",
                begin_date,
                end_date,
                location_code=self.config.code,
            )
            water_temp_task = noaa.NoaaApi.temperature(
                station_id,
                "water_temperature",
                begin_date,
                end_date,
                location_code=self.config.code,
            )
            # Wait for both requests to complete
            air_temp, water_temp = await asyncio.gather(air_temp_task, water_temp_task)
            # Concatenate the results
            temp_data = pd.concat([air_temp, water_temp], axis=1)

            # Remove outliers using the helper method
            self.live_temps = self._remove_outliers(temp_data)
            # Ensure we always store naive timestamps
            now = utc_now()
            assert now.tzinfo is None, "utc_now() must return naive datetime"
            self._live_temps_timestamp = now

            # Log info about the most recent data point
            last_time = latest_time_value(self.live_temps)
            age_str = "unknown"
            if last_time is not None:
                now = utc_now()  # utc_now returns naive datetime in UTC
                # Ensure both datetimes are naive for subtraction
                if hasattr(last_time, "tzinfo") and last_time.tzinfo is not None:
                    last_time = last_time.replace(tzinfo=None)
                age = now - last_time
                age_str = str(datetime.timedelta(seconds=int(age.total_seconds())))
            logging.info(
                f"[{self.config.code}] Fetched live temps. Last datapoint age: {age_str}"
            )
        except noaa.NoaaApiError as e:
            logging.warning(f"[{self.config.code}] Live temp fetch error: {e}")

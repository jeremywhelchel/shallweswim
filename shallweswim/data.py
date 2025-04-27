"""Data fetching and management for ShallWeSwim application.

This module handles data retrieval from NOAA APIs, data processing,
and provides the necessary data for plotting and presentation.
"""

# Standard library imports
import asyncio
import datetime
import logging
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Local imports
from shallweswim import config as config_lib
from shallweswim import feeds
from shallweswim import plot
from shallweswim import util
from shallweswim.types import (
    CurrentInfo,
    DatasetName,
    LegacyChartInfo,
    TideEntry,
    TideInfo,
)
from shallweswim.util import utc_now

# Data expiration periods
EXPIRATION_PERIODS = {
    # Tidal predictions already cover a wide past/present window
    "tides": datetime.timedelta(hours=24),
    # Current predictions already cover a wide past/present window
    "currents": datetime.timedelta(hours=24),
    # Live temperature readings occur every 6 minutes, and are
    # generally already 5 minutes old when a new reading first appears
    "live_temps": datetime.timedelta(minutes=10),
    # Hourly fetch historic temps + generate charts
    "historic_temps": datetime.timedelta(hours=3),
}


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

        # Dictionary mapping dataset names to their corresponding feeds
        # This is the single source of truth for all feed instances and data
        self._feeds: Dict[DatasetName, Optional[feeds.Feed]] = {
            "tides": self._configure_tides_feed(),
            "currents": self._configure_currents_feed(),
            "live_temps": self._configure_live_temps_feed(),
            "historic_temps": self._configure_historic_temps_feed(),
        }

        # Background update task
        self._update_task: Optional[asyncio.Task[None]] = None

    def _configure_live_temps_feed(self) -> Optional[feeds.Feed]:
        """Configure the live temperature feed.

        Returns:
            Configured feed or None if configuration is not available
        """
        if not hasattr(self.config, "temp_source") or not self.config.temp_source:
            return None

        temp_config = self.config.temp_source
        if not hasattr(temp_config, "station") or not temp_config.station:
            return None

        # Type check to ensure we're passing the right config type
        if not isinstance(temp_config, config_lib.NoaaTempSource):
            return None

        return feeds.NoaaTempFeed(
            location_config=self.config,
            config=temp_config,
            # Start 24 hours ago to get a full day of data
            start=utc_now() - datetime.timedelta(hours=24),
            # End at current time
            end=utc_now(),
            # Use 6-minute interval for live temps
            interval="6-min",
            # Set expiration interval to match our existing settings
            expiration_interval=EXPIRATION_PERIODS["live_temps"],
        )

    def _configure_historic_temps_feed(self) -> Optional[feeds.Feed]:
        """Configure the historical temperature feed.

        Returns:
            Configured feed or None if configuration is not available
        """
        if not hasattr(self.config, "temp_source") or not self.config.temp_source:
            return None

        temp_config = self.config.temp_source
        if not hasattr(temp_config, "station") or not temp_config.station:
            return None

        # Type check to ensure we're passing the right config type
        if not isinstance(temp_config, config_lib.NoaaTempSource):
            return None

        # Get the start year from config or use default 2011
        start_year = 2011
        if hasattr(temp_config, "start_year") and temp_config.start_year:
            start_year = temp_config.start_year

        return feeds.HistoricalTempsFeed(
            location_config=self.config,
            config=temp_config,
            # Use the start year we determined
            start_year=start_year,
            # End at current year
            end_year=utc_now().year,
            # Set expiration interval to match our existing settings
            expiration_interval=EXPIRATION_PERIODS["historic_temps"],
        )

    def _configure_tides_feed(self) -> Optional[feeds.Feed]:
        """Configure the tides feed.

        Returns:
            Configured feed or None if configuration is not available
        """
        if not hasattr(self.config, "tide_source") or not self.config.tide_source:
            return None

        tide_config = self.config.tide_source
        if not hasattr(tide_config, "station") or not tide_config.station:
            return None

        return feeds.NoaaTidesFeed(
            location_config=self.config,
            config=tide_config,
            # Set expiration interval to match our existing settings
            expiration_interval=EXPIRATION_PERIODS["tides"],
        )

    def _configure_currents_feed(self) -> Optional[feeds.Feed]:
        """Configure the currents feed.

        Returns:
            Configured feed or None if configuration is not available
        """
        if (
            not hasattr(self.config, "currents_source")
            or not self.config.currents_source
        ):
            return None

        currents_config = self.config.currents_source
        if not hasattr(currents_config, "stations") or not currents_config.stations:
            return None

        return feeds.MultiStationCurrentsFeed(
            location_config=self.config,
            config=currents_config,
            # Set expiration interval to match our existing settings
            expiration_interval=EXPIRATION_PERIODS["currents"],
        )

    @property
    def ready(self) -> bool:
        """Check if all configured datasets have been fetched and are not expired.

        Returns:
            True if all configured datasets have been fetched and are not expired, False otherwise
        """
        # Only check feeds that have been configured (non-None values in _feeds)
        configured_feeds = [feed for feed in self._feeds.values() if feed is not None]

        # If no feeds are configured, we're technically ready (nothing to wait for)
        if not configured_feeds:
            return True

        # Check if any configured feed is expired
        for feed in configured_feeds:
            if feed.is_expired:
                return False

        # All configured feeds exist and are not expired
        return True

    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until all configured feeds have data available.

        This method waits until all configured feeds have successfully fetched data and are ready to use.
        It's useful for coordinating operations that need feed data to be available before proceeding.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            True if all feeds are ready, False if timeout occurred
        """
        # Only consider feeds that are configured (not None)
        configured_feeds = [feed for feed in self._feeds.values() if feed is not None]

        if not configured_feeds:
            return True  # No feeds to wait for

        try:
            # Use asyncio.wait_for to apply a timeout to the gather operation
            await asyncio.wait_for(
                asyncio.gather(*[feed.wait_until_ready() for feed in configured_feeds]),
                timeout,
            )
            logging.debug(f"[{self.config.code}] All feeds are ready")
            return True
        except asyncio.TimeoutError:
            logging.warning(
                f"[{self.config.code}] Timeout waiting for feeds to be ready"
            )
            return False

    def _expired(self, dataset: DatasetName) -> bool:
        """Check if a dataset has expired and needs to be refreshed.

        Uses the feed's built-in expiration logic for all datasets.

        Args:
            dataset: The dataset to check

        Returns:
            True if the dataset is expired or missing, False otherwise
        """
        # Get the feed for this dataset from our dictionary
        feed = self._feeds.get(dataset)

        # If the feed doesn't exist yet or is expired, the data needs refreshing
        if feed is None:
            return True

        return feed.is_expired

    async def _update_dataset(self, dataset: DatasetName) -> None:
        """Update a specific dataset if it has expired.

        Uses the feed's update method to fetch fresh data.
        Data is accessed directly from the feed's values property when needed.

        Args:
            dataset: The dataset to update
        """
        # Get the feed for this dataset
        feed = self._feeds.get(dataset)

        # If there's no feed configured, we can't update the dataset
        if feed is None:
            logging.debug(f"[{self.config.code}] No feed configured for {dataset}")
            return

        try:
            # Update the feed to get fresh data
            logging.info(f"[{self.config.code}] Fetching {dataset}")
            await feed.update()

            # No need to update separate data cache variables
            # Data is accessed directly from feed.values when needed
        except Exception as e:
            logging.warning(
                f"[{self.config.code}] {dataset.capitalize()} fetch error: {e}"
            )
            # Re-raise to ensure error is not silently ignored
            raise

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
                    # Check and update tides and currents separately
                    if self._expired("tides"):
                        await self._update_dataset("tides")

                    if self._expired("currents"):
                        await self._update_dataset("currents")

                    if self._expired("live_temps"):
                        await self._update_dataset("live_temps")
                        # Only generate plot if we have valid data
                        live_temps_feed = self._feeds.get("live_temps")
                        live_temps_data = (
                            live_temps_feed.values
                            if live_temps_feed is not None
                            else None
                        )
                        if live_temps_data is not None and len(live_temps_data) >= 2:
                            plot.generate_and_save_live_temp_plot(
                                live_temps_data,
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
                        historic_temps_feed = self._feeds.get("historic_temps")
                        historic_temps_data = (
                            historic_temps_feed.values
                            if historic_temps_feed is not None
                            else None
                        )
                        if (
                            historic_temps_data is not None
                            and len(historic_temps_data) >= 10
                        ):
                            plot.generate_and_save_historic_plots(
                                historic_temps_data,
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
        # Get tides data from the feed
        tides_feed = self._feeds.get("tides")
        tides_data = tides_feed.values if tides_feed is not None else None

        if tides_data is None:
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
                tides_data.index.tz is None
            ), "Tide DataFrame should use naive datetimes"

            # When we reset_index, the DatetimeIndex becomes a column named 'time'
            # Convert to pandas Timestamp for proper slicing (to satisfy type checking)
            now_ts = pd.Timestamp(now)
            past_tide_dicts = (
                tides_data[:now_ts].tail(1).reset_index().to_dict(orient="records")
            )
            next_tide_dicts = (
                tides_data[now_ts:].head(2).reset_index().to_dict(orient="records")
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

        # Get tides data from the feed
        tides_feed = self._feeds.get("tides")
        tides_data = tides_feed.values if tides_feed is not None else None
        assert tides_data is not None, "Tide data is required for legacy chart info"

        # Handle the timezone compatibility for DataFrame index lookup
        # Check if the DataFrame has a timezone-naive index
        if tides_data.index.tz is None:
            # If index is naive, we need a naive datetime for slicing
            t_for_lookup = t.replace(tzinfo=None) if t.tzinfo is not None else t
        else:
            # If index has timezone info, ensure it's in the same timezone
            t_for_lookup = t

        row = tides_data.loc[tides_data.index.asof(t_for_lookup)]
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

        # Get currents data from the feed
        currents_feed = self._feeds.get("currents")
        currents_data = currents_feed.values if currents_feed is not None else None
        assert currents_data is not None, "Current data must be loaded first"

        # Create a working copy of the current data for analysis
        df = currents_data.copy()

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
        # Get live temperature data from the feed
        live_temps_feed = self._feeds.get("live_temps")
        live_temps_data = (
            live_temps_feed.values if live_temps_feed is not None else None
        )

        if live_temps_data is None:
            return datetime.time(0), 0.0

        ((time, temp),) = live_temps_data.tail(1)["water_temp"].items()
        return time, temp

    def _handle_task_exception(self, task: asyncio.Task[Any]) -> None:
        """Handle exceptions from asyncio tasks to prevent them from being silently ignored.

        This callback is attached to asyncio tasks and will log any exceptions that occur,
        ensuring they're not silently swallowed by the event loop.

        Args:
            task: The asyncio task that completed (successfully or with an exception)

        Raises:
            Exception: Re-raises any exception from the task to ensure failures are visible
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
            # Re-raise the exception to follow the project principle of failing fast
            raise

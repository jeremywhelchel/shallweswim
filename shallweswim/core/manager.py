"""Data fetching and management for ShallWeSwim application.

This module handles data retrieval from NOAA APIs, data processing,
and provides the necessary data for plotting and presentation.
"""

# Standard library imports
import asyncio
import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# Third-party imports
import pandas as pd

from shallweswim import api_types, feeds

# Local imports
from shallweswim import config as config_lib
from shallweswim.clients.base import BaseApiClient
from shallweswim.core import queries, updater
from shallweswim.types import (
    CurrentInfo,
    LegacyChartInfo,
    TemperatureReading,
    TideInfo,
)
from shallweswim.util import utc_now


# Lazy import wrappers for plot functions to avoid loading matplotlib/seaborn in main process
def _generate_live_temp_plot(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Wrapper that lazily imports plot module for subprocess execution."""
    from shallweswim import plot

    return plot.generate_live_temp_plot(*args, **kwargs)


def _generate_historic_temp_plots(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Wrapper that lazily imports plot module for subprocess execution."""
    from shallweswim import plot

    return plot.generate_historic_temp_plots(*args, **kwargs)


# Constants
# Default year to start historical temperature data collection
DEFAULT_HISTORIC_TEMPS_START_YEAR = 2011

# Hard timeout for plot generation in ProcessPoolExecutor (seconds).
# If a worker hasn't finished after this long, we assume it's hung and
# drop tracking so the location can retry. The orphaned worker process
# cannot be killed (Python limitation) but the pool continues working.
PLOT_HARD_TIMEOUT = 300.0

# Data expiration periods
EXPIRATION_PERIODS: dict[feeds.FeedName, datetime.timedelta] = {
    # Tidal predictions already cover a wide past/present window
    feeds.FEED_TIDES: datetime.timedelta(hours=24),
    # Current predictions already cover a wide past/present window
    feeds.FEED_CURRENTS: datetime.timedelta(hours=24),
    # Live temperature readings occur every 6 minutes, and are
    # generally already 5 minutes old when a new reading first appears
    feeds.FEED_LIVE_TEMPS: datetime.timedelta(minutes=10),
    # Hourly fetch historic temps + generate charts
    feeds.FEED_HISTORIC_TEMPS: datetime.timedelta(hours=3),
}


class LocationDataManager:
    """LocationDataManager for ShallWeSwim application.

    This class manages all the data feeds for one location. It handles fetching, processing,
    and storing data from various NOAA sources, including tides, currents, and temperature
    readings. It maintains data freshness and provides methods to access processed data
    for the web application.
    """

    def __init__(
        self,
        config: config_lib.LocationConfig,
        clients: dict[str, BaseApiClient],
        process_pool: ProcessPoolExecutor,  # For CPU-bound tasks like plotting
    ):
        """Initialize the Data manager for a specific location.

        Args:
            config: Location configuration object.
            clients: Dictionary of initialized API client instances.
            process_pool: A ProcessPoolExecutor for offloading CPU-bound work.
        """
        self.config = config
        self.clients = clients
        self.process_pool = process_pool

        # Dictionary mapping dataset names to their corresponding feeds
        # This is the single source of truth for all feed instances and data
        self._feeds: dict[feeds.FeedName, feeds.Feed | None] = {
            feeds.FEED_TIDES: self._configure_tides_feed(),
            feeds.FEED_CURRENTS: self._configure_currents_feed(),
            feeds.FEED_LIVE_TEMPS: self._configure_live_temps_feed(),
            feeds.FEED_HISTORIC_TEMPS: self._configure_historic_temps_feed(),
        }

        # Background update task
        self._update_task: asyncio.Task[None] | None = None
        self._ready_event = asyncio.Event()

        # In-memory storage for generated plots (eliminates filesystem race condition)
        self._plots: dict[feeds.PlotName, bytes] = {}

        # Track in-flight plot futures to avoid submitting duplicate tasks
        # to the process pool while previous ones are still running.
        # Stores (future, submitted_at) so we can detect hung workers.
        self._pending_plot_futures: dict[
            feeds.FeedName, tuple[asyncio.Future[Any], datetime.datetime]
        ] = {}

        # Timestamp of when each plot was last generated, used to detect
        # when feed data is newer than the current plot.
        self._plot_generated_at: dict[feeds.FeedName, datetime.datetime] = {}

    def _configure_live_temps_feed(self) -> feeds.Feed | None:
        """Configure the live temperature feed.

        Returns:
            Configured feed or None if configuration is not available or disabled

        Raises:
            TypeError: If an unsupported temperature source type is provided
        """
        if not self.config.temp_source:
            return None

        temp_config = self.config.temp_source
        if not temp_config.live_enabled:
            self.log(
                f"Live temperature data disabled for {self.config.code}", logging.INFO
            )
            return None

        # Use the factory function to create the appropriate feed
        try:
            return feeds.create_temp_feed(
                location_config=self.config,
                temp_config=temp_config,
                clients=self.clients,  # Pass the clients dictionary explicitly
                # Start 24 hours ago to get a full day of data
                start=utc_now() - datetime.timedelta(hours=24),
                # End at current time
                end=utc_now(),
                # Use 6-minute interval for live temps
                interval="6-min",
                # Set expiration interval to match our existing settings
                expiration_interval=EXPIRATION_PERIODS[feeds.FEED_LIVE_TEMPS],
            )
        except TypeError as e:
            # Re-raise with more context about what we were trying to do
            raise TypeError(f"Error configuring live temperature feed: {e}") from e

    def _configure_historic_temps_feed(self) -> feeds.Feed | None:
        """Configure the historical temperature feed.

        Returns:
            Configured feed or None if configuration is not available or disabled

        Raises:
            TypeError: If an unsupported temperature source type is provided
        """
        if not self.config.temp_source:
            return None

        temp_config = self.config.temp_source

        # Check if historical temperature data is enabled for this source
        if not temp_config.historic_enabled:
            self.log(
                f"Historical temperature data disabled for {self.config.code}",
                logging.INFO,
            )
            return None

        # Get the start year from config or use default
        start_year = DEFAULT_HISTORIC_TEMPS_START_YEAR
        if temp_config.start_year:
            start_year = temp_config.start_year

        # Get the end year from config or use current year
        end_year = utc_now().year
        if temp_config.end_year:
            end_year = temp_config.end_year

        try:
            # Use HistoricalTempsFeed which internally uses our factory function
            return feeds.HistoricalTempsFeed(
                location_config=self.config,
                feed_config=temp_config,
                # Use the start year we determined
                start_year=start_year,
                # Use the end year we determined
                end_year=end_year,
                # Set expiration interval to match our existing settings
                expiration_interval=EXPIRATION_PERIODS[feeds.FEED_HISTORIC_TEMPS],
                clients=self.clients,  # Pass the clients dict
            )
        except TypeError as e:
            # Re-raise with more context about what we were trying to do
            raise TypeError(
                f"Error configuring historical temperature feed: {e}"
            ) from e

    def _configure_tides_feed(self) -> feeds.Feed | None:
        """Configure the tides feed.

        Returns:
            Configured feed or None if configuration is not available
        """
        if not self.config.tide_source:
            return None

        tide_config = self.config.tide_source
        if not tide_config.station:
            return None

        # Use the factory function to create the appropriate feed
        try:
            return feeds.create_tide_feed(
                location_config=self.config,
                tide_config=tide_config,
                expiration_interval=EXPIRATION_PERIODS[feeds.FEED_TIDES],
            )
        except TypeError as e:
            # Re-raise with more context about what we were trying to do
            raise TypeError(f"Error configuring tide feed: {e}") from e

    def _configure_currents_feed(self) -> feeds.Feed | None:
        """Configure the currents feed.

        Returns:
            Configured feed or None if configuration is not available
        """
        if not self.config.currents_source or not self.config.currents_source:
            return None

        currents_config = self.config.currents_source

        try:
            return feeds.create_current_feed(
                location_config=self.config,
                current_config=currents_config,
                expiration_interval=EXPIRATION_PERIODS[feeds.FEED_CURRENTS],
                clients=self.clients,
            )
        except TypeError as e:
            # Re-raise with more context about what we were trying to do
            raise TypeError(f"Error configuring currents feed: {e}") from e

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

        # All configured feeds must not be expired
        return all(not feed.is_expired for feed in configured_feeds)

    @property
    def has_data(self) -> bool:
        """Check if any configured feed has data (fresh or stale).

        This is a lenient check used for health endpoints - returns True if
        the location can serve any data at all, even if some feeds are expired.

        Returns:
            True if at least one feed has data, False if no data available
        """
        for feed in self._feeds.values():
            if feed is not None and feed._data is not None:
                return True
        return False

    def has_feed_data(self, feed_name: feeds.FeedName) -> bool:
        """Check if a specific feed has data available.

        Use this to check before calling query functions that assert data exists.
        This prevents AssertionError for expected conditions (no data yet).

        Args:
            feed_name: Name of the feed to check (use constants from feeds module)

        Returns:
            True if the feed exists and has data, False otherwise
        """
        feed = self._feeds.get(feed_name)
        return feed is not None and feed._data is not None

    def has_feed(self, feed_name: feeds.FeedName | str) -> bool:
        """Check if a feed name is known for this manager.

        Args:
            feed_name: Name of the feed to check.

        Returns:
            True if the feed name is part of this manager's feed set, even if
            the feed is not configured for this location.
        """
        try:
            normalized_feed_name = feeds.FeedName(feed_name)
        except ValueError:
            return False
        return normalized_feed_name in self._feeds

    def get_feed_values(self, feed_name: feeds.FeedName | str) -> pd.DataFrame:
        """Get feed data by name.

        Args:
            feed_name: Name of the feed to retrieve.

        Returns:
            Feed data as a DataFrame.

        Raises:
            KeyError: If the feed name is unknown.
            DataUnavailableError: If the feed is known but has no data.
        """
        try:
            normalized_feed_name = feeds.FeedName(feed_name)
        except ValueError as e:
            raise KeyError(feed_name) from e

        if normalized_feed_name not in self._feeds:
            raise KeyError(feed_name)

        feed = self._feeds[normalized_feed_name]
        if feed is None or feed._data is None:
            raise queries.DataUnavailableError(
                f"Feed '{normalized_feed_name}' data not available"
            )

        return feed.values

    @property
    def status(self) -> api_types.LocationStatus:
        """Get a Pydantic model with the status of all configured feeds.

        Returns:
            A LocationStatus object containing a dictionary mapping feed names
            to their FeedStatus objects.
        """
        status_dict: dict[str, api_types.FeedStatus] = {}

        # Add status for each configured feed
        for name, feed in self._feeds.items():
            if feed is not None:
                status_dict[name] = feed.status

        return api_types.LocationStatus(feeds=status_dict)

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message with standardized formatting including location code.

        Args:
            message: The message to log
            level: The logging level (default: INFO)
        """
        log_message = f"[{self.config.code}] {message}"
        logging.log(level, log_message)

    async def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait until the initial data update is complete and the manager is ready.

        NOTE: This method's complexity stems from the need to simultaneously monitor
        both the `_ready_event` and the completion/exception state of the background
        `_update_task`. This ensures that if the `_update_task` fails with an
        exception during its initial run (before setting `_ready_event`), this
        method raises that specific exception immediately, rather than waiting for
        the timeout. This aligns with the project's 'fail fast' principle for
        internal errors.

        TODO: Explore potential simplifications or alternative designs for this
        synchronization logic in the future (e.g., using a dedicated error future).

        Monitors both the ready event and the background update task, failing fast
        on any internal errors.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            True if the manager is ready within the timeout.

        Raises:
            Exception: If the background update task raises an exception.
            RuntimeError: If the background update task finishes unexpectedly
                          without setting the ready event.
        """
        if self._ready_event.is_set():
            self.log("Manager already ready.", level=logging.DEBUG)
            return True

        if self._update_task is None:
            raise RuntimeError("Update task has not been started.")

        if self._update_task.done():
            self.log(
                "Update task finished before ready event was set.",
                level=logging.WARNING,
            )
            # Task is already done, check for exceptions
            exc = self._update_task.exception()
            if exc:
                self.log(
                    f"Update task failed with exception: {exc}", level=logging.ERROR
                )
                raise exc  # Re-raise the original exception
            else:
                # Should not happen if ready_event wasn't set, but handle defensively
                raise RuntimeError(
                    "Data update task finished unexpectedly before data was ready."
                )

        # Create waiters for the ready event and the update task completion
        ready_waiter = asyncio.create_task(
            self._ready_event.wait(), name=f"ReadyWait_{self.config.code}"
        )
        # We wait directly on the existing update task future
        update_task_waiter = self._update_task

        waiters = {ready_waiter, update_task_waiter}

        self.log(
            f"Waiting for ready event or update task completion (timeout={timeout}s)...",
            level=logging.DEBUG,
        )
        done, pending = await asyncio.wait(
            waiters,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # --- Cleanup and Result Handling ---

        # Cancel any pending waiters (if one finished, the other might still be pending)
        for task in pending:
            # Don't cancel the main update task, just the waiter we created
            if task is ready_waiter:
                self.log(
                    f"Cancelling pending ready_waiter task: {task.get_name()}",
                    level=logging.DEBUG,
                )
                task.cancel()
                # Suppress CancelledError if cancellation is successful
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if not done:
            # This means timeout occurred before either task completed
            self.log(
                f"Timed out after {timeout}s waiting for ready signal or task completion.",
                level=logging.WARNING,
            )
            return False

        # --- Check which task completed ---

        if update_task_waiter in done:
            # The main update task finished first. Check for exceptions.
            self.log(
                "Update task finished while waiting for ready signal.",
                level=logging.WARNING,
            )
            exc = update_task_waiter.exception()
            if exc:
                self.log(
                    f"Update task failed with exception: {exc}", level=logging.ERROR
                )
                raise exc  # Re-raise the original exception
            else:
                # Task finished without error, but ready event wasn't set.
                raise RuntimeError(
                    "Data update task finished unexpectedly before data was ready."
                )

        if ready_waiter in done:
            # The ready event was set successfully
            if self._ready_event.is_set():
                self.log("Ready event received.", level=logging.DEBUG)
                return True
            else:
                # Should not be possible if ready_waiter completed normally
                self.log(
                    "Ready waiter finished but event not set!", level=logging.ERROR
                )
                raise RuntimeError(
                    "Internal error: Ready waiter finished but event not set."
                )

        # Should be unreachable, but handle defensively
        self.log("Unexpected state reached in wait_until_ready.", level=logging.ERROR)
        return False

    def _get_feed_data(
        self, feed_name: feeds.FeedName, min_rows: int = 0
    ) -> Any | None:
        """Get data from a feed if it exists and meets minimum size.

        Returns:
            The feed's DataFrame values, or None if unavailable/too small.
        """
        try:
            values = self.get_feed_values(feed_name)
        except (KeyError, queries.DataUnavailableError):
            return None
        if values is not None and len(values) >= min_rows:
            return values
        return None

    def _needs_plot(
        self, feed_name: feeds.FeedName, plot_keys: list[feeds.PlotName]
    ) -> bool:
        """Check if a plot needs (re)generation.

        A plot needs generation if:
        - It's not already in-flight in the process pool
        - AND either: any plot key is missing, OR feed data is newer than the plot

        Existing plots continue to be served while regenerating.
        """
        pending = self._pending_plot_futures.get(feed_name)
        if pending is not None:
            return False

        # Missing plots always need generation
        if any(self._plots.get(key) is None for key in plot_keys):
            return True

        # Regenerate if feed data is newer than the plot
        feed = self._feeds.get(feed_name)
        if feed is None or feed._fetch_timestamp is None:
            return False
        generated_at = self._plot_generated_at.get(feed_name)
        return generated_at is None or feed._fetch_timestamp > generated_at

    def _generate_plots(self, loop: asyncio.AbstractEventLoop) -> None:
        """Submit and collect plot generation tasks (non-blocking).

        Plot generation is decoupled from feed expiration: plots are regenerated
        when missing (first run or previous failure) or when feed data is newer
        than the current plot. Existing plots continue to be served while stale.

        This method never awaits process pool results. It checks for completed
        futures from previous iterations and submits new work if needed.
        ProcessPoolExecutor futures cannot be cancelled (Python limitation),
        so we never use asyncio.wait_for/gather on them — instead we poll
        future.done() each loop tick.
        """
        self._collect_completed_plots()
        self._submit_plot_tasks(loop)

    def _collect_completed_plots(self) -> None:
        """Harvest results from completed plot futures.

        For each pending future:
        - Done with result: store plot bytes, update timestamp, remove from tracking
        - Done with exception: log error, remove from tracking
        - Still running past PLOT_HARD_TIMEOUT: log error, remove from tracking
          (orphaned worker cannot be killed but location can retry)
        - Still running within timeout: leave in tracking (blocks resubmission)
        """
        now = utc_now()
        completed: list[feeds.FeedName] = []

        for feed_name, (future, submitted_at) in self._pending_plot_futures.items():
            if future.done():
                completed.append(feed_name)
                try:
                    result = future.result()
                except Exception as e:
                    self.log(
                        f"Plot generation failed for {feed_name}: {e}",
                        level=logging.ERROR,
                    )
                    continue

                # Store results based on feed type
                if feed_name == feeds.FEED_LIVE_TEMPS:
                    self._plots[feeds.PLOT_LIVE_TEMPS] = result
                elif feed_name == feeds.FEED_HISTORIC_TEMPS:
                    self._plots[feeds.PLOT_HISTORIC_TEMPS_2MO] = result["2mo"]
                    self._plots[feeds.PLOT_HISTORIC_TEMPS_12MO] = result["12mo"]

                self._plot_generated_at[feed_name] = now
                self.log(f"Plot generation completed for {feed_name}.")

            elif (now - submitted_at).total_seconds() > PLOT_HARD_TIMEOUT:
                completed.append(feed_name)
                self.log(
                    f"Plot worker for {feed_name} stuck for "
                    f">{PLOT_HARD_TIMEOUT}s, abandoning",
                    level=logging.ERROR,
                )

        for feed_name in completed:
            del self._pending_plot_futures[feed_name]

    def _submit_plot_tasks(self, loop: asyncio.AbstractEventLoop) -> None:
        """Submit new plot generation tasks to the process pool if needed.

        Checks each plot type and submits work for any that need regeneration.
        Returns immediately — results are collected by _collect_completed_plots()
        on the next loop iteration.
        """
        need_live = self._needs_plot(feeds.FEED_LIVE_TEMPS, [feeds.PLOT_LIVE_TEMPS])
        need_historic = self._needs_plot(
            feeds.FEED_HISTORIC_TEMPS,
            [feeds.PLOT_HISTORIC_TEMPS_2MO, feeds.PLOT_HISTORIC_TEMPS_12MO],
        )

        if not need_live and not need_historic:
            return

        temp_source_name = (
            self.config.temp_source.name if self.config.temp_source else None
        )
        now = utc_now()

        if need_live:
            data = self._get_feed_data(feeds.FEED_LIVE_TEMPS, min_rows=2)
            if data is not None:
                self.log("Submitting live temps plot generation.")
                future = loop.run_in_executor(
                    self.process_pool,
                    _generate_live_temp_plot,
                    data,
                    self.config.code,
                    temp_source_name,
                )
                self._pending_plot_futures[feeds.FEED_LIVE_TEMPS] = (future, now)

        if need_historic:
            data = self._get_feed_data(feeds.FEED_HISTORIC_TEMPS, min_rows=10)
            if data is not None:
                self.log("Submitting historic temps plot generation.")
                future = loop.run_in_executor(
                    self.process_pool,
                    _generate_historic_temp_plots,
                    data,
                    self.config.code,
                    temp_source_name,
                )
                self._pending_plot_futures[feeds.FEED_HISTORIC_TEMPS] = (future, now)

    async def __update_loop(self) -> None:
        """Background task that refreshes feed data and regenerates plots.

        Runs continuously, checking each feed's expiration on every iteration.
        Feed updates respect the feed's own scheduling semantics (see feeds.py).
        Plot generation is decoupled — see _generate_plots().
        """
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    # Update each feed (feed.update() is a no-op if not expired)
                    for feed_name in (
                        feeds.FEED_TIDES,
                        feeds.FEED_CURRENTS,
                        feeds.FEED_LIVE_TEMPS,
                        feeds.FEED_HISTORIC_TEMPS,
                    ):
                        await updater.update_dataset(
                            self._feeds, self.clients, feed_name
                        )

                    self._generate_plots(loop)

                except Exception as e:
                    self.log(f"Error in data update loop: {e}", level=logging.ERROR)

                # Signal readiness after first iteration completes
                if not self._ready_event.is_set():
                    self._ready_event.set()

                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise

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

        self.log("Starting data fetch task")
        self._update_task = asyncio.create_task(self.__update_loop())
        self._update_task.set_name(task_name)

        # Add exception handling to the task
        self._update_task.add_done_callback(self._handle_task_exception)

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the background data fetching process.

        This cancels the background task if it's running and waits for it to finish.
        It's important to call this method when the LocationDataManager
        is no longer needed to prevent task leaks.

        Args:
            timeout: Max seconds to wait for task cancellation. If exceeded,
                    the task is abandoned (thread may still be running but
                    won't block shutdown).
        """
        if self._update_task and not self._update_task.done():
            self.log("Stopping data fetch task")
            self._update_task.cancel()
            try:
                # Wait for the task to finish cancelling, but don't block forever.
                # If the task is stuck in asyncio.to_thread(), the thread can't
                # be interrupted - it will complete eventually but we shouldn't
                # block teardown waiting for it.
                await asyncio.wait_for(self._update_task, timeout=timeout)
            except asyncio.CancelledError:
                # This is expected, signifies successful cancellation
                self.log("Data fetch task successfully cancelled")
            except TimeoutError:
                # Task didn't finish in time - likely stuck in a blocking thread
                self.log(
                    f"Data fetch task did not finish within {timeout}s, abandoning",
                    level=logging.WARNING,
                )
            except Exception as e:
                # Log any other unexpected error during task finalization
                self.log(
                    f"Error awaiting cancelled data fetch task: {e}",
                    level=logging.ERROR,
                )
                # Optionally re-raise depending on desired shutdown behavior
                # raise

    def get_current_temperature(self) -> TemperatureReading:
        """Get the most recent water temperature reading.

        Returns:
            A TemperatureReading object with timestamp and temperature

        Raises:
            AssertionError: If no temperature data is available
        """
        return queries.get_current_temperature(self._feeds)

    def get_current_tide_info(self) -> TideInfo:
        """Get the previous tide and upcoming tides relative to current time.

        Returns:
            A TideInfo object with past and next tide entries

        Raises:
            AssertionError: If tide data feed is missing
        """
        return queries.get_current_tide_info(self._feeds, self.config)

    def get_chart_info(self, t: datetime.datetime | None = None) -> LegacyChartInfo:
        """Generate chart information based on tide data for the specified time.

        Args:
            t: The time to generate chart info for, defaults to current time

        Returns:
            A LegacyChartInfo object with chart filename and metadata

        Raises:
            AssertionError: If tide data is not available
        """
        return queries.get_chart_info(self._feeds, self.config, t)

    def get_current_flow_info(self) -> CurrentInfo:
        """Get the latest observed current information.

        Returns:
            A CurrentInfo object with the most recent current observation

        Raises:
            AssertionError: If current data is not available
        """
        return queries.get_current_flow_info(self._feeds)

    def get_plot(self, plot_type: feeds.PlotName) -> bytes | None:
        """Get a generated plot by type.

        Args:
            plot_type: The plot to retrieve (e.g., PlotName.LIVE_TEMPS)

        Returns:
            The plot as SVG bytes, or None if not yet generated
        """
        return self._plots.get(plot_type)

    def predict_flow_at_time(self, t: datetime.datetime | None = None) -> CurrentInfo:
        """Predict tidal current conditions for a specific time.

        Args:
            t: Time to predict current for, defaults to current time

        Returns:
            A CurrentInfo object with current prediction

        Raises:
            AssertionError: If current data is not available
        """
        return queries.predict_flow_at_time(self._feeds, self.config, t)

    def _handle_task_exception(self, task: asyncio.Task[Any]) -> None:
        """Handle exceptions from asyncio tasks with appropriate logging levels."""
        updater.handle_task_exception(task, self.log)

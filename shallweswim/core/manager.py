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
from shallweswim import api_types, feeds, plot

# Local imports
from shallweswim import config as config_lib
from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,
    StationUnavailableError,
)
from shallweswim.core import queries
from shallweswim.types import (
    CurrentInfo,
    LegacyChartInfo,
    TemperatureReading,
    TideInfo,
)
from shallweswim.util import utc_now

# Constants
# Default year to start historical temperature data collection
DEFAULT_HISTORIC_TEMPS_START_YEAR = 2011

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
        self._feeds: dict[str, feeds.Feed | None] = {
            "tides": self._configure_tides_feed(),
            "currents": self._configure_currents_feed(),
            "live_temps": self._configure_live_temps_feed(),
            "historic_temps": self._configure_historic_temps_feed(),
        }

        # Background update task
        self._update_task: asyncio.Task[None] | None = None
        self._ready_event = asyncio.Event()

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
                expiration_interval=EXPIRATION_PERIODS["live_temps"],
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
                expiration_interval=EXPIRATION_PERIODS["historic_temps"],
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
                expiration_interval=EXPIRATION_PERIODS["tides"],
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
                expiration_interval=EXPIRATION_PERIODS["currents"],
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

    def _expired(self, dataset: str) -> bool:
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

    async def _update_dataset(self, dataset: str) -> None:
        """Update a specific dataset if it has expired.

        Args:
            dataset: The dataset to update
        """
        # Check if the dataset is expired
        if not self._expired(dataset):
            return

        # Update the dataset
        feed = self._feeds[dataset]
        if feed is not None:
            await feed.update(clients=self.clients)

    async def __update_loop(self) -> None:
        """Background asyncio task that continuously updates data.

        This runs as an asyncio task and periodically checks if datasets
        have expired. If so, it fetches new data and generates updated plots
        using the shared process pool for CPU-bound plotting tasks.

        This method catches and logs exceptions to prevent silent failures in the event loop,
        but will re-raise them to ensure they're not silently ignored.
        """
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    plot_tasks = []  # Tasks for parallel execution

                    # Check and update tides and currents separately
                    if self._expired("tides"):
                        await self._update_dataset("tides")
                    if self._expired("currents"):
                        await self._update_dataset("currents")

                    if self._expired("live_temps"):
                        await self._update_dataset("live_temps")
                        live_temps_feed = self._feeds.get("live_temps")
                        live_temps_data = (
                            live_temps_feed.values
                            if live_temps_feed is not None
                            else None
                        )
                        temp_source_name = (
                            self.config.temp_source.name
                            if self.config.temp_source
                            else None
                        )
                        if live_temps_data is not None and len(live_temps_data) >= 2:
                            self.log("Submitting live temps plot generation.")
                            task = loop.run_in_executor(
                                self.process_pool,
                                plot.generate_and_save_live_temp_plot,
                                live_temps_data,
                                self.config.code,
                                temp_source_name,
                            )
                            plot_tasks.append(task)

                    if self._expired("historic_temps"):
                        await self._update_dataset("historic_temps")
                        historic_temps_feed = self._feeds.get("historic_temps")
                        historic_temps_data = (
                            historic_temps_feed.values
                            if historic_temps_feed is not None
                            else None
                        )
                        temp_source_name = (
                            self.config.temp_source.name
                            if self.config.temp_source
                            else None
                        )
                        if (
                            historic_temps_data is not None
                            and len(historic_temps_data) >= 10
                        ):
                            self.log("Submitting historic temps plot generation.")
                            task = loop.run_in_executor(
                                self.process_pool,
                                plot.generate_and_save_historic_plots,
                                historic_temps_data,
                                self.config.code,
                                temp_source_name,
                            )
                            plot_tasks.append(task)

                    # Wait for any submitted plotting tasks to complete in parallel
                    if plot_tasks:
                        self.log(f"Waiting for {len(plot_tasks)} plot task(s)...")
                        try:
                            await asyncio.gather(*plot_tasks)
                            self.log(f"Completed {len(plot_tasks)} plot task(s).")
                        except Exception as e:
                            self.log(
                                f"Error during parallel plot generation: {e}",
                                level=logging.ERROR,
                            )
                            # Decide whether to raise e or just log

                except Exception as e:
                    # Log the exception but don't swallow it - let it propagate
                    self.log(f"Error in data update loop: {e}", level=logging.ERROR)
                    # Re-raise the exception to ensure error is not silently ignored
                    raise

                # Set the ready event after the first successful update
                if not self._ready_event.is_set():
                    self._ready_event.set()

                # Sleep for 1 second before the next update check
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # This is expected when the task is cancelled, so we let it propagate
            raise
        except Exception as e:
            # Log any unexpected exceptions at the outer level as well
            self.log(f"Fatal error in data update loop: {e}", level=logging.ERROR)
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

        self.log("Starting data fetch task")
        self._update_task = asyncio.create_task(self.__update_loop())
        self._update_task.set_name(task_name)

        # Add exception handling to the task
        self._update_task.add_done_callback(self._handle_task_exception)

    async def stop(self) -> None:
        """Stop the background data fetching process.

        This cancels the background task if it's running and waits for it to finish.
        It's important to call this method when the LocationDataManager
        is no longer needed to prevent task leaks.
        """
        if self._update_task and not self._update_task.done():
            self.log("Stopping data fetch task")
            self._update_task.cancel()
            try:
                # Wait for the task to actually finish cancelling
                await self._update_task
            except asyncio.CancelledError:
                # This is expected, signifies successful cancellation
                self.log("Data fetch task successfully cancelled")
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
        """Handle exceptions from asyncio tasks with appropriate logging levels.

        This callback is attached to asyncio tasks and logs exceptions at the
        appropriate level based on whether they're expected operational conditions
        or unexpected errors requiring investigation.

        Log levels:
        - WARNING: Expected conditions (station unavailable) - does NOT trigger alerts
        - ERROR: Unexpected conditions (API changes, bugs) - triggers GCP alerts

        Note: This is a done_callback. Re-raising here doesn't crash the service -
        it just triggers asyncio's exception handler. The feed stays stale and
        retries on the next interval regardless.

        Args:
            task: The asyncio task that completed (successfully or with an exception)
        """
        try:
            # If the task raised an exception, this will re-raise it
            task.result()
        except asyncio.CancelledError:
            # Task was cancelled, which is normal during shutdown
            self.log(f"Task {task.get_name()} was cancelled", level=logging.DEBUG)
        except StationUnavailableError as e:
            # Expected operational condition - station has no data
            # Log as WARNING (does NOT trigger GCP alerts)
            # Feed stays stale, will retry on next interval
            self.log(f"Station unavailable: {e}", level=logging.WARNING)
        except BaseClientError as e:
            # Unexpected upstream issue - potential API change or parsing error
            # Log as ERROR (triggers GCP alerts)
            # Feed stays stale, will retry on next interval
            self.log(f"Upstream error: {e}", level=logging.ERROR)
            raise  # Re-raise for extra visibility in asyncio's exception handler
        except Exception as e:
            # Internal error - bug in our code
            # Log as ERROR (triggers GCP alerts)
            self.log(
                f"Unhandled exception in task {task.get_name()}: {e}",
                level=logging.ERROR,
            )
            raise  # Re-raise for extra visibility in asyncio's exception handler

"""API handlers for ShallWeSwim application.

This module contains FastAPI route handlers for the API endpoints and data management.
"""

# Standard library imports
import asyncio
import io
import logging
import warnings

# Third-party imports
import aiohttp
import fastapi
from fastapi import HTTPException

# Local imports
from shallweswim import config as config_lib
from shallweswim import data as data_lib
from shallweswim import plot, types, util
from shallweswim.api_types import (
    CurrentInfo,
    CurrentsResponse,
    LegacyChartInfo,
    LocationConditions,
    LocationInfo,
    LocationStatus,
    TemperatureInfo,
    TideEntry,
    TideInfo,
)
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi

# Data store for location data will be stored in app.state.data_managers


# Filter the specific Pydantic serialization warning globally for production
# Note: Tests might handle this separately (e.g., via pytest markers/config)
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
    # Optionally target the specific module if needed, but keeping broad for now
    # module="pydantic.type_adapter"
)


async def initialize_location_data(
    location_codes: list[str],
    app: fastapi.FastAPI,
    wait_for_data: bool = False,
    timeout: float | None = 30.0,
) -> dict[str, data_lib.LocationDataManager]:
    """Initialize data for the specified locations.

    This function handles initialization of LocationDataManager objects for the specified locations.
    It can be used by both the main application and tests.

    Args:
        location_codes: List of location codes to initialize
        app: FastAPI application instance
        data_dict: Optional existing data dictionary to populate (creates new if None)
        wait_for_data: Whether to wait for data to be loaded before returning
        timeout: Maximum time in seconds to wait for data to be ready (None for no timeout)

    Returns:
        Dictionary mapping location codes to initialized LocationDataManager objects

    Raises:
        AssertionError: If a location's configuration cannot be found or if required data isn't loaded
    """
    # Retrieve the shared session from app state
    assert hasattr(app.state, "http_session"), "HTTP session not found in app state"
    session: aiohttp.ClientSession = app.state.http_session

    # Retrieve the process pool from app state
    assert hasattr(app.state, "process_pool"), "Process pool not found in app state"
    process_pool = app.state.process_pool

    # Create API client instances using the shared session
    api_clients: dict[str, BaseApiClient] = {
        "coops": CoopsApi(session=session),
        "nwis": NwisApi(session=session),
        "ndbc": NdbcApi(session=session),
    }

    # Initialize app.state.data_managers if it doesn't exist yet
    if not hasattr(app.state, "data_managers"):
        # Create an empty dictionary that will be populated with LocationDataManager objects
        app.state.data_managers = {}

    # Initialize each location
    for code in location_codes:
        # Get location config
        cfg = config_lib.get(code)
        assert cfg is not None, f"Config for location '{code}' not found"

        # Initialize data for this location, passing clients and process pool
        app.state.data_managers[code] = data_lib.LocationDataManager(
            cfg, clients=api_clients, process_pool=process_pool
        )
        app.state.data_managers[code].start()

    # Optionally wait for data to be fully loaded
    if wait_for_data:
        logging.info(f"Waiting for data to be loaded (timeout: {timeout}s)")

        # Create tasks for waiting on each location's data
        wait_tasks = []
        for code in location_codes:
            logging.info(f"Waiting for {code} data to load...")
            loc_data = app.state.data_managers[code]
            wait_tasks.append(loc_data.wait_until_ready(timeout=timeout))

        # Wait for all locations to be ready concurrently
        # This will raise an exception immediately if any task fails
        await asyncio.gather(*wait_tasks)

    # Create a properly typed dictionary to return
    result: dict[str, data_lib.LocationDataManager] = app.state.data_managers
    return result


def register_routes(app: fastapi.FastAPI) -> None:
    """Register API routes with the FastAPI application.

    Args:
        app: The FastAPI application
    """

    # Helper function to validate location
    def validate_location(loc: str) -> config_lib.LocationConfig:
        cfg = config_lib.get(loc)
        if not cfg:
            logging.warning(f"[{loc}] Bad location request")
            raise HTTPException(status_code=404, detail=f"Location '{loc}' not found")
        return cfg

    @app.get("/api/{location}/conditions", response_model=LocationConditions)
    async def location_conditions(location: str) -> LocationConditions:
        """API endpoint that returns tide and temperature data for a specific location.

        Args:
            location: Location code (e.g., 'nyc')

        Returns:
            JSON response with tide and temperature information

        Raises:
            HTTPException: If the location is not configured
        """
        logging.info(f"[{location}] Processing conditions request")
        cfg = validate_location(location)

        # Get the data manager for this location
        data_manager = app.state.data_managers[location]

        # Check if location has data before attempting to serve
        # Use has_data (not ready) to serve stale data during brief refresh windows
        # Background updater handles freshness - user-facing endpoints serve any available data
        if not data_manager.has_data:
            logging.warning(f"[{location}] No data available for conditions request")
            raise HTTPException(
                status_code=503,
                detail=f"{cfg.name} data temporarily unavailable",
            )

        # Create location info
        location_info = LocationInfo(
            code=location, name=cfg.name, swim_location=cfg.swim_location
        )

        # Initialize temperature and tides as None
        temperature_info = None
        tides_info = None
        current_info = None

        # Prepare temperature information if available
        if cfg.temp_source is not None and cfg.temp_source.live_enabled:
            # Directly attempt to get the temperature reading. If it fails (returns None),
            # it will raise a ValueError, aligning with fail-fast.
            temp_reading = data_manager.get_current_temperature()

            # cfg.temp_source is guaranteed to exist and have a name due to the outer check.
            # If not, it's an internal error and AttributeError is appropriate.
            temperature_info = TemperatureInfo(
                timestamp=temp_reading.timestamp.isoformat(),
                water_temp=temp_reading.temperature,
                units="F",
                station_name=cfg.temp_source.name,
            )

        # Add tide data only if the location has a tide source
        if cfg.tide_source:
            # Get tide information
            tide_info = data_manager.get_current_tide_info()

            # Create Pydantic model instances
            past_tides = [
                TideEntry(
                    time=tide.time.isoformat(),
                    type=tide.type,
                    prediction=tide.prediction,
                )
                for tide in tide_info.past
            ]

            next_tides = [
                TideEntry(
                    time=tide.time.isoformat(),
                    type=tide.type,
                    prediction=tide.prediction,
                )
                for tide in tide_info.next
            ]

            tides_info = TideInfo(past=past_tides, next=next_tides)

        # Fetch Current Data (if configured)
        if cfg.currents_source:
            # Use the source_type to determine which method to call
            match cfg.currents_source.source_type:
                case types.DataSourceType.PREDICTION:
                    # For prediction sources (like NOAA CO-OPS), use predict_flow_at_time
                    current_info_internal = data_manager.predict_flow_at_time()
                case types.DataSourceType.OBSERVATION:
                    # For observation sources (like USGS NWIS), use get_current_flow_info
                    current_info_internal = data_manager.get_current_flow_info()
                case _:
                    # If source_type is not recognized, raise an error
                    raise ValueError(
                        f"Unknown current source type: {cfg.currents_source.source_type}"
                    )

            current_info = CurrentInfo(
                timestamp=current_info_internal.timestamp.isoformat(),
                direction=current_info_internal.direction,
                magnitude=current_info_internal.magnitude,
                magnitude_pct=current_info_internal.magnitude_pct,
                state_description=current_info_internal.state_description,
                source_type=current_info_internal.source_type,
            )

        # Return structured response using Pydantic models
        return LocationConditions(
            location=location_info,
            temperature=temperature_info,
            tides=tides_info,
            current=current_info,
        )

    @app.get("/api/{location}/current_tide_plot")
    async def current_tide_plot(
        location: str, shift: int = 0
    ) -> fastapi.responses.Response:
        """Generate and serve a tide and current plot for the specified location.

        Args:
            location: Location code (e.g., "nyc", "san")
            shift: Time shift in minutes from current time

        Returns:
            SVG image response with tide and current visualization
        """
        logging.info(
            f"[{location}] Processing current tide plot request with shift={shift}"
        )
        # Get location config to access the timezone
        cfg = validate_location(location)

        # Calculate effective time with shift relative to the location's timezone
        ts = util.effective_time(cfg.timezone, shift_minutes=shift)

        # Generate the tide/current plot
        try:
            # Get data directly from feeds - will raise AttributeError if feed is None
            # which will be caught by the try-except block
            data_manager = app.state.data_managers[location]
            tides_data = data_manager._feeds.get("tides").values
            currents_data = data_manager._feeds.get("currents").values

            # Offload plotting to the process pool
            pool = app.state.process_pool
            loop = asyncio.get_running_loop()
            fig = await loop.run_in_executor(
                pool,
                plot.create_tide_current_plot,  # Function to run
                tides_data,  # Argument 1
                currents_data,  # Argument 2
                ts,  # Argument 3
                cfg,  # Argument 4
            )
        except AssertionError as e:
            # The function now raises assertions instead of returning None
            raise HTTPException(status_code=404, detail=str(e)) from e

        # Convert figure to SVG in a StringIO buffer
        svg_io = io.StringIO()
        fig.savefig(svg_io, format="svg", bbox_inches="tight", transparent=False)
        svg_io.seek(0)

        return fastapi.responses.Response(
            content=svg_io.getvalue(), media_type="image/svg+xml"
        )

    @app.get("/api/healthy", status_code=200)
    async def healthy_status() -> bool:
        """API endpoint for service health check (used by Cloud Run).

        Returns 200 if at least one location can serve data (fresh or stale).
        Returns 503 only if NO location has any data available.

        This lenient check ensures single station outages don't mark the entire
        service unhealthy. For detailed per-feed health status, use /api/status.

        Returns:
            True if service can serve at least one location
            Status code 200 if healthy, 503 if not healthy
        """
        logging.info("[api] Processing health status request")

        # Check if data managers exist and are initialized
        if not app.state.data_managers:
            logging.warning("[api] No locations configured")
            raise HTTPException(
                status_code=503, detail="Service not healthy - no locations configured"
            )

        # Check if at least one location has data
        locations_with_data = []
        locations_without_data = []

        for loc_code, loc_data in app.state.data_managers.items():
            if not loc_data:
                logging.warning(f"[{loc_code}] Location not in data dictionary")
                raise HTTPException(
                    status_code=503,
                    detail="Service not healthy - location data missing",
                )

            if loc_data.has_data:
                locations_with_data.append(loc_code)
            else:
                locations_without_data.append(loc_code)

        # Log status for visibility
        if locations_without_data:
            logging.info(
                f"[/api/healthy] Locations without data: {locations_without_data}"
            )
        if locations_with_data:
            logging.info(f"[/api/healthy] Locations with data: {locations_with_data}")

        # Healthy if at least one location can serve data
        if locations_with_data:
            logging.info(
                f"[api] Service healthy - {len(locations_with_data)} location(s) have data"
            )
            return True
        else:
            logging.warning(
                "[/api/healthy] No location has data available. Raising 503."
            )
            raise HTTPException(
                status_code=503,
                detail="Service not healthy - no location has data",
            )

    @app.get("/api/status", response_model=dict[str, LocationStatus])
    async def all_locations_status() -> dict[str, LocationStatus]:
        """API endpoint that returns status information for all configured locations.

        Returns:
            Dictionary mapping location codes to their status dictionaries

        Raises:
            HTTPException: If no locations are configured
        """
        logging.info("[api] Processing all locations status request")

        # Check if data managers exist and are initialized
        # This will raise AttributeError if app.state.data_managers doesn't exist
        if not app.state.data_managers:
            logging.warning("[api] No locations configured")
            raise HTTPException(status_code=404, detail="No locations configured")

        # Return status for each location
        status_dict = {
            code: manager.status for code, manager in app.state.data_managers.items()
        }
        return status_dict

    @app.get("/api/{location}/status", response_model=LocationStatus)
    async def location_status(location: str) -> LocationStatus:
        """API endpoint that returns status information for a specific location.

        Args:
            location: Location code (e.g., 'nyc')

        Returns:
            Status dictionary for the specified location

        Raises:
            HTTPException: If the location is not configured
        """
        logging.info(f"[{location}] Processing status request")

        # Validate location exists
        validate_location(location)

        # Check if location data exists
        if (
            location not in app.state.data_managers
            or not app.state.data_managers[location]
        ):
            logging.warning(f"[{location}] Location not in data dictionary")
            raise HTTPException(
                status_code=404, detail=f"Location '{location}' not found in data"
            )

        # Return the status dictionary for this location
        status_obj: LocationStatus = app.state.data_managers[location].status
        return status_obj

    @app.get("/api/{location}/currents", response_model=CurrentsResponse)
    async def location_currents(location: str, shift: int = 0) -> CurrentsResponse:
        """API endpoint that returns current predictions for a specific location.

        Args:
            location: Location code (e.g., 'nyc')
            shift: Time shift in minutes from current time (optional)

        Returns:
            CurrentsResponse object with current prediction details

        Raises:
            HTTPException: If the location is not configured or doesn't support currents
        """
        logging.info(f"[{location}] Processing currents request with shift={shift}")
        # Validate location exists
        cfg = validate_location(location)

        # Check if this location supports current predictions
        if not cfg.currents_source:
            raise HTTPException(
                status_code=404,
                detail=f"Location '{location}' does not support current predictions",
            )

        # Only NYC is fully supported for current predictions at this time
        if location != "nyc":
            raise HTTPException(
                status_code=501,
                detail=f"Current predictions for '{location}' are not fully implemented yet",
            )

        # Calculate effective time with shift relative to the location's timezone
        ts = util.effective_time(cfg.timezone, shift_minutes=shift)

        try:
            # Get current prediction information
            current_info = app.state.data_managers[location].predict_flow_at_time(ts)
        except ValueError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

        # Get legacy chart information
        chart_info = app.state.data_managers[location].get_chart_info(ts)

        # Get fwd/back shift values for navigation
        fwd = min(shift + 60, util.MAX_SHIFT_LIMIT)
        back = max(shift - 60, util.MIN_SHIFT_LIMIT)

        # Get current chart filename
        current_chart_filename = plot.get_current_chart_filename(
            current_info.direction.value,
            plot.bin_magnitude(current_info.magnitude_pct),
            location_code=location,
        )

        # Format current_info data for the API response
        current_prediction = CurrentInfo(
            timestamp=ts.isoformat(),
            direction=current_info.direction,
            magnitude=round(current_info.magnitude, 1),
            magnitude_pct=current_info.magnitude_pct,
            state_description=current_info.state_description,
            source_type=current_info.source_type,
        )

        # Format legacy chart data for the API response
        legacy_chart = LegacyChartInfo(
            hours_since_last_tide=round(chart_info.hours_since_last_tide, 1),
            last_tide_type=chart_info.last_tide_type,
            chart_filename=chart_info.chart_filename,
            map_title=chart_info.map_title,
        )

        # Return structured response
        return CurrentsResponse(
            location=LocationInfo(
                code=location, name=cfg.name, swim_location=cfg.swim_location
            ),
            timestamp=ts.isoformat(),
            current=current_prediction,
            legacy_chart=legacy_chart,
            current_chart_filename=current_chart_filename,
            navigation={
                "shift": shift,
                "next_hour": fwd,
                "prev_hour": back,
                "current_api_url": f"/api/{location}/currents",
                "plot_url": f"/api/{location}/current_tide_plot?shift={shift}",
            },
        )

    @app.get(
        "/api/{loc}/data/{feed_name}",
        # TODO: Re-enable response_model=pa_typing.DataFrame[TimeSeriesDataModel]
        # Removed due to FastAPI ResponseValidationError when validating specific
        # feed DataFrames (e.g., WaterTempDataModel) against the generic TimeSeriesDataModel.
        # Internal validation happens in feed.values anyway.
    )
    async def get_feed_data(loc: str, feed_name: str):  # type: ignore[no-untyped-def]
        # Originally returned pd.DataFrame, now dict for consistent serialization
        """Retrieve the timeseries data for a specific *feed* at a given *location*.

        Args:
            loc: The location code (e.g., 'nyc').
            feed_name: The data feed name (e.g., 'tides', 'live_temps', 'wind').

        Returns:
            A DataFrame representing the validated timeseries data for the feed.

        Raises:
            HTTPException(404): If the location or feed is not found or data is unavailable.
        """
        logging.info(f"Received request for feed '{feed_name}' at location '{loc}'")
        try:
            # Check if location exists
            if loc not in app.state.data_managers:
                logging.warning(
                    f"Location '{loc}' not found for feed '{feed_name}' request."
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Location '{loc}' not found or data not loaded",
                )
            location_data_manager = app.state.data_managers[loc]

            # Check if feed exists for the location
            if feed_name not in location_data_manager._feeds:
                logging.warning(f"Feed '{feed_name}' not found for location '{loc}'.")
                raise HTTPException(
                    status_code=404,
                    detail=f"Feed '{feed_name}' not found for location '{loc}'.",
                )
            feed = location_data_manager._feeds[feed_name]

            # Attempt to access feed data, catching other potential errors
            df = feed.values
            logging.info(
                f"Successfully retrieved and validated feed '{feed_name}' for location '{loc}'."
            )
            return df.to_dict(
                orient="index"
            )  # Return dict for consistent serialization

        except HTTPException:  # Re-raise HTTPExceptions directly
            raise
        except Exception as e:  # Catch other unexpected errors
            logging.exception(
                f"Error retrieving feed '{feed_name}' for location '{loc}': {e}"
            )
            # Catch potential errors during data fetching/processing
            raise HTTPException(
                status_code=500, detail="Internal server error retrieving feed data"
            ) from e

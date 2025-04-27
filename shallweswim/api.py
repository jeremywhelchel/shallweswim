"""API handlers for ShallWeSwim application.

This module contains FastAPI route handlers for the API endpoints and data management.
"""

# Standard library imports
import asyncio
import io
import logging
from typing import Optional

# Third-party imports
import fastapi
from fastapi import HTTPException

# Local imports
from shallweswim import config as config_lib, data as data_lib, plot, util
from shallweswim.types import (
    ApiTideEntry,
    CurrentPredictionInfo,
    CurrentsResponse,
    LegacyChartDetails,
    LocationConditions,
    LocationInfo,
    TemperatureInfo,
    TidesInfo,
)


# Global data store for location data
data: dict[str, data_lib.DataManager] = {}


async def initialize_location_data(
    location_codes: list[str],
    data_dict: Optional[dict[str, data_lib.DataManager]] = None,
    wait_for_data: bool = False,
    max_wait_retries: int = 15,
    retry_interval: int = 1,
) -> dict[str, data_lib.DataManager]:
    """Initialize data for the specified locations.

    This function handles initialization of DataManager objects for the specified locations.
    It can be used by both the main application and tests.

    Args:
        location_codes: List of location codes to initialize
        data_dict: Optional existing data dictionary to populate (creates new if None)
        wait_for_data: Whether to wait for data to be loaded before returning
        max_wait_retries: Max number of retries when waiting for data
        retry_interval: Time in seconds between retries

    Returns:
        Dictionary mapping location codes to initialized DataManager objects

    Raises:
        AssertionError: If a location's configuration cannot be found
    """
    # Use the provided data dictionary or create a new one
    if data_dict is None:
        data_dict = {}

    # Initialize each location
    for code in location_codes:
        # Get location config
        cfg = config_lib.get(code)
        assert cfg is not None, f"Config for location '{code}' not found"

        # Initialize data for this location
        data_dict[code] = data_lib.DataManager(cfg)
        data_dict[code].start()

    # Optionally wait for data to be fully loaded
    if wait_for_data:
        for code in location_codes:
            for i in range(max_wait_retries):
                # Check if data is ready using the new .ready property
                if data_dict[code].ready:
                    print(f"{code} data loaded successfully after {i+1} attempts")
                    break
                print(
                    f"Waiting for {code} data to load... attempt {i+1}/{max_wait_retries}"
                )
                await asyncio.sleep(retry_interval)

            # If we've exhausted all retries, verify basic data availability
            # This ensures we fail with helpful error messages if data isn't ready
            if not data_dict[code].ready:
                # Verify tide data was loaded (all locations should have tide data)
                assert (
                    data_dict[code].tides is not None
                ), f"{code} tide data was not loaded"

                # Only check currents if the location has current predictions enabled
                location_config = config_lib.get(code)
                if (
                    location_config is not None
                    and location_config.currents_source
                    and location_config.currents_source.predictions_available
                ):
                    assert (
                        data_dict[code].currents is not None
                    ), f"{code} current data was not loaded"

    return data_dict


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

        # Get current water temperature
        current_time, current_temp = data[location].live_temp_reading()

        # Get tide information
        tide_info = data[location].prev_next_tide()

        # Create Pydantic model instances
        past_tides = [
            ApiTideEntry(
                time=tide.time.isoformat(), type=tide.type, prediction=tide.prediction
            )
            for tide in tide_info.past_tides
        ]

        next_tides = [
            ApiTideEntry(
                time=tide.time.isoformat(), type=tide.type, prediction=tide.prediction
            )
            for tide in tide_info.next_tides
        ]

        # Return structured response using Pydantic models
        return LocationConditions(
            location=LocationInfo(
                code=location, name=cfg.name, swim_location=cfg.swim_location
            ),
            temperature=TemperatureInfo(
                timestamp=current_time.isoformat(),
                water_temp=current_temp,
                units="F",
                station_name=cfg.temp_source.name if cfg.temp_source else None,
            ),
            tides=TidesInfo(past=past_tides, next=next_tides),
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
            fig = plot.create_tide_current_plot(
                data[location].tides, data[location].currents, ts, cfg
            )
        except AssertionError as e:
            # The function now raises assertions instead of returning None
            raise HTTPException(status_code=404, detail=str(e))

        # Convert figure to SVG in a StringIO buffer
        svg_io = io.StringIO()
        fig.savefig(svg_io, format="svg", bbox_inches="tight", transparent=False)
        svg_io.seek(0)

        return fastapi.responses.Response(
            content=svg_io.getvalue(), media_type="image/svg+xml"
        )

    @app.get(
        "/api/ready",
        status_code=200,
        responses={503: {"description": "Service not ready"}},
    )
    async def ready_status() -> bool:
        """API endpoint that returns whether all locations' data is ready.

        Returns:
            Boolean indicating whether all locations' data is ready
            Status code 200 if ready, 503 if not ready
        """
        logging.info("[api] Checking readiness status for all locations")
        # Get all configured locations
        all_locations = list(config_lib.CONFIGS.keys())

        # Check if all locations are ready
        for loc_code in all_locations:
            # Skip if location not in the data dictionary
            if loc_code not in data or data[loc_code] is None:
                logging.warning(f"[{loc_code}] Location not in data dictionary")
                raise HTTPException(
                    status_code=503, detail="Service not ready - location data missing"
                )

            # Check if location data is ready
            if not data[loc_code].ready:
                logging.info(f"[{loc_code}] Location data not ready yet")
                raise HTTPException(
                    status_code=503, detail="Service not ready - data being loaded"
                )

        # All locations are ready
        logging.info("[api] All locations report ready status")
        return True

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
        if not cfg.currents_source or not cfg.currents_source.predictions_available:
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

        # Get current prediction information
        current_info = data[location].current_prediction(ts)

        # Get legacy chart information
        chart_info = data[location].legacy_chart_info(ts)

        # Get fwd/back shift values for navigation
        fwd = min(shift + 60, util.MAX_SHIFT_LIMIT)
        back = max(shift - 60, util.MIN_SHIFT_LIMIT)

        # Get current chart filename
        current_chart_filename = plot.get_current_chart_filename(
            current_info.direction,
            plot.bin_magnitude(current_info.magnitude_pct),
            location,
        )

        # Format current_info data for the API response
        current_prediction = CurrentPredictionInfo(
            timestamp=ts.isoformat(),
            direction=current_info.direction,
            magnitude=round(current_info.magnitude, 1),
            magnitude_pct=current_info.magnitude_pct,
            state_description=current_info.state_description,
        )

        # Format legacy chart data for the API response
        legacy_chart = LegacyChartDetails(
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

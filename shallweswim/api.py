"""API handlers for ShallWeSwim application.

This module contains FastAPI route handlers for the API endpoints.
"""

# Standard library imports
import logging

# Third-party imports
import fastapi
from fastapi import HTTPException

# Local imports
from shallweswim import config as config_lib, data as data_lib, plot, util
from shallweswim.types import (
    ApiTideEntry,
    CurrentPredictionInfo,
    CurrentsResponse,
    FreshnessInfo,
    LegacyChartDetails,
    LocationConditions,
    LocationInfo,
    TemperatureInfo,
    TidesInfo,
)


def register_routes(app: fastapi.FastAPI, data: dict[str, data_lib.Data]) -> None:
    """Register API routes with the FastAPI application.

    Args:
        app: The FastAPI application
        data: Dictionary mapping location codes to Data objects
    """

    # Helper function to validate location
    def validate_location(loc: str) -> config_lib.LocationConfig:
        cfg = config_lib.Get(loc)
        if not cfg:
            logging.warning("Bad location: %s", loc)
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
        cfg = validate_location(location)

        # Get current water temperature
        current_time, current_temp = data[location].LiveTempReading()

        # Get tide information
        tide_info = data[location].PrevNextTide()

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
                timestamp=current_time.isoformat(), water_temp=current_temp, units="F"
            ),
            tides=TidesInfo(past=past_tides, next=next_tides),
        )

    @app.get("/api/{location}/freshness", response_model=FreshnessInfo)
    async def location_freshness(location: str) -> FreshnessInfo:
        """API endpoint that returns data freshness information for a specific location.

        Args:
            location: Location code (e.g., 'nyc')

        Returns:
            FreshnessInfo object with timestamps of last data updates

        Raises:
            HTTPException: If the location is not configured
        """
        # Validate location exists
        validate_location(location)

        # Return freshness information
        return data[location].Freshness()

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
        # Validate location exists
        cfg = validate_location(location)

        # Check if this location supports current predictions
        if not cfg.current_predictions:
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

        # Calculate effective time with shift
        ts = util.EffectiveTime(shift)

        # Get current prediction information
        current_info = data[location].CurrentPrediction(ts)

        # Get legacy chart information
        chart_info = data[location].LegacyChartInfo(ts)

        # Get fwd/back shift values for navigation
        fwd = min(shift + 60, util.MAX_SHIFT_LIMIT)
        back = max(shift - 60, util.MIN_SHIFT_LIMIT)

        # Get current chart filename
        current_chart_filename = plot.GetCurrentChartFilename(
            current_info.direction, plot.BinMagnitude(current_info.magnitude_pct)
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
                "plot_url": f"/current_tide_plot?shift={shift}",
            },
        )

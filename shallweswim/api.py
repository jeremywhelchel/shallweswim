"""API handlers for ShallWeSwim application.

This module contains FastAPI route handlers for the API endpoints.
"""

# Standard library imports
import logging

# Third-party imports
import fastapi
from fastapi import HTTPException

# Local imports
from shallweswim import config, data as data_lib
from shallweswim.types import (
    ApiTideEntry,
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
        cfg = config.Get(location)
        if not cfg:
            logging.warning("Bad location: %s", location)
            raise HTTPException(
                status_code=404, detail=f"Location '{location}' not found"
            )

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

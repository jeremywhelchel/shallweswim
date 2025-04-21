#!/usr/bin/env python3
"""
Shall We Swim - FastAPI web application for displaying swimming conditions

This module contains the FastAPI application that serves tide, current, and temperature
data to help determine if swimming conditions are favorable.
"""

# Standard library imports
import contextlib
import datetime
import logging
import os
import time
from typing import AsyncGenerator

# Third-party imports
import fastapi
import google.cloud.logging
import uvicorn
from fastapi import HTTPException, responses, staticfiles, templating

# Local imports
from shallweswim import config, data as data_lib, api


data: dict[str, data_lib.Data] = {}


def initialize_location_data(
    location_codes: list[str],
    data_dict: dict[str, data_lib.Data] | None = None,
    wait_for_data: bool = False,
    max_wait_retries: int = 45,
    retry_interval: int = 1,
) -> dict[str, data_lib.Data]:
    """Initialize data for the specified locations.

    This function handles initialization of Data objects for the specified locations.
    It can be used by both the main application and tests.

    Args:
        location_codes: List of location codes to initialize
        data_dict: Optional existing data dictionary to populate (creates new if None)
        wait_for_data: Whether to wait for data to be loaded before returning
        max_wait_retries: Max number of retries when waiting for data
        retry_interval: Time in seconds between retries

    Returns:
        Dictionary mapping location codes to initialized Data objects

    Raises:
        AssertionError: If a location's configuration cannot be found
    """
    # Use the provided data dictionary or create a new one
    if data_dict is None:
        data_dict = {}

    # Initialize each location
    for code in location_codes:
        # Get location config
        cfg = config.Get(code)
        assert cfg is not None, f"Config for location '{code}' not found"

        # Initialize data for this location
        data_dict[code] = data_lib.Data(cfg)
        data_dict[code].Start()

    # Optionally wait for data to be fully loaded
    if wait_for_data:
        for code in location_codes:
            for i in range(max_wait_retries):
                # Check that essential data is loaded
                if (
                    data_dict[code].tides is not None
                    and data_dict[code].currents is not None
                ):
                    print(f"{code} data loaded successfully after {i+1} attempts")
                    break
                print(
                    f"Waiting for {code} data to load... attempt {i+1}/{max_wait_retries}"
                )
                time.sleep(retry_interval)

            # Verify tide data was loaded (all locations should have tide data)
            assert data_dict[code].tides is not None, f"{code} tide data was not loaded"

            # Only check currents if the location has current predictions enabled
            location_config = config.Get(code)
            if location_config is not None and location_config.current_predictions:
                assert (
                    data_dict[code].currents is not None
                ), f"{code} current data was not loaded"

    return data_dict


@contextlib.asynccontextmanager
async def lifespan(_app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
    """Initialize data sources during application startup.

    This loads data for all configured locations and starts data collection.

    Args:
        _app: The FastAPI application instance (not used)

    Yields:
        None when setup is complete
    """
    # Initialize data for all configured locations
    # Don't wait for data to load since this will block application startup
    initialize_location_data(
        location_codes=list(config.CONFIGS.keys()),
        data_dict=data,
        wait_for_data=False,  # Don't block app startup waiting for data
    )
    yield
    # No cleanup needed, but would be handled here if necessary


app = fastapi.FastAPI(lifespan=lifespan)

app.mount(
    "/static", staticfiles.StaticFiles(directory="shallweswim/static"), name="static"
)

templates = templating.Jinja2Templates(directory="shallweswim/templates")

# Register API routes
api.register_routes(app, data)


@app.get("/")
async def index() -> responses.RedirectResponse:
    """Redirect root path to default location (NYC).

    TODO: Use cookies to redirect to last used or saved location.
    """
    return responses.RedirectResponse("/nyc")


@app.get("/embed")
async def embed(request: fastapi.Request) -> responses.HTMLResponse:
    """Serve the embed view for embedding in other websites.

    Currently hardcoded to NYC location. Future enhancement: support other locations.
    Data will be fetched client-side via API.
    """
    return templates.TemplateResponse(
        request=request,
        name="embed.html",
        context=dict(
            config=config.Get("nyc"),
        ),
    )


@app.get("/{location}/currents")
async def location_water_current(
    request: fastapi.Request, location: str, shift: int = 0
) -> responses.HTMLResponse:
    """Serve the water current visualization page for the specified location.

    Args:
        request: FastAPI request object
        location: Location code (e.g., "nyc", "san")
        shift: Time shift in minutes from current time

    Returns:
        HTML response with current water conditions
    """
    # Check if location exists
    if location not in data:
        raise HTTPException(status_code=404, detail=f"Location {location} not found")

    # Get location config
    location_config = config.Get(location)
    if not location_config:
        raise HTTPException(
            status_code=404, detail=f"Configuration for {location} not found"
        )

    return templates.TemplateResponse(
        request=request,
        name="current.html",
        context=dict(
            config=location_config,
            request=request,
        ),
    )


# Legacy route for backward compatibility
@app.get("/current")
async def water_current(
    request: fastapi.Request, shift: int = 0
) -> responses.RedirectResponse:
    """Redirect to the locationized currents page."""
    # Redirect to the locationized version with the same shift parameter
    redirect_url = f"/nyc/currents"
    if shift != 0:
        redirect_url += f"?shift={shift}"
    return responses.RedirectResponse(redirect_url)


def fmt_datetime(timestamp: datetime.datetime) -> str:
    """Format a datetime object for display in templates.

    Args:
        timestamp: Datetime to format

    Returns:
        Formatted string like 'Monday, April 20 at 8:12 PM'
    """
    return timestamp.strftime("%A, %B %-d at %-I:%M %p")


# Register the datetime formatter with Jinja2
templates.env.filters["fmt_datetime"] = fmt_datetime


@app.get("/favicon.ico")
async def favicon() -> responses.RedirectResponse:
    """Redirect favicon requests to the static file."""
    return responses.RedirectResponse(
        "/static/favicon.ico",
    )


@app.get("/robots.txt")
async def robots() -> responses.RedirectResponse:
    """Redirect robots.txt requests to the static file."""
    return responses.RedirectResponse(
        "/static/robots.txt",
    )


@app.get("/{location}")
async def index_w_location(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve the main page for a specific location.

    Args:
        request: FastAPI request object
        location: Location code (e.g., 'nyc')

    Returns:
        HTML response with location-specific configuration

    Raises:
        HTTPException: If the location is not configured
    """
    cfg = config.Get(location)
    if not cfg:
        logging.warning("Bad location: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    # Just pass location config - data will be fetched client-side via API
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=dict(
            config=cfg,
        ),
    )


def start_app() -> fastapi.FastAPI:
    """Initialize and return the FastAPI application.

    Sets up logging based on the environment (Google Cloud Run or local).

    Returns:
        Configured FastAPI application instance
    """
    # If running in Google Cloud Run, use cloud logging
    if "K_SERVICE" in os.environ:
        # Setup Google Cloud logging
        # By default this captures all logs at INFO level and higher
        log_client = google.cloud.logging.Client()  # type: ignore[no-untyped-call]
        log_client.get_default_handler()  # type: ignore[no-untyped-call]
        log_client.setup_logging()  # type: ignore[no-untyped-call]
        logging.info("Using google cloud logging")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Using standard logging")

    logging.info("Starting app")
    return app


if __name__ == "__main__":
    """Run the application directly with uvicorn when executed as a script."""
    logging.info("Running uvicorn app")
    uvicorn.run(
        "main:start_app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        log_level="info",
        reload=True,
    )

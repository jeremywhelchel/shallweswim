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
from shallweswim import config, data as data_lib, plot, api


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
    """
    current_time, current_temp = data["nyc"].LiveTempReading()
    tide_info = data["nyc"].PrevNextTide()
    return templates.TemplateResponse(
        request=request,
        name="embed.html",
        context=dict(
            config=config.Get("nyc"),
            current_time=current_time,
            current_temp=current_temp,
            past_tides=tide_info.past_tides,
            next_tides=tide_info.next_tides,
        ),
    )


MIN_SHIFT_LIMIT = -180  # 3 hours
MAX_SHIFT_LIMIT = 1260  # 21 hours


def EffectiveTime(shift: int = 0) -> datetime.datetime:
    """Return the effective time for displaying charts based on query parameters.

    Args:
        shift: Time shift in minutes from current time (negative for past, positive for future)
              Will be clamped between MIN_SHIFT_LIMIT and MAX_SHIFT_LIMIT

    Returns:
        A datetime object representing the effective time to display
    """
    t = data_lib.Now()

    # Clamp the shift limit
    shift = max(MIN_SHIFT_LIMIT, min(shift, MAX_SHIFT_LIMIT))
    t = t + datetime.timedelta(minutes=shift)

    return t


@app.get("/current_tide_plot")
async def current_tide_plot(shift: int = 0) -> responses.Response:
    """Generate and serve an SVG plot of current tide and current data.

    Args:
        shift: Time shift in minutes from current time

    Returns:
        SVG image response with tide and current visualization
    """
    ts = EffectiveTime(shift)
    image = plot.GenerateTideCurrentPlot(data["nyc"].tides, data["nyc"].currents, ts)
    assert image
    return responses.Response(content=image.getvalue(), media_type="image/svg+xml")


@app.get("/current")
async def water_current(
    request: fastapi.Request, shift: int = 0
) -> responses.HTMLResponse:
    """Serve the water current visualization page.

    Args:
        request: FastAPI request object
        shift: Time shift in minutes from current time

    Returns:
        HTML response with current water conditions
    """
    ts = EffectiveTime(shift)

    chart_info = data["nyc"].LegacyChartInfo(ts)

    current_info = data["nyc"].CurrentPrediction(ts)

    # Get fwd/back shift values
    fwd = min(shift + 60, MAX_SHIFT_LIMIT)
    back = max(shift - 60, MIN_SHIFT_LIMIT)

    return templates.TemplateResponse(
        request=request,
        name="current.html",
        context=dict(
            config=config.Get("nyc"),
            last_tide_hrs_ago=round(chart_info.hours_since_last_tide, 1),
            last_tide_type=chart_info.last_tide_type,
            tide_chart_filename=chart_info.chart_filename,
            map_title=chart_info.map_title,
            ts=ts,
            ef=current_info.direction,
            magnitude=round(current_info.magnitude, 1),
            msg=current_info.state_description,
            shift=shift,
            fwd=fwd,
            back=back,
            current_chart_filename=plot.GetCurrentChartFilename(
                current_info.direction, plot.BinMagnitude(current_info.magnitude_pct)
            ),
        ),
    )


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
        HTML response with location-specific data

    Raises:
        HTTPException: If the location is not configured
    """
    cfg = config.Get(location)
    if not cfg:
        logging.warning("Bad location: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    current_time, current_temp = data[location].LiveTempReading()
    tide_info = data[location].PrevNextTide()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=dict(
            config=cfg,
            current_time=current_time,
            current_temp=current_temp,
            past_tides=tide_info.past_tides,
            next_tides=tide_info.next_tides,
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

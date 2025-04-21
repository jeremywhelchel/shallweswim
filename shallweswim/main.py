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
from typing import AsyncGenerator

# Third-party imports
import fastapi
import google.cloud.logging
import uvicorn
from fastapi import HTTPException, responses, staticfiles, templating

# Local imports
from shallweswim import config, data as data_lib, plot
from shallweswim.types import FreshnessInfo


data = {}


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
    """Initialize data sources during application startup.

    This loads data for all configured locations and starts data collection.

    Args:
        app: The FastAPI application instance

    Yields:
        None when setup is complete
    """
    for code, cfg in config.CONFIGS.items():
        data[code] = data_lib.Data(cfg)
        data[code].Start()
    yield
    # No cleanup needed, but would be handled here if necessary


app = fastapi.FastAPI(lifespan=lifespan)

app.mount(
    "/static", staticfiles.StaticFiles(directory="shallweswim/static"), name="static"
)

templates = templating.Jinja2Templates(directory="shallweswim/templates")


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
    past_tides, next_tides = data["nyc"].PrevNextTide()
    return templates.TemplateResponse(
        request=request,
        name="embed.html",
        context=dict(
            config=config.Get("nyc"),
            current_time=current_time,
            current_temp=current_temp,
            past_tides=past_tides,
            next_tides=next_tides,
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

    (
        last_tide_hrs_ago,
        last_tide_type,
        tide_chart_filename,
        legacy_map_title,
    ) = data[
        "nyc"
    ].LegacyChartInfo(ts)

    (
        ef,
        magnitude,
        magnitude_pct,
        msg,
    ) = data[
        "nyc"
    ].CurrentPrediction(ts)

    # Get fwd/back shift values
    fwd = min(shift + 60, MAX_SHIFT_LIMIT)
    back = max(shift - 60, MIN_SHIFT_LIMIT)

    return templates.TemplateResponse(
        request=request,
        name="current.html",
        context=dict(
            config=config.Get("nyc"),
            last_tide_hrs_ago=round(last_tide_hrs_ago, 1),
            last_tide_type=last_tide_type,
            tide_chart_filename=tide_chart_filename,
            legacy_map_title=legacy_map_title,
            ts=ts,
            ef=ef,
            magnitude=round(magnitude, 1),
            msg=msg,
            shift=shift,
            fwd=fwd,
            back=back,
            current_chart_filename=plot.GetCurrentChartFilename(
                ef, plot.BinMagnitude(magnitude_pct)
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


@app.get("/freshness")
async def freshness() -> FreshnessInfo:
    """Return data freshness information for the NYC location.

    Returns:
        FreshnessInfo object with timestamps of last data updates
    """
    return data["nyc"].Freshness()


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
        logging.warning(f"Bad location: {location}")
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    current_time, current_temp = data[location].LiveTempReading()
    past_tides, next_tides = data[location].PrevNextTide()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=dict(
            config=cfg,
            current_time=current_time,
            current_temp=current_temp,
            past_tides=past_tides,
            next_tides=next_tides,
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

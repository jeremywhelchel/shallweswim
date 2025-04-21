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
from fastapi import HTTPException, Request, Response, responses, staticfiles, templating
from typing import Awaitable, Callable

# Local imports
from shallweswim import config, api


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
    api.initialize_location_data(
        location_codes=list(config.CONFIGS.keys()),
        data_dict=api.data,
        wait_for_data=False,  # Don't block app startup waiting for data
    )
    yield
    # No cleanup needed, but would be handled here if necessary


app = fastapi.FastAPI(lifespan=lifespan)


# API response headers for preventing caching
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


# Add a response header modifier for API routes
@app.middleware("http")
async def add_cache_control_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add cache control headers to API responses to prevent caching."""
    response = await call_next(request)

    # Add no-cache headers only to API routes
    if request.url.path.startswith("/api/"):
        for name, value in NO_CACHE_HEADERS.items():
            response.headers[name] = value

    return response


app.mount(
    "/static", staticfiles.StaticFiles(directory="shallweswim/static"), name="static"
)

templates = templating.Jinja2Templates(directory="shallweswim/templates")

# Register API routes
api.register_routes(app)


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
    # Get location config
    location_config = config.Get(location)
    if not location_config:
        raise HTTPException(
            status_code=404, detail=f"Configuration for {location} not found"
        )

    # Check if location data exists
    if location not in api.data:
        raise HTTPException(
            status_code=404, detail=f"Data for location {location} not found"
        )

    return templates.TemplateResponse(
        request=request,
        name="current.html",
        context=dict(
            config=location_config,
            request=request,
            shift=shift,  # Pass the shift parameter to the template
        ),
    )


# Legacy route for backward compatibility
@app.get("/current")
async def water_current(shift: int = 0) -> responses.RedirectResponse:
    """Redirect to the locationized currents page.

    Args:
        shift: Time shift in minutes from current time

    Returns:
        Redirect response to the location-specific currents page
    """
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

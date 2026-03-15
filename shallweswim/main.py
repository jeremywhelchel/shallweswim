#!/usr/bin/env python3
"""
Shall We Swim - FastAPI web application for displaying swimming conditions

This module contains the FastAPI application that serves tide, current, and temperature
data to help determine if swimming conditions are favorable.
"""

# Standard library imports
import argparse
import contextlib
import datetime
import logging
import os
import signal
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from concurrent.futures import ProcessPoolExecutor
from typing import (
    Any,
)

# Third-party imports
import aiohttp
import fastapi
import uvicorn
from fastapi import HTTPException, Request, Response, responses, templating

# Local imports
from shallweswim import api, config

# Local imports
from shallweswim.assets import AssetManager, FingerprintStaticFiles, load_asset_manifest
from shallweswim.logging_utils import setup_logging


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None]:
    """Initialize data sources during application startup.

    This loads data for all configured locations and starts data collection.
    It also creates the shared HTTP client session.

    Args:
        app: The FastAPI application instance

    Yields:
        None when setup is complete
    """
    # Create a process pool for CPU-bound tasks (e.g., plotting)
    pool = ProcessPoolExecutor()
    app.state.process_pool = pool

    async with aiohttp.ClientSession() as session:
        # Store the shared session in app state
        app.state.http_session = session

        # Initialize data for all configured locations
        await api.initialize_location_data(
            location_codes=list(config.CONFIGS.keys()),
            app=app,  # Pass the app instance
            wait_for_data=False,  # Don't block app startup waiting for data
        )

        yield  # Run the app

        # Shutdown handling
        logging.info("-----------------------------------------------")
        logging.info("Shutting down app")

        pool.shutdown(wait=True)

        # Stop all data managers to properly clean up background tasks
        if hasattr(app.state, "data_managers"):
            for location_code, data_manager in app.state.data_managers.items():
                logging.info(f"Stopping data manager for {location_code}")
                await data_manager.stop()

        # Context manager exits here, closing the session


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


# Mount static files handler
app.mount(
    "/static",
    FingerprintStaticFiles(directory="shallweswim/static", html=True, app=app),
    name="static",
)

templates = templating.Jinja2Templates(directory="shallweswim/templates")


# Register API routes
api.register_routes(app)


# ======================================================================
# Root routes - These must come before parameterized routes to avoid conflicts
# ======================================================================


@app.get("/")
async def root_index() -> responses.RedirectResponse:
    """Redirect root path to default location (NYC).

    TODO: Use cookies to redirect to last used or saved location.
    """
    return responses.RedirectResponse("/nyc", status_code=301)


@app.get("/all")
async def view_all_locations(request: fastapi.Request) -> responses.HTMLResponse:
    """Serve a landing page showing all swimming locations with their current water temperatures.

    Args:
        request: FastAPI request object

    Returns:
        HTML response with all locations and their water temperatures
    """
    return templates.TemplateResponse(
        request=request,
        name="all_locations.html",
        context={
            "all_locations": config.CONFIGS,
        },
    )


@app.get("/embed")
async def root_embed_redirect() -> responses.RedirectResponse:
    """Redirect /embed to the default location embed page (NYC).

    Returns:
        Redirect response to the NYC embed page
    """
    return responses.RedirectResponse("/nyc/embed", status_code=301)


# ======================================================================
# Static file routes - These should come before parameterized routes
# ======================================================================

# Define static file redirects for cleaner implementation
STATIC_FILES = [
    ("/favicon.ico", "/static/favicon.ico"),
    ("/apple-touch-icon.png", "/static/apple-touch-icon.png"),
    ("/apple-touch-icon-precomposed.png", "/static/apple-touch-icon.png"),
    ("/manifest.json", "/static/manifest.json"),
    ("/robots.txt", "/static/robots.txt"),
]

# Register each static file route individually
for path, target in STATIC_FILES:
    # Use a closure to capture the current value of target
    def create_static_route(
        target_path: str,
    ) -> Callable[[], Coroutine[Any, Any, responses.RedirectResponse]]:
        async def static_route() -> responses.RedirectResponse:
            return responses.RedirectResponse(target_path, status_code=301)

        return static_route

    # Register the route with FastAPI
    app.get(path)(create_static_route(target))


# ======================================================================
# Location-specific routes - These must come after static and root routes
# ======================================================================


@app.get("/{location}")
async def location_index(
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
    cfg = config.get(location)
    if not cfg:
        logging.warning("Bad location: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    # Just pass location config - data will be fetched client-side via API
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "config": cfg,
            "all_locations": config.CONFIGS,
        },
    )


@app.get("/{location}/embed")
async def location_embed(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve the embed view for a specific location for embedding in other websites.

    Args:
        request: FastAPI request object
        location: Location code (e.g., 'nyc')

    Returns:
        HTML response with location-specific embed

    Raises:
        HTTPException: If the location is not configured
    """
    cfg = config.get(location)
    if not cfg:
        logging.warning("Bad location for embed: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    return templates.TemplateResponse(
        request=request,
        name="embed.html",
        context={
            "config": cfg,
        },
    )


@app.get("/{location}/widget")
async def location_widget(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve a widget view for a specific location.

    Args:
        request: FastAPI request object
        location: Location code (e.g., 'nyc')

    Returns:
        HTML response with location-specific widget

    Raises:
        HTTPException: If the location is not configured
    """
    cfg = config.get(location)
    if not cfg:
        logging.warning("Bad location for widget: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    return templates.TemplateResponse(
        request=request,
        name="widget.html",
        context={
            "config": cfg,
        },
    )


@app.get("/{location}/currents")
async def location_currents(
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
    cfg = config.get(location)
    if not cfg:
        logging.warning("Bad location for currents: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    # Calculate the shifted time
    now = datetime.datetime.now(datetime.UTC)
    shifted_time = now + datetime.timedelta(minutes=shift)

    # Format the time for display
    shifted_time_local = shifted_time.astimezone(cfg.timezone)
    formatted_time = shifted_time_local.strftime("%Y-%m-%d %H:%M")

    return templates.TemplateResponse(
        request=request,
        name="current.html",
        context={
            "config": cfg,
            "shift": shift,
            "shifted_time": formatted_time,
        },
    )


# ======================================================================
# Template helpers
# ======================================================================


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


# Function to be used in templates
def static_url(file_path: str) -> str:
    """Generate a URL for a static file, using fingerprinting if available.

    Args:
        file_path: Path to the static file, relative to the static directory

    Returns:
        URL for the static file
    """
    # Use the AssetManager if available
    if hasattr(app.state, "asset_manager"):
        asset_url: str = app.state.asset_manager.static_url(file_path)
        return asset_url
    else:
        # If we don't have an asset manager, just return the original path
        default_url: str = f"/static/{file_path}"
        return default_url


# Register the static_url function with Jinja2
templates.env.globals["static_url"] = static_url


# ======================================================================
# Application initialization and signal handling
# ======================================================================


def setup_signal_handlers() -> None:
    """Set up signal handlers to log when specific signals are received.

    Note: Only registers a handler for SIGTERM, as handling SIGINT would
    interfere with the default Ctrl+C behavior that uvicorn relies on.
    """
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def sigterm_handler(sig: int, frame: Any) -> None:
        # Log the signal
        logging.warning("Received SIGTERM signal, beginning shutdown")

        # Call original handler if it was a callable
        if callable(original_sigterm_handler):
            original_sigterm_handler(sig, frame)

    # Only register handler for SIGTERM (which Google Cloud Run sends)
    # We don't handle SIGINT here to avoid interfering with uvicorn's Ctrl+C handling
    signal.signal(signal.SIGTERM, sigterm_handler)


def start_app(asset_manifest: str | None = None) -> fastapi.FastAPI:
    """Initialize and return the FastAPI application.

    Args:
        asset_manifest: Optional path to asset manifest file for fingerprinting

    Returns:
        Configured FastAPI application instance
    """
    logging.info("Starting app")

    # Set up signal handlers to log signals
    setup_signal_handlers()

    # Load asset manifest if provided
    if asset_manifest:
        logging.info(f"Loading asset manifest: {asset_manifest}")
        manifest = load_asset_manifest(asset_manifest)
        if manifest is None:
            # Fail fast and loud - never silently fall back
            raise RuntimeError(f"Failed to load asset manifest: {asset_manifest}")

        # Create an AssetManager and store it in app.state
        app.state.asset_manager = AssetManager()
        app.state.asset_manager.manifest = manifest
        logging.info("Asset fingerprinting enabled")

    return app


# Define the argument parser at module level
parser = argparse.ArgumentParser(description="Run the Shall We Swim application")
parser.add_argument(
    "--asset-manifest",
    type=str,
    help="Path to asset manifest file for fingerprint-based cache busting",
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind the server to",
)
parser.add_argument(
    "--port",
    type=int,
    default=int(os.environ.get("PORT", 8080)),
    help="Port to bind the server to",
)


_parsed_args = parser.parse_args()


def create_app() -> fastapi.FastAPI:
    """Factory function for creating the FastAPI application.

    This function is used by uvicorn to create the application instance.
    It reads the asset manifest path from parsed arguments if available.

    Returns:
        FastAPI application instance
    """
    # Get the asset manifest path from parsed arguments if available
    assert _parsed_args is not None
    asset_manifest = _parsed_args.asset_manifest

    logging.info(f"create_app() called, asset_manifest = {asset_manifest}")
    return start_app(asset_manifest=asset_manifest)


if __name__ == "__main__":
    """Run the application directly with uvicorn when executed as a script."""
    # Set up logging first, before any logging calls
    setup_logging()
    logging.info("***********************************************")
    logging.info("Running uvicorn app")
    logging.info(f"Command line arguments: {_parsed_args}")

    # Run the application with uvicorn
    uvicorn.run(
        "shallweswim.main:create_app",
        host=_parsed_args.host,
        port=_parsed_args.port,
        log_level="info",
        reload=True,
        factory=True,
    )

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
import json
import logging
import os
import re
import signal
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import (
    Any,
    cast,
)

# Third-party imports
import aiohttp
import fastapi
import uvicorn
from fastapi import HTTPException, Request, Response, responses, templating

# Local imports
from shallweswim import api, canonical, config

# Local imports
from shallweswim.assets import AssetManager, FingerprintStaticFiles, load_asset_manifest
from shallweswim.compression import SelectiveGZipMiddleware
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
    # Create a process pool for CPU-bound tasks (e.g., plotting).
    # Bound to CPU count to prevent over-subscription on the machine.
    pool = ProcessPoolExecutor(max_workers=os.cpu_count())
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
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1000)


# API response headers for preventing caching
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}
APP_SHELL_CACHE_HEADERS = {
    "Cache-Control": "no-cache, must-revalidate",
}
APP_ASSET_CACHE_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
}
DEFAULT_FRONTEND_DIST = Path("frontend/dist")
ROOT_DIV_RE = re.compile(r"(<div\s+id=[\"']root[\"']\s*></div>)", re.IGNORECASE)
TITLE_RE = re.compile(r"\s*<title>.*?</title>", re.IGNORECASE | re.DOTALL)
HEAD_CLOSE_RE = re.compile(r"\s*</head>", re.IGNORECASE)
SOCIAL_IMAGE_PATH = "/static/android-chrome-512x512.png"


@app.middleware("http")
async def redirect_duplicate_hosts(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Redirect duplicate production hostnames to the canonical apex host."""
    redirect_url = canonical.canonical_redirect_url(
        hostname=request.url.hostname,
        path=request.url.path,
        query=request.url.query,
    )
    if redirect_url:
        return responses.RedirectResponse(redirect_url, status_code=301)

    return await call_next(request)


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


@app.get("/legacy")
async def legacy_root_redirect() -> responses.RedirectResponse:
    """Redirect the legacy root path to the default legacy location page."""
    return responses.RedirectResponse("/legacy/nyc", status_code=301)


@app.get("/all", include_in_schema=False)
async def legacy_all_redirect() -> responses.RedirectResponse:
    """Redirect the historical all-locations URL to the canonical route."""
    return responses.RedirectResponse("/locations", status_code=301)


@app.get("/legacy/all")
async def legacy_all_locations(request: fastapi.Request) -> responses.HTMLResponse:
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
            "canonical_url": canonical.canonical_url("/locations"),
        },
    )


@app.get("/legacy/embed")
async def legacy_embed_redirect() -> responses.RedirectResponse:
    """Redirect /legacy/embed to the default legacy location embed page.

    Returns:
        Redirect response to the NYC embed page
    """
    return responses.RedirectResponse("/legacy/nyc/embed", status_code=301)


@app.get("/embed")
async def root_embed_redirect() -> responses.RedirectResponse:
    """Redirect the existing embed route to the default location embed page."""
    return responses.RedirectResponse("/nyc/embed", status_code=301)


# ======================================================================
# Static file routes - These should come before parameterized routes
# ======================================================================

# Define static file redirects for cleaner implementation
STATIC_FILE_REDIRECTS = [
    ("/favicon.ico", "/static/favicon.ico"),
    ("/apple-touch-icon.png", "/static/apple-touch-icon.png"),
    ("/apple-touch-icon-precomposed.png", "/static/apple-touch-icon.png"),
]

# Register each static file route individually
for path, target in STATIC_FILE_REDIRECTS:
    # Use a closure to capture the current value of target
    def create_static_route(
        target_path: str,
    ) -> Callable[[], Coroutine[Any, Any, responses.RedirectResponse]]:
        async def static_route() -> responses.RedirectResponse:
            return responses.RedirectResponse(target_path, status_code=301)

        return static_route

    # Register the route with FastAPI
    app.get(path)(create_static_route(target))


@app.get("/manifest.json")
async def manifest_json() -> responses.FileResponse:
    """Serve the canonical web app manifest for the root-mounted React app."""
    return responses.FileResponse(
        "shallweswim/static/manifest.json",
        media_type="application/manifest+json",
        headers=APP_SHELL_CACHE_HEADERS,
    )


@app.get("/robots.txt")
async def robots_txt() -> responses.PlainTextResponse:
    """Serve crawler directives for the canonical site."""
    return responses.PlainTextResponse(canonical.robots_txt())


@app.get("/sitemap.xml")
async def sitemap_xml() -> responses.Response:
    """Serve a sitemap of canonical, indexable pages."""
    urls = [canonical.canonical_url("/locations")]
    urls.extend(
        canonical.canonical_url(f"/{loc_code}") for loc_code in sorted(config.CONFIGS)
    )
    return responses.Response(
        content=canonical.sitemap_xml(urls), media_type="application/xml"
    )


# ======================================================================
# React app routes - These must come before parameterized location routes
# ======================================================================


def frontend_dist_dir() -> Path:
    """Return the configured frontend build output directory."""
    return Path(getattr(app.state, "frontend_dist", DEFAULT_FRONTEND_DIST))


def frontend_index_path() -> Path:
    """Return the built React app shell path."""
    return frontend_dist_dir() / "index.html"


def frontend_index_snapshot(index: Path) -> tuple[str, str]:
    """Return the built React shell cache key and HTML."""
    cache_key = str(index)
    cache = getattr(app.state, "frontend_index_cache", None)
    if cache is not None and cache["key"] == cache_key:
        return cache_key, cast(str, cache["html"])

    index_html = index.read_text(encoding="utf-8")
    app.state.frontend_index_cache = {"key": cache_key, "html": index_html}
    return cache_key, index_html


def frontend_assets_dir() -> Path:
    """Return the resolved built React app assets directory."""
    return (frontend_dist_dir() / "assets").resolve()


def app_shell_missing_response() -> responses.PlainTextResponse:
    """Return a clear local-development response for a missing frontend build."""
    return responses.PlainTextResponse(
        "Frontend app shell has not been built. Run "
        "`corepack pnpm@10.18.3 --dir frontend build` from the repository root, "
        "then retry the requested app route.",
        status_code=404,
        headers=APP_SHELL_CACHE_HEADERS,
    )


@app.get("/assets/{asset_path:path}", include_in_schema=False)
async def frontend_asset(asset_path: str) -> responses.FileResponse:
    """Serve Vite-built immutable assets for the React app."""
    assets_dir = frontend_assets_dir()
    asset = (assets_dir / asset_path).resolve()

    try:
        asset.relative_to(assets_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Frontend asset not found") from exc

    if not asset.is_file():
        raise HTTPException(status_code=404, detail="Frontend asset not found")

    return responses.FileResponse(asset, headers=APP_ASSET_CACHE_HEADERS)


def _json_script(data: dict[str, Any]) -> str:
    """Serialize JSON-LD without allowing accidental script termination."""
    return json.dumps(data, ensure_ascii=False, indent=2).replace("</", "<\\/")


def _root_config() -> config.LocationConfig:
    """Return the configured default app location."""
    cfg = config.get(config.DEFAULT_LOCATION_CODE)
    if cfg is None:
        raise RuntimeError(
            f"Default location is not configured: {config.DEFAULT_LOCATION_CODE}"
        )
    return cfg


def _page_title(cfg: config.LocationConfig | None, *, locations_page: bool) -> str:
    """Build the durable app shell page title."""
    if locations_page:
        return "Open water swimming locations | shall we swim?"
    if cfg is None:
        raise RuntimeError("Location config is required for location page titles")
    return f"{cfg.swim_location} swim conditions | shall we swim?"


def _page_description(
    cfg: config.LocationConfig | None, *, locations_page: bool
) -> str:
    """Build the durable app shell meta description."""
    if locations_page:
        return (
            "Browse open water swimming condition pages and JSON data APIs for "
            "configured swim locations."
        )
    if cfg is None:
        raise RuntimeError("Location config is required for location page descriptions")
    return (
        f"{cfg.description}. Water temperature, tide, and current data for "
        f"{cfg.swim_location} in {cfg.name}."
    )


def _head_metadata(
    *,
    title: str,
    description: str,
    canonical_url: str,
    alternate_api_url: str,
    json_ld: dict[str, Any],
) -> str:
    """Build route-specific metadata for the React app shell head."""
    return templates.env.get_template("app_shell_head.html").render(
        title=title,
        description=description,
        canonical_url=canonical_url,
        alternate_api_url=alternate_api_url,
        social_image_url=canonical.canonical_url(SOCIAL_IMAGE_PATH),
        social_image_alt="shall we swim? open water conditions",
        json_ld=_json_script(json_ld),
    )


def _json_ld(
    *,
    path: str,
    title: str,
    description: str,
    cfg: config.LocationConfig | None,
    include_website: bool,
) -> dict[str, Any]:
    """Build conservative JSON-LD for a durable app route."""
    page_url = canonical.canonical_url(path)
    graph: list[dict[str, Any]] = []

    if include_website:
        graph.append(
            {
                "@type": "WebSite",
                "@id": f"{canonical.CANONICAL_BASE_URL}/#website",
                "name": "shall we swim?",
                "url": f"{canonical.CANONICAL_BASE_URL}/",
            }
        )

    page: dict[str, Any] = {
        "@type": "WebPage",
        "@id": f"{page_url}#webpage",
        "url": page_url,
        "name": title,
        "description": description,
    }

    if cfg is not None:
        page["about"] = {"@id": f"{page_url}#place"}
        graph.append(
            {
                "@type": "Place",
                "@id": f"{page_url}#place",
                "name": cfg.swim_location,
                "url": page_url,
                "geo": {
                    "@type": "GeoCoordinates",
                    "latitude": cfg.latitude,
                    "longitude": cfg.longitude,
                },
            }
        )

    graph.append(page)
    return {"@context": "https://schema.org", "@graph": graph}


def _noscript_fallback(
    *, cfg: config.LocationConfig | None, locations_page: bool
) -> str:
    """Build compact no-JavaScript fallback content for app routes."""
    if locations_page:
        return templates.env.get_template("app_shell_noscript.html").render(
            locations_page=True,
            locations=config.CONFIGS.items(),
        )

    if cfg is None:
        raise RuntimeError("Location config is required for location fallback")

    return templates.env.get_template("app_shell_noscript.html").render(
        locations_page=False,
        config=cfg,
    )


def _render_frontend_shell(
    *,
    index_html: str,
    path: str,
    cfg: config.LocationConfig | None,
    locations_page: bool = False,
    include_website: bool = False,
) -> str:
    """Render route-specific durable HTML around the built Vite shell."""
    title = _page_title(cfg, locations_page=locations_page)
    description = _page_description(cfg, locations_page=locations_page)
    page_url = canonical.canonical_url(path)
    if locations_page:
        alternate_api_path = "/api/locations"
    elif cfg is not None:
        alternate_api_path = f"/api/{cfg.code}/conditions"
    else:
        raise RuntimeError("Location config is required for location API metadata")
    alternate_api_url = canonical.canonical_url(alternate_api_path)
    metadata = _head_metadata(
        title=title,
        description=description,
        canonical_url=page_url,
        alternate_api_url=alternate_api_url,
        json_ld=_json_ld(
            path=path,
            title=title,
            description=description,
            cfg=cfg,
            include_website=include_website,
        ),
    )
    fallback = _noscript_fallback(cfg=cfg, locations_page=locations_page)

    rendered = TITLE_RE.sub("", index_html, count=1)

    if HEAD_CLOSE_RE.search(rendered):
        rendered = HEAD_CLOSE_RE.sub(f"\n{metadata}\n  </head>", rendered, count=1)
    else:
        rendered = f"<head>\n{metadata}\n</head>\n{rendered}"

    rendered, replacements = ROOT_DIV_RE.subn(
        lambda match: f"{match.group(1)}\n{fallback}", rendered, count=1
    )
    if replacements == 0:
        rendered = f'{rendered}<div id="root"></div>\n{fallback}'

    return rendered


def frontend_app_shell_response(
    *,
    path: str,
    cfg: config.LocationConfig | None,
    locations_page: bool = False,
    include_website: bool = False,
) -> responses.Response:
    """Serve route-specific durable HTML for canonical React app routes."""
    index = frontend_index_path()
    if not index.is_file():
        return app_shell_missing_response()

    index_key, index_html = frontend_index_snapshot(index)
    shell_cache_key = (
        index_key,
        path,
        cfg.code if cfg is not None else "",
        locations_page,
        include_website,
    )
    shell_cache = getattr(app.state, "frontend_shell_cache", {})
    if shell_cache_key in shell_cache:
        rendered = cast(str, shell_cache[shell_cache_key])
    else:
        rendered = _render_frontend_shell(
            index_html=index_html,
            path=path,
            cfg=cfg,
            locations_page=locations_page,
            include_website=include_website,
        )
        shell_cache[shell_cache_key] = rendered
        app.state.frontend_shell_cache = shell_cache

    return responses.HTMLResponse(
        rendered,
        media_type="text/html",
        headers=APP_SHELL_CACHE_HEADERS,
    )


@app.get("/", include_in_schema=False)
async def frontend_root() -> responses.Response:
    """Serve the React app shell for the default location dashboard."""
    return frontend_app_shell_response(
        path="/",
        cfg=_root_config(),
        include_website=True,
    )


@app.get("/locations", include_in_schema=False)
async def frontend_locations() -> responses.Response:
    """Serve the React app shell for the all-locations route."""
    return frontend_app_shell_response(
        path="/locations",
        cfg=None,
        locations_page=True,
    )


# ======================================================================
# Legacy Jinja routes - These must come after static and root routes
# ======================================================================


@app.get("/legacy/{location}")
async def legacy_location_index(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve the legacy Jinja main page for a specific location.

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
            "canonical_url": canonical.canonical_url(f"/{cfg.code}"),
        },
    )


@app.get("/legacy/{location}/embed")
async def legacy_location_embed(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve the legacy embed view for a specific location.

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
            "canonical_url": canonical.canonical_url(f"/legacy/{cfg.code}/embed"),
        },
    )


@app.get("/{location}/embed")
async def location_embed(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve the existing legacy embed view at its historical URL."""
    cfg = config.get(location)
    if not cfg:
        logging.warning("Bad location for embed: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    return templates.TemplateResponse(
        request=request,
        name="embed.html",
        context={
            "config": cfg,
            "canonical_url": canonical.canonical_url(f"/{cfg.code}/embed"),
        },
    )


@app.get("/legacy/{location}/widget")
async def legacy_location_widget(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
    """Serve a legacy widget view for a specific location.

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


@app.get("/legacy/{location}/currents")
async def legacy_location_currents(
    request: fastapi.Request, location: str, shift: int = 0
) -> responses.HTMLResponse:
    """Serve the legacy water current visualization page for the specified location.

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
            "canonical_url": canonical.canonical_url(f"/{cfg.code}"),
        },
    )


# ======================================================================
# Root-mounted React location routes - These must come after reserved routes
# ======================================================================


@app.get("/{location}", include_in_schema=False)
async def frontend_location(location: str) -> responses.Response:
    """Serve the React app shell for canonical location dashboards."""
    cfg = config.get(location)
    if not cfg:
        logging.warning("Bad location: %s", location)
        raise HTTPException(status_code=404, detail=f"Bad location: {location}")

    return frontend_app_shell_response(path=f"/{cfg.code}", cfg=cfg)


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
cast(Any, templates.env.globals)["static_url"] = static_url


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


def start_app(
    asset_manifest: str | None = None,
    frontend_dist: str | None = None,
    require_frontend_dist: bool = False,
) -> fastapi.FastAPI:
    """Initialize and return the FastAPI application.

    Args:
        asset_manifest: Optional path to asset manifest file for fingerprinting
        frontend_dist: Optional path to the built frontend app directory
        require_frontend_dist: Fail startup if the frontend app shell is missing

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

    if frontend_dist:
        app.state.frontend_dist = frontend_dist

    if require_frontend_dist and not frontend_index_path().is_file():
        raise RuntimeError(
            "Frontend app shell is missing. Expected built index.html at "
            f"{frontend_index_path()}"
        )

    return app


# Define the argument parser at module level
parser = argparse.ArgumentParser(description="Run the Shall We Swim application")
parser.add_argument(
    "--asset-manifest",
    type=str,
    help="Path to asset manifest file for fingerprint-based cache busting",
)
parser.add_argument(
    "--frontend-dist",
    type=str,
    default=str(DEFAULT_FRONTEND_DIST),
    help="Path to built frontend app output directory",
)
parser.add_argument(
    "--require-frontend-dist",
    action="store_true",
    help="Fail startup if the built frontend app shell is missing",
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


_parsed_args, _unknown_args = parser.parse_known_args()


def create_app() -> fastapi.FastAPI:
    """Factory function for creating the FastAPI application.

    This function is used by uvicorn to create the application instance.
    It reads the asset manifest path from parsed arguments if available.

    Returns:
        FastAPI application instance
    """
    # Get the asset manifest path from parsed arguments if available
    if _parsed_args is None:
        raise RuntimeError("Application arguments have not been parsed")
    asset_manifest = _parsed_args.asset_manifest

    logging.info(f"create_app() called, asset_manifest = {asset_manifest}")
    return start_app(
        asset_manifest=asset_manifest,
        frontend_dist=_parsed_args.frontend_dist,
        require_frontend_dist=_parsed_args.require_frontend_dist,
    )


if __name__ == "__main__":
    """Run the application directly with uvicorn when executed as a script."""
    _parsed_args = parser.parse_args()

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

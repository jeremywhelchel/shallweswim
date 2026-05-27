"""API handlers for ShallWeSwim application.

This module contains FastAPI route handlers for the API endpoints and data management.
"""

# Standard library imports
import asyncio
import dataclasses
import datetime
import io
import logging
import urllib.parse
import warnings

# Third-party imports
import aiohttp
import fastapi
from fastapi import HTTPException

# Local imports
from shallweswim import config as config_lib
from shallweswim import data as data_lib
from shallweswim import types, util
from shallweswim.api_types import (
    AppBootstrapLocation,
    AppBootstrapResponse,
    AppExternalIntegrations,
    AppFeatureFlags,
    AppLocationMetadata,
    AppPresentationLink,
    AppSourceCitations,
    AppTemperaturePlotConfig,
    AppWebcamConfig,
    CurrentInfo,
    CurrentRange,
    CurrentRangePoint,
    CurrentsResponse,
    LegacyChartInfo,
    LocationConditions,
    LocationInfo,
    LocationStatus,
    LocationSummary,
    NavigationInfo,
    TemperatureInfo,
    TideEntry,
    TideInfo,
    TideState,
    TransitRouteConfig,
)
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi
from shallweswim.core.feeds import (
    FEED_CURRENTS,
    FEED_LIVE_TEMPS,
    FEED_TIDES,
    PLOT_HISTORIC_TEMPS_2MO,
    PLOT_HISTORIC_TEMPS_12MO,
    PLOT_LIVE_TEMPS,
    PlotName,
)
from shallweswim.core.queries import DataUnavailableError

# Data store for location data will be stored in app.state.data_managers


def validate_location(loc: str) -> config_lib.LocationConfig:
    """Return location config or raise the API's standard 404."""
    cfg = config_lib.get(loc)
    if not cfg:
        logging.warning(f"[{loc}] Bad location request")
        raise HTTPException(status_code=404, detail=f"Location '{loc}' not found")
    return cfg


def _create_tide_current_plot(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Wrapper that lazily imports plot module for subprocess execution.

    This avoids loading matplotlib/seaborn/scipy in the main process.
    The heavy imports only happen in the subprocess pool worker.
    """
    from shallweswim import plot

    return plot.create_tide_current_plot(*args, **kwargs)


# Timeout for on-demand plot generation (seconds)
# Shorter than background (60s) since user is waiting
PLOT_TIMEOUT = 30.0
APP_NAME = "shall we swim?"
APP_SHORT_NAME = "shallweswim"
APP_THEME_COLOR = "#000099"
APP_BACKGROUND_COLOR = "#000099"

GITHUB_SOURCE_URL = "https://github.com/jeremywhelchel/shallweswim"

# Filter the specific Pydantic serialization warning globally for production
# Note: Tests might handle this separately (e.g., via pytest markers/config)
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
    # Optionally target the specific module if needed, but keeping broad for now
    # module="pydantic.type_adapter"
)


def api_current_range(current_range: types.CurrentRange | None) -> CurrentRange | None:
    """Convert internal current range context to the API model."""
    if current_range is None:
        return None

    return CurrentRange(
        slack=CurrentRangePoint(
            timestamp=current_range.slack.timestamp,
            magnitude=current_range.slack.magnitude,
            units=current_range.slack.units,
            phase=current_range.slack.phase,
        ),
        peak=CurrentRangePoint(
            timestamp=current_range.peak.timestamp,
            magnitude=current_range.peak.magnitude,
            units=current_range.peak.units,
            phase=current_range.peak.phase,
        ),
    )


@dataclasses.dataclass(frozen=True)
class LocationRequestContext:
    """Location request context after validating the location code."""

    cfg: config_lib.LocationConfig
    data_manager: data_lib.LocationDataManager


@dataclasses.dataclass(frozen=True)
class ResolvedLocationTime:
    """Location request context after validating planner time parameters."""

    cfg: config_lib.LocationConfig
    data_manager: data_lib.LocationDataManager
    time_query: util.EffectiveTimeQuery

    @property
    def timestamp(self) -> datetime.datetime:
        return self.time_query.timestamp


def resolve_location_context(
    app: fastapi.FastAPI,
    location: str,
) -> LocationRequestContext:
    """Resolve shared location config and manager for API routes."""
    cfg = validate_location(location)
    return LocationRequestContext(
        cfg=cfg,
        data_manager=app.state.data_managers[location],
    )


def resolve_location_time(
    app: fastapi.FastAPI,
    location: str,
    *,
    shift: int = 0,
    at: str | None = None,
) -> ResolvedLocationTime:
    """Resolve shared location-local planner time for time-aware API routes."""
    location_context = resolve_location_context(app, location)
    try:
        time_query = util.effective_time_query(
            location_context.cfg.timezone, shift_minutes=shift, at=at
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ResolvedLocationTime(
        cfg=location_context.cfg,
        data_manager=location_context.data_manager,
        time_query=time_query,
    )


def api_location_info(location: str, cfg: config_lib.LocationConfig) -> LocationInfo:
    """Convert location config to the API location summary model."""
    return LocationInfo(code=location, name=cfg.name, swim_location=cfg.swim_location)


def api_tide_entry(tide: types.TideEntry) -> TideEntry:
    """Convert an internal tide event to the API model."""
    return TideEntry(
        time=tide.time.isoformat(),
        type=tide.type,
        prediction=tide.prediction,
    )


def api_tide_state(tide_state: types.TideState | None) -> TideState | None:
    """Convert an internal point-in-time tide state to the API model."""
    if tide_state is None:
        return None

    return TideState(
        timestamp=tide_state.timestamp,
        estimated_height=tide_state.estimated_height,
        units=tide_state.units,
        trend=tide_state.trend,
        height_pct=tide_state.height_pct,
    )


def api_tide_info_at_time(
    data_manager: data_lib.LocationDataManager,
    timestamp: datetime.datetime,
) -> TideInfo:
    """Build tide API data for one location-local timestamp."""
    tide_info = data_manager.get_tide_info_at_time(timestamp)
    return TideInfo(
        past=[api_tide_entry(tide) for tide in tide_info.past],
        next=[api_tide_entry(tide) for tide in tide_info.next],
        state=api_tide_state(data_manager.predict_tide_at_time(timestamp)),
    )


def api_current_info(
    current_info: types.CurrentInfo, *, magnitude_digits: int | None = None
) -> CurrentInfo:
    """Convert internal current state to the API model."""
    magnitude = current_info.magnitude
    if magnitude_digits is not None:
        magnitude = round(magnitude, magnitude_digits)

    return CurrentInfo(
        timestamp=current_info.timestamp.isoformat(),
        direction=current_info.direction,
        phase=current_info.phase,
        strength=current_info.strength,
        trend=current_info.trend,
        magnitude=magnitude,
        magnitude_pct=current_info.magnitude_pct,
        state_description=current_info.state_description,
        range=api_current_range(current_info.range),
        source_type=current_info.source_type,
    )


def api_temperature_info(
    cfg: config_lib.LocationConfig,
    data_manager: data_lib.LocationDataManager,
) -> TemperatureInfo | None:
    """Build observed temperature data for the conditions endpoint."""
    if not (
        cfg.temp_source is not None
        and cfg.temp_source.live_enabled
        and data_manager.has_feed_data(FEED_LIVE_TEMPS)
    ):
        return None

    temp_reading = data_manager.get_current_temperature()
    return TemperatureInfo(
        timestamp=temp_reading.timestamp.isoformat(),
        water_temp=temp_reading.temperature,
        units="F",
        station_name=cfg.temp_source.name,
    )


def api_conditions_tide_info(
    ctx: ResolvedLocationTime,
) -> TideInfo | None:
    """Build tide data for the conditions endpoint at the resolved time."""
    if not (ctx.cfg.tide_source and ctx.data_manager.has_feed_data(FEED_TIDES)):
        return None

    return api_tide_info_at_time(ctx.data_manager, ctx.timestamp)


def api_conditions_current_info(
    ctx: ResolvedLocationTime,
) -> CurrentInfo | None:
    """Build current data for the conditions endpoint at the resolved time.

    Prediction sources use the requested planner time. Observation sources
    remain latest-observation data until we have forecast/prediction support for
    that source type.
    """
    if not (ctx.cfg.currents_source and ctx.data_manager.has_feed_data(FEED_CURRENTS)):
        return None

    match ctx.cfg.currents_source.source_type:
        case types.DataSourceType.PREDICTION:
            current_info = ctx.data_manager.predict_flow_at_time(ctx.timestamp)
        case types.DataSourceType.OBSERVATION:
            current_info = ctx.data_manager.get_current_flow_info()
        case _:
            raise ValueError(
                f"Unknown current source type: {ctx.cfg.currents_source.source_type}"
            )

    return api_current_info(current_info)


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
        RuntimeError: If required app startup state is missing.
        ValueError: If a requested location code is not configured.
    """
    # Retrieve the shared session from app state
    if not hasattr(app.state, "http_session") or app.state.http_session is None:
        raise RuntimeError("HTTP session not found in app state")
    session: aiohttp.ClientSession = app.state.http_session

    # Retrieve the process pool from app state
    if not hasattr(app.state, "process_pool") or app.state.process_pool is None:
        raise RuntimeError("Process pool not found in app state")
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
        if cfg is None:
            raise ValueError(f"Config for location '{code}' not found")

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

    def timezone_name(cfg: config_lib.LocationConfig) -> str:
        """Return a stable IANA timezone name for a location config."""
        zone = getattr(cfg.timezone, "zone", None)
        if isinstance(zone, str):
            return zone
        return str(cfg.timezone)

    def presentation_link(
        link: config_lib.PresentationLinkConfig | None,
    ) -> AppPresentationLink | None:
        """Convert internal presentation link config to the public API model."""
        if link is None:
            return None
        return AppPresentationLink(
            label=link.label,
            url=link.url,
            description=link.description,
        )

    def webcam_config(
        webcam: config_lib.WebcamConfig | None,
    ) -> AppWebcamConfig | None:
        """Convert internal webcam config to the public API model."""
        if webcam is None:
            return None
        return AppWebcamConfig(
            provider=webcam.provider,
            label=webcam.label,
            embed_url=webcam.embed_url,
            script_url=webcam.script_url,
            watch_url=webcam.watch_url,
            channel_id=webcam.channel_id,
            note=webcam.note,
            source=presentation_link(webcam.source),
            alternative=presentation_link(webcam.alternative),
        )

    def app_location_bootstrap(
        cfg: config_lib.LocationConfig,
    ) -> AppBootstrapLocation:
        """Build presentation bootstrap metadata for a location."""
        temp_enabled = cfg.temp_source is not None and cfg.temp_source.live_enabled
        tides_enabled = cfg.tide_source is not None
        currents_enabled = cfg.currents_source is not None
        webcam_enabled = cfg.presentation.webcam is not None
        transit_enabled = cfg.presentation.transit is not None

        webcam = webcam_config(cfg.presentation.webcam)
        transit_routes: list[TransitRouteConfig] = []
        if cfg.presentation.transit is not None:
            transit_routes = [
                TransitRouteConfig(
                    label=route.label,
                    goodservice_route_id=route.goodservice_route_id,
                    goodservice_direction=route.goodservice_direction,
                    icon_url=route.icon_url,
                )
                for route in cfg.presentation.transit.routes
            ]

        return AppBootstrapLocation(
            metadata=AppLocationMetadata(
                code=cfg.code,
                name=cfg.name,
                nav_label=cfg.name,
                swim_location=cfg.swim_location,
                swim_location_link=cfg.swim_location_link,
                description=cfg.description,
                latitude=cfg.latitude,
                longitude=cfg.longitude,
                timezone=timezone_name(cfg),
                features=AppFeatureFlags(
                    temperature=temp_enabled,
                    tides=tides_enabled,
                    currents=currents_enabled,
                    webcam=webcam_enabled,
                    transit=transit_enabled,
                    windy=True,
                ),
                temperature_plots=AppTemperaturePlotConfig(
                    live=temp_enabled,
                    historic=(
                        cfg.temp_source is not None and cfg.temp_source.historic_enabled
                    ),
                ),
                citations=AppSourceCitations(
                    temperature=cfg.temp_source.citation if cfg.temp_source else None,
                    tides=cfg.tide_source.citation if cfg.tide_source else None,
                    currents=(
                        cfg.currents_source.citation if cfg.currents_source else None
                    ),
                ),
            ),
            integrations=AppExternalIntegrations(
                webcam=webcam,
                transit_routes=transit_routes,
                transit_source=(
                    presentation_link(cfg.presentation.transit.source)
                    if cfg.presentation.transit
                    else None
                ),
            ),
        )

    @app.get("/api/app/bootstrap", response_model=AppBootstrapResponse)
    async def app_bootstrap() -> AppBootstrapResponse:
        """Return non-secret presentation metadata for the React app."""
        location_order = list(config_lib.CONFIGS.keys())
        return AppBootstrapResponse(
            app_name=APP_NAME,
            short_name=APP_SHORT_NAME,
            default_location_code=config_lib.DEFAULT_LOCATION_CODE,
            location_order=location_order,
            source_code_link=AppPresentationLink(
                label="jeremywhelchel/shallweswim",
                url=GITHUB_SOURCE_URL,
                description="Site source on github:",
            ),
            locations={
                code: app_location_bootstrap(cfg)
                for code, cfg in config_lib.CONFIGS.items()
            },
        )

    @app.get("/api/{location}/conditions", response_model=LocationConditions)
    async def location_conditions(
        location: str, shift: int = 0, at: str | None = None
    ) -> LocationConditions:
        """API endpoint that returns tide and temperature data for a specific location.

        Args:
            location: Location code (e.g., 'nyc')
            shift: Time shift in minutes from current time (optional)
            at: Location-local ISO-8601 timestamp within 24 hours; overrides shift

        Returns:
            JSON response with tide and temperature information

        Raises:
            HTTPException: If the location is not configured
        """
        logging.info(
            f"[{location}] Processing conditions request with shift={shift}, at={at}"
        )
        ctx = resolve_location_time(app, location, shift=shift, at=at)

        # Check if location has data before attempting to serve
        # Use has_data (not ready) to serve stale data during brief refresh windows
        # Background updater handles freshness - user-facing endpoints serve any available data
        if not ctx.data_manager.has_data:
            logging.warning(f"[{location}] No data available for conditions request")
            raise HTTPException(
                status_code=503,
                detail=f"{ctx.cfg.name} data temporarily unavailable",
            )

        # Return structured response using Pydantic models
        return LocationConditions(
            location=api_location_info(location, ctx.cfg),
            temperature=api_temperature_info(ctx.cfg, ctx.data_manager),
            tides=api_conditions_tide_info(ctx),
            current=api_conditions_current_info(ctx),
        )

    @app.get("/api/{location}/plots/live_temps")
    async def get_live_temps_plot(location: str) -> fastapi.responses.Response:
        """Serve the live temperature plot for the specified location.

        Args:
            location: Location code (e.g., "nyc", "sfo")

        Returns:
            SVG image response with live temperature visualization
        """
        logging.info(f"[{location}] Processing live temps plot request")
        validate_location(location)

        data_manager = app.state.data_managers[location]
        plot_bytes = data_manager.get_plot(PLOT_LIVE_TEMPS)

        if plot_bytes is None:
            raise HTTPException(
                status_code=503,
                detail=f"Live temperature plot not yet available for {location}",
            )

        return fastapi.responses.Response(
            content=plot_bytes, media_type="image/svg+xml"
        )

    @app.get("/api/{location}/plots/historic_temps")
    async def get_historic_temps_plot(
        location: str, period: str = "2mo"
    ) -> fastapi.responses.Response:
        """Serve a historic temperature plot for the specified location.

        Args:
            location: Location code (e.g., "nyc", "sfo")
            period: Time period - "2mo" for 2-month or "12mo" for full year

        Returns:
            SVG image response with historic temperature visualization
        """
        logging.info(
            f"[{location}] Processing historic temps plot request, period={period}"
        )
        validate_location(location)

        period_to_plot: dict[str, PlotName] = {
            "2mo": PLOT_HISTORIC_TEMPS_2MO,
            "12mo": PLOT_HISTORIC_TEMPS_12MO,
        }
        plot_name = period_to_plot.get(period)
        if plot_name is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period '{period}'. Must be '2mo' or '12mo'.",
            )

        data_manager = app.state.data_managers[location]
        plot_bytes = data_manager.get_plot(plot_name)

        if plot_bytes is None:
            raise HTTPException(
                status_code=503,
                detail=f"Historic temperature plot ({period}) not yet available for {location}",
            )

        return fastapi.responses.Response(
            content=plot_bytes, media_type="image/svg+xml"
        )

    @app.get("/api/{location}/plots/current_tide")
    async def get_current_tide_plot(
        location: str, shift: int = 0, at: str | None = None
    ) -> fastapi.responses.Response:
        """Generate and serve a tide and current plot for the specified location.

        Args:
            location: Location code (e.g., "nyc", "san")
            shift: Time shift in minutes from current time
            at: Location-local ISO-8601 timestamp within 24 hours; overrides shift

        Returns:
            SVG image response with tide and current visualization
        """
        logging.info(
            f"[{location}] Processing current tide plot request with shift={shift}, at={at}"
        )
        ctx = resolve_location_time(app, location, shift=shift, at=at)

        # Check both required feeds have data before attempting to generate plot
        if not (
            ctx.data_manager.has_feed_data(FEED_TIDES)
            and ctx.data_manager.has_feed_data(FEED_CURRENTS)
        ):
            raise HTTPException(
                status_code=503,
                detail=f"{ctx.cfg.name} tide/current data temporarily unavailable",
            )

        try:
            tides_data = ctx.data_manager.get_feed_values(FEED_TIDES)
            currents_data = ctx.data_manager.get_feed_values(FEED_CURRENTS)

            if len(tides_data) < 2 or len(currents_data) < 2:
                logging.warning(
                    f"[{location}] Insufficient tide/current data for plot generation"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"{ctx.cfg.name} tide/current data temporarily unavailable",
                )

            # TODO: Cache or precompute common shift values for this plot. This
            # endpoint is the remaining user-facing path that does Matplotlib
            # work per request, even though it is offloaded to the process pool.
            pool = app.state.process_pool
            loop = asyncio.get_running_loop()
            fig = await asyncio.wait_for(
                loop.run_in_executor(
                    pool,
                    _create_tide_current_plot,  # Function to run (lazy imports plot)
                    tides_data,  # Argument 1
                    currents_data,  # Argument 2
                    ctx.timestamp,  # Argument 3
                    ctx.cfg,  # Argument 4
                ),
                timeout=PLOT_TIMEOUT,
            )
        except TimeoutError as e:
            logging.error(
                f"[{location}] Plot generation timed out after {PLOT_TIMEOUT}s"
            )
            raise HTTPException(
                status_code=503, detail="Plot generation timed out"
            ) from e
        except DataUnavailableError as e:
            raise HTTPException(
                status_code=503,
                detail=f"{ctx.cfg.name} tide/current data temporarily unavailable",
            ) from e
        except (RuntimeError, ValueError) as e:
            logging.exception(f"[{location}] Internal plot generation error")
            raise HTTPException(
                status_code=500, detail="Internal server error generating plot"
            ) from e

        # Convert figure to SVG in a StringIO buffer
        svg_io = io.StringIO()
        fig.savefig(svg_io, format="svg", bbox_inches="tight", transparent=False)
        svg_io.seek(0)

        return fastapi.responses.Response(
            content=svg_io.getvalue(), media_type="image/svg+xml"
        )

    @app.get("/api/health", status_code=200)
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

        # Missing manager state means startup/config initialization failed.
        if not hasattr(app.state, "data_managers") or not app.state.data_managers:
            logging.error("[api] No location data managers initialized")
            raise HTTPException(
                status_code=500,
                detail="Internal server error - no location data managers initialized",
            )

        # Return status for each location
        status_dict = {
            code: manager.status for code, manager in app.state.data_managers.items()
        }
        return status_dict

    @app.get("/api/locations", response_model=list[LocationSummary])
    async def list_locations() -> list[LocationSummary]:
        """API endpoint that returns a list of all configured swimming locations.

        Returns a summary of each location including code, name, coordinates,
        and whether data is currently available.

        Returns:
            List of LocationSummary objects for all configured locations
        """
        logging.info("[api] Processing locations list request")

        locations = []
        for code, cfg in config_lib.CONFIGS.items():
            # Check if location has data available
            has_data = False
            if code in app.state.data_managers:
                manager = app.state.data_managers[code]
                if manager is not None:
                    has_data = manager.has_data

            locations.append(
                LocationSummary(
                    code=cfg.code,
                    name=cfg.name,
                    swim_location=cfg.swim_location,
                    latitude=cfg.latitude,
                    longitude=cfg.longitude,
                    has_data=has_data,
                )
            )

        return locations

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
            logging.error(f"[{location}] Configured location missing data manager")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error - location '{location}' data manager missing",
            )

        # Return the status dictionary for this location
        status_obj: LocationStatus = app.state.data_managers[location].status
        return status_obj

    @app.get("/api/{location}/currents", response_model=CurrentsResponse)
    async def location_currents(
        location: str, shift: int = 0, at: str | None = None
    ) -> CurrentsResponse:
        """API endpoint that returns current predictions for a specific location.

        Args:
            location: Location code (e.g., 'nyc')
            shift: Time shift in minutes from current time (optional)
            at: Location-local ISO-8601 timestamp within 24 hours; overrides shift

        Returns:
            CurrentsResponse object with current prediction details

        Raises:
            HTTPException: If the location is not configured or doesn't support currents
        """
        logging.info(
            f"[{location}] Processing currents request with shift={shift}, at={at}"
        )
        location_context = resolve_location_context(app, location)

        # Check if this location supports current predictions
        if not location_context.cfg.currents_source:
            raise HTTPException(
                status_code=404,
                detail=f"Location '{location}' does not support current predictions",
            )

        # Only PREDICTION-type sources support time-shifted current predictions
        if (
            location_context.cfg.currents_source.source_type
            != types.DataSourceType.PREDICTION
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Current predictions for '{location}' are not available (observation-only)",
            )

        ctx = resolve_location_time(app, location, shift=shift, at=at)
        resolved_shift = ctx.time_query.shift_minutes

        try:
            # Get current prediction information
            current_info = ctx.data_manager.predict_flow_at_time(ctx.timestamp)
        except DataUnavailableError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

        # Get fwd/back shift values for navigation
        fwd = min(resolved_shift + 60, util.MAX_SHIFT_LIMIT)
        back = max(resolved_shift - 60, util.MIN_SHIFT_LIMIT)

        current_prediction = api_current_info(current_info, magnitude_digits=1)

        # Get chart data only if this location has chart assets configured
        legacy_chart = None
        current_chart_filename = None
        if ctx.cfg.currents_source.has_static_charts:
            try:
                chart_info = ctx.data_manager.get_chart_info(ctx.timestamp)
            except DataUnavailableError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            legacy_chart = LegacyChartInfo(
                hours_since_last_tide=round(chart_info.hours_since_last_tide, 1),
                last_tide_type=chart_info.last_tide_type,
                chart_filename=chart_info.chart_filename,
                map_title=chart_info.map_title,
            )
            if (
                current_info.direction is not None
                and current_info.magnitude_pct is not None
            ):
                current_chart_filename = util.get_current_chart_filename(
                    current_info.direction.value,
                    util.bin_magnitude(current_info.magnitude_pct),
                    location_code=location,
                )

        plot_query = (
            urllib.parse.urlencode({"at": ctx.time_query.at})
            if ctx.time_query.at is not None
            else urllib.parse.urlencode({"shift": resolved_shift})
        )
        navigation = NavigationInfo(
            shift=resolved_shift,
            next_hour=fwd,
            prev_hour=back,
            current_api_url=f"/api/{location}/currents",
            plot_url=f"/api/{location}/plots/current_tide?{plot_query}",
            at=ctx.time_query.at,
        )

        # Return structured response
        return CurrentsResponse(
            location=api_location_info(location, ctx.cfg),
            timestamp=ctx.timestamp.isoformat(),
            current=current_prediction,
            legacy_chart=legacy_chart,
            current_chart_filename=current_chart_filename,
            navigation=navigation,
        )

    @app.get(
        "/api/{loc}/data/{feed_name}",
        include_in_schema=False,
        # TODO: Re-enable response_model=pa_typing.DataFrame[TimeSeriesDataModel]
        # Removed due to FastAPI ResponseValidationError when validating specific
        # feed DataFrames (e.g., WaterTempDataModel) against the generic TimeSeriesDataModel.
        # Internal validation happens in feed.values anyway.
    )
    async def get_debug_feed_data(loc: str, feed_name: str):  # type: ignore[no-untyped-def]
        # Originally returned pd.DataFrame, now dict for consistent serialization
        """Debug endpoint for raw cached feed data at a given location.

        This endpoint is intentionally excluded from OpenAPI because it exposes
        internal feed cache shape rather than a stable public API contract.

        Args:
            loc: The location code (e.g., 'nyc').
            feed_name: The data feed name (e.g., 'tides', 'live_temps').

        Returns:
            A dict representation of the validated feed DataFrame.

        Raises:
            HTTPException(404): If the location or feed is not configured.
            HTTPException(503): If the feed is configured but data is unavailable.
            HTTPException(500): If app state is missing a manager for a configured location.
        """
        logging.info(f"Received request for feed '{feed_name}' at location '{loc}'")
        try:
            # Check if the location is configured.
            if not config_lib.get(loc):
                logging.warning(
                    f"Location '{loc}' not found for feed '{feed_name}' request."
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Location '{loc}' not found",
                )

            if (
                not hasattr(app.state, "data_managers")
                or loc not in app.state.data_managers
            ):
                logging.error(
                    f"Configured location '{loc}' has no data manager for feed '{feed_name}' request."
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error - location '{loc}' data manager missing",
                )
            location_data_manager = app.state.data_managers[loc]

            # Check if feed exists for the location
            if not location_data_manager.has_feed(feed_name):
                logging.warning(f"Feed '{feed_name}' not found for location '{loc}'.")
                raise HTTPException(
                    status_code=404,
                    detail=f"Feed '{feed_name}' not found for location '{loc}'.",
                )

            try:
                df = location_data_manager.get_feed_values(feed_name)
            except DataUnavailableError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Feed '{feed_name}' data temporarily unavailable for location '{loc}'",
                ) from e

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

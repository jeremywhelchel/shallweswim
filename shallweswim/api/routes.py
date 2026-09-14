"""API handlers for ShallWeSwim application.

This module contains FastAPI route handlers for the API endpoints.

Every route serves the generation the instance has loaded: it resolves a
location's manager from `app.state.snapshot` on each request, so the generation
an elected request loads is visible to the next one. Nothing here contacts a
provider.
"""

# Standard library imports
import asyncio
import dataclasses
import datetime
import html
import io
import logging
import urllib.parse
from collections.abc import Mapping
from typing import Literal

# Third-party imports
import fastapi
from fastapi import HTTPException

# Local imports
from shallweswim import config as config_lib
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
    AppWindyConfig,
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
from shallweswim.core.serving import LocationServing

# The loaded generation lives in app.state.snapshot; see `serving_managers`.

_NO_MANAGERS: Mapping[str, LocationServing] = {}


def serving_managers(app: fastapi.FastAPI) -> Mapping[str, LocationServing]:
    """Return the loaded generation's per-location managers.

    Read on every request rather than captured once, so a generation an elected
    request loads is served by the next request. The mapping is empty while no
    generation is loaded, which every caller answers as unavailable data.
    """
    state = getattr(app.state, "snapshot", None)
    return _NO_MANAGERS if state is None else state.managers


def location_status_response(
    app: fastapi.FastAPI, manager: LocationServing
) -> LocationStatus:
    """Stamp one location's per-feed status with the generation it came from.

    The three generation fields are the same for every location of one
    response: an instance serves one whole generation at a time. They are None
    only when no generation is loaded, and then there is no manager to report.
    """
    state = getattr(app.state, "snapshot", None)
    if state is None:
        return manager.status
    return manager.status.model_copy(
        update={
            "generation_id": state.generation_id,
            "published_at": state.published_at,
            "loaded_at": state.loaded_at,
        }
    )


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


def _create_tide_plot(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Wrapper that lazily imports plot module for subprocess execution."""
    from shallweswim import plot

    return plot.create_tide_plot(*args, **kwargs)


# Timeout for on-demand plot generation (seconds)
# Shorter than background (60s) since user is waiting
PLOT_TIMEOUT = 30.0
APP_NAME = "shall we swim?"
APP_SHORT_NAME = "shallweswim"
APP_THEME_COLOR = "#000099"
APP_BACKGROUND_COLOR = "#000099"

GITHUB_SOURCE_URL = "https://github.com/jeremywhelchel/shallweswim"


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
    data_manager: LocationServing


@dataclasses.dataclass(frozen=True)
class ResolvedLocationTime:
    """Location request context after validating planner time parameters."""

    cfg: config_lib.LocationConfig
    data_manager: LocationServing
    time_query: util.EffectiveTimeQuery

    @property
    def timestamp(self) -> datetime.datetime:
        return self.time_query.timestamp


def resolve_location_context(
    app: fastapi.FastAPI,
    location: str,
) -> LocationRequestContext:
    """Resolve shared location config and manager for API routes.

    Raises:
        HTTPException: 404 if the location is not configured; 503 if the loaded
            generation does not carry it or carries no data for it. Serving
            stale data is deliberate: freshness is the publishing job's
            business, and a user-facing route serves whatever it has.
    """
    cfg = validate_location(location)
    data_manager = serving_managers(app).get(location)
    if data_manager is None or not data_manager.has_data:
        logging.warning(f"[{location}] No data available in the loaded generation")
        raise HTTPException(
            status_code=503,
            detail=f"{cfg.name} data temporarily unavailable",
        )
    return LocationRequestContext(cfg=cfg, data_manager=data_manager)


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
        time=tide.time,
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
    data_manager: LocationServing,
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
        timestamp=current_info.timestamp,
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
    data_manager: LocationServing,
) -> TemperatureInfo | None:
    """Build observed temperature data for the conditions endpoint."""
    if not (
        cfg.live_temp_source is not None
        and cfg.live_temp_source.live_enabled
        and data_manager.has_feed_data(FEED_LIVE_TEMPS)
    ):
        return None

    temp_reading = data_manager.get_current_temperature()
    water_temp_f = temp_reading.temperature
    water_temp_c = round(util.f_to_c(water_temp_f), 1)
    return TemperatureInfo(
        timestamp=temp_reading.timestamp,
        water_temp_f=water_temp_f,
        water_temp_c=water_temp_c,
        station_name=cfg.live_temp_source.name,
    )


def app_source_citations(cfg: config_lib.LocationConfig) -> AppSourceCitations:
    """Build presentation citations, de-duplicating equivalent temp sources."""
    temperature: str | None = None
    live_temperature: str | None = None
    historical_temperature: str | None = None
    location_info_label = cfg.location_info_source or cfg.swim_location
    for row in cfg.temperature_source_citations:
        if row.label == "Temperature":
            temperature = row.html
        elif row.label == "Live temperature":
            live_temperature = row.html
        elif row.label == "Historical temperature":
            historical_temperature = row.html

    return AppSourceCitations(
        temperature=temperature,
        live_temperature=live_temperature,
        historical_temperature=historical_temperature,
        location_info=(
            "Location info: "
            f'<a href="{html.escape(cfg.swim_location_link, quote=True)}" '
            'target="_blank" rel="noopener noreferrer">'
            f"{html.escape(location_info_label)}</a>"
        ),
        tides=cfg.tide_source.citation if cfg.tide_source else None,
        currents=cfg.currents_source.citation if cfg.currents_source else None,
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
        temp_enabled = (
            cfg.live_temp_source is not None and cfg.live_temp_source.live_enabled
        )
        historic_temp_enabled = (
            cfg.historic_temp_source is not None
            and cfg.historic_temp_source.historic_enabled
        )
        tides_enabled = cfg.tide_source is not None
        currents_enabled = cfg.currents_source is not None
        prediction_currents_enabled = (
            cfg.currents_source is not None
            and cfg.currents_source.source_type == types.DataSourceType.PREDICTION
        )
        water_movement_detail_plot_type: Literal["current_tide", "tide"] | None = None
        if tides_enabled:
            water_movement_detail_plot_type = (
                "current_tide" if prediction_currents_enabled else "tide"
            )
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
                nav_label=cfg.nav_label or cfg.name,
                swim_location=cfg.swim_location,
                swim_location_link=cfg.swim_location_link,
                description=cfg.description,
                latitude=cfg.latitude,
                longitude=cfg.longitude,
                timezone=timezone_name(cfg),
                default_temperature_unit=cfg.default_temperature_unit,
                temperature_note=cfg.temperature_note,
                temperature_source_at_swim_location=(
                    cfg.presentation.temperature_source_at_swim_location
                ),
                water_movement_note=cfg.water_movement_note,
                features=AppFeatureFlags(
                    temperature=temp_enabled,
                    tides=tides_enabled,
                    currents=currents_enabled,
                    water_movement_planning=(
                        tides_enabled or prediction_currents_enabled
                    ),
                    water_movement_detail=tides_enabled,
                    water_movement_detail_plot_type=water_movement_detail_plot_type,
                    webcam=webcam_enabled,
                    transit=transit_enabled,
                    windy=cfg.presentation.windy.enabled,
                ),
                temperature_plots=AppTemperaturePlotConfig(
                    live=temp_enabled,
                    historic=historic_temp_enabled,
                ),
                citations=app_source_citations(cfg),
            ),
            integrations=AppExternalIntegrations(
                webcam=webcam,
                transit_routes=transit_routes,
                transit_source=(
                    presentation_link(cfg.presentation.transit.source)
                    if cfg.presentation.transit
                    else None
                ),
                water_quality_info=presentation_link(
                    cfg.presentation.water_quality_info
                ),
                windy=(
                    AppWindyConfig(
                        overlay=cfg.presentation.windy.overlay,
                        product=cfg.presentation.windy.product,
                        level=cfg.presentation.windy.level,
                        zoom=cfg.presentation.windy.zoom,
                        metric_wind=cfg.presentation.windy.metric_wind,
                        metric_temp=cfg.presentation.windy.metric_temp,
                    )
                    if cfg.presentation.windy.enabled
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
        ctx = resolve_location_context(app, location)
        plot_bytes = ctx.data_manager.get_plot(PLOT_LIVE_TEMPS)

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

        ctx = resolve_location_context(app, location)
        plot_bytes = ctx.data_manager.get_plot(plot_name)

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

    @app.get("/api/{location}/plots/tide")
    async def get_tide_plot(
        location: str, shift: int = 0, at: str | None = None
    ) -> fastapi.responses.Response:
        """Generate and serve a tide-only plot for the specified location."""
        ctx = resolve_location_time(app, location, shift=shift, at=at)

        if not ctx.data_manager.has_feed_data(FEED_TIDES):
            raise HTTPException(
                status_code=503,
                detail=f"{ctx.cfg.name} tide data temporarily unavailable",
            )

        try:
            tides_data = ctx.data_manager.get_feed_values(FEED_TIDES)

            if len(tides_data) < 2:
                logging.warning(f"[{location}] Insufficient tide data for plot")
                raise HTTPException(
                    status_code=503,
                    detail=f"{ctx.cfg.name} tide data temporarily unavailable",
                )

            pool = app.state.process_pool
            loop = asyncio.get_running_loop()
            fig = await asyncio.wait_for(
                loop.run_in_executor(
                    pool,
                    _create_tide_plot,
                    tides_data,
                    ctx.timestamp,
                    ctx.cfg,
                ),
                timeout=PLOT_TIMEOUT,
            )
        except TimeoutError as e:
            logging.error(
                f"[{location}] Tide plot generation timed out after {PLOT_TIMEOUT}s"
            )
            raise HTTPException(
                status_code=503, detail="Plot generation timed out"
            ) from e
        except DataUnavailableError as e:
            raise HTTPException(
                status_code=503,
                detail=f"{ctx.cfg.name} tide data temporarily unavailable",
            ) from e
        except (RuntimeError, ValueError) as e:
            logging.exception(f"[{location}] Internal tide plot generation error")
            raise HTTPException(
                status_code=500, detail="Internal server error generating plot"
            ) from e

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

        Returns 200 once a generation is loaded and at least one of its
        locations can serve data (fresh or stale). Returns 503 while no
        generation is loaded, or if NO location in it has any data.

        This lenient check ensures single station outages don't mark the entire
        service unhealthy. For detailed per-feed health status, use /api/status.

        Returns:
            True if service can serve at least one location
            Status code 200 if healthy, 503 if not healthy
        """
        managers = serving_managers(app)
        if not managers:
            logging.warning("[api] No published generation loaded")
            raise HTTPException(
                status_code=503,
                detail="Service not healthy - no published generation loaded",
            )

        # Healthy if at least one location of the generation can serve data
        if any(manager.has_data for manager in managers.values()):
            return True

        logging.warning("[/api/healthy] No location has data available. Raising 503.")
        raise HTTPException(
            status_code=503,
            detail="Service not healthy - no location has data",
        )

    @app.get("/api/status", response_model=dict[str, LocationStatus])
    async def all_locations_status() -> dict[str, LocationStatus]:
        """API endpoint that returns status for every location of the generation.

        Returns:
            Dictionary mapping location codes to their status dictionaries,
            empty while no generation is loaded.
        """
        return {
            code: location_status_response(app, manager)
            for code, manager in serving_managers(app).items()
        }

    @app.get("/api/locations", response_model=list[LocationSummary])
    async def list_locations() -> list[LocationSummary]:
        """API endpoint that returns a list of all configured swimming locations.

        Returns a summary of each location including code, name, coordinates,
        and whether data is currently available.

        Returns:
            List of LocationSummary objects for all configured locations
        """
        managers = serving_managers(app)
        locations = []
        for code, cfg in config_lib.CONFIGS.items():
            # A configured location the loaded generation does not carry has
            # no data to offer, exactly as one whose feeds are all empty.
            manager = managers.get(code)
            has_data = manager is not None and manager.has_data

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
            HTTPException: If the location is not configured, or the loaded
                generation does not carry it.
        """
        ctx = resolve_location_context(app, location)
        return location_status_response(app, ctx.data_manager)

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
            at=ctx.timestamp if ctx.time_query.at is not None else None,
        )

        # Return structured response
        return CurrentsResponse(
            location=api_location_info(location, ctx.cfg),
            timestamp=ctx.timestamp,
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
            HTTPException(503): If the feed is configured but data is
                unavailable, or the loaded generation does not carry the
                location.
        """
        try:
            location_data_manager = resolve_location_context(app, loc).data_manager

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

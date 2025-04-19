#!/usr/bin/env python3

import contextlib
import datetime
from typing import Any, AsyncGenerator

import fastapi
from fastapi import HTTPException
from fastapi import responses
from fastapi import staticfiles
from fastapi import templating

import google.cloud.logging
import logging
import os
import uvicorn

from shallweswim import config
from shallweswim import data as data_lib
from shallweswim import plot


data = {}


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
    # XXX Try not to reload this if already there...
    for code, cfg in config.CONFIGS.items():
        data[code] = data_lib.Data(cfg)
        data[code].Start()
    yield
    # Nothing to cleanup, but it would happen here


app = fastapi.FastAPI(lifespan=lifespan)

app.mount(
    "/static", staticfiles.StaticFiles(directory="shallweswim/static"), name="static"
)

templates = templating.Jinja2Templates(directory="shallweswim/templates")


# TODO use some cookie to redirect to last used or saved location
@app.get("/")
async def index() -> responses.RedirectResponse:
    return responses.RedirectResponse("/nyc")


# XXX locationify properly
@app.get("/embed")
async def embed(request: fastapi.Request) -> responses.HTMLResponse:
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
    """Return the effective time for displaying charts based on query parameters."""

    # XXX Add an optional absolute time parameter here.
    t = data_lib.Now()

    # Clamp the shift limit
    shift = max(MIN_SHIFT_LIMIT, min(shift, MAX_SHIFT_LIMIT))
    t = t + datetime.timedelta(minutes=shift)

    return t


@app.get("/current_tide_plot")
async def current_tide_plot(shift: int = 0) -> responses.Response:
    ts = EffectiveTime(shift)
    image = plot.GenerateTideCurrentPlot(data["nyc"].tides, data["nyc"].currents, ts)
    assert image
    return responses.Response(content=image.getvalue(), media_type="image/svg+xml")


@app.get("/current")
async def water_current(
    request: fastapi.Request, shift: int = 0
) -> responses.HTMLResponse:
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
    return timestamp.strftime("%A, %B %-d at %-I:%M %p")


templates.env.filters["fmt_datetime"] = fmt_datetime


# XXX locationify


@app.get("/freshness")
async def freshness() -> dict[str, Any]:
    return data["nyc"].Freshness()


@app.get("/favicon.ico")
async def favicon() -> responses.RedirectResponse:
    return responses.RedirectResponse(
        "/static/favicon.ico",
    )
    # return redirect(url_for("static", filename="favicon.ico"))


@app.get("/robots.txt")
async def robots() -> responses.RedirectResponse:
    return responses.RedirectResponse(
        "/static/robots.txt",
    )
    # return redirect(url_for("static", filename="robots.txt"))


@app.get("/{location}")
async def index_w_location(
    request: fastapi.Request, location: str
) -> responses.HTMLResponse:
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


if __name__ == "__main__":  # Run uvicorn app directly
    logging.info("Running uvicorn app")
    uvicorn.run(
        "main:start_app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        log_level="info",
        reload=True,
    )

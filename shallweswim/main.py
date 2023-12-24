#!/usr/bin/env python3

import datetime
from flask import Flask, jsonify, Response, redirect, render_template, request, url_for
import google.cloud.logging
import logging
import os

from shallweswim import config
from shallweswim import data as data_lib
from shallweswim import plot


# XXX automate. or use lifetime setup thing or something
data = {code: data_lib.Data(cfg) for code, cfg in config.CONFIGS.items()}

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True


@app.route("/<location>")
def index_w_location(location: str):
    cfg = config.Get(location)
    current_time, current_temp = data[location].LiveTempReading()
    past_tides, next_tides = data[location].PrevNextTide()
    return render_template(
        "index.html",
        config=cfg,
        current_time=current_time,
        current_temp=current_temp,
        past_tides=past_tides,
        next_tides=next_tides,
    )


@app.route("/")
def index():
    return index_w_location("nyc")


@app.route("/embed")
def embed():
    current_time, current_temp = data["nyc"].LiveTempReading()
    past_tides, next_tides = data["nyc"].PrevNextTide()
    return render_template(
        "embed.html",
        config=config.Get("nyc"),
        current_time=current_time,
        current_temp=current_temp,
        past_tides=past_tides,
        next_tides=next_tides,
    )


MIN_SHIFT_LIMIT = -180  # 3 hours
MAX_SHIFT_LIMIT = 1260  # 21 hours


def EffectiveTime() -> datetime.datetime:
    """Return the effective time for displaying charts based on query parameters."""

    # XXX Add an optional absolute time parameter here.
    t = data_lib.Now()

    # Optionally shift current time
    shift = request.args.get("shift", 0, int)
    # Clamp the shift limit
    shift = max(MIN_SHIFT_LIMIT, min(shift, MAX_SHIFT_LIMIT))
    t = t + datetime.timedelta(minutes=shift)

    return t


@app.route("/current_tide_plot")
def current_tide_plot():
    ts = EffectiveTime()
    image = plot.GenerateTideCurrentPlot(data["nyc"].tides, data["nyc"].currents, ts)
    return Response(image, mimetype="image/svg+xml")


@app.route("/current")
def water_current():
    ts = EffectiveTime()

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
    shift = request.args.get("shift", 0, int)
    fwd = min(shift + 60, MAX_SHIFT_LIMIT)
    back = max(shift - 60, MIN_SHIFT_LIMIT)

    return render_template(
        "current.html",
        config=config.Get("nyc"),
        last_tide_hrs_ago=round(last_tide_hrs_ago, 1),
        last_tide_type=last_tide_type,
        tide_chart_filename=tide_chart_filename,
        legacy_map_title=legacy_map_title,
        ts=ts,
        ef=ef,
        magnitude=round(magnitude, 1),
        msg=msg,
        fwd=fwd,
        back=back,
        current_chart_filename=plot.GetCurrentChartFilename(
            ef, plot.BinMagnitude(magnitude_pct)
        ),
        query_string=request.query_string.decode(),
    )


@app.template_filter()
def fmt_datetime(timestamp):
    return timestamp.strftime("%A, %B %-d at %-I:%M %p")


@app.route("/freshness")
def freshness():
    return jsonify(data["nyc"].Freshness())


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.ico"))


def start_app():
    # If running in Google Cloud Run, use cloud logging
    if "K_SERVICE" in os.environ:
        # Setup Google Cloud logging
        # By default this captures all logs at INFO level and higher
        log_client = google.cloud.logging.Client()
        log_client.get_default_handler()
        log_client.setup_logging()
        logging.info("Using google cloud logging")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Using standard logging")

    logging.info("Starting app")
    for d in data.values():
        d.Start()
    return app


if __name__ == "__main__":  # Run Flask dev-server directly
    logging.info("Running app.run()")
    start_app()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

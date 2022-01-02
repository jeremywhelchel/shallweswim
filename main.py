#!/usr/bin/env python3

import datetime
from flask import Flask, jsonify, Response, redirect, render_template, request, url_for
import google.cloud.logging
import logging
import os

import data as data_lib

data = data_lib.Data()

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True


@app.route("/")
def index():
    current_time, current_temp = data.LiveTempReading()
    past_tides, next_tides = data.PrevNextTide()
    return render_template(
        "index.html",
        current_time=current_time,
        current_temp=current_temp,
        past_tides=past_tides,
        next_tides=next_tides,
    )


@app.route("/current")
def water_current():
    # Optionally shift current time
    t = data_lib.Now()
    shift = request.args.get("shift")
    if shift:
        shift = int(shift)
        t = t + datetime.timedelta(minutes=shift)
    else:
        shift = 0
    fwd = shift + 60
    back = shift - 60

    # XXX
    data_lib.GenerateTideCurrentPlot(data.tides, data.currents, t)

    (
        last_tide_hrs_ago,
        last_tide_type,
        tide_chart_filename,
        legacy_map_title,
    ) = data.LegacyChartInfo(t)

    (
        ts,
        ef,
        magnitude,
        magnitude_pct,
        msg,
    ) = data.CurrentPrediction(t)

    data_lib.GenerateCurrentChart(ef, magnitude_pct)  # XXX pass in magnitude and e/f

    return render_template(
        "current.html",
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
    )


@app.template_filter()
def fmt_datetime(timestamp):
    return timestamp.strftime("%A, %B %-d at %-I:%M %p")


@app.route("/freshness")
def freshness():
    return jsonify(data.Freshness())


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
    data.Start()
    return app


if __name__ == "__main__":  # Run Flask dev-server directly
    logging.info("Running app.run()")
    start_app()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

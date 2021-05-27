#!/usr/bin/env python3

from flask import Flask, jsonify, Response, redirect, render_template, url_for
import google.cloud.logging
import logging
import os

import data as data_lib

data = data_lib.Data()

app = Flask(__name__)


@app.route("/")
def index():
    current_time, current_temp = data.CurrentReading()
    past_tides, next_tides = data.PrevNextTide()
    return render_template(
        "index.html",
        current_time=current_time,
        current_temp=current_temp,
        past_tides=past_tides,
        next_tides=next_tides,
    )


@app.template_filter()
def fmt_datetime(timestamp):
    return timestamp.strftime("%A, %B %-d at %-I:%M %p")


@app.route("/freshness")
def freshness():
    return jsonify(data.Freshness())


@app.route("/plot")
def plot():
    plot = data.LiveTempPlot()
    return Response(plot, mimetype="image/svg+xml")


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.ico"))


# XXX This is used for gunicorn.
def start_app():
    logging.getLogger().setLevel(logging.INFO)  # XXX Main thing
    gunicorn_logger = logging.getLogger('gunicorn.info')
    gunicorn_logger.setLevel(logging.INFO)
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info('INFO LOG')
    app.logger.warning('WARN LOG')
    gunicorn_logger.warn('WARN LOG')

    logging.info('Starting app')
    app.logger.info('App logger')
    # XXX There may be a better way to kick off this thread in gunicorn
    data.Start()
    return app


if __name__ == "__main__":  # Run Flask dev-server directly
    # Setup Google Cloud logging
    # By default this captures all logs at INFO level and higher
    log_client = google.cloud.logging.Client()
    log_client.get_default_handler()
    log_client.setup_logging()

    logging.info("Running app.run()")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

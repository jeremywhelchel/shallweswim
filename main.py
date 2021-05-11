from flask import Flask, Response, redirect, render_template, send_file, url_for
from google.cloud import storage
import matplotlib.dates as md
from matplotlib.figure import Figure

import base64
import datetime
import io
import logging
import os
import pandas as pd
import pytz
import seaborn as sns
import urllib

app = Flask(__name__)

def load_temps():
    # XXX Fetch live data. not from bucket
    print('Loading data')
    logging.info('Loading data')
    client = storage.Client()
    bucket = client.get_bucket('watertemp')
    blob = bucket.get_blob('full_data.parquet')
    blob_str = blob.download_as_string()
    buf = io.BytesIO(blob_str)
    temps = pd.read_parquet(buf)
    print('Loaded data:\n', temps.head())
    return temps


def make_plot(temps) -> bytes:
    data = temps.loc['2020-04':'2020-05'].rolling(3*24).mean().dropna(how='all')
    data.index = data.index.map(lambda ts: ts.replace(year=2021))

    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(16,8))
    #fig = Figure(figsize=(12,6))
    ax = fig.subplots()

    sns.set_theme()
    sns.axes_style('darkgrid')
    print(sns.axes_style())

    ax = sns.lineplot(data=data, ax=ax)
    ax.set_title('Battery NYC Water Tempeatures April+May 2011-2021')
    ax.set(xlabel='date', ylabel='water temp')
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))

    # Current year
    line = ax.lines[10]
    line.set_linewidth(3)
    line.set_linestyle('-')
    line.set_color('r')

    line = ax.legend().get_lines()[-1]
    line.set_linewidth(3)
    line.set_linestyle('-')
    line.set_color('r')

    # Save it to a temporary buffer.
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    return buf.getvalue()


historic_temps = None
historic_plot = None

# XXX Make flask data refresh thread...
# every ~hour here?
def reload_data():
    print('Reloading data')
    global historic_temps, historic_plot
    historic_temps = load_temps()
    historic_plot = make_plot(historic_temps)
    print('Reloading data DONE')

reload_data()


BATTERY_STATION = '8518750'
CONEY_STATION = '8517741'

BASE_URL = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
EASTERN_TZ = pytz.timezone('US/Eastern')


def recent_temps() -> pd.DataFrame:
    params = {
        'product': 'water_temperature',
        'application': 'jeremys-chart',
        'begin_date': (datetime.datetime.today() -
            datetime.timedelta(days=2)).strftime(format='%Y%m%d'),
        'end_date': datetime.datetime.today().strftime(format='%Y%m%d'),
        'station': BATTERY_STATION,
        'time_zone': 'GMT',  # XXX lst_ldt
        'units': 'english',
        #'interval': 'h',  6 min interval...
        'format': 'csv'
    }
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    print(url)
    ret = (
        pd.read_csv(url)
        .assign(time=lambda x: pd.to_datetime(x['Date Time'], utc=True))
        .set_index('time')
        # Convert UTC -> Eastern, then drop the timezone info
        .tz_convert(EASTERN_TZ)
        .tz_localize(None)
        .rename(columns={' Water Temperature': 'water_temp'})
    )
    print(ret)
    return ret


def current_reading() -> (pd.Timestamp, float):
    (time, temp), = (
        recent_temps()
        .tail(1)
        ['water_temp']
        .items()
    )
    return time, temp


def tides():
    base_url = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
    params = {
        'product': 'predictions',
        'datum': 'MLLW',
        'application': 'jeremys-chart',
        'begin_date': (datetime.datetime.today() -
            datetime.timedelta(days=1)).strftime(format='%Y%m%d'),
        'end_date': (datetime.datetime.today() +
            datetime.timedelta(days=1)).strftime(format='%Y%m%d'),
        'station': CONEY_STATION,
        'time_zone': 'lst_ldt',
        'units': 'english',
        'interval': 'hilo',  #6 min interval...
        'format': 'csv'
    }
    url = base_url + '?' + urllib.parse.urlencode(params)
    print(url)
    tides = (
        pd.read_csv(url)
        .assign(time=lambda x: pd.to_datetime(x['Date Time'], utc=True))#.dt.astimezone(EASTERN_TZ))
        .set_index('time')
        #.tz_convert(EASTERN_TZ)
        .tz_localize(None)
        .rename(columns={' Prediction': 'prediction', ' Type': 'type'})
        .assign(type=lambda x: x['type'].map({'L': 'low', 'H': 'high'}))
        [['prediction','type']]
    )
    now = datetime.datetime.now(tz=EASTERN_TZ).replace(tzinfo=None)
    past_tides = tides[:now].tail(1).reset_index().to_dict(orient='records')
    next_tides = tides[now:].head(2).reset_index().to_dict(orient='records')
    print(past_tides)
    print(next_tides)
    return past_tides, next_tides



@app.route("/")
def index():
    current_time, current_temp = current_reading()
    past_tides, next_tides = tides()
    return render_template('index.html',
            current_time=current_time,
            current_temp=current_temp,
            past_tides=past_tides,
            next_tides=next_tides,
            )


@app.template_filter()
def fmt_datetime(timestamp):
    return timestamp.strftime('%A, %B %-d at %-I:%M %p')


@app.route("/chart")
def chart():
    print(temps.head())
    data = temps.loc['2020-04':'2020-05'].rolling(3*24).mean().dropna(how='all')
    data.index = data.index.map(lambda ts: ts.replace(year=2021))
    print(data.head())
    return('<pre>%s</pre>' % str(data.head()))


@app.route("/plot")
def plot():
    return Response(historic_plot, mimetype='image/svg+xml')


@app.route("/versions")
def versions():
    pd.show_versions('versions')
    with open('versions') as fp:
        versions = fp.read()
    return('<pre>%s</pre>' % versions)


@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))


if __name__ == "__main__":
    print('Running app.run()')
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

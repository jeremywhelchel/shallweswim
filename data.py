"""Data fetching and management."""

from matplotlib.figure import Figure
from typing import Optional, Sequence, Tuple
import datetime
import io
import logging
import matplotlib.dates as md
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import threading
import time
import urllib


EASTERN_TZ = pytz.timezone("US/Eastern")

sns.set_theme()
sns.axes_style("darkgrid")


class NoaaApiError(Exception):
    """Error in a NOAA API Call."""


class NoaaApi(object):
    """Static class to fetch data from the NOAA Tides and Currents API.

    API is documented here: https://api.tidesandcurrents.noaa.gov/api/prod/
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    BASE_PARAMS = {
        "application": "shallweswim",
        "time_zone": "lst_ldt",
        "units": "english",
        "format": "csv",
    }
    STATIONS = {
        "battery": 8518750,
        "coney": 8517741,  # tide predictions only
        "coney_channel": "ACT3876",  # current predictions only
        "rockaway_inlet": "NYH1905",  # current predictions only
    }

    @classmethod
    def _Request(cls, params: dict) -> pd.DataFrame:
        url_params = dict(cls.BASE_PARAMS, **params)
        url = cls.BASE_URL + "?" + urllib.parse.urlencode(url_params)
        logging.info(f"NOAA API: {url}")
        try:
            df = pd.read_csv(url)
        except urllib.error.URLError as e:
            raise NoaaApiError(e)
        if len(df) == 1:
            raise NoaaApiError(df.iloc[0].values[0])
        return df

    @classmethod
    def Tides(
        cls,
        station: str = STATIONS["coney"],
    ) -> pd.DataFrame:
        """Return tide predictions from yesterday to two days from now."""
        return (
            cls._Request(
                {
                    "product": "predictions",
                    "datum": "MLLW",
                    "begin_date": (
                        datetime.datetime.today() - datetime.timedelta(days=1)
                    ).strftime("%Y%m%d"),
                    "end_date": (
                        datetime.datetime.today() + datetime.timedelta(days=2)
                    ).strftime("%Y%m%d"),
                    "station": station,
                    "interval": "hilo",
                }
            )
            .pipe(cls._FixTime)
            .rename(columns={" Prediction": "prediction", " Type": "type"})
            .assign(type=lambda x: x["type"].map({"L": "low", "H": "high"}))[
                ["prediction", "type"]
            ]
        )

    @classmethod
    def Currents(
        cls,
        station: str,
    ) -> pd.DataFrame:
        return (
            cls._Request(
                {
                    "product": "currents_predictions",
                    "datum": "MLLW",
                    "begin_date": (
                        datetime.datetime.today() - datetime.timedelta(days=1)
                    ).strftime("%Y%m%d"),
                    "end_date": (
                        datetime.datetime.today() + datetime.timedelta(days=2)
                    ).strftime("%Y%m%d"),
                    "station": station,
                    "interval": "MAX_SLACK",
                }
            )
            .rename(columns={"Time": "Date Time"})  # XXX param to FixTime
            .pipe(cls._FixTime)
            .rename(
                columns={
                    " Depth": "depth",
                    " Type": "type",
                    " Velocity_Major": "velocity",
                    " meanFloodDir": "mean_flood_dir",
                    " Bin": "bin",
                }
            )
            # Data is just flood/slack/ebb datapoints. This creates a smooth curve
            .resample("60s")
            .interpolate("polynomial", order=2)
            # XXX Normalize to proper interval to make mean make sense... (one is at depth, the other isn't...)
            # probably doesn't really matter though, tbh
        )

    @classmethod
    def Temperature(
        cls,
        product: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch buoy temperature dataset."""
        assert product in ["air_temperature", "water_temperature"], product
        return (
            cls._Request(
                {
                    "product": product,
                    "begin_date": begin_date.strftime("%Y%m%d"),
                    "end_date": end_date.strftime("%Y%m%d"),
                    "station": cls.STATIONS["battery"],
                    # No 'interval' specified...returns 6-minute intervals
                    "interval": interval,
                }
            )
            .pipe(cls._FixTime)
            .rename(
                columns={
                    " Water Temperature": "water_temp",
                    " Air Temperature": "air_temp",
                }
            )
            .drop(columns=[" X", " N", " R "])  # No idea what these mean
        )

    @classmethod
    def _FixTime(cls, df):
        return (
            df.assign(time=lambda x: pd.to_datetime(x["Date Time"], utc=True))
            .drop(columns="Date Time")
            .set_index("time")
            # Drop timezone info. Already in local time (LST/LDT in request)
            .tz_localize(None)
        )


def Now():
    return datetime.datetime.now(tz=EASTERN_TZ).replace(tzinfo=None)


def LatestTimeValue(df: Optional[pd.DataFrame]) -> Optional[datetime.datetime]:
    if df is None:
        return None
    return df.index[-1].to_pydatetime()


def PivotYear(df):
    """Move year dimension to columns."""
    df = df.assign(year=df.index.year)
    df.index = pd.to_datetime(
        # Use 2020-indexing because it's a leap year
        df.index.strftime("2020-%m-%d %H:%M:%S")
    )
    return df.set_index("year", append=True).unstack("year")


class Data(object):
    """Data management for shallweswim webpage."""

    def __init__(self):
        self.tides = None
        self.historic_temps = None
        self.live_temps = None

        self._tides_timestamp = None
        self._live_temps_timestamp = None
        self._historic_temps_timestamp = None

        self.expirations = {
            # XXX tidesandcurrents
            # Tidal predictions already cover a wide past/present window
            "tides": datetime.timedelta(hours=24),
            # Live temperature readings ouccur every 6 minutes, and are
            # generally already 5 minutes old when a new reading first appears.
            "live_temps": datetime.timedelta(minutes=10),
            # Hourly fetch historic temps + generate charts
            "historic_temps": datetime.timedelta(hours=3),
        }

    def _Expired(self, dataset: str) -> bool:
        age_seconds = self.Freshness()[dataset]["fetch"]["age_seconds"]
        return not age_seconds or (
            age_seconds > self.expirations[dataset].total_seconds()
        )

    def _Update(self):
        """Daemon thread to continuously updating data."""
        while True:
            if self._Expired("tides"):
                self._FetchTides()
                GenerateTideCurrentPlot(self.tides, self.currents)

            if self._Expired("live_temps"):
                self._FetchLiveTemps()
                GenerateLiveTempPlot(self.live_temps)

                # XXX Do currents here too...?

            # XXX Flag this
            if self._Expired("historic_temps"):
                self._FetchHistoricTemps()
                GenerateHistoricPlots(self.historic_temps)

            # XXX Can probably be increased to 1s even... but would need to add API spam buffer
            time.sleep(60)

    def Start(self):
        """Start the background data fetching process."""
        for thread in threading.enumerate():
            assert (
                thread.name != "DataUpdateThread"
            ), "Data update thread already running"

        logging.info("Starting data fetch thread")
        self._update_thread = threading.Thread(
            target=self._Update, name="DataUpdateThread", daemon=True
        )
        self._update_thread.start()

    def PrevNextTide(self):
        """Return previous tide and next two tides."""
        if self.tides is None:
            unknown = {"time": datetime.time(0)}
            return [unknown], [unknown, unknown]
        past_tides = self.tides[: Now()].tail(1).reset_index().to_dict(orient="records")
        next_tides = self.tides[Now() :].head(2).reset_index().to_dict(orient="records")
        return past_tides, next_tides

    def CurrentPrediction(self) -> Tuple[float, str]:
        # XXX cleanup
        t = Now()
        row = self.tides.loc[self.tides.index.asof(t)]
        offset = t - row.name
        offset_hrs = offset.seconds / (60 * 60)
        filename = "%s+%s.png" % (row["type"], round(offset_hrs))

        print("It is %0.1f hours since the last %s tide" % (offset_hrs, row["type"]))
        print(t, row["type"], offset, offset_hrs, filename)
        return offset_hrs, filename

    def LiveTempReading(self) -> Tuple[pd.Timestamp, float]:
        if self.live_temps is None:
            return datetime.time(0), 0.0
        ((time, temp),) = self.live_temps.tail(1)["water_temp"].items()
        return time, temp

    def Freshness(self):
        # XXX Consistent dtype
        # XXX EST timezone for timestamps
        ret = {
            "tides": {
                "fetch": {"time": self._tides_timestamp},
                "latest_value": {"time": LatestTimeValue(self.tides)},
            },
            "live_temps": {
                "fetch": {"time": self._live_temps_timestamp},
                "latest_value": {"time": LatestTimeValue(self.live_temps)},
            },
            "historic_temps": {
                "fetch": {"time": self._historic_temps_timestamp},
                "latest_value": {"time": LatestTimeValue(self.historic_temps)},
            },
        }

        # Calculate current ages
        now = Now()
        for dataset, info in ret.items():
            for label in list(info.keys()):
                freshness = info[label]["time"]
                if freshness:
                    age = now - freshness
                    age_sec = age.total_seconds()
                    age_str = str(datetime.timedelta(seconds=int(age_sec)))
                else:
                    age = None
                    age_sec = None
                    age_str = None
                ret[dataset][label]["age"] = age_str
                ret[dataset][label]["age_seconds"] = age_sec
        return ret

    def _FetchTides(self):
        logging.info("Fetching tides")
        try:
            self.tides = NoaaApi.Tides()

            # XXX Currents. Explain Mean
            currents_coney = NoaaApi.Currents(NoaaApi.STATIONS['coney_channel'])
            currents_ri = NoaaApi.Currents(NoaaApi.STATIONS['rockaway_inlet'])
            self.currents = pd.concat([currents_coney,
                currents_ri]).groupby(level=0).mean()

            self._tides_timestamp = Now()
        except NoaaApiError as e:
            logging.warning(f"Tide fetch error: {e}")

    def _FetchHistoricTemps(self):
        """Get hourly temp data since 2011."""
        logging.info("Fetching historic temps")
        try:
            year_frames = []
            for year in range(2011, Now().year + 1):
                begin_date = datetime.date(year, 1, 1)
                end_date = datetime.date(year, 12, 31)
                year_frames.append(
                    pd.concat(
                        [
                            NoaaApi.Temperature(
                                "air_temperature", begin_date, end_date, interval="h"
                            ),
                            NoaaApi.Temperature(
                                "water_temperature", begin_date, end_date, interval="h"
                            ),
                        ],
                        axis=1,
                    )
                )
            self.historic_temps = (
                pd.concat(year_frames)
                # These samples have erroneous data
                # XXX Find a way to identify / prune outliers automatically
                .drop(pd.to_datetime("2017-05-23 11:00:00"))
                .drop(pd.to_datetime("2017-05-23 12:00:00"))
                .drop(pd.to_datetime("2020-05-22 13:00:00"))
                .resample("H")
                .first()
            )
            self._historic_temps_timestamp = Now()
        except NoaaApiError as e:
            logging.warning(f"Historic temp fetch error: {e}")

    # XXX Test by disabling local wifi briefly
    def _FetchLiveTemps(self):
        """Get last N days of air and water temperatures."""
        logging.info("Fetching live temps")
        begin_date = datetime.datetime.today() - datetime.timedelta(days=8)
        end_date = datetime.datetime.today()
        # XXX Resample to 6min
        try:
            self.live_temps = (
                pd.concat(
                    [
                        NoaaApi.Temperature("air_temperature", begin_date, end_date),
                        NoaaApi.Temperature("water_temperature", begin_date, end_date),
                    ],
                    axis=1,
                )
                # Drop a bad reading
                # XXX Find an automated way to drop these solo outliers
                # .drop(pd.to_datetime("2021-05-18 22:24:00"))
            )
            self._live_temps_timestamp = Now()
            age = self.Freshness()["live_temps"]["latest_value"]["age"]
            logging.info(f"Fetched live temps. Last datapoint age: {age}")
        except NoaaApiError as e:
            logging.warning(f"Live temp fetch error: {e}")


def MultiYearPlot(df: pd.DataFrame, fig: Figure, title: str, subtitle: str):
    ax = sns.lineplot(data=df, ax=fig.subplots())

    fig.suptitle(title, fontsize=24)
    ax.set_title(subtitle, fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Water Temp (°F)", fontsize=18)

    # Current year
    # XXX When we have other than 10 years, need to select properly.
    line = ax.lines[10]
    line.set_linewidth(3)
    line.set_linestyle("-")
    line.set_color("r")

    line = ax.legend().get_lines()[-1]
    line.set_linewidth(3)
    line.set_linestyle("-")
    line.set_color("r")
    return ax


def LiveTempPlot(
    df: pd.DataFrame,
    fig: Figure,
    title: str,
    subtitle: str,
    time_fmt: str,
):
    ax = fig.subplots()
    sns.lineplot(data=df, ax=ax)
    ax.xaxis.set_major_formatter(md.DateFormatter(time_fmt))

    fig.suptitle(title, fontsize=24)
    ax.set_title(subtitle, fontsize=18)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Water Temp (°F)", fontsize=18)

    # This gets confusing to plot on a second axis, since temps don't align
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Air Temp', fontsize=18)
    # ax2.grid(False)
    # sns.lineplot(
    #     data=df["air_temp"],
    #     ax=ax2,
    #     color="r",
    # )
    return ax


def GenerateLiveTempPlot(live_temps):
    if live_temps is None:
        return
    plot_filename = "static/plots/live_temps.svg"
    logging.info("Generating live temp plot: %s", plot_filename)
    raw = live_temps["water_temp"]
    trend = raw.rolling(10 * 2, center=True).mean()
    df = pd.DataFrame(
        {
            "live": raw,
            "trend (2-hr)": trend,
        }
    ).tail(10 * 24 * 2)
    fig = Figure(figsize=(16, 8))
    LiveTempPlot(
        df,
        fig,
        "Battery NYC Water Temperature",
        "48-hour, live",
        "%a %-I %p",
    )
    fig.savefig(plot_filename, format="svg", bbox_inches='tight')


def GenerateHistoricPlots(hist_temps):
    if hist_temps is None:
        return
    year_df = PivotYear(hist_temps)

    # 2 Month plot
    two_mo_plot_filename = "static/plots/historic_temps_2mo_24h_mean.svg"
    logging.info("Generating 2 month plot: %s", two_mo_plot_filename)
    df = (
        year_df["water_temp"]
        .loc[
            Now().date().replace(year=2020)
            - datetime.timedelta(days=30) : Now().date().replace(year=2020)
            + datetime.timedelta(days=30)
        ]
        .rolling(24, center=True)
        .mean()
    )
    fig = Figure(figsize=(16, 8))
    ax = MultiYearPlot(
        df,
        fig,
        "Battery NYC Water Temperature",
        "2 month, all years, 24-hour mean",
    )
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))
    fig.savefig(two_mo_plot_filename, format="svg", bbox_inches='tight')

    # Full year
    yr_plot_filename = "static/plots/historic_temps_12mo_24h_mean.svg"
    logging.info("Generating full time plot: %s", yr_plot_filename)
    df = (
        year_df["water_temp"]
        .rolling(24, center=True)
        .mean()
        # Kludge to prevent seaborn from connecting over nan gaps.
        .fillna(np.inf)
    )
    fig = Figure(figsize=(16, 8))
    ax = MultiYearPlot(
        df,
        fig,
        "Battery NYC Water Temperature",
        "all years, 24-hour mean",
    )
    ax.xaxis.set_major_locator(md.MonthLocator(bymonthday=1))
    # X labels between gridlines
    ax.set_xticklabels("")
    ax.xaxis.set_minor_locator(md.MonthLocator(bymonthday=15))
    ax.xaxis.set_minor_formatter(md.DateFormatter("%b"))
    fig.savefig(yr_plot_filename, format="svg", bbox_inches='tight')


def GenerateTideCurrentPlot(tides, currents):
    if tides is None or currents is None:
        return
    plot_filename = "static/plots/tides_currents.svg"
    logging.info("Generating tide and current plot: %s", plot_filename)

    # XXX Do this directly in tide dataset?
    tides = tides.resample('60s').interpolate('polynomial', order=2)

    df = pd.DataFrame({
        'tide': tides['prediction'],
        'tide_type': tides['type'],
        'current': currents['velocity'],
    })
    df = df[Now()-datetime.timedelta(hours=3):
            Now()+datetime.timedelta(hours=9)
            ]

    fig = Figure(figsize=(16,8))

    ax = fig.subplots()

    sns.lineplot(data=df['tide'], ax=ax, color='b')
    ax.set_ylabel('Tide (ft)', color='b')

    # XXX Align the 0 line on both

    ax2=ax.twinx()
    ax2.xaxis.set_major_formatter(md.DateFormatter('%a %-I %p'))
    sns.lineplot(data=df['current'],
                 ax=ax2, color='g')
    ax2.grid(False)
    ax2.set_ylabel('Current (kts)', color='g')

    # Attempts to line up 0 on both axises...
    #ax.set_ylim(-5,5)  naturally -1,5
    #ax2.set_ylim(-2,2)  naturally -1.5 to 1

    # Draw lines at 0
    ax.axhline(0, color='b', linestyle=':', alpha=0.8)#, linewidth=0.8)
    ax2.axhline(0, color='g', linestyle=':', alpha=0.8)#, linewidth=0.8)

    ax.axvline(Now(), color='r')

    # Useful plot that indicates how the current (which?) LEADS the tide
    # XXX
    # Peak flood ~2hrs before high tide
    # Peak ebb XX mins before low tide

    # Label high and low tide points
    tt = df[df['tide_type'].notnull()][['tide','tide_type']]
    tt['tide_type']=tt['tide_type'] + ' tide'
    sns.scatterplot(data=tt, ax=ax, legend=False)
    for t, row in tt.iterrows():
      ax.annotate(row['tide_type'], (t, row['tide']), color='b', xytext=(-24,8),
              textcoords='offset pixels')

    fig.savefig(plot_filename, format="svg", bbox_inches='tight')

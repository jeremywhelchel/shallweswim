#!/usr/bin/env python3
"""Data fetching and management."""

from matplotlib.figure import Figure
from typing import Sequence, Tuple
import datetime
import io
import logging
import matplotlib.dates as md
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import time
import urllib


EASTERN_TZ = pytz.timezone("US/Eastern")

sns.set_theme()
sns.axes_style("darkgrid")


class NoaaApi(object):
    """Static class to fetch data from the NOAA Tides and Currents API.

    API is documented here: https://api.tidesandcurrents.noaa.gov/api/prod/
    """

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    BASE_PARAMS = {
        "application": "sws.today",
        "time_zone": "lst_ldt",
        "units": "english",
        "format": "csv",
    }
    STATIONS = {
        "battery": 8518750,
        "coney": 8517741,
    }

    @classmethod
    def _Request(cls, params: dict) -> pd.DataFrame:
        url_params = dict(cls.BASE_PARAMS, **params)
        url = cls.BASE_URL + "?" + urllib.parse.urlencode(url_params)
        logging.info("Fetching NOAA API: %s", url)
        return pd.read_csv(url)

    @classmethod
    def Tides(cls) -> pd.DataFrame:
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
                    "station": cls.STATIONS["coney"],
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
    def Temperature(
        cls,
        product: str,
        begin_date: datetime.date,
        end_date: datetime.date,
        interval: str = None,
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


def PivotYear(df):
    """Move year dimension to columns."""
    df = df.assign(year=df.index.year)
    df.index = pd.to_datetime(
        # Use 2020-indexing because it's a leap year
        df.index.strftime("2020-%m-%d %H:%M:%S")
    )
    return df.set_index("year", append=True).unstack("year")


class Data(object):
    def __init__(self):
        self.tides = None

        self.live_temps = None
        self.live_temps_expiration = None

    def Start(self):
        self._DailyFetch()

    def PrevNextTide(self):
        """Return previous tide and next two tides."""
        assert self.tides is not None
        past_tides = self.tides[: Now()].tail(1).reset_index().to_dict(orient="records")
        next_tides = self.tides[Now() :].head(2).reset_index().to_dict(orient="records")
        return past_tides, next_tides

    # XXX Cache last temps and use those if Noaa call fails
    # XXX Test by disabling local wifi briefly
    def LiveTemps(self) -> pd.DataFrame:
        """Get last N days of air and water temperatures."""
        if self.live_temps is not None and (time.time() < self.live_temps_expiration):
            return self.live_temps

        begin_date = datetime.datetime.today() - datetime.timedelta(days=8)
        end_date = datetime.datetime.today()
        # XXX Resample to 6min
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
            .drop(pd.to_datetime("2021-05-18 22:24:00"))
        )
        self.live_temps_expiration = time.time() + 60
        return self.live_temps

    def CurrentReading(self) -> Tuple[pd.Timestamp, float]:
        ((time, temp),) = self.LiveTemps().tail(1)["water_temp"].items()
        return time, temp

    def LiveTempPlot(self) -> bytes:
        raw = self.LiveTemps()["water_temp"]
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
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        return buf.getvalue()

    def _DailyFetch(self):
        self.tides = NoaaApi.Tides()

    def HistoricalTemps(self):
        """Get hourly temp data since 2011."""
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
        return (
            pd.concat(year_frames)
            # These samples have erroneous data
            # XXX Find a way to identify / prune outliers automatically
            .drop(pd.to_datetime("2017-05-23 11:00:00"))
            .drop(pd.to_datetime("2017-05-23 12:00:00"))
            .drop(pd.to_datetime("2020-05-22 13:00:00"))
            .resample("H")
            .first()
        )


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


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    print("Local data execution.")

    data = Data()
    data.Start()

    print("Tides:")
    prev_tide, next_tide = data.PrevNextTide()
    print(prev_tide, next_tide)

    print("Live temps:")
    live_temps = data.LiveTemps()
    print(live_temps)


if __name__ == "__main__":
    main()

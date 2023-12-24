"""Data fetching and management."""

from concurrent import futures
from typing import Optional, Tuple
import datetime
import logging
import pandas as pd
import threading
import time

from shallweswim import config as config_lib
from shallweswim import noaa
from shallweswim import plot
from shallweswim import util


NOAA_STATIONS = {
    "battery": 8518750,  # temperature
    "coney": 8517741,  # tide predictions only
    "coney_channel": "ACT3876",  # current predictions only
    "rockaway_inlet": "NYH1905",  # current predictions only
}

Now = util.Now


def LatestTimeValue(df: Optional[pd.DataFrame]) -> Optional[datetime.datetime]:
    if df is None:
        return None
    return df.index[-1].to_pydatetime()


class Data(object):
    """Data management for shallweswim webpage."""

    def __init__(self, config: config_lib.LocationConfig):
        self.config = config

        self.tides = None
        self.currents = None
        self.historic_temps = None
        self.live_temps = None

        self._tides_timestamp = None
        self._live_temps_timestamp = None
        self._historic_temps_timestamp = None

        self.expirations = {
            # Tidal predictions already cover a wide past/present window
            "tides_and_currents": datetime.timedelta(hours=24),
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
            if self._Expired("tides_and_currents"):
                self._FetchTidesAndCurrents()

            if self._Expired("live_temps"):
                self._FetchLiveTemps()
                plot.GenerateLiveTempPlot(
                    self.live_temps, self.config.temp_station_name
                )

            if self._Expired("historic_temps"):
                self._FetchHistoricTemps()
                plot.GenerateHistoricPlots(
                    self.historic_temps, self.config.temp_station_name
                )

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

    def LegacyChartInfo(
        self, t: Optional[datetime.datetime] = None
    ) -> Tuple[float, str, str, str]:
        if not t:
            t = Now()
        row = self.tides.loc[self.tides.index.asof(t)]
        # XXX Break this into a function
        tide_type = row["type"]
        offset = t - row.name
        offset_hrs = offset.seconds / (60 * 60)
        if offset_hrs > 5.5:
            chart_num = 0
            INVERT = {"high": "low", "low": "high"}
            chart_type = INVERT[tide_type]
            # XXX last tide type will be wrong here in the chart
        else:
            chart_num = round(offset_hrs)
            chart_type = tide_type

        legacy_map_title = "%s Water at New York" % chart_type.capitalize()
        if chart_num:
            legacy_map_title = (
                "%i Hour%s after " % (chart_num, "s" if chart_num > 1 else "")
                + legacy_map_title
            )
        filename = "%s+%s.png" % (chart_type, chart_num)

        return offset_hrs, tide_type, filename, legacy_map_title

    def CurrentPrediction(
        self, t: Optional[datetime.datetime] = None
    ) -> Tuple[str, float, float, str]:
        if not t:
            t = Now()

        v = self.currents["velocity"]
        df = pd.DataFrame(
            {
                "v": v,
                "magnitude": v.abs(),
                "slope": v.abs().shift(-1) - v.abs().shift(1),
            }
        )  # XXX trend...
        df["ef"] = (df["v"] > 0).map({True: "flooding", False: "ebbing"})

        # XXX This doesn't work when df has a stronger current in a different
        # tidal cycle. Try to only look at +/- til we get to 0 or something
        df["mag_pct"] = df.groupby("ef")["magnitude"].rank(pct=True)

        row = df[t:].iloc[0]

        STRONG_THRESHOLD = 0.85  # 30% on either side of peak
        WEAK_THRESHOLD = 0.15  # 30% on either side of bottom
        magnitude = row["magnitude"]

        # XXX Move to template
        if row["mag_pct"] < WEAK_THRESHOLD:
            msg = "at its weakest (slack)"
        elif row["mag_pct"] > STRONG_THRESHOLD:
            msg = "at its strongest"
        elif row["slope"] < 0:
            msg = "getting weaker"
        elif row["slope"] > 0:
            msg = "getting stronger"
        else:
            raise ValueError(row)

        tstr = t.strftime("%A, %B %-d at %-I:%M %p")
        ef = row["ef"]

        # Return mag_pct, for determining arrow size
        return ef, magnitude, row["mag_pct"], msg

    def LiveTempReading(self) -> Tuple[pd.Timestamp, float]:
        if self.live_temps is None:
            return datetime.time(0), 0.0
        ((time, temp),) = self.live_temps.tail(1)["water_temp"].items()
        return time, temp

    def Freshness(self):
        # XXX Consistent dtype
        # XXX EST timezone for timestamps
        ret = {
            "tides_and_currents": {
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

    def _FetchTidesAndCurrents(self):
        logging.info("Fetching tides and currents")
        try:
            self.tides = noaa.NoaaApi.Tides(station=self.config.tide_station)

            currents = [
                noaa.NoaaApi.Currents(stn) for stn in self.config.currents_stations
            ]
            self.currents = pd.concat(currents)[["velocity"]].groupby(level=0).mean()

            self._tides_timestamp = Now()
        except noaa.NoaaApiError as e:
            logging.warning(f"Tide fetch error: {e}")

    def _FetchHistoricTempYear(self, year):
        begin_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        return pd.concat(
            [
                noaa.NoaaApi.Temperature(
                    self.config.temp_station,
                    "air_temperature",
                    begin_date,
                    end_date,
                    interval="h",
                ),
                noaa.NoaaApi.Temperature(
                    self.config.temp_station,
                    "water_temperature",
                    begin_date,
                    end_date,
                    interval="h",
                ),
            ],
            axis=1,
        )

    def _FetchHistoricTemps(self):
        """Get hourly temp data since 2011."""
        logging.info("Fetching historic temps")
        try:
            years = range(2011, Now().year + 1)
            threadpool = futures.ThreadPoolExecutor(len(years))
            year_frames = threadpool.map(self._FetchHistoricTempYear, years)
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
        except noaa.NoaaApiError as e:
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
                        noaa.NoaaApi.Temperature(
                            self.config.temp_station,
                            "air_temperature",
                            begin_date,
                            end_date,
                        ),
                        noaa.NoaaApi.Temperature(
                            self.config.temp_station,
                            "water_temperature",
                            begin_date,
                            end_date,
                        ),
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
        except noaa.NoaaApiError as e:
            logging.warning(f"Live temp fetch error: {e}")

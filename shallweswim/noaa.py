"""NOAA tides and current API client."""

from typing import Optional, Sequence, Tuple, Union
import datetime
import logging
import numpy as np
import os
import pandas as pd
import time
import urllib


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
        station: str | int,
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
        station: int | str,
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
                    "station": station,
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

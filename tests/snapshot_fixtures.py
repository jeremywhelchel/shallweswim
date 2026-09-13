"""Deterministic served frames and snapshots for the snapshot tests.

Frames have the shape the feeds publish: a naive station-local `time` index
and the columns each Pandera model requires.
"""

import datetime
import math

import numpy as np
import pandas as pd

from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.model import (
    FeedMetadata,
    FeedSnapshot,
    LocationSnapshot,
    PlotSnapshot,
    Snapshot,
)
from shallweswim.types import TIDE_TYPE_CATEGORIES

FETCHED_AT = datetime.datetime(2026, 6, 1, 12, 0, 5, tzinfo=datetime.UTC)


def _naive_index(times: list[datetime.datetime]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times, name="time")


def live_temps_frame(offset: float = 0.0) -> pd.DataFrame:
    """A day of six-minute readings; `offset` shifts every value."""
    start = datetime.datetime(2026, 6, 1)
    times = [start + datetime.timedelta(minutes=6 * i) for i in range(240)]
    values = [68.0 + offset + math.sin(i / 20) for i in range(240)]
    return pd.DataFrame({"water_temp": values}, index=_naive_index(times))


def historic_temps_frame() -> pd.DataFrame:
    """Ten days resampled hourly, as the historical feed publishes, with gaps."""
    start = datetime.datetime(2026, 1, 1)
    times = [start + datetime.timedelta(hours=i) for i in range(240) if i % 7 != 3]
    values = [50.0 + (i % 24) / 4 for i in range(len(times))]
    frame = pd.DataFrame({"water_temp": values}, index=_naive_index(times))
    frame.loc[frame.index[10:14], "water_temp"] = np.nan
    return frame.resample("h").mean()


def tides_frame() -> pd.DataFrame:
    start = datetime.datetime(2026, 6, 1, 3, 12)
    times = [start + datetime.timedelta(hours=6 * i, minutes=13 * i) for i in range(4)]
    return pd.DataFrame(
        {
            "prediction": [-0.5, 1.2, -0.3, 1.4],
            "type": pd.Categorical(
                ["low", "high", "low", "high"], categories=TIDE_TYPE_CATEGORIES
            ),
        },
        index=_naive_index(times),
    )


def currents_frame() -> pd.DataFrame:
    start = datetime.datetime(2026, 6, 1)
    times = [start + datetime.timedelta(minutes=i) for i in range(360)]
    values = [1.5 * math.sin(i * math.pi / 360) for i in range(360)]
    return pd.DataFrame({"velocity": values}, index=_naive_index(times))


FRAMES = {
    FeedName.LIVE_TEMPS: live_temps_frame,
    FeedName.HISTORIC_TEMPS: historic_temps_frame,
    FeedName.TIDES: tides_frame,
    FeedName.CURRENTS: currents_frame,
}


def feed_metadata(
    feed_name: FeedName,
    frame: pd.DataFrame,
    *,
    fetch_timestamp: datetime.datetime = FETCHED_AT,
) -> FeedMetadata:
    return FeedMetadata(
        source_identity=f"coops:{feed_name}:8518750",
        fetch_timestamp=fetch_timestamp,
        next_fetch_after=fetch_timestamp + datetime.timedelta(minutes=10),
        expiration_seconds=600.0,
        record_count=len(frame),
        consecutive_failures=0,
        last_error=None,
        timezone="US/Eastern",
        historical=None,
    )


def sample_snapshot(
    *,
    live_offset: float = 0.0,
    live_fetch_timestamp: datetime.datetime = FETCHED_AT,
) -> Snapshot:
    """One location with every feed and plot; ten distinct objects in all."""
    frames = {name: build() for name, build in FRAMES.items()}
    frames[FeedName.LIVE_TEMPS] = live_temps_frame(live_offset)
    feeds = {
        name: FeedSnapshot(
            frame=frame,
            metadata=feed_metadata(
                name,
                frame,
                fetch_timestamp=(
                    live_fetch_timestamp if name is FeedName.LIVE_TEMPS else FETCHED_AT
                ),
            ),
        )
        for name, frame in frames.items()
    }
    plots = {
        PlotName.LIVE_TEMPS: PlotSnapshot(
            data=b"<svg>live</svg>",
            feed=FeedName.LIVE_TEMPS,
            feed_fetch_timestamp=live_fetch_timestamp,
        ),
        PlotName.HISTORIC_TEMPS_2MO: PlotSnapshot(
            data=b"<svg>2mo</svg>",
            feed=FeedName.HISTORIC_TEMPS,
            feed_fetch_timestamp=FETCHED_AT,
        ),
        PlotName.HISTORIC_TEMPS_12MO: PlotSnapshot(
            data=b"<svg>12mo</svg>",
            feed=FeedName.HISTORIC_TEMPS,
            feed_fetch_timestamp=FETCHED_AT,
        ),
    }
    return Snapshot(locations={"nyc": LocationSnapshot(feeds=feeds, plots=plots)})


SAMPLE_OBJECT_COUNT = len(FRAMES) + 3

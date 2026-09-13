"""Exact Parquet round trips for served feed frames.

A served frame is naive station-local, indexed by `time`, and already
validated by its feed's Pandera model. Serialization moves the index into a
`time` column so Parquet stores it like any other column; loading restores the
index, restores documented categoricals, and validates the frame against the
same model the feed used, so a loaded frame is equivalent to the served one.
`DatetimeIndex.freq` is metadata Parquet cannot carry and nothing served reads
it; a loaded index has no freq.
"""

from io import BytesIO

import pandas as pd
from pandera.pandas import DataFrameModel

from shallweswim import dataframe_models as df_models
from shallweswim.core.feeds import FeedName
from shallweswim.types import TIDE_TYPE_CATEGORIES

TIME_COLUMN = "time"

# Feeds expose their Pandera model through an instance property; loading has
# no feed instance, so the model is bound to the feed name here.
FEED_DATA_MODELS: dict[FeedName, type[DataFrameModel]] = {
    FeedName.TIDES: df_models.TidePredictionDataModel,
    FeedName.CURRENTS: df_models.CurrentDataModel,
    FeedName.LIVE_TEMPS: df_models.WaterTempDataModel,
    FeedName.HISTORIC_TEMPS: df_models.WaterTempDataModel,
}

TIDE_TYPE_DTYPE = pd.CategoricalDtype(TIDE_TYPE_CATEGORIES, ordered=False)


def frame_to_parquet(frame: pd.DataFrame) -> bytes:
    """Serialize a served frame with its `time` index as a `time` column.

    Args:
        frame: A feed's served frame: naive `time` index, validated columns.

    Returns:
        Parquet bytes preserving every column dtype.

    Raises:
        ValueError: If the frame is not indexed by naive datetimes named `time`.
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.name != TIME_COLUMN:
        raise ValueError(f"Served frame must have a DatetimeIndex named {TIME_COLUMN}")
    if index.tz is not None:
        raise ValueError("Served frame index must be timezone naive")
    if TIME_COLUMN in frame.columns:
        raise ValueError(f"Served frame must not have a {TIME_COLUMN} column")
    output = BytesIO()
    frame.reset_index().to_parquet(output, index=False, engine="pyarrow")
    return output.getvalue()


def parquet_to_frame(data: bytes, feed_name: FeedName) -> pd.DataFrame:
    """Restore a served frame and validate it as the named feed would.

    Args:
        data: Parquet bytes produced by `frame_to_parquet`.
        feed_name: The feed the frame belongs to, selecting its Pandera model
            and the categorical columns to restore.

    Returns:
        The frame with its naive `time` index restored, validated against the
        feed's Pandera model.

    Raises:
        ValueError: If the `time` column is missing or timezone-aware.
        pandera.errors.SchemaErrors: If the frame fails the feed's model.
    """
    table = pd.read_parquet(BytesIO(data), engine="pyarrow")
    if TIME_COLUMN not in table.columns:
        raise ValueError(f"Snapshot frame has no {TIME_COLUMN} column")
    frame = table.set_index(TIME_COLUMN)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Snapshot frame {TIME_COLUMN} column must hold datetimes")
    if frame.index.tz is not None:
        raise ValueError(f"Snapshot frame {TIME_COLUMN} column must be timezone naive")
    if feed_name is FeedName.TIDES:
        frame["type"] = frame["type"].astype(TIDE_TYPE_DTYPE)
    return FEED_DATA_MODELS[feed_name].validate(frame, lazy=True)

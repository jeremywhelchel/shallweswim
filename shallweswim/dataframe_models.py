"""Pandera DataFrame models for validating internal data structures."""

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pa_typing

from shallweswim.types import TIDE_TYPE_CATEGORIES


class TimeSeriesDataModel(pa.DataFrameModel):
    """Pandera DataFrameModel for internal timeseries data."""

    time: pa_typing.Index[pa.DateTime] = pa.Field(
        nullable=False, unique=True, check_name=True
    )

    @pa.dataframe_check(error="DataFrame must have at least one row")
    def check_not_empty(cls, df: pd.DataFrame) -> bool:
        """Check that the dataframe is not empty."""
        return not df.empty

    @pa.check("time", error="Index not sorted")
    def check_index_monotonic(cls, idx: pd.Index) -> bool:
        return bool(idx.is_monotonic_increasing)

    @pa.check("time", error="Index must be timezone naive")
    def check_index_tz_naive(cls, idx: pd.Index) -> bool:
        return idx.dt.tz is None

    class Config:
        """Pandera model configuration."""

        strict = True  # Disallow columns not specified in the schema
        # IMPORTANT: Coercion is explicitly disabled for maximum strictness.
        # Data must match the defined types exactly.
        coerce = False
        # Controls how the DataFrame is serialized to JSON for FastAPI output.
        to_format = "dict"
        to_format_kwargs = {"orient": "index"}


class WaterTempDataModel(TimeSeriesDataModel):

    water_temp: pa_typing.Series[float] = pa.Field(nullable=True)

    @pa.check("water_temp", error="water_temp all NaN")
    def check_water_temp_not_all_nan(cls, series: pd.Series) -> bool:
        return not series.isna().all()


class CurrentDataModel(TimeSeriesDataModel):
    velocity: pa_typing.Series[float] = pa.Field(nullable=True)

    @pa.check("velocity", error="velocity all NaN")
    def check_velocity_not_all_nan(cls, series: pd.Series) -> bool:
        return not series.isna().all()


class TidePredictionDataModel(TimeSeriesDataModel):
    prediction: pa_typing.Series[float] = pa.Field(nullable=False)
    type: pd.CategoricalDtype = pa.Field(
        nullable=False,
        isin=TIDE_TYPE_CATEGORIES,  # ← reject anything not in your list
        dtype_kwargs={"categories": TIDE_TYPE_CATEGORIES, "ordered": False},
    )

    @pa.check("prediction", error="prediction all NaN")
    def check_prediction_not_all_nan(cls, series: pd.Series) -> bool:
        return not series.isna().all()

    # May be redundant with the isin check. But added for parity.
    @pa.check("type", error="type all NaN")
    def check_type_not_all_nan(cls, series: pd.Series) -> bool:
        return not series.isna().all()

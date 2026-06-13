"""Local harmonic tide prediction helpers.

This module is intentionally small and data-driven. Offline scripts fit harmonic
coefficients from observed water levels and write them to JSON; runtime feeds
load that compact model and generate the short high/low prediction window the
app needs.
"""

import datetime
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
from scipy import optimize
from scipy.signal import find_peaks

from shallweswim import types

MODEL_SCHEMA_VERSION = 1

# Common tidal constituent speeds in degrees per solar hour. These are the
# frequencies the offline fitter uses by default; model files persist the exact
# subset used so runtime prediction does not depend on this list.
DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR: dict[str, float] = {
    "M2": 28.9841042,
    "S2": 30.0,
    "N2": 28.4397295,
    "K2": 30.0821373,
    "K1": 15.0410686,
    "O1": 13.9430356,
    "P1": 14.9589314,
    "Q1": 13.3986609,
    "M4": 57.9682084,
    "MS4": 58.9841042,
    "MN4": 57.4238337,
}

DEFAULT_FIT_CONSTITUENTS = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1")


@dataclass(frozen=True)
class HarmonicConstituent:
    """One harmonic term stored as linear cosine/sine coefficients."""

    name: str
    speed_degrees_per_hour: float
    cos: float
    sin: float

    @property
    def speed_radians_per_hour(self) -> float:
        """Angular speed in radians per solar hour."""
        return math.radians(self.speed_degrees_per_hour)


@dataclass(frozen=True)
class HarmonicTideModel:
    """Compact local tide model derived from observed gauge data."""

    name: str
    epoch_utc: datetime.datetime
    intercept: float
    constituents: tuple[HarmonicConstituent, ...]
    height_units: str
    height_datum: str
    source: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HarmonicTideModel":
        """Load and validate a harmonic tide model from decoded JSON."""
        if raw.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported harmonic tide model schema: {raw.get('schema_version')}"
            )

        epoch = datetime.datetime.fromisoformat(
            raw["epoch_utc"].replace("Z", "+00:00")
        ).astimezone(datetime.UTC)

        constituents = tuple(
            HarmonicConstituent(
                name=item["name"],
                speed_degrees_per_hour=float(item["speed_degrees_per_hour"]),
                cos=float(item["cos"]),
                sin=float(item["sin"]),
            )
            for item in raw["constituents"]
        )
        if not constituents:
            raise ValueError("Harmonic tide model must include constituents")

        return cls(
            name=raw["name"],
            epoch_utc=epoch,
            intercept=float(raw["intercept"]),
            constituents=constituents,
            height_units=raw["height_units"],
            height_datum=raw["height_datum"],
            source=raw["source"],
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a stable JSON-compatible dictionary."""
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "name": self.name,
            "epoch_utc": self.epoch_utc.isoformat().replace("+00:00", "Z"),
            "intercept": self.intercept,
            "height_units": self.height_units,
            "height_datum": self.height_datum,
            "source": self.source,
            "constituents": [
                {
                    "name": constituent.name,
                    "speed_degrees_per_hour": constituent.speed_degrees_per_hour,
                    "cos": constituent.cos,
                    "sin": constituent.sin,
                }
                for constituent in self.constituents
            ],
            "metadata": self.metadata,
        }

    def predict_utc(self, times_utc: np.ndarray) -> np.ndarray:
        """Predict tide level for timezone-aware UTC datetimes."""
        hours = np.array(
            [(time - self.epoch_utc).total_seconds() / 3600.0 for time in times_utc],
            dtype=float,
        )
        values = np.full(hours.shape, self.intercept, dtype=float)
        for constituent in self.constituents:
            angle = constituent.speed_radians_per_hour * hours
            values += constituent.cos * np.cos(angle) + constituent.sin * np.sin(angle)
        return values


def constituent_amplitude_phase(
    constituent: HarmonicConstituent,
) -> dict[str, float | str]:
    """Return amplitude and phase diagnostics for one fitted constituent."""
    return {
        "name": constituent.name,
        "amplitude": float(np.hypot(constituent.cos, constituent.sin)),
        "phase_degrees": float(np.degrees(np.atan2(constituent.sin, constituent.cos))),
    }


def load_model(model_path: str) -> HarmonicTideModel:
    """Load a harmonic model from a filesystem path or package resource."""
    path = Path(model_path)
    if path.is_absolute() or path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = json.loads(
            resources.files("shallweswim")
            .joinpath(model_path)
            .read_text(encoding="utf-8")
        )
    return HarmonicTideModel.from_dict(raw)


def fit_harmonic_model(
    observations: pd.DataFrame,
    *,
    name: str,
    source: str,
    height_units: str,
    height_datum: str,
    constituent_speeds: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> HarmonicTideModel:
    """Fit harmonic coefficients from observed water levels.

    Args:
        observations: DataFrame indexed by UTC datetimes with a ``value`` column.
        name: Human-readable model name.
        source: Observation source used to derive the model.
        height_units: Units for predicted values.
        height_datum: Datum description for predicted values.
        constituent_speeds: Optional constituent speed map.
        metadata: Optional model metadata to persist.

    Returns:
        Fitted harmonic model.
    """
    if observations.empty:
        raise ValueError("Cannot fit harmonic tide model with no observations")
    if "value" not in observations.columns:
        raise ValueError("Observations must include a value column")

    clean = observations[["value"]].dropna().sort_index()
    if clean.empty:
        raise ValueError("Cannot fit harmonic tide model with only null observations")

    index = clean.index
    if index.tz is None:
        index = index.tz_localize(datetime.UTC)
    else:
        index = index.tz_convert(datetime.UTC)

    epoch = index.min().to_pydatetime().astimezone(datetime.UTC)
    hours = np.array(
        [
            (ts.to_pydatetime().astimezone(datetime.UTC) - epoch).total_seconds()
            / 3600.0
            for ts in index
        ],
        dtype=float,
    )
    y = clean["value"].to_numpy(dtype=float)
    speeds = constituent_speeds or {
        name: DEFAULT_CONSTITUENT_SPEEDS_DEGREES_PER_HOUR[name]
        for name in DEFAULT_FIT_CONSTITUENTS
    }

    columns = [np.ones_like(hours)]
    ordered_speeds = list(speeds.items())
    for _, speed in ordered_speeds:
        angle = np.deg2rad(speed) * hours
        columns.extend((np.cos(angle), np.sin(angle)))

    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    constituents: list[HarmonicConstituent] = []
    coefficient_index = 1
    for constituent_name, speed in ordered_speeds:
        constituents.append(
            HarmonicConstituent(
                name=constituent_name,
                speed_degrees_per_hour=speed,
                cos=float(coefficients[coefficient_index]),
                sin=float(coefficients[coefficient_index + 1]),
            )
        )
        coefficient_index += 2

    model_metadata = dict(metadata or {})
    model_metadata["fit_condition_number"] = float(np.linalg.cond(design))

    return HarmonicTideModel(
        name=name,
        epoch_utc=epoch,
        intercept=float(coefficients[0]),
        constituents=tuple(constituents),
        height_units=height_units,
        height_datum=height_datum,
        source=source,
        metadata=model_metadata,
    )


def predict_high_low_events(
    model: HarmonicTideModel,
    *,
    start_utc: datetime.datetime,
    end_utc: datetime.datetime,
    timezone: datetime.tzinfo,
    sample_minutes: int = 5,
) -> pd.DataFrame:
    """Generate local-naive high/low tide events for a UTC time window."""
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=datetime.UTC)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=datetime.UTC)
    start_utc = start_utc.astimezone(datetime.UTC)
    end_utc = end_utc.astimezone(datetime.UTC)
    if start_utc >= end_utc:
        raise ValueError("start_utc must be before end_utc")

    # Evaluate with margins so extrema near requested boundaries are still
    # detected, then trim refined events back to the requested window.
    margin = datetime.timedelta(hours=8)
    sample_start = start_utc - margin
    sample_end = end_utc + margin
    step = datetime.timedelta(minutes=sample_minutes)
    sample_count = int((sample_end - sample_start) / step) + 1
    sample_times = np.array(
        [sample_start + i * step for i in range(sample_count)], dtype=object
    )
    sample_values = model.predict_utc(sample_times)

    min_distance = max(1, int(datetime.timedelta(hours=4) / step))
    high_indexes, _ = find_peaks(sample_values, distance=min_distance)
    low_indexes, _ = find_peaks(-sample_values, distance=min_distance)

    events: list[tuple[datetime.datetime, float, str]] = []
    for peak_index in high_indexes:
        events.append(
            _refine_extremum(
                model,
                sample_times,
                int(peak_index),
                kind=types.TideCategory.HIGH.value,
            )
        )
    for peak_index in low_indexes:
        events.append(
            _refine_extremum(
                model,
                sample_times,
                int(peak_index),
                kind=types.TideCategory.LOW.value,
            )
        )

    rows = [
        (
            _to_location_naive(event_time, timezone),
            prediction,
            event_type,
        )
        for event_time, prediction, event_type in events
        if start_utc <= event_time <= end_utc
    ]
    rows.sort(key=lambda row: row[0])

    df = pd.DataFrame(rows, columns=["time", "prediction", "type"]).set_index("time")
    # The app's derived tide-state curve is minute-resolution. Keep the refined
    # heights, but publish event timestamps on minute boundaries like NOAA does.
    df.index = pd.DatetimeIndex(df.index).round("min")
    df["type"] = pd.Categorical(df["type"], categories=types.TIDE_TYPE_CATEGORIES)
    return df


def _refine_extremum(
    model: HarmonicTideModel,
    sample_times: np.ndarray,
    peak_index: int,
    *,
    kind: str,
) -> tuple[datetime.datetime, float, str]:
    """Refine a sampled high/low event with scalar optimization."""
    left_index = max(0, peak_index - 1)
    right_index = min(len(sample_times) - 1, peak_index + 1)
    left = sample_times[left_index]
    right = sample_times[right_index]

    def objective(seconds: float) -> float:
        at = left + datetime.timedelta(seconds=float(seconds))
        value = model.predict_utc(np.array([at], dtype=object))[0]
        return -value if kind == types.TideCategory.HIGH.value else value

    result = optimize.minimize_scalar(
        objective,
        bounds=(0.0, (right - left).total_seconds()),
        method="bounded",
        options={"xatol": 1.0},
    )
    event_time = left + datetime.timedelta(seconds=float(result.x))
    prediction = model.predict_utc(np.array([event_time], dtype=object))[0]
    return event_time, float(prediction), kind


def _to_location_naive(
    event_time_utc: datetime.datetime, timezone: datetime.tzinfo
) -> datetime.datetime:
    """Convert a UTC event timestamp to the app's location-local naive form."""
    local = event_time_utc.astimezone(timezone)
    if isinstance(timezone, pytz.BaseTzInfo):
        local = timezone.normalize(local)
    return local.replace(tzinfo=None)

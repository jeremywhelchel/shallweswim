"""Query functions for retrieving current data from feeds.

These functions extract and process data from feeds to answer specific queries
about temperature, tides, and currents.
"""

import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from shallweswim import config as config_lib
from shallweswim.core import feeds
from shallweswim.core.feeds import (
    FEED_CURRENTS,
    FEED_LIVE_TEMPS,
    FEED_TIDES,
)
from shallweswim.types import (
    CurrentDirection,
    CurrentInfo,
    CurrentPhase,
    CurrentStrength,
    CurrentTrend,
    DataSourceType,
    LegacyChartInfo,
    TemperatureReading,
    TideCategory,
    TideEntry,
    TideInfo,
    TideState,
    TideTrend,
)

# =============================================================================
# Exceptions
# =============================================================================


class DataUnavailableError(Exception):
    """Raised when feed data is requested but not currently available.

    This is an expected operational condition (not a bug) that occurs when:
    - A station has no recent data (StationUnavailableError during fetch)
    - Data hasn't been fetched yet (startup race)

    API routes should catch this and return HTTP 503.
    """

    pass


# =============================================================================
# Helper functions
# =============================================================================


SLACK_MAGNITUDE_THRESHOLD_KNOTS = 0.2
LIGHT_CURRENT_MAX_PCT = 1 / 3
MODERATE_CURRENT_MAX_PCT = 2 / 3
TIDE_CURVE_FREQUENCY = "60s"


def get_feed_data(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None], feed_name: feeds.FeedName
) -> pd.DataFrame:
    """Get data from a feed, raising if unavailable.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects
        feed_name: The name of the feed to get data from

    Returns:
        The feed data as a pandas DataFrame

    Raises:
        DataUnavailableError: If the feed data is not available
    """
    feed = feeds_dict.get(feed_name)
    if feed is None or feed._data is None:
        raise DataUnavailableError(f"Feed '{feed_name}' data not available")
    return feed.values


def get_latest_row(df: pd.DataFrame) -> pd.Series:
    """Get the latest row from a DataFrame.

    Args:
        df: DataFrame with a DatetimeIndex

    Returns:
        The latest row as a pandas Series
    """
    return df.iloc[-1]


def get_row_at_time(df: pd.DataFrame, t: datetime.datetime) -> pd.Series:
    """Get the row closest to the specified time.

    Args:
        df: DataFrame with a DatetimeIndex
        t: Time to find the closest row for

    Returns:
        The row closest to the specified time as a pandas Series
    """
    # All times should be assumed to be naive
    return cast(pd.Series, df.loc[df.index.asof(t)])


def _require_naive_datetime_index(df: pd.DataFrame, context: str) -> None:
    """Validate that a query input frame uses local naive datetimes."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataUnavailableError(f"{context} DataFrame must use a DatetimeIndex")
    if df.index.tz is not None:
        raise DataUnavailableError(f"{context} DataFrame should use naive datetimes")


def record_to_tide_entry(record: dict[str, Any]) -> TideEntry:
    """Convert a record dictionary to a TideEntry object."""
    return TideEntry(
        time=record["time"],
        type=TideCategory(record["type"]),
        prediction=record["prediction"],
    )


def prepare_tide_prediction_frame(tides_data: pd.DataFrame) -> pd.DataFrame:
    """Precompute a minute-resolution tide-height curve from high/low events.

    The current tide feed stores NOAA high/low predictions only. This prepares a
    derived frame that can later support cheap point-in-time tide-state lookups
    without changing the upstream NOAA request shape.
    """
    if len(tides_data) < 2:
        raise DataUnavailableError("At least two tide predictions are required")

    if not isinstance(tides_data.index, pd.DatetimeIndex):
        raise ValueError("Tide prediction DataFrame must use a DatetimeIndex")
    if tides_data.index.tz is not None:
        raise ValueError("Tide prediction DataFrame should use naive datetimes")

    tides = tides_data.sort_index()
    interpolation_method = "polynomial" if len(tides) >= 3 else "linear"
    interpolation_kwargs: dict[str, int] = (
        {"order": 2} if interpolation_method == "polynomial" else {}
    )

    df = (
        tides[["prediction"]]
        .resample(TIDE_CURVE_FREQUENCY)
        .interpolate(
            interpolation_method,
            **interpolation_kwargs,
        )
    )

    event_times = tides.index
    curve_times = df.index
    if not isinstance(event_times, pd.DatetimeIndex) or not isinstance(
        curve_times, pd.DatetimeIndex
    ):
        raise DataUnavailableError("Tide prediction DataFrame must use datetimes")

    event_ns = event_times.asi8
    curve_ns = curve_times.asi8
    previous_positions = np.clip(
        np.searchsorted(event_ns, curve_ns, side="right") - 1,
        0,
        len(event_ns) - 1,
    )
    next_positions = np.clip(
        np.searchsorted(event_ns, curve_ns, side="left"),
        0,
        len(event_ns) - 1,
    )

    previous_heights = tides["prediction"].to_numpy(dtype=float)[previous_positions]
    next_heights = tides["prediction"].to_numpy(dtype=float)[next_positions]
    low_heights = np.minimum(previous_heights, next_heights)
    high_heights = np.maximum(previous_heights, next_heights)
    height_ranges = high_heights - low_heights

    df["height_pct"] = np.divide(
        df["prediction"].to_numpy(dtype=float) - low_heights,
        height_ranges,
        out=np.zeros(len(df), dtype=float),
        where=height_ranges > 0,
    )
    df["height_pct"] = df["height_pct"].clip(0.0, 1.0)
    df["trend"] = np.select(
        [next_heights > previous_heights, next_heights < previous_heights],
        ["rising", "falling"],
        default="steady",
    )

    return df


def predict_tide_from_precomputed_frame(
    df: pd.DataFrame,
    config: config_lib.LocationConfig,
    t: datetime.datetime | None = None,
) -> TideState:
    """Estimate point-in-time tide state from a precomputed tide frame.

    Args:
        df: Precomputed tide prediction DataFrame from prepare_tide_prediction_frame().
        config: Location configuration.
        t: Time to estimate tide state for, defaults to current local time.

    Returns:
        Estimated tide state for the closest available minute at or before `t`.

    Raises:
        ValueError: If input datetime has timezone info or the frame contract is invalid.
    """
    if not t:
        t = config.local_now()

    if t.tzinfo is not None:
        raise ValueError("Input datetime must be naive")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Tide prediction DataFrame must use a DatetimeIndex")
    if df.index.tz is not None:
        raise ValueError("Tide prediction DataFrame should use naive datetimes")

    row = get_row_at_time(df, t)
    timestamp = row.name
    if not isinstance(timestamp, datetime.datetime):
        raise ValueError("Tide prediction row index must contain datetimes")

    height_pct = row["height_pct"]

    return TideState(
        timestamp=timestamp,
        estimated_height=float(row["prediction"]),
        units="ft",
        trend=TideTrend(row["trend"]),
        height_pct=None if pd.isna(height_pct) else float(height_pct),
    )


def prepare_current_prediction_frame(currents_data: pd.DataFrame) -> pd.DataFrame:
    """Precompute derived current prediction columns for fast point-in-time lookup."""
    df = currents_data.copy()
    _require_naive_datetime_index(df, "Current prediction")

    # Convert raw velocity to magnitude and direction
    df["v"] = df["velocity"]
    df["magnitude"] = df["v"].abs()
    velocities = df["v"].to_numpy(dtype=float)
    magnitudes = np.abs(velocities)
    index = cast(pd.DatetimeIndex, df.index)

    # Add a slope column to track if current is strengthening or weakening
    df["raw_slope"] = df["v"].diff()

    # Create a directionally correct slope column
    conditions = [
        (df["v"] > 0),  # flooding
        (df["v"] < 0),  # ebbing
    ]
    choices = [
        df["raw_slope"],  # for flooding, use raw slope
        -df["raw_slope"],  # for ebbing, negate the slope
    ]
    df["slope"] = np.select(conditions, choices, default=0)

    # Mark direction as flood or ebb
    conditions = [
        (df["v"] > 0),  # flooding
        (df["v"] < 0),  # ebbing
    ]
    choices = [CurrentDirection.FLOODING.value, CurrentDirection.EBBING.value]
    df["direction"] = np.select(conditions, choices, default="")

    direction_values = df["direction"].to_numpy(dtype=str)
    local_mag_pct = np.zeros(len(df), dtype=float)
    segment_ids = np.full(len(df), np.nan)
    segment_phases = np.full(len(df), None, dtype=object)
    segment_peak_times = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    segment_peak_magnitudes = np.full(len(df), np.nan)
    segment_start_slack_times = np.full(
        len(df), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    segment_start_slack_magnitudes = np.full(len(df), np.nan)
    segment_end_slack_times = np.full(
        len(df), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    segment_end_slack_magnitudes = np.full(len(df), np.nan)

    def interpolated_slack_time(
        left_position: int,
        right_position: int,
    ) -> datetime.datetime | None:
        left_time = df.index[left_position]
        right_time = df.index[right_position]
        if not isinstance(left_time, datetime.datetime) or not isinstance(
            right_time, datetime.datetime
        ):
            return None

        left_velocity = float(df.iloc[left_position]["v"])
        right_velocity = float(df.iloc[right_position]["v"])
        velocity_delta = right_velocity - left_velocity
        if velocity_delta == 0:
            return None

        zero_fraction = -left_velocity / velocity_delta
        if zero_fraction < 0 or zero_fraction > 1:
            return None

        return left_time + (right_time - left_time) * zero_fraction

    def slack_boundary_before(position: int) -> tuple[datetime.datetime, float] | None:
        if position == 0:
            return None

        # Use an explicit slack row when the prediction curve provides one.
        # Otherwise store an interpolated zero-crossing as the semantic slack
        # boundary for this segment.
        previous_time = df.index[position - 1]
        if magnitudes[position - 1] < SLACK_MAGNITUDE_THRESHOLD_KNOTS:
            return previous_time, float(magnitudes[position - 1])  # type: ignore[return-value]

        slack_time = interpolated_slack_time(position - 1, position)
        return (slack_time, 0.0) if slack_time is not None else None

    def slack_boundary_after(position: int) -> tuple[datetime.datetime, float] | None:
        if position >= len(df) - 1:
            return None

        # Mirror slack_boundary_before: prefer real slack rows, and fall back to
        # zero-crossing interpolation when adjacent non-slack rows change sign.
        next_time = df.index[position + 1]
        if magnitudes[position + 1] < SLACK_MAGNITUDE_THRESHOLD_KNOTS:
            return next_time, float(magnitudes[position + 1])  # type: ignore[return-value]

        slack_time = interpolated_slack_time(position, position + 1)
        return (slack_time, 0.0) if slack_time is not None else None

    non_slack = (magnitudes >= SLACK_MAGNITUDE_THRESHOLD_KNOTS) & (
        direction_values != ""
    )
    previous_non_slack = np.concatenate(([False], non_slack[:-1]))
    previous_directions = np.concatenate(([""], direction_values[:-1]))
    segment_starts = non_slack & (
        ~previous_non_slack | (direction_values != previous_directions)
    )
    segment_marker = np.cumsum(segment_starts)
    segment_lookup = np.where(non_slack, segment_marker, 0)

    for segment_id in range(1, int(segment_marker.max()) + 1):
        positions = np.flatnonzero(segment_lookup == segment_id)
        if len(positions) == 0:
            continue

        active_direction = direction_values[positions[0]]
        peak_position = positions[int(magnitudes[positions].argmax())]
        peak_time = index[peak_position]
        peak_magnitude = float(magnitudes[peak_position])

        start_position = positions[0]
        end_position = positions[-1]
        start_slack = slack_boundary_before(start_position)
        end_slack = slack_boundary_after(end_position)

        segment_ids[positions] = segment_id
        segment_phases[positions] = (
            CurrentPhase.FLOOD.value
            if active_direction == CurrentDirection.FLOODING.value
            else CurrentPhase.EBB.value
        )
        segment_peak_times[positions] = np.datetime64(peak_time, "ns")
        segment_peak_magnitudes[positions] = peak_magnitude
        if peak_magnitude > 0:
            local_mag_pct[positions] = np.minimum(
                magnitudes[positions] / peak_magnitude, 1.0
            )

        if start_slack is not None:
            slack_time, slack_magnitude = start_slack
            segment_start_slack_times[positions] = np.datetime64(slack_time, "ns")
            segment_start_slack_magnitudes[positions] = slack_magnitude
        if end_slack is not None:
            slack_time, slack_magnitude = end_slack
            segment_end_slack_times[positions] = np.datetime64(slack_time, "ns")
            segment_end_slack_magnitudes[positions] = slack_magnitude

    df["local_mag_pct"] = local_mag_pct
    df["segment_id"] = segment_ids
    df["segment_phase"] = segment_phases
    df["segment_peak_time"] = segment_peak_times
    df["segment_peak_magnitude"] = segment_peak_magnitudes
    df["segment_start_slack_time"] = segment_start_slack_times
    df["segment_start_slack_magnitude"] = segment_start_slack_magnitudes
    df["segment_end_slack_time"] = segment_end_slack_times
    df["segment_end_slack_magnitude"] = segment_end_slack_magnitudes
    return df


def _next_non_slack_direction(
    df: pd.DataFrame,
    t: datetime.datetime,
) -> CurrentDirection | None:
    """Find the next predicted flood/ebb direction above the slack threshold."""
    future = df[(df.index > t) & (df["magnitude"] >= SLACK_MAGNITUDE_THRESHOLD_KNOTS)]
    if future.empty:
        return None

    direction = future.iloc[0]["direction"]
    if direction == CurrentDirection.FLOODING.value:
        return CurrentDirection.FLOODING
    if direction == CurrentDirection.EBBING.value:
        return CurrentDirection.EBBING
    return None


def _phase_for_current(
    df: pd.DataFrame,
    row_time: datetime.datetime,
    magnitude: float,
    direction_str: str,
) -> CurrentPhase:
    """Return the compact current phase for a prediction row."""
    if magnitude < SLACK_MAGNITUDE_THRESHOLD_KNOTS:
        next_direction = _next_non_slack_direction(df, row_time)
        match next_direction:
            case CurrentDirection.FLOODING:
                return CurrentPhase.SLACK_BEFORE_FLOOD
            case CurrentDirection.EBBING:
                return CurrentPhase.SLACK_BEFORE_EBB
            case None:
                return CurrentPhase.SLACK

    if direction_str == CurrentDirection.FLOODING.value:
        return CurrentPhase.FLOOD
    if direction_str == CurrentDirection.EBBING.value:
        return CurrentPhase.EBB
    return CurrentPhase.SLACK


def _strength_for_current(
    phase: CurrentPhase,
    magnitude_pct: float,
) -> CurrentStrength | None:
    """Return the display strength bucket for a non-slack tidal current."""
    if phase in {
        CurrentPhase.SLACK_BEFORE_FLOOD,
        CurrentPhase.SLACK_BEFORE_EBB,
        CurrentPhase.SLACK,
    }:
        return None
    if pd.isna(magnitude_pct):
        return None
    if magnitude_pct < LIGHT_CURRENT_MAX_PCT:
        return CurrentStrength.LIGHT
    if magnitude_pct < MODERATE_CURRENT_MAX_PCT:
        return CurrentStrength.MODERATE
    return CurrentStrength.STRONG


def _trend_for_current(
    phase: CurrentPhase,
    slope: float,
) -> CurrentTrend | None:
    """Return the display trend for a non-slack tidal current."""
    if phase in {
        CurrentPhase.SLACK_BEFORE_FLOOD,
        CurrentPhase.SLACK_BEFORE_EBB,
        CurrentPhase.SLACK,
    }:
        return None
    if slope > 0:
        return CurrentTrend.BUILDING
    if slope < 0:
        return CurrentTrend.EASING
    return CurrentTrend.STEADY


def _state_description_for_current(
    phase: CurrentPhase,
    strength: CurrentStrength | None,
    trend: CurrentTrend | None,
) -> str:
    """Return a compact phrase for user-facing current state text."""
    match phase:
        case CurrentPhase.SLACK_BEFORE_FLOOD:
            return "slack before flood"
        case CurrentPhase.SLACK_BEFORE_EBB:
            return "slack before ebb"
        case CurrentPhase.SLACK:
            return "slack"
        case CurrentPhase.FLOOD | CurrentPhase.EBB:
            base = phase.value
            if strength is not None:
                base = f"{strength.value} {base}"
            if trend is None:
                return base
            return f"{base} and {trend.value}"
    raise ValueError(f"Unhandled current phase: {phase}")


# =============================================================================
# Query functions
# =============================================================================


def get_current_temperature(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None],
) -> TemperatureReading:
    """Get the most recent water temperature reading.

    Retrieves the latest temperature data from the configured temperature source.
    The temperature is rounded to 1 decimal place for consistency.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects

    Returns:
        A TemperatureReading object containing:
            - timestamp: datetime of when the reading was taken
            - temperature: float representing the water temperature in degrees Celsius

    Raises:
        DataUnavailableError: If no temperature data is available or the feed is not configured
    """
    # Get live temperature data from the feed
    live_temps_data = get_feed_data(feeds_dict, FEED_LIVE_TEMPS)

    # Get the latest temperature reading
    latest_row = get_latest_row(live_temps_data)
    time = latest_row.name
    temp = latest_row["water_temp"]

    # Round temperature to 1 decimal place to avoid excessive precision
    rounded_temp = round(temp, 1)  # type: ignore[call-overload]

    return TemperatureReading(timestamp=time, temperature=rounded_temp)  # type: ignore[arg-type]


def get_current_tide_info(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None],
    config: config_lib.LocationConfig,
) -> TideInfo:
    """Get the previous tide and upcoming tides relative to current time.

    Retrieves the most recent tide before current time and the next two
    upcoming tides from the tide predictions data. All times are naive datetimes
    in the location's timezone.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects
        config: Location configuration

    Returns:
        A TideInfo object containing:
            - past: List of TideEntry objects with the most recent tide information
            - next: List of TideEntry objects with the next two upcoming tides

    Raises:
        DataUnavailableError: If tide data feed is missing or not properly configured
    """
    # Get tides data from the feed
    tides_data = get_feed_data(feeds_dict, FEED_TIDES)

    # Get current time in the location's timezone as a naive datetime and convert to pandas Timestamp for slicing
    now = config.local_now()
    now_ts = pd.Timestamp(now)

    # Ensure DataFrame has no timezone info for consistent comparison
    _require_naive_datetime_index(tides_data, "Tide")

    # Extract past and future tide data
    past_tides_df = tides_data[:now_ts].tail(1)
    next_tides_df = tides_data[now_ts:].head(2)

    # Convert DataFrames to dictionaries for processing
    past_tide_dicts = past_tides_df.reset_index().to_dict(orient="records")
    next_tide_dicts = next_tides_df.reset_index().to_dict(orient="records")

    # Convert DataFrame records to TideEntry objects using the helper function
    past_tides = [
        record_to_tide_entry(cast(dict[str, Any], record)) for record in past_tide_dicts
    ]
    next_tides = [
        record_to_tide_entry(cast(dict[str, Any], record)) for record in next_tide_dicts
    ]

    return TideInfo(past=past_tides, next=next_tides)


def get_chart_info(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None],
    config: config_lib.LocationConfig,
    t: datetime.datetime | None = None,
) -> LegacyChartInfo:
    """Generate chart information based on tide data for the specified time.

    Calculates the time since the last tide event and generates appropriate
    chart information including filenames and titles. This supports the legacy
    chart display system.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects
        config: Location configuration
        t: The time to generate chart info for, defaults to current time in location's timezone

    Returns:
        A LegacyChartInfo object containing:
            - hours_since_last_tide: Number of hours since last tide
            - last_tide_type: Type of last tide ("high" or "low")
            - chart_filename: Filename for the chart image
            - map_title: Formatted title for the map display

    Raises:
        DataUnavailableError: If tide data is not available
    """
    if not t:
        t = config.local_now()

    # Get tides data from the feed
    tides_data = get_feed_data(feeds_dict, FEED_TIDES)

    # Get the row closest to the specified time
    row = get_row_at_time(tides_data, t)
    tide_type = row["type"]

    # Get the timestamp from the row
    row_time = row.name

    offset = t - row_time  # type: ignore[operator]
    offset_hrs = offset.seconds / (60 * 60)
    if offset_hrs > 5.5:
        chart_num = 0
        INVERT = {"high": "low", "low": "high"}
        chart_type = INVERT[tide_type]  # type: ignore[index]
    else:
        chart_num = round(offset_hrs)
        chart_type = tide_type

    legacy_map_title = f"{chart_type.capitalize()} Water at New York"  # type: ignore[attr-defined]
    if chart_num:
        suffix = "s" if chart_num > 1 else ""
        legacy_map_title = f"{chart_num} Hour{suffix} after " + legacy_map_title
    filename = f"{chart_type}+{chart_num}.png"

    return LegacyChartInfo(
        hours_since_last_tide=offset_hrs,
        last_tide_type=tide_type,  # type: ignore[arg-type]
        chart_filename=filename,
        map_title=legacy_map_title,
    )


def get_current_flow_info(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None],
) -> CurrentInfo:
    """Get the latest observed current information.

    Retrieves the most recent current observation from the configured current source.
    This returns actual observed data, not predictions.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects

    Returns:
        A CurrentInfo object containing the timestamp, magnitude, and source type
        of the most recent current observation

    Raises:
        DataUnavailableError: If current data is not available or not properly loaded
    """
    # Get currents data from the feed
    currents_data = get_feed_data(feeds_dict, FEED_CURRENTS)

    # Get the latest current reading
    latest_reading = get_latest_row(currents_data)
    latest_timestamp = latest_reading.name  # Timestamp is in the index

    return CurrentInfo(
        timestamp=latest_timestamp,  # type: ignore[arg-type]
        source_type=DataSourceType.OBSERVATION,
        magnitude=latest_reading["velocity"],  # type: ignore[arg-type]
    )


def predict_flow_from_precomputed_frame(
    df: pd.DataFrame,
    config: config_lib.LocationConfig,
    t: datetime.datetime | None = None,
) -> CurrentInfo:
    """Predict tidal current conditions for a specific time.

    Uses a precomputed current prediction frame from
    prepare_current_prediction_frame() to determine the state, direction, and
    magnitude of the tidal current at the specified time.

    NOTE: This method is currently specific to TIDAL current systems.
    River current predictions will require a separate implementation.

    Args:
        df: Precomputed current prediction DataFrame
        config: Location configuration
        t: Time to predict current for, defaults to current time in location's timezone.
           Must be a naive datetime in the location's timezone.

    Returns:
        A CurrentInfo object containing:
            - direction: Direction of current (FLOODING or EBBING)
            - phase: Compact phase (flood, ebb, slack_before_flood, slack_before_ebb, or slack)
            - magnitude: Magnitude of current in knots
            - magnitude_pct: Relative magnitude percentage (0.0-1.0) compared to the
              peak in the current continuous flood or ebb segment
            - state_description: Display-ready current state phrase
            - source_type: Always PREDICTION for this method

    Raises:
        DataUnavailableError: If current data is not available or not properly loaded.
        ValueError: If input datetime has timezone info.
    """
    if not t:
        t = config.local_now()

    # Ensure we're using a naive datetime for DataFrame slicing
    if t.tzinfo is not None:
        raise ValueError("Input datetime must be naive")
    _require_naive_datetime_index(df, "Current prediction")

    # Fetch only the scalar columns needed for the public response. The
    # precomputed frame also carries segment metadata for future range labels,
    # and materializing a full mixed-type pandas row on every request is
    # measurably slower.
    row_time = df.index.asof(t)
    if not isinstance(row_time, datetime.datetime):
        raise DataUnavailableError(
            "Current prediction row index must contain datetimes"
        )

    magnitude = float(df.at[row_time, "magnitude"])
    local_pct = float(df.at[row_time, "local_mag_pct"])
    slope = float(df.at[row_time, "slope"])
    direction_str = str(df.at[row_time, "direction"])

    phase = _phase_for_current(df, row_time, magnitude, direction_str)
    strength = _strength_for_current(phase, local_pct)
    trend = _trend_for_current(phase, slope)
    state_description = _state_description_for_current(phase, strength, trend)

    if not direction_str:
        if phase == CurrentPhase.SLACK_BEFORE_FLOOD:
            direction_str = CurrentDirection.FLOODING.value
        elif phase == CurrentPhase.SLACK_BEFORE_EBB:
            direction_str = CurrentDirection.EBBING.value
    direction = CurrentDirection(direction_str) if direction_str else None

    return CurrentInfo(
        timestamp=t,
        source_type=DataSourceType.PREDICTION,
        direction=direction,
        phase=phase,
        strength=strength,
        trend=trend,
        magnitude=magnitude,
        magnitude_pct=local_pct,
        state_description=state_description,
    )


def predict_flow_at_time(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None],
    config: config_lib.LocationConfig,
    t: datetime.datetime | None = None,
) -> CurrentInfo:
    """Predict tidal current conditions for a specific time.

    This convenience path derives the current prediction frame on demand. Runtime
    managers should prefer prepare_current_prediction_frame() plus
    predict_flow_from_precomputed_frame() so repeated requests are cheap.
    """
    currents_data = get_feed_data(feeds_dict, FEED_CURRENTS)
    prediction_frame = prepare_current_prediction_frame(currents_data)
    return predict_flow_from_precomputed_frame(prediction_frame, config, t)

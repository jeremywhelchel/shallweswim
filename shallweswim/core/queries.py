"""Query functions for retrieving current data from feeds.

These functions extract and process data from feeds to answer specific queries
about temperature, tides, and currents.
"""

import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
    DataSourceType,
    LegacyChartInfo,
    TemperatureReading,
    TideCategory,
    TideEntry,
    TideInfo,
)

# =============================================================================
# Helper functions
# =============================================================================


def get_feed_data(
    feeds_dict: dict[str, feeds.Feed | None], feed_name: str
) -> pd.DataFrame:
    """Get data from a feed and assert that it's available.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects
        feed_name: The name of the feed to get data from

    Returns:
        The feed data as a pandas DataFrame

    Raises:
        AssertionError: If the feed data is not available
    """
    feed = feeds_dict.get(feed_name)
    data = feed.values if feed is not None else None
    assert data is not None
    return data


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
    return df.loc[df.index.asof(t)]


def record_to_tide_entry(record: dict[str, Any]) -> TideEntry:
    """Convert a record dictionary to a TideEntry object."""
    return TideEntry(
        time=record["time"],
        type=TideCategory(record["type"]),
        prediction=record["prediction"],
    )


def _process_local_magnitude_pct(
    df: pd.DataFrame,
    current_df: pd.DataFrame,
    direction: str,
    invert: bool = False,
) -> pd.DataFrame:
    """Process local magnitude percentages for current data.

    Calculates magnitude percentages relative to local peaks for a given
    current direction. This provides more meaningful context than global
    percentages since tidal currents vary in strength throughout the day.

    Args:
        df: The main DataFrame to update with local magnitude percentages
        current_df: DataFrame containing only the current direction's data
        direction: The current direction being processed ('flood' or 'ebb')
        invert: Whether to invert the magnitude values (for ebb currents)

    Returns:
        Updated DataFrame with local_mag_pct values for the specified direction
    """
    if current_df.empty:
        return df

    # Get magnitude values for peak detection
    magnitudes: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.asarray(
        current_df["magnitude"].values
    )
    if invert:
        magnitudes = -magnitudes

    # Find peaks in the magnitude data
    # Using a minimum height threshold to avoid detecting noise
    peaks, _ = find_peaks(magnitudes, height=0.1)

    if len(peaks) == 0:
        # No peaks found, use global percentage as fallback
        df.loc[current_df.index, "local_mag_pct"] = df.loc[current_df.index, "mag_pct"]
        return df

    # Get the timestamps and magnitudes at peak locations
    peak_times = current_df.index[peaks]
    peak_magnitudes = current_df["magnitude"].iloc[peaks].values

    # For each row in the current direction, find the nearest peak
    # and calculate the percentage relative to that peak
    for idx in current_df.index:
        # Find the nearest peak (before or after)
        time_diffs = abs(peak_times - idx)
        nearest_peak_idx = time_diffs.argmin()
        nearest_peak_magnitude = peak_magnitudes[nearest_peak_idx]

        # Calculate percentage relative to nearest peak
        if nearest_peak_magnitude > 0:
            local_pct = current_df.loc[idx, "magnitude"] / nearest_peak_magnitude
            df.loc[idx, "local_mag_pct"] = min(local_pct, 1.0)  # Cap at 100%

    return df


# =============================================================================
# Query functions
# =============================================================================


def get_current_temperature(
    feeds_dict: dict[str, feeds.Feed | None],
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
        AssertionError: If no temperature data is available or the feed is not configured
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
    feeds_dict: dict[str, feeds.Feed | None],
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
        AssertionError: If tide data feed is missing or not properly configured
    """
    # Get tides data from the feed
    tides_data = get_feed_data(feeds_dict, FEED_TIDES)

    # Get current time in the location's timezone as a naive datetime and convert to pandas Timestamp for slicing
    now = config.local_now()
    now_ts = pd.Timestamp(now)

    # Ensure DataFrame has no timezone info for consistent comparison
    assert tides_data.index.tz is None, "Tide DataFrame should use naive datetimes"  # type: ignore[attr-defined]

    # Extract past and future tide data
    past_tides_df = tides_data[:now_ts].tail(1)
    next_tides_df = tides_data[now_ts:].head(2)

    # Convert DataFrames to dictionaries for processing
    past_tide_dicts = past_tides_df.reset_index().to_dict(orient="records")
    next_tide_dicts = next_tides_df.reset_index().to_dict(orient="records")

    # Convert DataFrame records to TideEntry objects using the helper function
    past_tides = [record_to_tide_entry(record) for record in past_tide_dicts]
    next_tides = [record_to_tide_entry(record) for record in next_tide_dicts]

    return TideInfo(past=past_tides, next=next_tides)


def get_chart_info(
    feeds_dict: dict[str, feeds.Feed | None],
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
        AssertionError: If tide data is not available
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
    feeds_dict: dict[str, feeds.Feed | None],
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
        AssertionError: If current data is not available or not properly loaded
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


def predict_flow_at_time(
    feeds_dict: dict[str, feeds.Feed | None],
    config: config_lib.LocationConfig,
    t: datetime.datetime | None = None,
) -> CurrentInfo:
    """Predict tidal current conditions for a specific time.

    Analyzes current prediction data to determine the state, direction, and magnitude
    of the tidal current at the specified time. Includes contextual information about
    whether the current is strengthening, weakening, or at peak/slack.

    NOTE: This method is currently specific to TIDAL current systems.
    River current predictions will require a separate implementation.

    Args:
        feeds_dict: Dictionary mapping feed names to Feed objects
        config: Location configuration
        t: Time to predict current for, defaults to current time in location's timezone.
           Must be a naive datetime in the location's timezone.

    Returns:
        A CurrentInfo object containing:
            - direction: Direction of current (FLOODING or EBBING)
            - magnitude: Magnitude of current in knots
            - magnitude_pct: Relative magnitude percentage (0.0-1.0) compared to local peaks
            - state_description: Text description of current state
            - source_type: Always PREDICTION for this method

    Raises:
        AssertionError: If current data is not available or input datetime has timezone info
    """
    if not t:
        t = config.local_now()

    # Get currents data from the feed
    currents_data = get_feed_data(feeds_dict, FEED_CURRENTS)

    # Create a working copy of the current data for analysis
    df = currents_data.copy()

    # Convert raw velocity to magnitude and direction
    df["v"] = df["velocity"]
    df["magnitude"] = df["v"].abs()

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

    # Initialize column for local magnitude percentage
    df["local_mag_pct"] = 0.0

    # Process flood and ebb separately
    flood_df: pd.DataFrame = df[df["v"] > 0].copy()  # type: ignore[assignment]
    ebb_df: pd.DataFrame = df[df["v"] < 0].copy()  # type: ignore[assignment]

    # Calculate mag_pct for global magnitude ranking (needed for fallback)
    df["mag_pct"] = df.groupby("direction")["magnitude"].rank(pct=True)

    # Calculate magnitude percentages relative to local peaks
    df = _process_local_magnitude_pct(df, flood_df, CurrentDirection.FLOODING.value)
    df = _process_local_magnitude_pct(
        df,
        ebb_df,
        CurrentDirection.EBBING.value,
        invert=True,
    )

    # Ensure we're using a naive datetime for DataFrame slicing
    assert t.tzinfo is None, "Input datetime must be naive"
    assert df.index.tz is None, "DataFrame should use naive datetimes"  # type: ignore[attr-defined]

    # Get the row at or after the specified time
    row = get_row_at_time(df, t)

    # Constants for determining current state
    STRONG_THRESHOLD = 0.85
    SLACK_THRESHOLD = 0.2
    magnitude = row["magnitude"]
    local_pct = row["local_mag_pct"]

    # Determine state description
    if magnitude < SLACK_THRESHOLD:
        msg = "at its weakest (slack)"
    elif local_pct > STRONG_THRESHOLD:
        msg = "at its strongest"
    elif row["slope"] < 0:
        msg = "getting weaker"
    elif row["slope"] > 0:
        msg = "getting stronger"
    else:
        msg = "stable"

    direction_str = row["direction"]

    return CurrentInfo(
        timestamp=t,
        source_type=DataSourceType.PREDICTION,
        direction=CurrentDirection(direction_str),
        magnitude=magnitude,  # type: ignore[arg-type]
        magnitude_pct=row["local_mag_pct"],  # type: ignore[arg-type]
        state_description=msg,
    )

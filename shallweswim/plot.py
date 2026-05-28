"""Generation of plots and charts for ShallWeSwim application.

This module handles the creation of visualizations for tides, currents, and temperature
data. It generates both static charts and dynamic plots based on data fetched from NOAA.

This module is organized into three main sections:
1. Utility functions for common plotting operations
2. Temperature plotting functions for historical and live temperature data
3. Tide and current plotting functions for water movement visualization
"""

# Standard library imports
import datetime
import io
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import cast

# Third-party imports
import matplotlib.dates as md
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import]
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import find_peaks

# Local imports
from shallweswim import config as config_lib
from shallweswim import types, util

# Re-export from util for backwards compatibility
from shallweswim.util import get_current_chart_filename

# Set default Seaborn theme settings for consistent plot appearance
sns.set_theme()
sns.axes_style("darkgrid")

# Constants for plot sizes
STANDARD_FIGURE_SIZE = (16, 8)  # Standard plot size in inches
CURRENT_CHART_SIZE = (16, 6)  # Size for current charts (2596 x 967 pixels)
MAX_HISTORIC_TEMP_PLOT_GAP = pd.Timedelta(hours=48)
HISTORIC_TEMP_PLOT_ARTIFACT_WINDOW = pd.Timedelta(days=7)
MAX_HISTORIC_TEMP_PLOT_SPIKE_RESIDUAL_F = 12.0
MAX_HISTORIC_TEMP_PLOT_CROSS_YEAR_RESIDUAL_F = 10.0
HISTORIC_TEMP_PLOT_VOLATILITY_WINDOW = pd.Timedelta(hours=48)
MAX_HISTORIC_TEMP_PLOT_SMOOTHED_RANGE_F = 6.0
MIN_HISTORIC_TEMP_PLOT_SEGMENT = pd.Timedelta(hours=48)
HISTORIC_TEMP_LINE_STYLES = ["--", ":", "-."]
HISTORIC_TEMP_COLOR_PALETTE = sns.color_palette(n_colors=20)

# Font size constants
TITLE_FONT_SIZE = 24
SUBTITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 16

# Color constants
CURRENT_FLOOD_COLOR = "#3f8f46"  # Green for flooding currents
CURRENT_EBB_COLOR = "#d14a3a"  # Red for ebbing currents
TIDE_COLOR = "#2323b8"  # Blue for tide data
HIGHLIGHT_COLOR = "#ff4000"  # Orange for highlighting points of interest
NOW_MARKER_COLOR = "#111827"  # Dark marker for current time
PLANNED_MARKER_COLOR = "#db2777"  # Accent marker for planner time
PLOT_BACKGROUND_COLOR = "#f8fafc"
PLOT_GRID_COLOR = "#cbd5e1"
PLOT_SPINE_COLOR = "#94a3b8"

#############################################################
# UTILITY FUNCTIONS                                        #
#############################################################


def create_standard_figure() -> Figure:
    """Create a standard figure with consistent size.

    Returns:
        Figure object with standard size
    """
    return Figure(figsize=STANDARD_FIGURE_SIZE)


def add_celsius_axis(ax: Axes) -> Axes:
    """Add a secondary y-axis with Celsius temperature scale.

    Args:
        ax: Primary axis with Fahrenheit scale

    Returns:
        Secondary axis with Celsius scale
    """
    ax2 = ax.twinx()
    fmin, fmax = ax.get_ylim()
    ax2.set_ylim(util.f_to_c(fmin), util.f_to_c(fmax))
    ax2.set_ylabel("Water Temp (°C)", fontsize=LABEL_FONT_SIZE)
    ax2.grid(None)
    return ax2


def save_fig(
    fig: Figure,
    dst: str | io.StringIO,
    fmt: str = "svg",
    location_code: str = "unknown",
) -> None:
    """Save a matplotlib figure to a file or string buffer.

    Handles path adjustments for running from different working directories
    and creates directories as needed.

    Args:
        fig: Figure object to save
        dst: Destination path (string) or buffer
        fmt: Format to save in ('svg', 'png', etc.)

    Raises:
        ValueError: If a string path doesn't start with 'static/'
    """
    # If running outside the 'shallweswim' directory, prepend it to all paths
    if isinstance(dst, str):
        if not dst.startswith("static/"):
            raise ValueError(f"Plot path must start with 'static/': {dst}")
        logging.info(f"[{location_code}] Saving plot to {dst} in {fmt} format")
        if not os.path.exists("static/") and os.path.exists("shallweswim/static/"):
            dst = f"shallweswim/{dst}"
            logging.debug(f"[{location_code}] Path adjusted to {dst}")
        # Create directory if it doesn't exist
        dirname = os.path.dirname(dst)
        if not os.path.isdir(dirname):
            logging.debug(f"[{location_code}] Creating directory: {dirname}")
            os.mkdir(dirname)
    else:
        logging.debug(f"[{location_code}] Saving plot to memory buffer in {fmt} format")

    # Save the figure and log success
    # For file outputs, use atomic write (temp file + rename) to prevent race conditions
    # when the file is being served while being regenerated
    tmp_path: str | None = None
    try:
        if isinstance(dst, str):
            # Write to temp file in same directory (same filesystem for atomic rename)
            file_dirname = os.path.dirname(dst)
            fd, tmp_path = tempfile.mkstemp(dir=file_dirname, suffix=f".{fmt}.tmp")
            os.close(fd)  # Close fd, savefig will open it
            fig.savefig(tmp_path, format=fmt, bbox_inches="tight", transparent=False)
            os.rename(tmp_path, dst)  # Atomic on POSIX
            abs_path = os.path.abspath(dst)
            logging.info(f"[{location_code}] Plot saved to absolute path: {abs_path}")
        else:
            # Memory buffer - write directly
            fig.savefig(dst, format=fmt, bbox_inches="tight", transparent=False)

        logging.debug(f"[{location_code}] Plot saved successfully")
    except Exception as e:
        # Clean up temp file on error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logging.error(f"[{location_code}] Error saving plot: {e}")
        raise


def fig_to_bytes(fig: Figure, fmt: str = "svg") -> bytes:
    """Convert a matplotlib Figure to bytes.

    Args:
        fig: Figure object to convert
        fmt: Format to save in ('svg', 'png', etc.)

    Returns:
        bytes: The figure as bytes in the specified format
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", transparent=False)
    buf.seek(0)
    return buf.getvalue()


#############################################################
# TEMPERATURE PLOTTING FUNCTIONS                           #
#############################################################


def multi_year_plot(df: pd.DataFrame, fig: Figure, title: str, subtitle: str) -> Axes:
    """Create a multi-year line plot for temperature data.

    Historical station feeds can have long gaps. Plot with matplotlib directly
    so NaN gaps remain in the line data and render as visual breaks.

    Args:
        df: DataFrame containing temperature data with date index
        fig: Figure object to draw the plot on
        title: Main title for the plot
        subtitle: Subtitle/description for the plot

    Returns:
        Axes object with the configured plot
    """
    ax = fig.subplots()
    current_year = util.utc_now().year
    for column in df.columns:
        year = int(column)
        is_current_year = year == current_year
        historic_style_index = year % len(HISTORIC_TEMP_LINE_STYLES)
        historic_color = HISTORIC_TEMP_COLOR_PALETTE[
            year % len(HISTORIC_TEMP_COLOR_PALETTE)
        ]
        ax.plot(
            df.index,
            pd.to_numeric(df[column], errors="coerce"),
            label=str(column),
            linestyle=(
                "-"
                if is_current_year
                else HISTORIC_TEMP_LINE_STYLES[historic_style_index]
            ),
            linewidth=3 if is_current_year else 1.2,
            alpha=1.0 if is_current_year else 0.75,
            color="r" if is_current_year else historic_color,
        )

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
    ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Water Temp (°F)", fontsize=LABEL_FONT_SIZE)

    # Add second Y axis with Celsius
    add_celsius_axis(ax)

    ax.legend(loc="upper right")
    return ax


def _historic_temperature_plot_frame(water_temp_by_year: pd.DataFrame) -> pd.DataFrame:
    """Prepare historical temperatures for trend plotting.

    Small station data hiccups are interpolated before smoothing so they do not
    fragment the trend line. Longer outages remain missing so the plotted line
    breaks instead of drawing a false diagonal across the gap. The smoothed
    frame is also masked for visual artifacts: raw spikes, cross-year
    anomalies, sharp smoothed jumps, and tiny isolated plot segments.
    """
    smoothed_frame = _historic_temperature_smoothed_plot_frame(water_temp_by_year)
    cross_year_mask = _historic_temperature_plot_cross_year_artifact_mask(
        smoothed_frame
    )
    cross_year_suppressed = smoothed_frame.mask(cross_year_mask)
    volatility_suppressed = cross_year_suppressed.mask(
        _historic_temperature_plot_volatility_artifact_mask(cross_year_suppressed)
    )
    return _remove_short_historic_temperature_plot_segments(volatility_suppressed)


def _historic_temperature_smoothed_plot_frame(
    water_temp_by_year: pd.DataFrame,
) -> pd.DataFrame:
    gap_limit_rows = _gap_limit_rows(
        water_temp_by_year.index, MAX_HISTORIC_TEMP_PLOT_GAP
    )
    interpolated_columns = {}
    long_gap_masks = {}

    for column in water_temp_by_year.columns:
        source = pd.to_numeric(water_temp_by_year[column], errors="coerce")
        source = _suppress_historic_temperature_plot_spike_artifacts(source)
        long_gap_mask = _long_missing_gap_mask(source, gap_limit_rows)
        interpolated = source.interpolate(method="time", limit_area="inside")
        interpolated[long_gap_mask] = np.nan

        interpolated_columns[column] = interpolated
        long_gap_masks[column] = long_gap_mask

    interpolated_frame = pd.DataFrame(
        interpolated_columns, index=water_temp_by_year.index
    )
    long_gap_mask_frame = pd.DataFrame(long_gap_masks, index=water_temp_by_year.index)
    return interpolated_frame.rolling(24, center=True).mean().mask(long_gap_mask_frame)


def _historic_temperature_plot_cross_year_artifact_mask(
    smoothed_frame: pd.DataFrame,
) -> pd.DataFrame:
    seasonal_median = smoothed_frame.median(axis=1, skipna=True)
    mask = (
        smoothed_frame.sub(seasonal_median, axis=0).abs()
        > MAX_HISTORIC_TEMP_PLOT_CROSS_YEAR_RESIDUAL_F
    )
    return mask.rolling(24, center=True, min_periods=1).max().fillna(False).astype(bool)


def _historic_temperature_plot_volatility_artifact_mask(
    smoothed_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return smoothed plot segments that still move too sharply."""
    window_rows = _gap_limit_rows(
        smoothed_frame.index, HISTORIC_TEMP_PLOT_VOLATILITY_WINDOW
    )
    min_periods = max(3, window_rows // 3)
    rolling = smoothed_frame.rolling(
        window_rows,
        center=True,
        min_periods=min_periods,
    )
    rolling_range = rolling.max() - rolling.min()
    return rolling_range > MAX_HISTORIC_TEMP_PLOT_SMOOTHED_RANGE_F


def _remove_short_historic_temperature_plot_segments(
    plot_frame: pd.DataFrame,
) -> pd.DataFrame:
    return plot_frame.mask(_short_historic_temperature_plot_segment_mask(plot_frame))


def _short_historic_temperature_plot_segment_mask(
    plot_frame: pd.DataFrame,
) -> pd.DataFrame:
    min_segment_rows = _gap_limit_rows(plot_frame.index, MIN_HISTORIC_TEMP_PLOT_SEGMENT)
    masks = {}
    for column in plot_frame.columns:
        valid = plot_frame[column].notna()
        mask = pd.Series(False, index=plot_frame.index)
        if valid.any():
            segment_groups = valid.ne(valid.shift(fill_value=False)).cumsum()
            for _, segment in valid.groupby(segment_groups):
                if bool(segment.iloc[0]) and len(segment) < min_segment_rows:
                    mask.loc[segment.index] = True
        masks[column] = mask
    return pd.DataFrame(masks, index=plot_frame.index)


def _suppress_historic_temperature_plot_spike_artifacts(series: pd.Series) -> pd.Series:
    """Remove isolated historic temperature spikes before smoothing plots."""
    return series.mask(_historic_temperature_plot_spike_artifact_mask(series))


def _historic_temperature_plot_spike_artifact_mask(series: pd.Series) -> pd.Series:
    """Return isolated historic temperature spikes before smoothing plots."""
    window_rows = _gap_limit_rows(series.index, HISTORIC_TEMP_PLOT_ARTIFACT_WINDOW)
    if len(series.dropna()) < window_rows:
        return pd.Series(False, index=series.index)

    rolling_median = series.rolling(
        window_rows,
        center=True,
        min_periods=max(3, window_rows // 3),
    ).median()
    artifact_mask = (
        series.sub(rolling_median).abs() > MAX_HISTORIC_TEMP_PLOT_SPIKE_RESIDUAL_F
    )
    return artifact_mask


def _historic_temperature_plot_spike_artifact_counts(
    water_temp_by_year: pd.DataFrame,
) -> dict[str, int]:
    counts = {}
    for column in water_temp_by_year.columns:
        source = pd.to_numeric(water_temp_by_year[column], errors="coerce")
        artifact_count = int(
            _historic_temperature_plot_spike_artifact_mask(source).sum()
        )
        if artifact_count:
            counts[str(column)] = artifact_count
    return counts


def _historic_temperature_plot_cross_year_artifact_counts(
    water_temp_by_year: pd.DataFrame,
) -> dict[str, int]:
    smoothed_frame = _historic_temperature_smoothed_plot_frame(water_temp_by_year)
    mask = _historic_temperature_plot_cross_year_artifact_mask(smoothed_frame)
    return {
        str(column): int(count)
        for column, count in mask.sum().items()
        if int(count) > 0
    }


def _historic_temperature_plot_volatility_artifact_counts(
    water_temp_by_year: pd.DataFrame,
) -> dict[str, int]:
    smoothed_frame = _historic_temperature_smoothed_plot_frame(water_temp_by_year)
    cross_year_mask = _historic_temperature_plot_cross_year_artifact_mask(
        smoothed_frame
    )
    cross_year_suppressed = smoothed_frame.mask(cross_year_mask)
    mask = _historic_temperature_plot_volatility_artifact_mask(cross_year_suppressed)
    return {
        str(column): int(count)
        for column, count in mask.sum().items()
        if int(count) > 0
    }


def _historic_temperature_short_segment_counts(
    water_temp_by_year: pd.DataFrame,
) -> dict[str, int]:
    smoothed_frame = _historic_temperature_smoothed_plot_frame(water_temp_by_year)
    cross_year_mask = _historic_temperature_plot_cross_year_artifact_mask(
        smoothed_frame
    )
    cross_year_suppressed = smoothed_frame.mask(cross_year_mask)
    volatility_mask = _historic_temperature_plot_volatility_artifact_mask(
        cross_year_suppressed
    )
    volatility_suppressed = cross_year_suppressed.mask(volatility_mask)
    mask = _short_historic_temperature_plot_segment_mask(volatility_suppressed)
    return {
        str(column): int(count)
        for column, count in mask.sum().items()
        if int(count) > 0
    }


def _log_historic_temperature_plot_artifact_counts(
    location_code: str, water_temp_by_year: pd.DataFrame
) -> None:
    counts = _historic_temperature_plot_spike_artifact_counts(water_temp_by_year)
    cross_year_counts = _historic_temperature_plot_cross_year_artifact_counts(
        water_temp_by_year
    )
    volatility_counts = _historic_temperature_plot_volatility_artifact_counts(
        water_temp_by_year
    )
    short_segment_counts = _historic_temperature_short_segment_counts(
        water_temp_by_year
    )
    if (
        not counts
        and not cross_year_counts
        and not volatility_counts
        and not short_segment_counts
    ):
        logging.debug(
            f"[{location_code}] Historical temperature plot visual artifact suppression "
            f"flagged 0 points "
            f"(window={HISTORIC_TEMP_PLOT_ARTIFACT_WINDOW}, "
            f"threshold={MAX_HISTORIC_TEMP_PLOT_SPIKE_RESIDUAL_F}°F, "
            f"cross_year_threshold={MAX_HISTORIC_TEMP_PLOT_CROSS_YEAR_RESIDUAL_F}°F, "
            f"volatility_window={HISTORIC_TEMP_PLOT_VOLATILITY_WINDOW}, "
            f"volatility_threshold={MAX_HISTORIC_TEMP_PLOT_SMOOTHED_RANGE_F}°F, "
            f"min_segment={MIN_HISTORIC_TEMP_PLOT_SEGMENT})"
        )
        return

    logging.info(
        f"[{location_code}] Historical temperature plot visual artifact suppression "
        f"flagged "
        f"{sum(counts.values())} raw points by year: {counts}; "
        f"{sum(cross_year_counts.values())} cross-year smoothed points by year: {cross_year_counts}; "
        f"{sum(volatility_counts.values())} volatile smoothed points by year: {volatility_counts}; "
        f"{sum(short_segment_counts.values())} short-segment smoothed points by year: {short_segment_counts} "
        f"(window={HISTORIC_TEMP_PLOT_ARTIFACT_WINDOW}, "
        f"threshold={MAX_HISTORIC_TEMP_PLOT_SPIKE_RESIDUAL_F}°F, "
        f"cross_year_threshold={MAX_HISTORIC_TEMP_PLOT_CROSS_YEAR_RESIDUAL_F}°F, "
        f"volatility_window={HISTORIC_TEMP_PLOT_VOLATILITY_WINDOW}, "
        f"volatility_threshold={MAX_HISTORIC_TEMP_PLOT_SMOOTHED_RANGE_F}°F, "
        f"min_segment={MIN_HISTORIC_TEMP_PLOT_SEGMENT})"
    )


def _gap_limit_rows(index: pd.Index, gap_limit: pd.Timedelta) -> int:
    fallback_rows = int(gap_limit / pd.Timedelta(hours=1))
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return fallback_rows

    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return fallback_rows

    step = diffs.median()
    if step <= pd.Timedelta(0):
        return fallback_rows
    return max(1, math.floor(gap_limit / step))


def _long_missing_gap_mask(series: pd.Series, gap_limit_rows: int) -> pd.Series:
    missing = series.isna()
    mask = pd.Series(False, index=series.index)
    if not missing.any():
        return mask

    gap_groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    for _, gap in missing.groupby(gap_groups):
        if bool(gap.iloc[0]) and len(gap) > gap_limit_rows:
            mask.loc[gap.index] = True
    return mask


def live_temp_plot(
    df: pd.DataFrame,
    fig: Figure,
    title: str,
    subtitle: str,
    time_fmt: str,
) -> Axes:
    """Create a plot of recent temperature data with custom time formatting.

    Args:
        df: DataFrame containing temperature data with datetime index
        fig: Figure object to draw the plot on
        title: Main title for the plot
        subtitle: Subtitle/description for the plot
        time_fmt: Format string for time labels on x-axis (e.g., '%a %-I %p')

    Returns:
        Axes object with the configured plot
    """
    ax = fig.subplots()
    sns.lineplot(data=df, ax=ax)
    ax.xaxis.set_major_formatter(md.DateFormatter(time_fmt))

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
    ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE)
    ax.set_xlabel("Time", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Water Temp (°F)", fontsize=LABEL_FONT_SIZE)

    # Add second Y axis with Celsius
    add_celsius_axis(ax)

    return ax


def create_live_temp_plot(live_temps: pd.DataFrame, station_name: str | None) -> Figure:
    """Create a plot of recent water temperature data.

    Creates a plot showing both raw temperature readings and a 2-hour
    rolling average trend line for the past 48 hours.

    Args:
        live_temps: DataFrame containing temperature data
        station_name: Station name for plot title or None

    Returns:
        Figure object with the plot
    """
    # Extract water temperature and calculate the rolling average
    raw = live_temps["water_temp"]

    # Take the last 48 hours (at most)
    raw = raw.sort_index().iloc[-96:].copy()

    # Calculate 2-hour rolling average
    trend = raw.rolling(4, center=True).mean().dropna()

    # Create the DataFrame with both raw data and trend line
    df = pd.DataFrame(
        {
            "Water Temp": raw,
            "2-hour MA": trend,
        }
    )

    # Create the figure and plot
    fig = create_standard_figure()
    live_temp_plot(
        df,
        fig,
        f"{station_name} Water Temperature" if station_name else "Water Temperature",
        "last 48 hours",
        "%a %-I %p",  # Mon 3 PM
    )

    return fig


def generate_live_temp_plot(
    live_temps: pd.DataFrame, location_code: str, station_name: str | None
) -> bytes:
    """Generate a plot of recent water temperature data and return as bytes.

    Args:
        live_temps: DataFrame containing temperature data
        location_code: Location identifier for logging
        station_name: Station name for plot title or None

    Returns:
        bytes: The plot as SVG bytes
    """
    if len(live_temps) < 2:
        raise ValueError("Insufficient temperature data for plotting")

    logging.info(f"[{location_code}] Generating live temperature plot")
    try:
        live_fig = create_live_temp_plot(live_temps, station_name)
        plot_bytes = fig_to_bytes(live_fig)
        logging.info(f"[{location_code}] Live temperature plot generated successfully")
        return plot_bytes
    except Exception as e:
        logging.error(f"[{location_code}] Error generating live temperature plot: {e}")
        raise


def create_historic_monthly_plot(
    hist_temps: pd.DataFrame, station_name: str | None
) -> Figure:
    """Create a plot showing historical temperature data centered around the current date.

    Creates a plot showing water temperature data for a 2-month period centered
    on the current date (±30 days) with a 24-hour rolling mean.

    Args:
        hist_temps: DataFrame containing historical temperature data
        station_name: Station name for plot title or None

    Returns:
        Figure object with the plot
    """
    # Make sure we have columns for each year
    # Also ensure the index is by day-of-year (no year component)
    year_df = util.pivot_year(hist_temps)

    # Get the current date and create a 2-month window centered on today
    # Using a reference year (2020) to avoid leap year issues
    ref_date = util.utc_now().date().replace(year=2020)
    start_date = pd.to_datetime(ref_date - datetime.timedelta(days=30))
    end_date = pd.to_datetime(ref_date + datetime.timedelta(days=30))

    # Get the water_temp column and apply 24-hour rolling mean.
    water_temp_by_year = cast(pd.DataFrame, year_df["water_temp"])
    df = (
        _historic_temperature_plot_frame(water_temp_by_year)
        .loc[start_date:end_date]
        .dropna(axis=1, how="all")
    )

    # Create the 2-month plot
    fig = create_standard_figure()
    ax = multi_year_plot(
        df,
        fig,
        f"{station_name} Water Temperature" if station_name else "Water Temperature",
        "2 month, all years, 24-hour mean",
    )

    # Set x-axis formatting with weekly ticks
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))

    return fig


def create_historic_yearly_plot(
    hist_temps: pd.DataFrame, station_name: str | None
) -> Figure:
    """Create a plot showing historical temperature data for the full year.

    Creates a plot showing water temperature data across the entire year
    with a 24-hour rolling mean.

    Args:
        hist_temps: DataFrame containing historical temperature data
        station_name: Station name for plot title or None

    Returns:
        Figure object with the plot
    """
    # Make sure we have columns for each year
    # Also ensure the index is by day-of-year (no year component)
    year_df = util.pivot_year(hist_temps)

    # Get the water_temp column and apply 24-hour rolling mean.
    water_temp_by_year = cast(pd.DataFrame, year_df["water_temp"])
    # Some years may have 0 visible data after plot artifact suppression.
    # All-NA columns will cause plotting errors, so we remove them here.
    df = _historic_temperature_plot_frame(water_temp_by_year).dropna(axis=1, how="all")

    # Create the yearly plot
    fig = create_standard_figure()
    ax = multi_year_plot(
        df,  # type: ignore[arg-type] # pyright thinks df might be a Series, but it's a DataFrame
        fig,
        f"{station_name} Water Temperature" if station_name else "Water Temperature",
        "all years, 24-hour mean",
    )

    # Set x-axis formatting with month locators
    ax.xaxis.set_major_locator(md.MonthLocator(bymonthday=1))
    # X labels between gridlines
    ax.set_xticklabels("")  # type: ignore[operator]
    ax.xaxis.set_minor_locator(md.MonthLocator(bymonthday=15))
    ax.xaxis.set_minor_formatter(md.DateFormatter("%b"))

    return fig


def generate_historic_temp_plots(
    hist_temps: pd.DataFrame, location_code: str, station_name: str | None
) -> dict[str, bytes]:
    """Generate historical temperature plots and return as bytes.

    Creates plots showing water temperature data across multiple years,
    including 2-month and full-year comparisons.

    Args:
        hist_temps: DataFrame containing historical temperature data
        location_code: Location identifier for logging
        station_name: Station name for plot titles or None

    Returns:
        dict mapping period names to SVG bytes:
            - "2mo": 2-month centered plot
            - "12mo": Full year plot
    """
    if len(hist_temps) < 10:
        raise ValueError("Insufficient historical temperature data for plotting")

    logging.info(f"[{location_code}] Generating historic temperature plots")
    logging.debug(
        f"[{location_code}] Historic temperature data shape: {hist_temps.shape}"
    )

    try:
        year_df = util.pivot_year(hist_temps)
        water_temp_by_year = cast(pd.DataFrame, year_df["water_temp"])
        _log_historic_temperature_plot_artifact_counts(
            location_code, water_temp_by_year
        )

        # Create monthly plot
        logging.debug(f"[{location_code}] Creating 2-month historic plot")
        monthly_fig = create_historic_monthly_plot(hist_temps, station_name)
        if monthly_fig is None:
            raise RuntimeError("Failed to create historic monthly plot")
        monthly_bytes = fig_to_bytes(monthly_fig)
        logging.info(f"[{location_code}] 2-month historic plot generated successfully")

        # Create yearly plot
        logging.debug(f"[{location_code}] Creating yearly historic plot")
        yearly_fig = create_historic_yearly_plot(hist_temps, station_name)
        if yearly_fig is None:
            raise RuntimeError("Failed to create historic yearly plot")
        yearly_bytes = fig_to_bytes(yearly_fig)
        logging.info(f"[{location_code}] Yearly historic plot generated successfully")

        return {"2mo": monthly_bytes, "12mo": yearly_bytes}
    except Exception as e:
        logging.error(
            f"[{location_code}] Error generating historic temperature plots: {e}"
        )
        raise


#############################################################
# TIDE AND CURRENT PLOTTING FUNCTIONS                      #
#############################################################


def _tide_current_plot_window(
    now: datetime.datetime,
    marker_time: datetime.datetime,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return a plot window that keeps now and planner time visible."""
    marker_margin = datetime.timedelta(hours=3)
    default_end_time = now + datetime.timedelta(hours=21)
    return (
        min(now, marker_time) - marker_margin,
        max(default_end_time, marker_time + marker_margin),
    )


def _finite_plot_values(series: pd.Series, context: str) -> pd.Series:
    """Return finite values for plotting, failing clearly when none exist."""
    finite_values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        raise ValueError(f"Insufficient finite {context} data for plotting")
    return finite_values


def _plot_annotation_box() -> dict[str, object]:
    """Return a readable annotation background for dense SVG plots."""
    return {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": PLOT_GRID_COLOR,
        "linewidth": 0.5,
        "alpha": 0.78,
    }


def _is_high_tide_type(tide_type: object) -> bool:
    """Return whether a tide type value represents high tide."""
    return tide_type in {types.TideCategory.HIGH, types.TideCategory.HIGH.value}


def create_tide_current_plot(
    tides: pd.DataFrame,
    currents: pd.DataFrame,
    t: datetime.datetime,
    location_config: config_lib.LocationConfig,
) -> Figure:
    """Create a plot showing tide and current data.

    Creates a dual-axis plot with tide height and current velocity, showing a 24-hour window
    from 3 hours in the past to 21 hours in the future. Marks the specified time with a
    vertical line.

    Args:
        tides: DataFrame with tide predictions
        currents: DataFrame with current predictions
        t: Datetime to mark on the plot
        location_config: Location configuration for timezone settings

    Returns:
        Figure object with the plot, or None if data is not available
    """
    if len(tides) < 2:
        raise ValueError("Insufficient tide data for plotting")
    if len(currents) < 2:
        raise ValueError("Insufficient current data for plotting")

    # Select a window around now, extending when needed to keep the planner
    # marker visible at the far end of the next-day planning range.
    now = location_config.local_now()
    start_time, end_time = _tide_current_plot_window(now, t)

    # Interpolate only the numeric 'prediction' column for a smoother plot line.
    # Do this *before* filtering to the window to allow interpolation to use
    # data outside the window edges for better accuracy.
    tide_predictions = tides[["prediction"]].resample("60s").mean()
    if len(tides) >= 3:
        tides_interpolated = tide_predictions.interpolate("polynomial", order=2)
    else:
        tides_interpolated = tide_predictions.interpolate("time")

    # Filter all relevant DataFrames to the desired plot window
    tides = tides[(tides.index >= start_time) & (tides.index <= end_time)]  # type: ignore[operator] # pyright confused about index type
    currents = currents[(currents.index >= start_time) & (currents.index <= end_time)]  # type: ignore[operator] # pyright confused about index type
    tides_interpolated_filtered = tides_interpolated[
        (tides_interpolated.index >= start_time)
        & (tides_interpolated.index <= end_time)
    ]
    tide_plot_values = _finite_plot_values(
        tides_interpolated_filtered.prediction,
        "tide",
    )

    fig = Figure(figsize=(15, 6.5))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.16)
    fig.set_facecolor("white")

    ax = fig.subplots()
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)
    major_ticks = pd.date_range(
        pd.Timestamp(start_time).ceil("3h"),
        pd.Timestamp(end_time).floor("3h"),
        freq="3h",
    ).to_pydatetime()
    major_tick_values = md.date2num(major_ticks)
    ax.set_xticks(major_tick_values)
    ax.xaxis.set_major_formatter(md.DateFormatter("%a %-I %p"))
    ax.set_ylabel(
        "Current Speed (kts)", fontsize=LABEL_FONT_SIZE, color=CURRENT_FLOOD_COLOR
    )
    ax.grid(
        True,
        which="major",
        alpha=0.4,
        color=PLOT_GRID_COLOR,
        linestyle="-",
        linewidth=0.7,
    )
    ax.tick_params(axis="x", labelsize=LABEL_FONT_SIZE - 5, pad=8)
    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE - 4, colors=CURRENT_FLOOD_COLOR)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Plot the current on the first Y axis (left)
    # Find transitions between flood and ebb by checking consecutive values
    is_ebb = currents.velocity < 0
    transitions = is_ebb != is_ebb.shift()  # True at each transition point

    # Prepare separate dataframes for flood and ebb
    # Add NaN values at transition points to prevent connecting lines
    ebb_df = currents.copy()
    ebb_df.loc[~is_ebb | transitions, "velocity"] = np.nan

    flood_df = currents.copy()
    flood_df.loc[is_ebb | transitions, "velocity"] = np.nan

    # Plot flood currents in green
    ax.plot(
        flood_df.index,
        flood_df.velocity,
        color=CURRENT_FLOOD_COLOR,
        label="Flood speed",
        linewidth=2.15,
    )

    # Plot ebb currents in red
    ax.plot(
        ebb_df.index,
        ebb_df.velocity,
        color=CURRENT_EBB_COLOR,
        label="Ebb speed",
        linewidth=2.15,
    )

    # Find and mark local maxima and minima for currents

    # Process flood currents (find local maxima)
    if not flood_df.velocity.isna().all():
        # Need to handle NaN values properly for peak detection
        flood_vel = flood_df.velocity.dropna()

        # Find peaks with a minimum prominence to avoid minor fluctuations
        peaks, _ = find_peaks(flood_vel.values, prominence=0.3)

        # Add markers and labels for each peak
        for peak_idx in peaks:
            peak_time = flood_vel.index[peak_idx]
            peak_val = flood_vel.iloc[peak_idx]

            ax.scatter(
                [peak_time],
                [peak_val],
                marker="o",  # type: ignore[arg-type]
                color=CURRENT_FLOOD_COLOR,
                s=40,
                zorder=10,
            )
            ax.annotate(
                f"{peak_val:.1f} kt",
                xy=(peak_time, peak_val),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE - 3,
                fontweight="semibold",
                color=CURRENT_FLOOD_COLOR,
                bbox=_plot_annotation_box(),
            )

    # Process ebb currents (find local minima by finding peaks in the negative)
    if not ebb_df.velocity.isna().all():
        # Need to handle NaN values properly for peak detection
        ebb_vel = ebb_df.velocity.dropna()

        # For ebb currents, we're looking for minima, so negate the values to find peaks
        # and use the same peak finding algorithm
        neg_ebb_vel = -ebb_vel.to_numpy(dtype=float)  # Make values positive for peaks
        peaks, _ = find_peaks(neg_ebb_vel, prominence=0.3)

        # Add markers and labels for each peak (which are actually valleys in original data)
        for peak_idx in peaks:
            peak_time = ebb_vel.index[peak_idx]
            peak_val = ebb_vel.iloc[peak_idx]  # This is already negative

            ax.scatter(
                [peak_time],
                [peak_val],
                marker="o",  # type: ignore[arg-type]
                color=CURRENT_EBB_COLOR,
                s=40,
                zorder=10,
            )
            ax.annotate(
                f"{peak_val:.1f} kt",
                xy=(peak_time, peak_val),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE - 3,
                fontweight="semibold",
                color=CURRENT_EBB_COLOR,
                bbox=_plot_annotation_box(),
            )

    # Draw a line at current 0
    ax.axhline(0, color=CURRENT_FLOOD_COLOR, linestyle=":", alpha=0.55)

    # Keep the current range sensible
    ax.set_ylim(-2.5, 2.5)

    # Add a second Y axis for the tide (right)
    ax2 = ax.twinx()
    ax2.set_xticks(major_tick_values)
    ax2.set_ylabel("Tide Height (ft)", fontsize=LABEL_FONT_SIZE, color=TIDE_COLOR)
    ax2.grid(False)
    ax2.tick_params(axis="y", labelsize=LABEL_FONT_SIZE - 4, colors=TIDE_COLOR)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # Plot the tide
    ax2.plot(
        tides_interpolated_filtered.index,  # Use interpolated index
        tides_interpolated_filtered.prediction,  # Use interpolated prediction values
        color=TIDE_COLOR,
        label="Tide height",
        linewidth=2.15,
    )

    # Draw a line at tide 0
    ax2.axhline(0, color=TIDE_COLOR, linestyle=":", alpha=0.55)

    # Add markers for extreme tides (smaller size)
    extreme_tides = tides[
        tides["type"].isin(types.TIDE_TYPE_CATEGORIES)
    ].copy()  # Use original filtered tides DF
    ax2.scatter(
        extreme_tides.index,
        extreme_tides.prediction,
        marker="o",  # type: ignore[arg-type]
        color=TIDE_COLOR,
        s=40,  # Smaller marker size
    )

    for _, row in extreme_tides.iterrows():
        is_high_tide = _is_high_tide_type(row.type)
        ax2.annotate(
            f"{row.prediction:.1f} ft {row.type}",
            xy=(row.name, row.prediction),  # type: ignore[arg-type] # pyright expects Sequence[float] but row.name is a pandas timestamp
            xytext=(
                0,
                14 if is_high_tide else -14,
            ),  # Reduced offset
            textcoords="offset points",
            ha="center",
            va="bottom" if is_high_tide else "top",
            fontsize=ANNOTATION_FONT_SIZE - 3,  # Smaller font size
            fontweight="semibold",
            color=TIDE_COLOR,
            bbox=_plot_annotation_box(),
        )

    # Optimize Y-limits so the plots don't get squished by outliers
    # Keep the tide range sensible
    ax2.set_ylim(
        max(tide_plot_values.min() - 1, -8),
        min(tide_plot_values.max() + 1, 8),
    )

    # Mark now and planner time. When they are effectively the same timestamp,
    # use one marker so the chart does not imply a difference.
    now_float = md.date2num(now)
    ax.axvline(
        x=now_float,
        color=NOW_MARKER_COLOR,
        linestyle="-",
        alpha=0.7,
        linewidth=1.8,
    )
    marker_gap_seconds = abs((t - now).total_seconds())
    labels_would_overlap = 60 < marker_gap_seconds <= 2 * 60 * 60
    ax.annotate(
        "now",
        xy=(now_float, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, 7),
        textcoords="offset points",
        color=NOW_MARKER_COLOR,
        fontsize=ANNOTATION_FONT_SIZE - 2,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": PLOT_GRID_COLOR,
            "linewidth": 0.5,
            "alpha": 0.9,
        },
    )

    t_float = md.date2num(t)
    if marker_gap_seconds > 60:
        ax.axvline(
            x=t_float,
            color=PLANNED_MARKER_COLOR,
            linestyle="--",
            alpha=0.85,
            linewidth=2.2,
        )
        ax.annotate(
            "planned",
            xy=(t_float, 1.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0, 22 if labels_would_overlap else 7),
            textcoords="offset points",
            color=PLANNED_MARKER_COLOR,
            fontsize=ANNOTATION_FONT_SIZE - 2,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": PLOT_GRID_COLOR,
                "linewidth": 0.5,
                "alpha": 0.9,
            },
        )

    ax.set_xlim(md.date2num(start_time), md.date2num(end_time))

    # No titles for the tide/current plot as per original implementation

    # Add a compact combined legend for the plotted lines.
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        fontsize=LABEL_FONT_SIZE - 2,
        loc="upper right",
        framealpha=0.9,
        facecolor="white",
        edgecolor=PLOT_GRID_COLOR,
    )

    ax.tick_params(which="major", length=8, width=2)
    ax.tick_params(which="minor", length=4, width=1)

    return fig


# Current chart utility values and functions


@dataclass
class ArrowPosition:
    """Position and direction information for current arrows."""

    x: int  # x coordinate on the map
    y: int  # y coordinate on the map
    angle: int  # flood direction in degrees


def create_current_chart(ef: str, magnitude_bin: int, _: str = "") -> Figure:
    """Create a current chart showing water movement over a map.

    Creates a chart with arrows indicating water movement direction and strength
    over a base map of the area. Arrow size and width are proportional to current strength.

    Args:
        ef: Current direction value
        magnitude_bin: Magnitude bin value (0-100)
        location_code: The 3-letter location code (e.g., 'nyc')

    Returns:
        Figure object with the current chart

    Raises:
        ValueError: If magnitude_bin is outside the valid range
        ValueError: If ef is an invalid CurrentDirection
    """
    if magnitude_bin < 0 or magnitude_bin > 100:
        raise ValueError(f"magnitude_bin must be between 0 and 100: {magnitude_bin}")
    magnitude_pct = magnitude_bin / 100

    fig = Figure(figsize=CURRENT_CHART_SIZE)  # Dimensions: 2596 x 967

    ax = fig.subplots()
    map_img = mpimg.imread("static/base_coney_map.png")
    ax.imshow(map_img)
    del map_img
    ax.axis("off")
    ax.grid(False)

    ax.annotate(
        "Grimaldos\nChair",
        (1850, 400),
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight="bold",
        color=HIGHLIGHT_COLOR,
    )

    # Arrow positions for the current chart
    ARROWS: list[ArrowPosition] = [
        ArrowPosition(600, 800, 260),
        ArrowPosition(900, 750, 260),
        ArrowPosition(1150, 850, 335),
        ArrowPosition(1300, 820, 20),
        ArrowPosition(1520, 700, 75),
        ArrowPosition(1800, 640, 75),
        ArrowPosition(2050, 600, 75),
        ArrowPosition(2350, 550, 75),
    ]
    length = 80 + 80 * magnitude_pct
    width = 4 + 12 * magnitude_pct

    if ef == types.CurrentDirection.FLOODING.value:
        flip = 0
        color = CURRENT_FLOOD_COLOR
    elif ef == types.CurrentDirection.EBBING.value:
        flip = 180
        color = CURRENT_EBB_COLOR
    else:
        raise ValueError(ef)

    for arrow in ARROWS:
        dx = length * math.sin(math.radians(arrow.angle + flip))
        dy = length * math.cos(math.radians(arrow.angle + flip))
        ax.arrow(
            arrow.x - dx / 2,
            arrow.y + dy / 2,
            dx,
            -dy,
            width=width,
            color=color,
            length_includes_head=True,
        )

    return fig


def generate_and_save_current_chart(
    ef: str, magnitude_bin: int, location_code: str = "nyc"
) -> None:
    """Generate and save a current chart showing water movement over a map.

    Args:
        ef: Current direction value
        magnitude_bin: Magnitude bin value (0-100)
        location_code: The 3-letter location code (e.g., 'nyc')

    Returns:
        None - Saves the chart to a file
    """
    logging.info(
        f"Generating {ef} current chart with magnitude {magnitude_bin} for {location_code}"
    )

    try:
        # Generate filename and create directory
        plot_filename = get_current_chart_filename(ef, magnitude_bin, location_code)

        # Ensure the directory exists
        os.makedirs(f"static/plots/{location_code}", exist_ok=True)
        logging.debug(
            f"[{location_code}] Ensured directory exists: static/plots/{location_code}"
        )

        # Create and save the figure
        fig = create_current_chart(ef, magnitude_bin)
        if fig is None:
            raise RuntimeError("Failed to create current chart")
        save_fig(fig, plot_filename, fmt="png", location_code=location_code)
        logging.info(
            f"[{location_code}] Current chart ({ef}, {magnitude_bin}) generated successfully"
        )
    except Exception as e:
        logging.error(
            f"[{location_code}] Error generating current chart ({ef}, {magnitude_bin}): {e}"
        )
        raise


def _extract_location_code(path: str) -> str:
    """Extract location code from a file path.

    Handles different path formats:
    - static/plots/nyc/file.png
    - shallweswim/static/plots/nyc/file.png
    - /path/to/static/plots/nyc/file.png

    Returns "unknown" if no location code can be extracted.
    """
    if not path:
        return "unknown"

    # Match location code in plots directory path
    match = re.search(r"(?:^|/)plots/([a-z]{3})/", path)
    if match:
        return match.group(1)

    # Try to match just the 3-letter location code
    match = re.search(r"/([a-z]{3})/[^/]+$", path)
    if match:
        return match.group(1)

    return "unknown"

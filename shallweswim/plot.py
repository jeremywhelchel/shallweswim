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
from dataclasses import dataclass
from typing import List, Optional, Union

# Third-party imports
import matplotlib.dates as md
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import find_peaks

# Local imports
from shallweswim import config as config_lib
from shallweswim import types
from shallweswim import util


# Set default Seaborn theme settings for consistent plot appearance
sns.set_theme()
sns.axes_style("darkgrid")

# Constants for plot sizes
STANDARD_FIGURE_SIZE = (16, 8)  # Standard plot size in inches
CURRENT_CHART_SIZE = (16, 6)  # Size for current charts (2596 × 967 pixels)

# Font size constants
TITLE_FONT_SIZE = 24
SUBTITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 16

# Color constants
CURRENT_FLOOD_COLOR = "g"  # Green for flooding currents
CURRENT_EBB_COLOR = "r"  # Red for ebbing currents
TIDE_COLOR = "b"  # Blue for tide data
HIGHLIGHT_COLOR = "#ff4000"  # Orange for highlighting points of interest
TIME_MARKER_COLOR = "r"  # Red for marking current time

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


def get_plot_filepath(
    location_code: str, plot_type: str, extension: str = "svg"
) -> str:
    """Generate standard file path for plots.

    Args:
        location_code: Location identifier
        plot_type: Type of plot (e.g., 'current_temp', 'historic_temp_yr')
        extension: File extension without dot (defaults to 'svg')

    Returns:
        Full path to the plot file
    """
    return f"static/plots/{location_code}/{plot_type}.{extension}"


def save_fig(
    fig: Figure,
    dst: Union[str, io.StringIO],
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
        AssertionError: If a string path doesn't start with 'static/'
    """
    # If running outside the 'shallweswim' directory, prepend it to all paths
    if isinstance(dst, str):
        assert dst.startswith("static/"), dst
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
    try:
        fig.savefig(dst, format=fmt, bbox_inches="tight", transparent=False)
        # Log the absolute path for file output location
        if isinstance(dst, str):
            abs_path = os.path.abspath(dst)
            logging.info(f"[{location_code}] Plot saved to absolute path: {abs_path}")

        logging.debug(f"[{location_code}] Plot saved successfully")
    except Exception as e:
        logging.error(f"[{location_code}] Error saving plot: {e}")
        raise


#############################################################
# TEMPERATURE PLOTTING FUNCTIONS                           #
#############################################################


def multi_year_plot(df: pd.DataFrame, fig: Figure, title: str, subtitle: str) -> Axes:
    """Create a multi-year line plot for temperature data.

    Args:
        df: DataFrame containing temperature data with date index
        fig: Figure object to draw the plot on
        title: Main title for the plot
        subtitle: Subtitle/description for the plot

    Returns:
        Axes object with the configured plot
    """
    ax = sns.lineplot(data=df, ax=fig.subplots())  # type: Axes

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
    ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Water Temp (°F)", fontsize=LABEL_FONT_SIZE)

    # Add second Y axis with Celsius
    add_celsius_axis(ax)

    # Make current year stand out with bold red line.
    data_line = [l for l in ax.lines if len(l.get_xdata())][-1]
    legend_line = ax.legend().get_lines()[-1]
    for line in [data_line, legend_line]:
        line.set_linewidth(3)
        line.set_linestyle("-")
        line.set_color("r")
    return ax


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


def create_live_temp_plot(
    live_temps: pd.DataFrame, station_name: Optional[str]
) -> Figure:
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


def generate_and_save_live_temp_plot(
    live_temps: pd.DataFrame, location_code: str, station_name: Optional[str]
) -> None:
    """Generate and save a plot of recent water temperature data.

    Args:
        live_temps: DataFrame containing temperature data
        location_code: Location identifier for file naming
        station_name: Station name for plot title or None

    Returns:
        None - Saves the plot to a file
    """
    # Assert we have sufficient data
    assert (
        live_temps is not None and len(live_temps) >= 2
    ), "Insufficient temperature data for plotting"

    logging.info(f"[{location_code}] Generating live temperature plot")
    try:
        live_fig = create_live_temp_plot(live_temps, station_name)
        # Save to file
        plot_filename = get_plot_filepath(location_code, "live_temps")
        save_fig(live_fig, plot_filename, location_code=location_code)
        logging.info(f"[{location_code}] Live temperature plot generated successfully")
    except Exception as e:
        logging.error(f"[{location_code}] Error generating live temperature plot: {e}")
        raise


def create_historic_monthly_plot(
    hist_temps: pd.DataFrame, station_name: Optional[str]
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

    # Get the water_temp column and apply 24-hour rolling mean
    df = year_df["water_temp"].loc[start_date:end_date].rolling(24, center=True).mean()

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
    hist_temps: pd.DataFrame, station_name: Optional[str]
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

    # Get the water_temp column, apply 24-hour rolling mean
    # and handle NaN values as in the original implementation
    df = (
        year_df["water_temp"]
        .rolling(24, center=True)
        .mean()
        # Kludge to prevent seaborn from connecting over nan gaps
        .fillna(np.inf)
        # Some years may have 0 data at this filtering level. All-NA columns
        # will cause plotting errors, so we remove them here.
        .dropna(axis=1, how="all")
    )

    # Create the yearly plot
    fig = create_standard_figure()
    ax = multi_year_plot(
        df,
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


def generate_and_save_historic_plots(
    hist_temps: pd.DataFrame, location_code: str, station_name: Optional[str]
) -> None:
    """Generate and save historical temperature plots.

    Creates plots showing water temperature data across multiple years,
    including 2-month and full-year comparisons.

    Args:
        hist_temps: DataFrame containing historical temperature data
        location_code: Location identifier for file naming
        station_name: Station name for plot titles or None

    Returns:
        None - Saves the plots to files
    """
    # Assert we have sufficient data
    assert (
        hist_temps is not None and len(hist_temps) >= 10
    ), "Insufficient historical temperature data for plotting"

    logging.info(f"[{location_code}] Generating historic temperature plots")
    logging.debug(
        f"[{location_code}] Historic temperature data shape: {hist_temps.shape}"
    )

    try:
        # Create and save monthly plot
        logging.debug(f"[{location_code}] Creating 2-month historic plot")
        monthly_fig = create_historic_monthly_plot(hist_temps, station_name)
        assert monthly_fig is not None, "Failed to create historic monthly plot"
        mon_plot_filename = get_plot_filepath(
            location_code, "historic_temps_2mo_24h_mean"
        )
        save_fig(monthly_fig, mon_plot_filename, location_code=location_code)
        logging.info(f"[{location_code}] 2-month historic plot generated successfully")

        # Create and save yearly plot
        logging.debug(f"[{location_code}] Creating yearly historic plot")
        yearly_fig = create_historic_yearly_plot(hist_temps, station_name)
        assert yearly_fig is not None, "Failed to create historic yearly plot"
        yr_plot_filename = get_plot_filepath(
            location_code, "historic_temps_12mo_24h_mean"
        )
        save_fig(yearly_fig, yr_plot_filename, location_code=location_code)
        logging.info(f"[{location_code}] Yearly historic plot generated successfully")
    except Exception as e:
        logging.error(
            f"[{location_code}] Error generating historic temperature plots: {e}"
        )
        raise


#############################################################
# TIDE AND CURRENT PLOTTING FUNCTIONS                      #
#############################################################


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
    # Assert that we have sufficient data
    assert tides is not None and len(tides) >= 2, "Insufficient tide data for plotting"
    assert (
        currents is not None and len(currents) >= 2
    ), "Insufficient current data for plotting"

    # Select a window of data around the current time
    start_time = location_config.local_now() - datetime.timedelta(hours=3)
    end_time = location_config.local_now() + datetime.timedelta(hours=21)

    # Interpolate only the numeric 'prediction' column for a smoother plot line.
    # Do this *before* filtering to the window to allow interpolation to use
    # data outside the window edges for better accuracy.
    tides_interpolated = (
        tides[["prediction"]].resample("60s").interpolate("polynomial", order=2)
    )

    # Filter all relevant DataFrames to the desired plot window
    tides = tides[(tides.index >= start_time) & (tides.index <= end_time)]
    currents = currents[(currents.index >= start_time) & (currents.index <= end_time)]
    tides_interpolated_filtered = tides_interpolated[
        (tides_interpolated.index >= start_time)
        & (tides_interpolated.index <= end_time)
    ]

    fig = create_standard_figure()

    ax = fig.subplots()
    ax.xaxis.set_major_formatter(md.DateFormatter("%a %-I %p"))
    ax.set_ylabel(
        "Current Speed (kts)", fontsize=LABEL_FONT_SIZE, color=CURRENT_FLOOD_COLOR
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.tick_params(labelsize=LABEL_FONT_SIZE - 4)

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
        label="Flood",
        linewidth=2,
    )

    # Plot ebb currents in red
    ax.plot(
        ebb_df.index,
        ebb_df.velocity,
        color=CURRENT_EBB_COLOR,
        label="Ebb",
        linewidth=2,
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
                f"{peak_val:.1f}kt",
                xy=(peak_time, peak_val),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE - 2,
                fontweight="bold",
                color=CURRENT_FLOOD_COLOR,
            )

    # Process ebb currents (find local minima by finding peaks in the negative)
    if not ebb_df.velocity.isna().all():
        # Need to handle NaN values properly for peak detection
        ebb_vel = ebb_df.velocity.dropna()

        # For ebb currents, we're looking for minima, so negate the values to find peaks
        # and use the same peak finding algorithm
        neg_ebb_vel = -ebb_vel.values  # Make negative values positive to find peaks
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
                f"{peak_val:.1f}kt",
                xy=(peak_time, peak_val),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE - 2,
                fontweight="bold",
                color=CURRENT_EBB_COLOR,
            )

    # Draw a line at current 0
    ax.axhline(0, color=CURRENT_FLOOD_COLOR, linestyle=":", alpha=0.8)

    # Keep the current range sensible
    ax.set_ylim(-2.5, 2.5)

    # Add a second Y axis for the tide (right)
    ax2 = ax.twinx()
    ax2.set_ylabel("Tide Height (ft)", fontsize=LABEL_FONT_SIZE, color=TIDE_COLOR)
    ax2.grid(False)
    ax2.tick_params(labelsize=LABEL_FONT_SIZE - 4)

    # Plot the tide
    ax2.plot(
        tides_interpolated_filtered.index,  # Use interpolated index
        tides_interpolated_filtered.prediction,  # Use interpolated prediction values
        color=TIDE_COLOR,
        label="Tide height",
        linewidth=2,
    )

    # Draw a line at tide 0
    ax2.axhline(0, color=TIDE_COLOR, linestyle=":", alpha=0.8)

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
        ax2.annotate(
            f"{row.prediction:.1f}ft {row.type}",
            xy=(row.name, row.prediction),
            xytext=(
                0,
                8 if row.type == types.TideCategory.HIGH else -15,
            ),  # Reduced offset
            textcoords="offset points",
            ha="center",
            va="center" if row.type == types.TideCategory.HIGH else "top",
            fontsize=ANNOTATION_FONT_SIZE - 2,  # Smaller font size
            fontweight="bold",
            color=TIDE_COLOR,
        )

    # Optimize Y-limits so the plots don't get squished by outliers
    # Keep the tide range sensible
    ax2.set_ylim(
        max(tides.prediction.min() - 1, -8),
        min(tides.prediction.max() + 1, 8),
    )

    # Mark where we are in time
    # Convert datetime to a float for axvline
    t_float = md.date2num(t)
    ax.axvline(
        x=t_float, color=TIME_MARKER_COLOR, linestyle="--", alpha=0.5, linewidth=2
    )

    # No titles for the tide/current plot as per original implementation

    # Add a legend for the current
    ax.legend(fontsize=LABEL_FONT_SIZE, loc="lower right")

    ax.tick_params(which="major", length=8, width=2)
    ax.tick_params(which="minor", length=4, width=1)

    return fig


def generate_and_save_tide_current_plot(
    tides: pd.DataFrame,
    currents: pd.DataFrame,
    t: datetime.datetime,
    location_config: config_lib.LocationConfig,
    filename: str,
) -> None:
    """Generate and save a plot showing tide and current data.

    Args:
        tides: DataFrame with tide predictions
        currents: DataFrame with current predictions
        t: Datetime to mark on the plot
        location_config: Location configuration for timezone settings
        filename: Filename to save the plot

    Returns:
        None
    """
    fig = create_tide_current_plot(tides, currents, t, location_config)
    assert fig is not None, "Failed to create tide and current plot"

    # Save the figure to the provided filename
    save_fig(fig, filename, location_code=location_config.code)


# Current chart utility values and functions


@dataclass
class ArrowPosition:
    """Position and direction information for current arrows."""

    x: int  # x coordinate on the map
    y: int  # y coordinate on the map
    angle: int  # flood direction in degrees


# Magnitude bins for discretizing current strengths
# TODO: Something off with these. Likely should be using the midpoint or some such
# Which image is representative for the full range?
MAGNITUDE_BINS = [0, 10, 30, 45, 55, 70, 90, 100]


def bin_magnitude(magnitude_pct: float) -> int:
    """Convert a magnitude percentage to a binned value.

    Maps a magnitude percentage (0.0-1.0) to one of the predefined bin values
    in MAGNITUDE_BINS to determine which current chart to display.

    Args:
        magnitude_pct: Magnitude as a percentage (0.0-1.0)

    Returns:
        The bin value (integer from MAGNITUDE_BINS)

    Raises:
        AssertionError: If magnitude_pct is outside the valid range
    """
    assert magnitude_pct >= 0 and magnitude_pct <= 1.0, magnitude_pct
    i = np.digitize([magnitude_pct * 100], MAGNITUDE_BINS, right=True)[0]
    return int(MAGNITUDE_BINS[i])


def get_current_chart_filename(
    ef: str, magnitude_bin: int, location_code: str = "nyc"
) -> str:
    """Generate a filename for a current chart.

    Args:
        ef: Current direction (flooding or ebbing)
        magnitude_bin: Binned magnitude value (from bin_magnitude)
        location_code: The 3-letter location code (e.g., 'nyc')

    Returns:
        Path to the PNG file for the specified current conditions
    """
    # Make sure the path starts with /static/ to be properly accessible from any route
    plot_filename = (
        f"/static/plots/{location_code}/current_chart_{ef}_{magnitude_bin}.png"
    )
    return plot_filename


def create_current_chart(ef: str, magnitude_bin: int, _: str = "") -> Figure:
    """Create a current chart showing water movement over a map.

    Creates a chart with arrows indicating water movement direction and strength
    over a base map of the area. Arrow size and width are proportional to current strength.

    Args:
        ef: Current direction (flooding or ebbing)
        magnitude_bin: Magnitude bin value (0-100)
        location_code: The 3-letter location code (e.g., 'nyc')

    Returns:
        Figure object with the current chart

    Raises:
        AssertionError: If magnitude_bin is outside the valid range
        ValueError: If ef is an invalid CurrentDirection
    """
    assert (magnitude_bin >= 0) and (magnitude_bin <= 100), magnitude_bin
    magnitude_pct = magnitude_bin / 100

    fig = Figure(figsize=CURRENT_CHART_SIZE)  # Dimensions: 2596 × 967

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
    ARROWS: List[ArrowPosition] = [
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

    if ef == "flooding":
        flip = 0
        color = CURRENT_FLOOD_COLOR
    elif ef == "ebbing":
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
        ef: Current direction (flooding or ebbing)
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
        assert fig is not None, "Failed to create current chart"
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

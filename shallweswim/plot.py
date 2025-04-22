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

# Local imports
from shallweswim import config as config_lib
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
    ax2.set_ylim(util.F2C(fmin), util.F2C(fmax))
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


def save_fig(fig: Figure, dst: Union[str, io.StringIO], fmt: str = "svg") -> None:
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
        if not os.path.exists("static/") and os.path.exists("shallweswim/static/"):
            dst = f"shallweswim/{dst}"
        # Create directory if it doesn't exist
        dirname = os.path.dirname(dst)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
    fig.savefig(dst, format=fmt, bbox_inches="tight", transparent=False)


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
    # Take the last 48 hours (at most)
    last_2days = live_temps.sort_index().iloc[-96:].copy()

    # Calculate 2-hour rolling average (for trend line, but drop nan values)
    ma = last_2days.rolling(4, center=True).mean().dropna()

    # Prepare a combined dataset with both raw readings and moving average
    df = pd.DataFrame(
        {
            "Water Temp": last_2days,
            "2-hour MA": ma,
        }
    )

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

    logging.info("Generating live temp plot for %s", location_code)
    fig = create_live_temp_plot(live_temps, station_name)
    plot_filename = get_plot_filepath(location_code, "current_temp")
    save_fig(fig, plot_filename)


def create_historic_monthly_plot(
    hist_temps: pd.DataFrame, station_name: Optional[str]
) -> Figure:
    """Create a plot showing historical temperature data for the next two months.

    Args:
        hist_temps: DataFrame containing historical temperature data
        station_name: Station name for plot title or None

    Returns:
        Figure object with the plot
    """
    # Make sure we have columns for each year
    # Also ensure the index is by day-of-year (no year component)
    df = util.PivotYear(hist_temps)

    # Take data for the next two months, relative to today
    today = datetime.datetime.now().timetuple().tm_yday
    start_date = today + 1
    end_date = min(today + 60, 365)
    df_mon = df.loc[start_date:end_date]

    # Create the 2-month plot
    fig = create_standard_figure()
    multi_year_plot(
        df_mon,
        fig,
        f"{station_name} Water Temperature" if station_name else "Water Temperature",
        "next 60 days",
    )
    return fig


def create_historic_yearly_plot(
    hist_temps: pd.DataFrame, station_name: Optional[str]
) -> Figure:
    """Create a plot showing historical temperature data for the full year.

    Args:
        hist_temps: DataFrame containing historical temperature data
        station_name: Station name for plot title or None

    Returns:
        Figure object with the plot
    """
    # Make sure we have columns for each year
    # Also ensure the index is by day-of-year (no year component)
    df = util.PivotYear(hist_temps)

    # Calculate 28-day rolling averages for each column in the dataframe
    smoothed = df.copy()
    # This is filling with 'inf' to create breaks in lines (instead of connecting
    # across missing data, which could be very misleading)
    smoothed = (
        smoothed.fillna(np.inf)
        .rolling(28, center=True)
        .mean()
        # After taking the mean, any columns that are all NA
        # will cause plotting errors, so we remove them here.
        .dropna(axis=1, how="all")
    )

    # Create the yearly plot
    fig = create_standard_figure()
    multi_year_plot(
        smoothed,
        fig,
        f"{station_name} Water Temperature" if station_name else "Water Temperature",
        "28-day rolling average",
    )
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

    logging.info("Generating historic temp plots for %s", location_code)

    # Create and save monthly plot
    monthly_fig = create_historic_monthly_plot(hist_temps, station_name)
    assert monthly_fig is not None, "Failed to create historic monthly plot"
    mon_plot_filename = get_plot_filepath(location_code, "historic_temp_2mo")
    save_fig(monthly_fig, mon_plot_filename)

    # Create and save yearly plot
    yearly_fig = create_historic_yearly_plot(hist_temps, station_name)
    assert yearly_fig is not None, "Failed to create historic yearly plot"
    yr_plot_filename = get_plot_filepath(location_code, "historic_temp_yr")
    save_fig(yearly_fig, yr_plot_filename)


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
    start_time = location_config.LocalNow() - datetime.timedelta(hours=3)
    end_time = location_config.LocalNow() + datetime.timedelta(hours=21)

    # Filter the DataFrames to only include data within our time window
    # Use pandas' query method which is type-safe
    tides = tides[tides.index >= start_time]
    tides = tides[tides.index <= end_time]
    currents = currents[currents.index >= start_time]
    currents = currents[currents.index <= end_time]

    fig = create_standard_figure()

    ax = fig.subplots()
    ax.xaxis.set_major_formatter(md.DateFormatter("%a %-I %p"))
    ax.set_ylabel("Tide height (ft)", fontsize=LABEL_FONT_SIZE)
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.tick_params(labelsize=LABEL_FONT_SIZE - 4)

    # First plot the tide
    ax.plot(
        tides.index,
        tides.prediction,
        color=TIDE_COLOR,
        label="Tide height",
        linewidth=2,
    )

    # Add markers for extreme tides
    extreme_tides = tides[tides.type.isin(["high", "low"])].copy()
    # Using 'o' string for marker is valid in matplotlib but mypy doesn't recognize it
    # so we need to suppress the type error
    ax.scatter(
        extreme_tides.index,
        extreme_tides.prediction,
        marker="o",  # type: ignore[arg-type]
        color=TIDE_COLOR,
        s=80,
    )

    for _, row in extreme_tides.iterrows():
        ax.annotate(
            f"{row.prediction:.1f}ft {row.type}",
            xy=(row.name, row.prediction),
            xytext=(0, 10 if row.type == "high" else -20),
            textcoords="offset points",
            ha="center",
            va="center" if row.type == "high" else "top",
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
            color=TIDE_COLOR,
        )

    # Add a second Y axis for the current
    ax2 = ax.twinx()
    ax2.set_ylabel("Current (knots)", fontsize=LABEL_FONT_SIZE)
    ax2.axhline(y=0, linestyle="-", alpha=0.2, color="lightgrey")
    ax2.tick_params(labelsize=LABEL_FONT_SIZE - 4)
    ax2.grid(False)

    # Plot the current, which uses the second (right) Y axis
    ebb = currents[currents.velocity < 0].copy()
    flood = currents[currents.velocity >= 0].copy()

    # Plot flood currents in green
    ax2.plot(
        flood.index,
        flood.velocity,
        color=CURRENT_FLOOD_COLOR,
        label="Flood",
        linewidth=2,
    )

    # Plot ebb currents in red
    ax2.plot(
        ebb.index,
        ebb.velocity,
        color=CURRENT_EBB_COLOR,
        label="Ebb",
        linewidth=2,
    )

    # Optimize Y-limits so the plots don't get squished by outliers
    # Keep the tide range sensible
    ax.set_ylim(
        max(tides.prediction.min() - 1, -8),
        min(tides.prediction.max() + 1, 8),
    )
    # Keep the current range sensible
    ax2.set_ylim(-2.5, 2.5)

    # Mark where we are in time
    # Convert datetime to a float for axvline
    t_float = md.date2num(t)
    ax.axvline(
        x=t_float, color=TIME_MARKER_COLOR, linestyle="--", alpha=0.5, linewidth=2
    )

    # Add title (using the time of the marker)
    # Use a safer approach to get the station name
    location_name = getattr(location_config, "StationName", location_config.code)
    title = f"{location_name} Tide & Current"
    subtitle = f"{t:%A, %B %-d, %Y}"  # Tuesday, April 3, 2023
    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
    ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE)

    # Add a legend for the current
    ax2.legend(fontsize=LABEL_FONT_SIZE, loc="lower right")

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
    save_fig(fig, filename)


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
    # Generate filename and create directory
    plot_filename = get_current_chart_filename(ef, magnitude_bin, location_code)
    logging.info(
        "Generating current map with pct %.2f: %s", magnitude_bin / 100, plot_filename
    )

    # Ensure the directory exists
    os.makedirs(f"static/plots/{location_code}", exist_ok=True)

    # Create and save the figure
    fig = create_current_chart(ef, magnitude_bin)
    assert fig is not None, "Failed to create current chart"
    save_fig(fig, plot_filename, fmt="png")

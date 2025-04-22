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


def generate_live_temp_plot(
    live_temps: Optional[pd.DataFrame], location_code: str, station_name: Optional[str]
) -> None:
    """Generate and save a plot of recent water temperature data.

    Creates a plot showing both raw temperature readings and a 2-hour
    rolling average trend line for the past 48 hours.

    Args:
        live_temps: DataFrame containing temperature data or None
        location_code: Location identifier for file naming
        station_name: Station name for plot title or None

    Returns:
        None - Saves the plot to a file if data is available
    """
    if live_temps is None or len(live_temps) < 2:
        return

    logging.info("Generating live temp plot for %s", location_code)
    plot_filename = f"static/plots/{location_code}/current_temp.svg"

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

    save_fig(fig, plot_filename)


def generate_historic_plots(
    hist_temps: Optional[pd.DataFrame], location_code: str, station_name: Optional[str]
) -> None:
    """Generate and save historical temperature plots.

    Creates plots showing water temperature data across multiple years,
    including 2-month and full-year comparisons.

    Args:
        hist_temps: DataFrame containing historical temperature data or None
        location_code: Location identifier for file naming
        station_name: Station name for plot titles or None

    Returns:
        None - Saves the plots to files if data is available
    """
    if hist_temps is None or len(hist_temps) < 10:
        return

    logging.info("Generating historic temp plots for %s", location_code)
    mon_plot_filename = f"static/plots/{location_code}/historic_temp_2mo.svg"
    yr_plot_filename = f"static/plots/{location_code}/historic_temp_yr.svg"

    # Make sure we have columns for each year
    # Also ensure the index is by day-of-year (no year component)
    df = util.PivotYear(hist_temps)

    # Create a rolling 5-day average plot for the last 2 months
    today = datetime.datetime.now().replace(year=2020)
    start_date = (today - datetime.timedelta(days=60)).replace(year=2020)

    # Calculate 5-day moving average (for smoothed trend line)
    # and select only the last 2 months
    smoothed = (
        df.rolling(5, center=True).mean()
        # Kludge to prevent seaborn from connecting over nan gaps.
        .fillna(np.inf)
        # Some years may have 0 data at this filtering level. All-NA columns
        # will cause plotting errors, so we remove them here.
        .dropna(axis=1, how="all")
    )
    smoothed = smoothed.loc[start_date:today]  # type: ignore[misc]

    fig = create_standard_figure()
    ax = multi_year_plot(
        smoothed,
        fig,
        f"{station_name} Water Temperature" if station_name else "Water Temperature",
        "last 60 days, 5-day mean",
    )
    # Ensure X-axis has reasonable ticks
    ax.xaxis.set_major_locator(md.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %-d"))
    # X labels between gridlines
    plt_ticks = ax.get_xticks()  # type: ignore[operator]
    ax.set_xticks(plt_ticks, minor=True)  # type: ignore[operator]
    ax.set_xticks(plt_ticks - 5, False)  # type: ignore[operator]
    save_fig(fig, mon_plot_filename)

    # Create a plot for the entire year
    # Use a 24-hour moving average to smooth out the lines
    df = (
        df.rolling(24, center=True).mean()
        # Kludge to prevent seaborn from connecting over nan gaps.
        .fillna(np.inf)
        # Some years may have 0 data at this filtering level. All-NA columns
        # will cause plotting errors, so we remove them here.
        .dropna(axis=1, how="all")
    )
    fig = create_standard_figure()
    ax = multi_year_plot(
        df,
        fig,
        f"{station_name} Water Temperature",
        "all years, 24-hour mean",
    )
    ax.xaxis.set_major_locator(md.MonthLocator(bymonthday=1))
    # X labels between gridlines
    ax.set_xticklabels("")  # type: ignore[operator]
    ax.xaxis.set_minor_locator(md.MonthLocator(bymonthday=15))
    ax.xaxis.set_minor_formatter(md.DateFormatter("%b"))
    save_fig(fig, yr_plot_filename)


#############################################################
# TIDE AND CURRENT PLOTTING FUNCTIONS                      #
#############################################################


def generate_tide_current_plot(
    tides: pd.DataFrame,
    currents: pd.DataFrame,
    t: datetime.datetime,
    location_config: config_lib.LocationConfig,
) -> Optional[io.StringIO]:
    """Generate a plot showing tide and current data.

    Creates a dual-axis plot with tide height and current velocity, showing a 24-hour window
    from 3 hours in the past to 21 hours in the future. Marks the specified time with a
    vertical line and labels high/low tide points.

    Args:
        tides: DataFrame containing tide predictions
        currents: DataFrame containing current predictions
        t: Time point to mark on the plot (required)
        location_config: Location configuration containing timezone information

    Returns:
        StringIO object containing SVG image data, or None if data is not available
    """
    if tides is None or currents is None:
        return None

    logging.info("Generating tide and current plot for: %s", t)

    # XXX Do this directly in tide dataset?
    tides = tides.resample("60s").interpolate("polynomial", order=2)

    df = pd.DataFrame(
        {
            "tide": tides["prediction"],
            "tide_type": tides["type"],
            "current": currents["velocity"],
        }
    )
    df = df[
        location_config.LocalNow()  # type: ignore[misc]
        - datetime.timedelta(hours=3) : location_config.LocalNow()  # type: ignore[misc]
        + datetime.timedelta(hours=21)
    ]

    fig = create_standard_figure()

    ax = fig.subplots()
    ax.xaxis.set_major_formatter(md.DateFormatter("%a %-I %p"))
    sns.lineplot(data=df["current"], ax=ax, color=CURRENT_FLOOD_COLOR)
    ax.set_ylabel("Current Speed (kts)", color=CURRENT_FLOOD_COLOR)

    # TODO: Align the 0 line on both axes

    ax2 = ax.twinx()
    sns.lineplot(data=df["tide"], ax=ax2, color=TIDE_COLOR)
    ax2.set_ylabel("Tide Height (ft)", color=TIDE_COLOR)
    ax2.grid(False)

    # Attempts to line up 0 on both axises...
    # ax.set_ylim(-2,2)  naturally -1.5 to 1
    # ax2.set_ylim(-5,5)  naturally -1,5

    # Draw lines at 0
    ax.axhline(
        0, color=CURRENT_FLOOD_COLOR, linestyle=":", alpha=0.8
    )  # , linewidth=0.8)
    ax2.axhline(0, color=TIDE_COLOR, linestyle=":", alpha=0.8)  # , linewidth=0.8)

    ax.axvline(t, color=TIME_MARKER_COLOR, linestyle="-", alpha=0.6)  # type: ignore[arg-type]

    # Useful plot that indicates how the current (which?) LEADS the tide
    # TODO:
    # Peak flood ~2hrs before high tide
    # Peak ebb XX mins before low tide

    # Label high and low tide points
    tt = df[df["tide_type"].notnull()][["tide", "tide_type"]]
    tt["tide_type"] = tt["tide_type"] + " tide"
    sns.scatterplot(data=tt, ax=ax2, legend=False)
    for t, row in tt.iterrows():
        ax2.annotate(
            row["tide_type"],
            (t, row["tide"]),  # type: ignore[arg-type]
            color=TIDE_COLOR,
            xytext=(-24, 8),
            textcoords="offset pixels",
        )

    svg_io = io.StringIO()
    save_fig(fig, svg_io)
    svg_io.seek(0)
    return svg_io


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


def generate_current_chart(
    ef: str, magnitude_bin: int, location_code: str = "nyc"
) -> None:
    """Generate a current chart showing water movement over a map.

    Creates a chart with arrows indicating water movement direction and strength
    over a base map of the area. Arrow size and width are proportional to current strength.

    Args:
        ef: Current direction (flooding or ebbing)
        magnitude_bin: Magnitude bin value (0-100)
        location_code: The 3-letter location code (e.g., 'nyc')

    Raises:
        AssertionError: If magnitude_bin is outside the valid range
        ValueError: If ef is an invalid CurrentDirection
    """
    assert (magnitude_bin >= 0) and (magnitude_bin <= 100), magnitude_bin
    magnitude_pct = magnitude_bin / 100

    fig = Figure(figsize=CURRENT_CHART_SIZE)  # Dimensions: 2596 × 967
    plot_filename = get_current_chart_filename(ef, magnitude_bin, location_code)
    logging.info(
        "Generating current map with pct %.2f: %s", magnitude_pct, plot_filename
    )

    # Ensure the directory exists
    os.makedirs(f"static/plots/{location_code}", exist_ok=True)

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

    save_fig(fig, plot_filename, fmt="png")

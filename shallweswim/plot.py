"""Generation SWS plots and charts."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, Union
import datetime
import io
import logging
import math
import matplotlib.image as mpimg
import matplotlib.dates as md
import numpy as np
import os
import pandas as pd
import seaborn as sns

from shallweswim import util


sns.set_theme()
sns.axes_style("darkgrid")


def MultiYearPlot(df: pd.DataFrame, fig: Figure, title: str, subtitle: str) -> Axes:
    ax = sns.lineplot(data=df, ax=fig.subplots())  # type: Axes

    fig.suptitle(title, fontsize=24)
    ax.set_title(subtitle, fontsize=18)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Water Temp (°F)", fontsize=18)

    # Add second Y axis with Celsius
    ax2 = ax.twinx()
    fmin, fmax = ax.get_ylim()
    ax2.set_ylim(util.F2C(fmin), util.F2C(fmax))
    ax2.set_ylabel("Water Temp (°C)", fontsize=18)
    ax2.grid(None)

    # Make current year stand out with bold red line.
    data_line = [l for l in ax.lines if len(l.get_xdata())][-1]
    legend_line = ax.legend().get_lines()[-1]
    for line in [data_line, legend_line]:
        line.set_linewidth(3)
        line.set_linestyle("-")
        line.set_color("r")
    return ax


def LiveTempPlot(
    df: pd.DataFrame,
    fig: Figure,
    title: str,
    subtitle: str,
    time_fmt: str,
) -> Axes:
    ax = fig.subplots()
    sns.lineplot(data=df, ax=ax)
    ax.xaxis.set_major_formatter(md.DateFormatter(time_fmt))

    fig.suptitle(title, fontsize=24)
    ax.set_title(subtitle, fontsize=18)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Water Temp (°F)", fontsize=18)

    # Add second Y axis with Celsius
    ax2 = ax.twinx()
    fmin, fmax = ax.get_ylim()
    ax2.set_ylim(util.F2C(fmin), util.F2C(fmax))
    ax2.set_ylabel("Water Temp (°C)", fontsize=18)
    ax2.grid(None)

    return ax


def SaveFig(fig: Figure, dst: Union[str, io.StringIO], fmt: str = "svg") -> None:
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


def GenerateLiveTempPlot(
    live_temps: pd.DataFrame | None, location_code: str, station_name: str | None
) -> None:
    if live_temps is None:
        return
    plot_filename = f"static/plots/{location_code}/live_temps.svg"
    logging.info("Generating live temp plot: %s", plot_filename)
    raw = live_temps["water_temp"]
    trend = raw.rolling(10 * 2, center=True).mean()
    df = pd.DataFrame(
        {
            "live": raw,
            "trend (2-hr)": trend,
        }
    ).tail(10 * 24 * 2)
    fig = Figure(figsize=(16, 8))
    LiveTempPlot(
        df,
        fig,
        f"{station_name} Water Temperature",
        "48-hour, live",
        "%a %-I %p",
    )
    SaveFig(fig, plot_filename)


def GenerateHistoricPlots(
    hist_temps: pd.DataFrame | None, location_code: str, station_name: str | None
) -> None:
    if hist_temps is None:
        return
    year_df = util.PivotYear(hist_temps)

    # 2 Month plot
    two_mo_plot_filename = (
        f"static/plots/{location_code}/historic_temps_2mo_24h_mean.svg"
    )
    logging.info("Generating 2 month plot: %s", two_mo_plot_filename)
    df = (
        year_df["water_temp"]
        .loc[
            util.Now().date().replace(year=2020)  # type: ignore[misc]
            - datetime.timedelta(days=30) : util.Now().date().replace(year=2020)  # type: ignore[misc]
            + datetime.timedelta(days=30)
        ]
        .rolling(24, center=True)
        .mean()
    )
    fig = Figure(figsize=(16, 8))
    ax = MultiYearPlot(
        df,
        fig,
        f"{station_name} Water Temperature",
        "2 month, all years, 24-hour mean",
    )
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))
    SaveFig(fig, two_mo_plot_filename)

    # Full year
    yr_plot_filename = f"static/plots/{location_code}/historic_temps_12mo_24h_mean.svg"
    logging.info("Generating full time plot: %s", yr_plot_filename)
    df = (
        year_df["water_temp"]
        .rolling(24, center=True)
        .mean()
        # Kludge to prevent seaborn from connecting over nan gaps.
        .fillna(np.inf)
        # Some years may have 0 data at this filtering level. All-NA columns
        # will cause plotting errors, so we remove them here.
        .dropna(axis=1, how="all")
    )
    fig = Figure(figsize=(16, 8))
    ax = MultiYearPlot(
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
    SaveFig(fig, yr_plot_filename)


# XXX Return a tide image. Dont write it to filesystem
def GenerateTideCurrentPlot(
    tides: pd.DataFrame, currents: pd.DataFrame, t: Optional[datetime.datetime] = None
) -> Optional[io.StringIO]:
    if tides is None or currents is None:
        return None
    if not t:
        t = util.Now()
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
    # XXX
    df = df[
        util.Now()  # type: ignore[misc]
        - datetime.timedelta(hours=3) : util.Now()  # type: ignore[misc]
        + datetime.timedelta(hours=21)
    ]

    fig = Figure(figsize=(16, 8))

    ax = fig.subplots()
    ax.xaxis.set_major_formatter(md.DateFormatter("%a %-I %p"))
    sns.lineplot(data=df["current"], ax=ax, color="g")
    ax.set_ylabel("Current Speed (kts)", color="g")

    # XXX Align the 0 line on both

    ax2 = ax.twinx()
    sns.lineplot(data=df["tide"], ax=ax2, color="b")
    ax2.set_ylabel("Tide Height (ft)", color="b")
    ax2.grid(False)

    # Attempts to line up 0 on both axises...
    # ax.set_ylim(-2,2)  naturally -1.5 to 1
    # ax2.set_ylim(-5,5)  naturally -1,5

    # Draw lines at 0
    ax.axhline(0, color="g", linestyle=":", alpha=0.8)  # , linewidth=0.8)
    ax2.axhline(0, color="b", linestyle=":", alpha=0.8)  # , linewidth=0.8)

    ax.axvline(t, color="r", linestyle="-", alpha=0.6)  # type: ignore[arg-type]

    # Useful plot that indicates how the current (which?) LEADS the tide
    # XXX
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
            color="b",
            xytext=(-24, 8),
            textcoords="offset pixels",
        )

    svg_io = io.StringIO()
    SaveFig(fig, svg_io)
    svg_io.seek(0)
    return svg_io


MAGNITUDE_BINS = [0, 10, 30, 45, 55, 70, 90, 100]  # XXX Something off with these


# XXX These bins are weird. Likely should be using the midpoint or some such
# Which image is representative for the full range?
def BinMagnitude(magnitude_pct: float) -> int:
    assert magnitude_pct >= 0 and magnitude_pct <= 1.0, magnitude_pct
    i = np.digitize([magnitude_pct * 100], MAGNITUDE_BINS, right=True)[0]
    return int(MAGNITUDE_BINS[i])


def GetCurrentChartFilename(ef: str, magnitude_bin: int) -> str:
    # magnitude_bin = BinMagnitude(magnitude_pct)
    plot_filename = f"static/plots/current_chart_{ef}_{magnitude_bin}.png"
    return plot_filename


def GenerateCurrentChart(ef: str, magnitude_bin: int) -> None:
    assert (magnitude_bin >= 0) and (magnitude_bin <= 100), magnitude_bin
    magnitude_pct = magnitude_bin / 100

    fig = Figure(figsize=(16, 6))  # Dims: 2596 × 967
    plot_filename = GetCurrentChartFilename(ef, magnitude_bin)
    logging.info(
        "Generating current map with pct %.2f: %s", magnitude_pct, plot_filename
    )

    ax = fig.subplots()
    map_img = mpimg.imread("static/base_coney_map.png")
    ax.imshow(map_img)
    del map_img
    ax.axis("off")
    ax.grid(False)

    ax.annotate(
        "Grimaldos\nChair", (1850, 400), fontsize=16, fontweight="bold", color="#ff4000"
    )

    # x,y (for centerpoint), flood_direction(degrees)
    ARROWS = [
        (600, 800, 260),
        (900, 750, 260),
        (1150, 850, 335),
        (1300, 820, 20),
        (1520, 700, 75),
        (1800, 640, 75),
        (2050, 600, 75),
        (2350, 550, 75),
    ]
    length = 80 + 80 * magnitude_pct
    width = 4 + 12 * magnitude_pct

    if ef == "flooding":
        flip = 0
        color = "g"
    elif ef == "ebbing":
        flip = 180
        color = "r"
    else:
        raise ValueError(ef)

    for x, y, angle in ARROWS:
        dx = length * math.sin(math.radians(angle + flip))
        dy = length * math.cos(math.radians(angle + flip))
        ax.arrow(
            x - dx / 2,
            y + dy / 2,
            dx,
            -dy,
            width=width,
            color=color,
            length_includes_head=True,
        )

    SaveFig(fig, plot_filename, fmt="png")

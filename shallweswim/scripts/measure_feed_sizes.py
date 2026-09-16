"""Measure serialized feed and plot sizes for the bundle.

This is an operational, network-fetching investigation script (not a test). It
fetches every configured feed for one or more locations through the existing
clients and feed classes, then reports row counts, in-memory DataFrame size,
and serialized Parquet size (SVG size for plots). It never writes any file to
disk and never writes to the observation archive; all serialization happens in
memory via ``io.BytesIO``.

It measures the serialized size of every current feed and plot, and whether
any combined per-location historical frame is large enough to justify finer
serving partitions than the one object per feed the bundle uses
(DATA_PIPELINE.md, "The bundle"). For historical temperature feeds, the
combined frame is also split by the feed's local-time calendar year and each
year is measured separately as an approximation of a per-year serving
partition. This differs from the archive's UTC year-boundary contract (see
``shallweswim/archive/capture.py``), which converts timestamps to UTC before
partitioning; the approximation here is sufficient for a sizing estimate.

Usage:
    uv run python -m shallweswim.scripts.measure_feed_sizes
    uv run python -m shallweswim.scripts.measure_feed_sizes --location nyc
    uv run python -m shallweswim.scripts.measure_feed_sizes --json
"""

import argparse
import asyncio
import io
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import aiohttp
import pandas as pd

from shallweswim import config, plot
from shallweswim.clients import create_api_clients
from shallweswim.clients.base import BaseApiClient
from shallweswim.config.locations import LocationConfig
from shallweswim.core import feeds
from shallweswim.core.manager import build_feeds


@dataclass(frozen=True)
class FeedMeasurement:
    """Measured size for one feed's fetched data."""

    location: str
    feed: str
    source: str | None
    rows: int
    memory_bytes: int
    parquet_bytes: int
    error: str | None = None


@dataclass(frozen=True)
class YearPartitionMeasurement:
    """Measured Parquet size for one calendar-year slice of a historical feed."""

    location: str
    year: int
    rows: int
    parquet_bytes: int


@dataclass(frozen=True)
class PlotMeasurement:
    """Measured SVG byte size for one generated plot."""

    location: str
    plot: str
    svg_bytes: int
    error: str | None = None


@dataclass(frozen=True)
class LocationReport:
    """All measurements collected for one location."""

    location: str
    feeds: list[FeedMeasurement] = field(default_factory=list)
    year_partitions: list[YearPartitionMeasurement] = field(default_factory=list)
    plots: list[PlotMeasurement] = field(default_factory=list)


def _parquet_bytes(df: pd.DataFrame) -> int:
    """Return the serialized Parquet size of a DataFrame in bytes.

    Serializes entirely in memory; nothing is written to disk.

    Args:
        df: DataFrame to serialize.

    Returns:
        Size of the Parquet-encoded bytes.
    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow")
    return buffer.getbuffer().nbytes


def _year_partitions(location: str, df: pd.DataFrame) -> list[YearPartitionMeasurement]:
    """Split a combined historical frame by local calendar year and measure each slice.

    Args:
        location: Location code, for labeling the resulting measurements.
        df: Combined historical temperature frame with a DatetimeIndex.

    Returns:
        One measurement per year present in the frame, sorted by year.
    """
    measurements = [
        YearPartitionMeasurement(
            location=location,
            year=int(year),
            rows=len(year_df),
            parquet_bytes=_parquet_bytes(year_df),
        )
        for year, year_df in df.groupby(df.index.year)
    ]
    return sorted(measurements, key=lambda measurement: measurement.year)


def _human_bytes(num_bytes: int) -> str:
    """Format a byte count using binary units for compact table display.

    Args:
        num_bytes: Size in bytes.

    Returns:
        A short human-readable size, such as ``"12.3KiB"``.
    """
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024:
            return f"{int(value)}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GiB"


def summarize(reports: list[LocationReport]) -> dict[str, Any]:
    """Aggregate totals across all measured locations.

    Args:
        reports: Per-location measurement reports.

    Returns:
        A dict with per-feed-type totals, grand totals, error counts, and the
        location with the largest combined historical temperature frame.
    """
    feed_totals: dict[str, dict[str, int]] = {}
    total_feed_parquet = 0
    total_feed_memory = 0
    total_rows = 0
    feed_errors = 0
    plot_errors = 0
    total_plot_bytes = 0
    largest_historic: tuple[str, int] | None = None

    for report in reports:
        for measurement in report.feeds:
            if measurement.error:
                feed_errors += 1
                continue
            bucket = feed_totals.setdefault(
                measurement.feed,
                {"rows": 0, "memory_bytes": 0, "parquet_bytes": 0, "locations": 0},
            )
            bucket["rows"] += measurement.rows
            bucket["memory_bytes"] += measurement.memory_bytes
            bucket["parquet_bytes"] += measurement.parquet_bytes
            bucket["locations"] += 1
            total_feed_parquet += measurement.parquet_bytes
            total_feed_memory += measurement.memory_bytes
            total_rows += measurement.rows
            if measurement.feed == feeds.FEED_HISTORIC_TEMPS.value and (
                largest_historic is None
                or measurement.parquet_bytes > largest_historic[1]
            ):
                largest_historic = (measurement.location, measurement.parquet_bytes)
        for plot_measurement in report.plots:
            if plot_measurement.error:
                plot_errors += 1
                continue
            total_plot_bytes += plot_measurement.svg_bytes

    return {
        "locations_measured": len(reports),
        "feed_totals": feed_totals,
        "total_feed_rows": total_rows,
        "total_feed_memory_bytes": total_feed_memory,
        "total_feed_parquet_bytes": total_feed_parquet,
        "total_plot_bytes": total_plot_bytes,
        "grand_total_bytes": total_feed_parquet + total_plot_bytes,
        "feed_errors": feed_errors,
        "plot_errors": plot_errors,
        "largest_combined_historic_frame": (
            {"location": largest_historic[0], "parquet_bytes": largest_historic[1]}
            if largest_historic
            else None
        ),
    }


def format_text(reports: list[LocationReport], summary: dict[str, Any]) -> str:
    """Render a human-readable table with per-location detail and totals.

    Args:
        reports: Per-location measurement reports.
        summary: Aggregation produced by :func:`summarize`.

    Returns:
        The full report as one string, ready to print.
    """
    header = f"{'location':<9}{'row':<20}{'source':<32}{'rows':>10}{'memory':>10}{'parquet':>10}"
    lines = [header, "-" * len(header)]

    for report in reports:
        for measurement in report.feeds:
            source = (measurement.source or "")[:31]
            if measurement.error:
                lines.append(
                    f"{measurement.location:<9}{measurement.feed:<20}{source:<32}"
                    f"{'ERROR':>10}{'':>10}{'':>10}  {measurement.error}"
                )
                continue
            lines.append(
                f"{measurement.location:<9}{measurement.feed:<20}{source:<32}"
                f"{measurement.rows:>10,}{_human_bytes(measurement.memory_bytes):>10}"
                f"{_human_bytes(measurement.parquet_bytes):>10}"
            )
        for year_measurement in report.year_partitions:
            row_label = f"  historic_temps/{year_measurement.year}"
            lines.append(
                f"{year_measurement.location:<9}{row_label:<20}{'':<32}"
                f"{year_measurement.rows:>10,}{'':>10}"
                f"{_human_bytes(year_measurement.parquet_bytes):>10}"
            )
        for plot_measurement in report.plots:
            row_label = f"plot:{plot_measurement.plot}"
            if plot_measurement.error:
                lines.append(
                    f"{plot_measurement.location:<9}{row_label:<20}{'':<32}"
                    f"{'ERROR':>10}{'':>10}{'':>10}  {plot_measurement.error}"
                )
                continue
            lines.append(
                f"{plot_measurement.location:<9}{row_label:<20}{'':<32}"
                f"{'':>10}{'':>10}{_human_bytes(plot_measurement.svg_bytes):>10}"
            )

    lines.append("-" * len(header))
    lines.append(f"Locations measured: {summary['locations_measured']}")
    lines.append(
        "Feed totals: "
        f"rows={summary['total_feed_rows']:,} "
        f"memory={_human_bytes(summary['total_feed_memory_bytes'])} "
        f"parquet={_human_bytes(summary['total_feed_parquet_bytes'])}"
    )
    lines.append(f"Plot totals: {_human_bytes(summary['total_plot_bytes'])}")
    lines.append(
        "Grand total (feed Parquet + plot SVG bytes): "
        f"{_human_bytes(summary['grand_total_bytes'])}"
    )
    largest = summary["largest_combined_historic_frame"]
    if largest:
        lines.append(
            "Largest combined historical temperature frame: "
            f"{largest['location']} ({_human_bytes(largest['parquet_bytes'])})"
        )
    if summary["feed_errors"] or summary["plot_errors"]:
        lines.append(
            f"WARNING: {summary['feed_errors']} feed error(s), "
            f"{summary['plot_errors']} plot error(s); see ERROR rows above."
        )
    return "\n".join(lines)


def reports_to_dict(
    reports: list[LocationReport], summary: dict[str, Any]
) -> dict[str, Any]:
    """Build a machine-readable report payload.

    Args:
        reports: Per-location measurement reports.
        summary: Aggregation produced by :func:`summarize`.

    Returns:
        A JSON-serializable dict with per-location detail and the summary.
    """
    return {
        "locations": [
            {
                "location": report.location,
                "feeds": [asdict(measurement) for measurement in report.feeds],
                "year_partitions": [
                    asdict(measurement) for measurement in report.year_partitions
                ],
                "plots": [asdict(measurement) for measurement in report.plots],
            }
            for report in reports
        ],
        "summary": summary,
    }


def _feed_plots(
    location_config: LocationConfig,
    fetched_values: dict[feeds.FeedName, pd.DataFrame],
) -> list[PlotMeasurement]:
    """Generate and measure plots for whichever temperature feeds were fetched."""
    code = location_config.code
    measurements: list[PlotMeasurement] = []

    if feeds.FEED_LIVE_TEMPS in fetched_values:
        source_name = (
            location_config.live_temp_source.name
            if location_config.live_temp_source
            else None
        )
        try:
            plot_bytes = plot.generate_live_temp_plot(
                fetched_values[feeds.FEED_LIVE_TEMPS], code, source_name
            )
            measurements.append(
                PlotMeasurement(
                    location=code,
                    plot=feeds.PLOT_LIVE_TEMPS.value,
                    svg_bytes=len(plot_bytes),
                )
            )
        except Exception as e:
            measurements.append(
                PlotMeasurement(
                    location=code,
                    plot=feeds.PLOT_LIVE_TEMPS.value,
                    svg_bytes=0,
                    error=f"{e.__class__.__name__}: {e}",
                )
            )

    if feeds.FEED_HISTORIC_TEMPS in fetched_values:
        historic_temp_source = location_config.historic_temp_source
        source_name = historic_temp_source.name if historic_temp_source else None
        policy = (
            historic_temp_source.historic_plot_policy if historic_temp_source else None
        )
        try:
            plot_bytes_by_period = plot.generate_historic_temp_plots(
                fetched_values[feeds.FEED_HISTORIC_TEMPS], code, source_name, policy
            )
            for period, plot_name in (
                ("2mo", feeds.PLOT_HISTORIC_TEMPS_2MO),
                ("12mo", feeds.PLOT_HISTORIC_TEMPS_12MO),
            ):
                measurements.append(
                    PlotMeasurement(
                        location=code,
                        plot=plot_name.value,
                        svg_bytes=len(plot_bytes_by_period[period]),
                    )
                )
        except Exception as e:
            for plot_name in (
                feeds.PLOT_HISTORIC_TEMPS_2MO,
                feeds.PLOT_HISTORIC_TEMPS_12MO,
            ):
                measurements.append(
                    PlotMeasurement(
                        location=code,
                        plot=plot_name.value,
                        svg_bytes=0,
                        error=f"{e.__class__.__name__}: {e}",
                    )
                )

    return measurements


async def _measure_location(
    location_config: LocationConfig,
    clients: dict[str, BaseApiClient],
) -> LocationReport:
    """Fetch and measure every configured feed and plot for one location.

    Args:
        location_config: Location configuration to measure.
        clients: Provider API clients keyed by provider name.

    Returns:
        The completed measurement report for this location.
    """
    code = location_config.code
    feed_instances = {
        feed_name: feed
        for feed_name, feed in build_feeds(location_config, clients).items()
        if feed is not None
    }
    feed_measurements: list[FeedMeasurement] = []
    year_measurements: list[YearPartitionMeasurement] = []
    fetched_values: dict[feeds.FeedName, pd.DataFrame] = {}

    for feed_name, feed in feed_instances.items():
        source_name = feed.feed_config.name or feed.feed_config.citation_key
        try:
            await feed.update(clients=clients, feed_name=feed_name)
            df = feed.values
        except Exception as e:
            feed_measurements.append(
                FeedMeasurement(
                    location=code,
                    feed=feed_name.value,
                    source=source_name,
                    rows=0,
                    memory_bytes=0,
                    parquet_bytes=0,
                    error=f"{e.__class__.__name__}: {e}",
                )
            )
            continue

        feed_measurements.append(
            FeedMeasurement(
                location=code,
                feed=feed_name.value,
                source=source_name,
                rows=len(df),
                memory_bytes=int(df.memory_usage(deep=True).sum()),
                parquet_bytes=_parquet_bytes(df),
            )
        )
        fetched_values[feed_name] = df
        if feed_name == feeds.FEED_HISTORIC_TEMPS:
            year_measurements.extend(_year_partitions(code, df))

    plot_measurements = _feed_plots(location_config, fetched_values)

    return LocationReport(
        location=code,
        feeds=feed_measurements,
        year_partitions=year_measurements,
        plots=plot_measurements,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch every configured location's feeds live and measure "
            "in-memory and serialized Parquet/SVG sizes. Never writes files "
            "to disk."
        )
    )
    parser.add_argument(
        "--location",
        choices=sorted(config.CONFIGS),
        help="Only measure this location (default: all configured locations).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    return parser.parse_args(argv)


def _disable_archive_capture() -> None:
    """Unset the archive bucket env var.

    Measurement runs must never write to the observation archive, even if the
    operator's environment configures it for production deployment.
    """
    os.environ.pop("SHALLWESWIM_ARCHIVE_BUCKET", None)


async def _async_main() -> None:
    _disable_archive_capture()
    args = _parse_args()
    location_configs = (
        [config.CONFIGS[args.location]]
        if args.location
        else list(config.CONFIGS.values())
    )

    async with aiohttp.ClientSession() as session:
        clients = create_api_clients(session)
        reports = [
            await _measure_location(location_config, clients)
            for location_config in location_configs
        ]

    summary = summarize(reports)
    if args.json:
        json.dump(reports_to_dict(reports, summary), sys.stdout, indent=2)
        print()
    else:
        print(format_text(reports, summary))

    if summary["feed_errors"] or summary["plot_errors"]:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

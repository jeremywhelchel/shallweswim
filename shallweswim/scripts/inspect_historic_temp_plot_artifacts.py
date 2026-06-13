"""Inspect historical temperature plot visual artifact suppression for one location.

This is an operator/debugging tool for tuning the historical temperature plot
pipeline. It fetches historical temperatures from the configured source, shapes
them the same way the plot code does, and reports how many points each visual
artifact stage suppresses from the rendered plot.
"""

import argparse
import asyncio
import datetime
from pathlib import Path
from typing import Any, cast

import aiohttp
import pandas as pd

from shallweswim import config, plot, util
from shallweswim.clients.base import BaseApiClient, StationUnavailableError
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.cspf import CspfApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi
from shallweswim.core import feeds
from shallweswim.core.manager import DEFAULT_HISTORIC_TEMPS_START_YEAR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect historical temperature plot visual artifact suppression for a location."
    )
    parser.add_argument(
        "location",
        help="Location code, such as bos, nyc, san, or chi.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="First historical year to fetch. Defaults to the location config or app default.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last historical year to fetch. Defaults to the location config or current year.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional directory for CSV output. Writes plot_suppressed_points.csv and "
            "final_plot_frame.csv."
        ),
    )
    return parser.parse_args()


def _api_clients(session: aiohttp.ClientSession) -> dict[str, BaseApiClient]:
    return {
        "coops": CoopsApi(session=session),
        "cspf": CspfApi(session=session),
        "nwis": NwisApi(session=session),
        "ndbc": NdbcApi(session=session),
    }


async def _fetch_historic_temps(
    location_config: config.LocationConfig,
    *,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    temp_source = location_config.historic_temp_source
    if temp_source is None:
        raise ValueError(f"{location_config.code} has no historical temperature source")

    unavailable_years: list[tuple[int, str]] = []
    dataframes: list[pd.DataFrame] = []

    async with aiohttp.ClientSession() as session:
        clients = _api_clients(session)
        for year in range(start_year, end_year + 1):
            start = datetime.datetime(year, 1, 1)
            if year == util.utc_now().year:
                end = util.utc_now()
            else:
                end = datetime.datetime(year, 12, 31, 23, 59, 59)

            temp_feed = feeds.create_temp_feed(
                location_config=location_config,
                temp_config=temp_source,
                start=start,
                end=end,
                interval="h",
                clients=clients,
            )
            try:
                dataframes.append(await temp_feed._fetch(clients=clients))
            except StationUnavailableError as e:
                unavailable_years.append((year, str(e)))

    if not dataframes:
        raise ValueError("No historical temperature data fetched")

    combined = pd.concat(dataframes).sort_index().resample("h").first()
    historic_feed = feeds.HistoricalTempsFeed(
        location_config=location_config,
        feed_config=temp_source,
        start_year=start_year,
        end_year=end_year,
        expiration_interval=None,
    )
    return historic_feed._remove_outliers(combined), unavailable_years


def _original_timestamp(pivot_timestamp: pd.Timestamp, year: int) -> str:
    try:
        return str(pivot_timestamp.replace(year=year))
    except ValueError:
        return ""


def _plot_suppressed_points_frame(
    *,
    source_frame: pd.DataFrame,
    smoothed_frame: pd.DataFrame,
    raw_mask: pd.DataFrame,
    cross_year_mask: pd.DataFrame,
    volatility_mask: pd.DataFrame,
    short_segment_mask: pd.DataFrame,
) -> pd.DataFrame:
    seasonal_median = smoothed_frame.median(axis=1, skipna=True)
    rows: list[dict[str, Any]] = []

    stages = [
        ("raw", raw_mask, source_frame),
        ("cross_year", cross_year_mask, smoothed_frame),
        ("volatility", volatility_mask, smoothed_frame),
        ("short_segment", short_segment_mask, smoothed_frame),
    ]

    for stage, mask, value_frame in stages:
        for column in mask.columns:
            year = int(column)
            hit_index = mask.index[mask[column].fillna(False)]
            for timestamp in hit_index:
                pivot_timestamp = pd.Timestamp(timestamp)
                source_value = source_frame.loc[timestamp, column]
                suppressed_value = value_frame.loc[timestamp, column]
                median_value = seasonal_median.loc[timestamp]
                rows.append(
                    {
                        "stage": stage,
                        "year": year,
                        "pivot_timestamp": pivot_timestamp,
                        "original_timestamp": _original_timestamp(
                            pivot_timestamp, year
                        ),
                        "source_temp_f": source_value,
                        "suppressed_value_f": suppressed_value,
                        "seasonal_median_f": median_value,
                        "seasonal_residual_f": suppressed_value - median_value,
                    }
                )

    return pd.DataFrame(rows)


def _counts_by_year(mask: pd.DataFrame) -> dict[str, int]:
    return {
        str(column): int(count)
        for column, count in mask.sum().items()
        if int(count) > 0
    }


def _build_visual_artifact_outputs(
    hist_temps: pd.DataFrame,
    policy: plot.HistoricTempPlotPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    year_df = util.pivot_year(hist_temps)
    source_frame = cast(pd.DataFrame, year_df["water_temp"])

    raw_masks = {}
    for column in source_frame.columns:
        source = pd.to_numeric(source_frame[column], errors="coerce")
        raw_masks[column] = plot._historic_temperature_plot_spike_artifact_mask(
            source, policy
        )
    raw_mask = pd.DataFrame(raw_masks, index=source_frame.index)

    smoothed_frame = plot._historic_temperature_smoothed_plot_frame(
        source_frame, policy
    )
    cross_year_mask = plot._historic_temperature_plot_cross_year_artifact_mask(
        smoothed_frame, policy
    )
    cross_year_suppressed = smoothed_frame.mask(cross_year_mask)
    volatility_mask = plot._historic_temperature_plot_volatility_artifact_mask(
        cross_year_suppressed, policy
    )
    volatility_suppressed = cross_year_suppressed.mask(volatility_mask)
    short_segment_mask = plot._short_historic_temperature_plot_segment_mask(
        volatility_suppressed, policy
    )
    final_plot_frame = plot._historic_temperature_plot_frame(source_frame, policy)

    plot_suppressed_points = _plot_suppressed_points_frame(
        source_frame=source_frame,
        smoothed_frame=smoothed_frame,
        raw_mask=raw_mask,
        cross_year_mask=cross_year_mask,
        volatility_mask=volatility_mask,
        short_segment_mask=short_segment_mask,
    )

    counts = {
        "raw": _counts_by_year(raw_mask),
        "cross_year": _counts_by_year(cross_year_mask),
        "volatility": _counts_by_year(volatility_mask),
        "short_segment": _counts_by_year(short_segment_mask),
    }
    return plot_suppressed_points, final_plot_frame, counts


async def _async_main() -> None:
    args = _parse_args()
    location_config = config.get(args.location)
    temp_source = location_config.historic_temp_source
    if temp_source is None:
        raise ValueError(f"{location_config.code} has no historical temperature source")

    start_year = (
        args.start_year or temp_source.start_year or DEFAULT_HISTORIC_TEMPS_START_YEAR
    )
    end_year = args.end_year or temp_source.end_year or util.utc_now().year
    if start_year > end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")

    hist_temps, unavailable_years = await _fetch_historic_temps(
        location_config,
        start_year=start_year,
        end_year=end_year,
    )
    policy = plot._resolve_historic_temperature_plot_policy(
        temp_source.historic_plot_policy
    )
    plot_suppressed_points, final_plot_frame, counts = _build_visual_artifact_outputs(
        hist_temps, policy
    )

    print(
        f"{location_config.code}: fetched {len(hist_temps)} hourly historical rows "
        f"for {start_year}-{end_year}"
    )
    if unavailable_years:
        print("Unavailable years:")
        for year, reason in unavailable_years:
            print(f"  {year}: {reason}")

    print("Plot-suppressed points by stage/year:")
    for stage, stage_counts in counts.items():
        print(f"  {stage}: {sum(stage_counts.values())} {stage_counts}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        suppressed_path = args.output_dir / "plot_suppressed_points.csv"
        final_path = args.output_dir / "final_plot_frame.csv"
        plot_suppressed_points.to_csv(suppressed_path, index=False)
        final_plot_frame.to_csv(final_path)
        print(f"Wrote {suppressed_path}")
        print(f"Wrote {final_path}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

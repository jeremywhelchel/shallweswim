"""Debug CSPF Sandettie historical temperature fetches."""

import argparse
import asyncio
import datetime
import time

import aiohttp

from shallweswim import config
from shallweswim.clients.cspf import CspfApi
from shallweswim.config.locations import CspfTempFeedConfig
from shallweswim.util import utc_now


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch CSPF Sandettie temperature data through shallweswim.clients.cspf."
    )
    parser.add_argument(
        "--location",
        default="dov",
        help="Location code with a CSPF historical temperature source.",
    )
    parser.add_argument("--start-year", type=int, help="First year to fetch.")
    parser.add_argument("--end-year", type=int, help="Last year to fetch.")
    return parser.parse_args()


def _year_end(year: int) -> datetime.datetime:
    if year == utc_now().year:
        return utc_now()
    return datetime.datetime(year, 12, 31, 23, 59, 59)


async def _main() -> None:
    args = _parse_args()
    location_config = config.get(args.location)
    if location_config is None:
        raise ValueError(f"Unknown location: {args.location}")
    temp_source = location_config.historic_temp_source
    if not isinstance(temp_source, CspfTempFeedConfig):
        raise ValueError(f"{args.location} does not use a CSPF historical temp source")

    start_year = args.start_year or temp_source.start_year or utc_now().year
    end_year = args.end_year or temp_source.end_year or utc_now().year

    print(
        f"target={location_config.code}:historical-temp "
        f"source={temp_source.station_slug} start_year={start_year} end_year={end_year}"
    )
    started = time.monotonic()
    total_rows = 0
    failures = 0

    async with aiohttp.ClientSession() as session:
        client = CspfApi(session=session)
        for year in range(start_year, end_year + 1):
            year_started = time.monotonic()
            try:
                df = await client.sandettie_temperature(
                    begin_date=datetime.datetime(year, 1, 1),
                    end_date=_year_end(year),
                    station_slug=temp_source.station_slug,
                    timezone=location_config.timezone,
                    location_code=location_config.code,
                )
            except Exception as e:
                failures += 1
                print(f"{year}: error={e.__class__.__name__}: {e}")
                continue

            total_rows += len(df)
            print(
                f"{year}: rows={len(df)} oldest={df.index.min()} newest={df.index.max()} "
                f"elapsed={time.monotonic() - year_started:.2f}s"
            )

    print(
        f"summary: years={end_year - start_year + 1} rows={total_rows} "
        f"failures={failures} elapsed={time.monotonic() - started:.2f}s"
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

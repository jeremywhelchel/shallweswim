"""Integration tests for CSPF Sandettie historical temperature pages.

Run with: uv run pytest tests/test_cspf_integration.py -v --run-integration
"""

import datetime

import aiohttp
import pandas as pd
import pytest
import pytz

from shallweswim.clients.cspf import CspfApi

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sandettie_monthly_temperature_fetch_is_dense_and_local_time() -> None:
    """CSPF monthly pages provide dense Dover-local Sandettie history."""
    timezone = pytz.timezone("Europe/London")
    async with aiohttp.ClientSession() as session:
        client = CspfApi(session)
        df = await client.sandettie_temperature(
            begin_date=datetime.datetime(2026, 1, 1),
            end_date=datetime.datetime(2026, 12, 31, 23, 59, 59),
            station_slug="sandettie-data",
            timezone=timezone,
            location_code="dov",
        )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 300
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing
    assert df.index.tz is None
    assert df.index.min() <= pd.Timestamp("2026-01-01 02:00:00")
    assert df.index.max() >= pd.Timestamp("2026-06-01")
    assert df["water_temp"].between(30.0, 80.0).all()

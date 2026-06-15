"""Integration tests for the Marine Institute Ireland API client.

Run with: uv run pytest tests/test_marine_institute_integration.py -v --run-integration
"""

import aiohttp
import pandas as pd
import pytest

from shallweswim import config
from shallweswim.clients.marine_institute import MarineInstituteApi

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kinsale_tide_high_low_fetch_matches_cork_config() -> None:
    """Marine Institute Kinsale high/low tides are usable for Cork/Sandycove."""
    cork = config.get("cor")
    assert cork is not None
    assert isinstance(cork.tide_source, config.MarineInstituteTideFeedConfig)

    async with aiohttp.ClientSession() as session:
        client = MarineInstituteApi(session)
        df = await client.tides(
            station_id=cork.tide_source.station_id,
            timezone=cork.timezone,
            height_offset_m=cork.tide_source.height_offset_m,
            location_code=cork.code,
        )

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 8
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing
    assert df.index.tz is None
    assert set(df["type"].astype(str)) == {"high", "low"}
    assert df["prediction"].between(0.0, 20.0).all()
    assert df["prediction"].max() - df["prediction"].min() > 5.0

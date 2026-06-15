"""Integration tests for Irish Lights MetOcean temperature observations.

Run with: uv run pytest tests/test_irish_lights_integration.py -v --run-integration
"""

import datetime

import aiohttp
import pandas as pd
import pytest

from shallweswim import config
from shallweswim.clients.irish_lights import IrishLightsApi

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cork_buoy_temperature_fetch_matches_config() -> None:
    """Irish Lights Cork Buoy observations are usable for Cork/Sandycove."""
    cork = config.get("cor")
    assert cork is not None
    assert isinstance(cork.live_temp_source, config.IrishLightsTempFeedConfig)

    end = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    begin = end - datetime.timedelta(days=2)
    async with aiohttp.ClientSession() as session:
        client = IrishLightsApi(session)
        df = await client.water_temperature(
            mmsi=cork.live_temp_source.mmsi,
            begin_date=begin,
            end_date=end,
            timezone=cork.timezone,
            location_code=cork.code,
            min_valid_temp_c=cork.live_temp_source.min_valid_temp_c,
            max_valid_temp_c=cork.live_temp_source.max_valid_temp_c,
        )

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 12
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing
    assert df.index.tz is None
    assert df["water_temp"].between(32.0, 77.0).all()

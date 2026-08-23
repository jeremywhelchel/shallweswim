"""Integration tests for USGS NWIS API client.

These tests interact with the live USGS NWIS API and may fail if the API
is down or changes. They are skipped by default and only run when the
--run-integration flag is passed to pytest.
"""

# Standard library imports
import datetime

# Third-party imports
import aiohttp
import pandas as pd
import pytest

# Local imports
from shallweswim.clients.base import StationUnavailableError
from shallweswim.clients.nwis import NwisApi, NwisApiError, NwisDataError

# Mark all tests in this file as integration tests that hit live services
pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_temperature() -> None:
    """Integration test for fetching configured Austin NWIS temperature data."""
    begin_date = datetime.date(2026, 6, 1)
    end_date = datetime.date(2026, 6, 2)

    async with aiohttp.ClientSession() as session:
        nwis_client = NwisApi(session)
        df = await nwis_client.temperature(
            site_no="08155500",  # Barton Springs (used in aus config)
            parameter_cd="00010",
            begin_date=begin_date,
            end_date=end_date,
            timezone="America/Chicago",
            location_code="aus",
        )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "water_temp" in df.columns
    assert df.index.name == "time"
    assert df["water_temp"].between(32.0, 100.0).all()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_missing_temp_column() -> None:
    """Integration test for a configured site missing a requested temp parameter."""
    begin_date = datetime.date(2026, 6, 1)
    end_date = datetime.date(2026, 6, 2)

    async with aiohttp.ClientSession() as session:
        nwis_client = NwisApi(session)
        with pytest.raises(StationUnavailableError, match="returned no data"):
            await nwis_client.temperature(
                site_no="03292494",  # SDF uses 00011; 00010 should be absent
                parameter_cd="00010",
                begin_date=begin_date,
                end_date=end_date,
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_nwis_currents() -> None:
    """Test fetching current data (velocity) from a specific NWIS site."""
    site_no = "03292494"  # Ohio River at Louisville Water Tower (used in sdf config)
    parameter_cd = "72255"  # Stream velocity, ft/sec (used in sdf config)
    timezone = "America/New_York"  # Example timezone

    async with aiohttp.ClientSession() as session:
        client = NwisApi(session)
        try:
            df = await client.currents(
                site_no=site_no,
                parameter_cd=parameter_cd,
                timezone=timezone,
                location_code="test-currents",
            )
        except NwisDataError as e:
            pytest.fail(f"NwisDataError encountered: {e}")
        except NwisApiError as e:
            pytest.fail(f"NwisApiError encountered: {e}")

        # Basic checks
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, (
            f"No current data returned for site {site_no}, param {parameter_cd}"
        )
        assert "velocity_fps" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        # Note: NWIS 'iv' service might only return one row
        print(f"\nReceived {len(df)} current readings for {site_no}:")
        print(df.head())

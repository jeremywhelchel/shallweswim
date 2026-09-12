"""External API client package and the shared client-set factory."""

import aiohttp

from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.cspf import CspfApi
from shallweswim.clients.irish_lights import IrishLightsApi
from shallweswim.clients.marine_institute import MarineInstituteApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi


def create_api_clients(session: aiohttp.ClientSession) -> dict[str, BaseApiClient]:
    """Build the provider client set every host process uses.

    Args:
        session: Shared aiohttp session owned by the calling host process.

    Returns:
        Provider API clients keyed by the provider names feeds look up.
    """
    return {
        "coops": CoopsApi(session=session),
        "cspf": CspfApi(session=session),
        "irish_lights": IrishLightsApi(session=session),
        "marine_institute": MarineInstituteApi(session=session),
        "nwis": NwisApi(session=session),
        "ndbc": NdbcApi(session=session),
    }

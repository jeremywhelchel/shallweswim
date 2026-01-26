"""Backwards compatibility - re-exports from core.manager."""

from shallweswim.core.manager import (
    DEFAULT_HISTORIC_TEMPS_START_YEAR,
    EXPIRATION_PERIODS,
    LocationDataManager,
)

__all__ = [
    "DEFAULT_HISTORIC_TEMPS_START_YEAR",
    "EXPIRATION_PERIODS",
    "LocationDataManager",
]

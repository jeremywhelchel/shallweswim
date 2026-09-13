"""Snapshot manifest and in-memory serving-state models.

The manifest models are the durable JSON contract: every field is validated
strictly and unknown fields are rejected, so a manifest written by one version
of the publisher cannot be silently misread by another. The in-memory models
hold the same metadata next to the frames and plot bytes they describe, before
serialization on publish and after deserialization on load.
"""

import dataclasses
import datetime
import re

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from shallweswim.api_types import HistoricalTempStatus
from shallweswim.core.feeds import FeedName, PlotName

SCHEMA_VERSION = 1

# A run id becomes a manifest key segment, so it is limited to characters that
# every object store accepts verbatim.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def generation_id(published_at: datetime.datetime, run_id: str) -> str:
    """Return the sortable, unique generation id `<published_at>-<run_id>`.

    Args:
        published_at: Timezone-aware publication instant; rendered in UTC to
            whole seconds, so overlapping publishers starting within the same
            second rely on distinct run ids to stay distinct.
        run_id: The publishing job's run identifier.

    Raises:
        ValueError: If `published_at` is naive or `run_id` is empty or contains
            characters that are not portable inside an object key.
    """
    if published_at.tzinfo is None:
        raise ValueError("published_at must be timezone-aware")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    stamp = published_at.astimezone(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{run_id}"


class _ManifestModel(BaseModel):
    """Shared strictness for every manifest component."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FeedMetadata(_ManifestModel):
    """Per-feed serving state that lives in the manifest rather than an object."""

    source_identity: str = Field(..., description="The feed's citation_key")
    fetch_timestamp: AwareDatetime = Field(
        ..., description="When the served frame was fetched"
    )
    next_fetch_after: AwareDatetime | None = Field(
        ..., description="Next scheduled refresh or retry, if the feed refreshes"
    )
    expiration_seconds: float | None = Field(
        ..., description="Configured refresh cadence in seconds, if any"
    )
    record_count: int = Field(..., ge=0, description="Rows in the served frame")
    consecutive_failures: int = Field(
        ..., ge=0, description="Failed fetch attempts since the last success"
    )
    last_error: str | None = Field(
        ..., description="Sanitized message of the last failed attempt, if any"
    )
    timezone: str = Field(
        ..., description="IANA zone name of the naive station-local time index"
    )
    historical: HistoricalTempStatus | None = Field(
        ..., description="Year diagnostics for the historical temperature feed"
    )


class FeedObject(FeedMetadata):
    """One served feed frame: its content-addressed object plus metadata."""

    key: str = Field(..., description="Content-addressed Parquet object key")
    size_bytes: int = Field(..., ge=0, description="Object size in bytes")


class PlotObject(_ManifestModel):
    """One plot SVG and the feed state it was drawn from."""

    key: str = Field(..., description="Content-addressed SVG object key")
    size_bytes: int = Field(..., ge=0, description="Object size in bytes")
    feed: FeedName = Field(..., description="The feed the plot was drawn from")
    feed_fetch_timestamp: AwareDatetime = Field(
        ..., description="That feed's fetch timestamp when the plot was drawn"
    )


class LocationManifest(_ManifestModel):
    """Every published feed and plot of one location."""

    feeds: dict[FeedName, FeedObject]
    plots: dict[PlotName, PlotObject]


class Manifest(_ManifestModel):
    """One immutable snapshot generation."""

    schema_version: int = Field(..., ge=1)
    generation_id: str
    published_at: AwareDatetime
    previous_generation_id: str | None
    locations: dict[str, LocationManifest]


class CurrentPointer(_ManifestModel):
    """The contents of `published/current.json`."""

    manifest_key: str
    generation_id: str


@dataclasses.dataclass(frozen=True)
class FeedSnapshot:
    """A served frame and the manifest metadata describing it."""

    frame: pd.DataFrame
    metadata: FeedMetadata


@dataclasses.dataclass(frozen=True)
class PlotSnapshot:
    """A plot SVG and the feed state it was drawn from."""

    data: bytes
    feed: FeedName
    feed_fetch_timestamp: datetime.datetime


@dataclasses.dataclass(frozen=True)
class LocationSnapshot:
    """One location's serving state, keyed the way the manager keys it."""

    feeds: dict[FeedName, FeedSnapshot]
    plots: dict[PlotName, PlotSnapshot]


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """The serving state of every location to publish as one generation."""

    locations: dict[str, LocationSnapshot]

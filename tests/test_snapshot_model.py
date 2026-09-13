"""Manifest model contract: strict JSON round trips and generation ids."""

import datetime
import json

import pytest
from pydantic import ValidationError

from shallweswim.api_types import HistoricalTempStatus
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.model import (
    SCHEMA_VERSION,
    CurrentPointer,
    FeedObject,
    LocationManifest,
    Manifest,
    PlotObject,
    generation_id,
)

PUBLISHED_AT = datetime.datetime(2026, 9, 13, 14, 5, 9, tzinfo=datetime.UTC)


def _manifest() -> Manifest:
    fetched = datetime.datetime(2026, 9, 13, 14, 0, 1, 250000, tzinfo=datetime.UTC)
    return Manifest(
        schema_version=SCHEMA_VERSION,
        generation_id="20260913T140509Z-run-1",
        published_at=PUBLISHED_AT,
        previous_generation_id="20260913T130509Z-run-0",
        locations={
            "nyc": LocationManifest(
                feeds={
                    FeedName.HISTORIC_TEMPS: FeedObject(
                        key="published/objects/sha256-ab.parquet",
                        size_bytes=1234,
                        source_identity="coops:temperature:8518750",
                        fetch_timestamp=fetched,
                        next_fetch_after=None,
                        expiration_seconds=None,
                        record_count=87600,
                        consecutive_failures=1,
                        last_error="Station unavailable",
                        timezone="US/Eastern",
                        historical=HistoricalTempStatus(
                            required_years=[2024, 2025, 2026],
                            available_years=[2024, 2025],
                            cached_years=[2024, 2025],
                            missing_years=[2026],
                            fetched_years=[2025],
                            failed_years={2026: "timeout"},
                        ),
                    )
                },
                plots={
                    PlotName.HISTORIC_TEMPS_2MO: PlotObject(
                        key="published/objects/sha256-cd.svg",
                        size_bytes=99,
                        feed=FeedName.HISTORIC_TEMPS,
                        feed_fetch_timestamp=fetched,
                    )
                },
            )
        },
    )


def test_manifest_json_round_trip_is_exact() -> None:
    manifest = _manifest()
    restored = Manifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    feed = restored.locations["nyc"].feeds[FeedName.HISTORIC_TEMPS]
    assert feed.fetch_timestamp.tzinfo is not None
    assert feed.historical is not None
    assert feed.historical.failed_years == {2026: "timeout"}


def test_current_pointer_json_round_trip() -> None:
    pointer = CurrentPointer(
        manifest_key="published/manifests/20260913T140509Z-run-1.json",
        generation_id="20260913T140509Z-run-1",
    )
    assert CurrentPointer.model_validate_json(pointer.model_dump_json()) == pointer


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("locations", "nyc"),
        ("locations", "nyc", "feeds", "historic_temps"),
        ("locations", "nyc", "plots", "historic_temps_2mo"),
    ],
)
def test_manifest_rejects_unknown_fields(path: tuple[str, ...]) -> None:
    payload = json.loads(_manifest().model_dump_json())
    target = payload
    for segment in path:
        target = target[segment]
    target["unexpected"] = 1

    with pytest.raises(ValidationError, match="unexpected"):
        Manifest.model_validate_json(json.dumps(payload))


def test_pointer_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        CurrentPointer.model_validate_json(
            '{"manifest_key": "k", "generation_id": "g", "extra": true}'
        )


def test_manifest_rejects_naive_timestamps_and_unknown_feeds() -> None:
    payload = json.loads(_manifest().model_dump_json())
    payload["published_at"] = "2026-09-13T14:05:09"
    with pytest.raises(ValidationError, match="published_at"):
        Manifest.model_validate_json(json.dumps(payload))

    payload = json.loads(_manifest().model_dump_json())
    feeds = payload["locations"]["nyc"]["feeds"]
    feeds["wind"] = feeds.pop("historic_temps")
    with pytest.raises(ValidationError, match="wind"):
        Manifest.model_validate_json(json.dumps(payload))


def test_generation_id_combines_utc_second_and_run_id() -> None:
    assert generation_id(PUBLISHED_AT, "run-a1b2") == "20260913T140509Z-run-a1b2"
    eastern = PUBLISHED_AT.astimezone(datetime.timezone(datetime.timedelta(hours=-4)))
    assert generation_id(eastern, "run-a1b2") == "20260913T140509Z-run-a1b2"


@pytest.mark.parametrize("run_id", ["", "run/1", "run 1"])
def test_generation_id_rejects_nonportable_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        generation_id(PUBLISHED_AT, run_id)


def test_generation_id_rejects_naive_publication_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        generation_id(PUBLISHED_AT.replace(tzinfo=None), "run-1")

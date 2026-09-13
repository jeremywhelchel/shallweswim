"""Exact Parquet round trips for every served feed type, and key derivation."""

import hashlib

import pandas as pd
import pandera.errors
import pytest

from shallweswim.core.feeds import FeedName
from shallweswim.snapshot.serialize import frame_to_parquet, parquet_to_frame
from shallweswim.snapshot.store import CURRENT_KEY, manifest_key, object_key
from shallweswim.types import TIDE_TYPE_CATEGORIES
from tests.snapshot_fixtures import FRAMES, tides_frame


@pytest.mark.parametrize(
    "feed_name", [FeedName.LIVE_TEMPS, FeedName.TIDES, FeedName.CURRENTS]
)
def test_round_trip_is_exact(feed_name: FeedName) -> None:
    original = FRAMES[feed_name]()
    assert original.index.freq is None

    restored = parquet_to_frame(frame_to_parquet(original), feed_name)

    pd.testing.assert_frame_equal(original, restored)
    assert restored.index.name == "time"
    assert restored.index.tz is None
    assert restored.index.dtype == original.index.dtype
    assert list(restored.dtypes) == list(original.dtypes)


def test_historical_round_trip_keeps_gaps_and_drops_only_freq() -> None:
    original = FRAMES[FeedName.HISTORIC_TEMPS]()
    assert original.index.freq is not None
    assert original["water_temp"].isna().any()

    restored = parquet_to_frame(frame_to_parquet(original), FeedName.HISTORIC_TEMPS)

    pd.testing.assert_frame_equal(original, restored, check_freq=False)
    assert restored.index.freq is None


def test_tide_type_categories_restored_when_partially_present() -> None:
    original = tides_frame().iloc[[1, 3]]
    assert set(original["type"]) == {"high"}

    restored = parquet_to_frame(frame_to_parquet(original), FeedName.TIDES)

    pd.testing.assert_frame_equal(original, restored)
    assert list(restored["type"].cat.categories) == TIDE_TYPE_CATEGORIES


def test_load_validates_against_the_feed_model() -> None:
    data = frame_to_parquet(FRAMES[FeedName.CURRENTS]())

    with pytest.raises(pandera.errors.SchemaErrors):
        parquet_to_frame(data, FeedName.LIVE_TEMPS)


def test_serialize_rejects_frames_that_are_not_served_frames() -> None:
    frame = FRAMES[FeedName.LIVE_TEMPS]()

    with pytest.raises(ValueError, match="timezone naive"):
        frame_to_parquet(frame.tz_localize("UTC"))
    with pytest.raises(ValueError, match="named time"):
        frame_to_parquet(frame.rename_axis("when"))


def test_load_rejects_frames_without_naive_time() -> None:
    frame = FRAMES[FeedName.LIVE_TEMPS]()
    aware = frame.tz_localize("UTC").reset_index()
    data = aware.to_parquet(index=False)
    with pytest.raises(ValueError, match="timezone naive"):
        parquet_to_frame(data, FeedName.LIVE_TEMPS)

    data = frame.reset_index(drop=True).to_parquet(index=False)
    with pytest.raises(ValueError, match="no time column"):
        parquet_to_frame(data, FeedName.LIVE_TEMPS)


def test_keys_are_content_addressed_under_the_published_prefix() -> None:
    digest = hashlib.sha256(b"payload").hexdigest()

    assert object_key(b"payload", "parquet") == (
        f"published/objects/sha256-{digest}.parquet"
    )
    assert object_key(b"payload", "svg") == f"published/objects/sha256-{digest}.svg"
    assert object_key(b"other", "svg") != object_key(b"payload", "svg")
    assert manifest_key("20260913T140509Z-run-1") == (
        "published/manifests/20260913T140509Z-run-1.json"
    )
    assert CURRENT_KEY == "published/current.json"

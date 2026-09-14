"""The shadow-mode comparison core.

Every test compares two in-memory serving states: a seeded fetching manager
and a `SnapshotLocationManager` loaded from a generation published into a
memory store. Nothing here touches the network or a real bucket; the command's
fetch-and-load plumbing is deliberately untested.
"""

import dataclasses
import datetime

import pandas as pd
import pytest

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.core.manager import LocationDataManager
from shallweswim.scripts.compare_snapshot import (
    FeedComparison,
    FeedOutcome,
    LocationComparison,
    _parse_args,
    compare_location,
    exit_code,
    format_report,
    is_configured,
)
from shallweswim.snapshot.load import load_current
from shallweswim.snapshot.manager import SnapshotLocationManager
from shallweswim.snapshot.model import FeedSnapshot, Snapshot
from shallweswim.snapshot.publish import publish
from shallweswim.snapshot.store import SnapshotStore
from shallweswim.types import TIDE_TYPE_CATEGORIES
from tests.conftest import TEST_CONFIG_FULL
from tests.snapshot_fixtures import (
    feed_failure,
    feed_metadata,
    live_temps_frame,
    sample_snapshot,
    seeded_manager,
    tides_frame,
)

LOADED_AT = datetime.datetime(2026, 6, 1, 12, 30, tzinfo=datetime.UTC)
# A local time inside every fixture frame's window.
LOCAL_T = datetime.datetime(2026, 6, 1, 4, 0)
# After the third fixture tide and before the fourth.
BEFORE_LAST_TIDE = datetime.datetime(2026, 6, 1, 21, 0)


async def _bundle(snapshot: Snapshot) -> SnapshotLocationManager:
    """Publish a snapshot into a memory store and load it back as the bundle."""
    store = SnapshotStore(MemoryObjectStore())
    result = await publish(store, snapshot, run_id="run-1", now=LOADED_AT)
    assert result.outcome == "success"
    loaded = await load_current(store)
    assert loaded is not None
    return SnapshotLocationManager(
        TEST_CONFIG_FULL,
        loaded.manifest.locations["nyc"],
        loaded.frames["nyc"],
        loaded.plots["nyc"],
        LOADED_AT,
    )


def _with_frame(
    snapshot: Snapshot, feed_name: FeedName, frame: pd.DataFrame
) -> Snapshot:
    """Return the snapshot with one feed's frame replaced."""
    location = snapshot.locations["nyc"]
    feeds = dict(location.feeds)
    feeds[feed_name] = FeedSnapshot(
        frame=frame, metadata=feed_metadata(feed_name, frame)
    )
    return Snapshot(
        locations={"nyc": dataclasses.replace(location, feeds=feeds)},
    )


def _clear_feed(manager: LocationDataManager, feed_name: FeedName) -> None:
    """Make a seeded manager look like a location whose fetch failed."""
    feed = manager._feeds[feed_name]
    assert feed is not None
    feed._data = None


def _outcomes(comparison: LocationComparison) -> dict[FeedName, FeedOutcome]:
    return {feed.feed: feed.outcome for feed in comparison.feeds}


def _feed(comparison: LocationComparison, feed_name: FeedName) -> FeedComparison:
    (found,) = [feed for feed in comparison.feeds if feed.feed is feed_name]
    return found


@pytest.mark.asyncio
async def test_identical_sides_match_on_every_feed() -> None:
    legacy = seeded_manager()
    bundle = await _bundle(sample_snapshot())

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    assert _outcomes(comparison) == dict.fromkeys(FeedName, FeedOutcome.MATCH)
    live = _feed(comparison, FeedName.LIVE_TEMPS)
    assert live.shared_count == 240
    assert live.differing_count == 0
    assert live.largest_difference == 0.0
    assert live.differences == ()
    assert live.derived_differences == ()
    assert live.lag == datetime.timedelta(0)
    assert comparison.legacy_plots == tuple(PlotName)
    assert comparison.bundle_plots == tuple(PlotName)
    assert comparison.ok
    assert exit_code([comparison]) == 0


@pytest.mark.asyncio
async def test_feed_only_the_legacy_manager_holds_is_missing() -> None:
    legacy = seeded_manager()
    bundle = await _bundle(
        sample_snapshot(
            failures={FeedName.LIVE_TEMPS: feed_failure(FeedName.LIVE_TEMPS)}
        )
    )

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    live = _feed(comparison, FeedName.LIVE_TEMPS)
    assert live.outcome is FeedOutcome.MISSING
    assert live.shared_count == 0
    assert live.lag is None
    assert live.legacy_last is not None
    assert live.bundle_first is None
    # The plot drawn from the failed feed is absent from the bundle too.
    assert PlotName.LIVE_TEMPS in comparison.legacy_plots
    assert PlotName.LIVE_TEMPS not in comparison.bundle_plots
    assert not comparison.ok
    assert exit_code([comparison]) == 1


@pytest.mark.asyncio
async def test_feed_only_the_bundle_holds_is_extra_and_passes() -> None:
    legacy = seeded_manager()
    _clear_feed(legacy, FeedName.CURRENTS)
    bundle = await _bundle(sample_snapshot())

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    currents = _feed(comparison, FeedName.CURRENTS)
    assert currents.outcome is FeedOutcome.EXTRA
    assert currents.legacy_first is None
    assert currents.bundle_last is not None
    assert comparison.ok
    assert exit_code([comparison]) == 0


@pytest.mark.asyncio
async def test_feed_neither_side_holds_is_absent_and_passes() -> None:
    legacy = seeded_manager()
    _clear_feed(legacy, FeedName.LIVE_TEMPS)
    bundle = await _bundle(
        sample_snapshot(
            failures={FeedName.LIVE_TEMPS: feed_failure(FeedName.LIVE_TEMPS)}
        )
    )

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    live = _feed(comparison, FeedName.LIVE_TEMPS)
    assert live.outcome is FeedOutcome.ABSENT
    assert live.shared_count == 0
    assert live.largest_difference is None
    assert comparison.ok
    assert exit_code([comparison]) == 0


@pytest.mark.asyncio
async def test_indexes_sharing_no_timestamp_are_disjoint() -> None:
    legacy = seeded_manager()
    stale = live_temps_frame()
    stale.index = stale.index - datetime.timedelta(days=10)
    bundle = await _bundle(_with_frame(sample_snapshot(), FeedName.LIVE_TEMPS, stale))

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    live = _feed(comparison, FeedName.LIVE_TEMPS)
    assert live.outcome is FeedOutcome.DISJOINT
    assert live.shared_count == 0
    assert live.differing_count == 0
    assert live.lag == datetime.timedelta(days=10)
    # The latest reading is a different answer, so it is not even asked.
    assert live.derived_differences == ()
    assert exit_code([comparison]) == 1


@pytest.mark.asyncio
async def test_differing_values_mismatch_and_report_both_sides() -> None:
    legacy = seeded_manager()
    bundle = await _bundle(sample_snapshot(live_offset=0.5))

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    live = _feed(comparison, FeedName.LIVE_TEMPS)
    assert live.outcome is FeedOutcome.MISMATCH
    assert live.shared_count == 240
    assert live.differing_count == 240
    assert live.largest_difference == pytest.approx(0.5)
    # Bounded to the first 20 rows, each carrying both values.
    assert len(live.differences) == 20
    first = live.differences[0]
    assert first.column == "water_temp"
    assert first.bundle == pytest.approx(float(first.legacy) + 0.5)
    # Both frames end at the same timestamp, so the latest reading is compared.
    assert [derived.answer for derived in live.derived_differences] == [
        "get_current_temperature"
    ]
    assert exit_code([comparison]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relative", "outcome"),
    [(5e-7, FeedOutcome.MATCH), (5e-6, FeedOutcome.MISMATCH)],
)
async def test_float_columns_use_a_relative_tolerance(
    relative: float, outcome: FeedOutcome
) -> None:
    legacy = seeded_manager()
    nudged = live_temps_frame()
    nudged.iloc[0, 0] = float(nudged.iloc[0, 0]) * (1 + relative)
    bundle = await _bundle(_with_frame(sample_snapshot(), FeedName.LIVE_TEMPS, nudged))

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    assert _feed(comparison, FeedName.LIVE_TEMPS).outcome is outcome


@pytest.mark.asyncio
async def test_a_differing_derived_answer_mismatches_an_equal_overlap() -> None:
    legacy = seeded_manager()
    tides = tides_frame()
    extra = pd.DataFrame(
        {
            "prediction": [-0.4],
            "type": pd.Categorical(["low"], categories=TIDE_TYPE_CATEGORIES),
        },
        index=pd.DatetimeIndex(
            [tides.index[-1] + datetime.timedelta(hours=6, minutes=13)], name="time"
        ),
    )
    bundle = await _bundle(
        _with_frame(sample_snapshot(), FeedName.TIDES, pd.concat([tides, extra]))
    )

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, BEFORE_LAST_TIDE)

    tide_comparison = _feed(comparison, FeedName.TIDES)
    assert tide_comparison.outcome is FeedOutcome.MISMATCH
    # Every shared timestamp agrees; only the next-tide answer differs.
    assert tide_comparison.shared_count == len(tides)
    assert tide_comparison.differing_count == 0
    assert tide_comparison.differences == ()
    assert "get_tide_info_at_time" in [
        derived.answer for derived in tide_comparison.derived_differences
    ]
    assert exit_code([comparison]) == 1


@pytest.mark.asyncio
async def test_lag_is_the_gap_between_the_two_latest_timestamps() -> None:
    legacy = seeded_manager()
    trailing = live_temps_frame().iloc[:-5]
    bundle = await _bundle(
        _with_frame(sample_snapshot(), FeedName.LIVE_TEMPS, trailing)
    )

    comparison = compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)

    live = _feed(comparison, FeedName.LIVE_TEMPS)
    assert live.outcome is FeedOutcome.MATCH
    assert live.shared_count == len(trailing)
    assert live.legacy_last == live.bundle_last + datetime.timedelta(minutes=30)
    assert live.lag == datetime.timedelta(minutes=30)
    # The two frames end at different timestamps, so the latest reading, which
    # would differ by construction, is not compared.
    assert live.derived_differences == ()
    assert exit_code([comparison]) == 0


@pytest.mark.asyncio
async def test_report_names_the_outcome_and_the_differing_rows() -> None:
    legacy = seeded_manager()
    bundle = await _bundle(sample_snapshot(live_offset=0.5))

    report = format_report(
        [compare_location(legacy, bundle, TEST_CONFIG_FULL, LOCAL_T)]
    )

    assert "nyc at 2026-06-01 04:00:00 local (FAILED)" in report
    assert "live_temps     mismatch" in report
    assert "tides          match" in report
    assert "water_temp" in report
    assert "get_current_temperature differs" in report
    assert "plots: legacy=live_temps,historic_temps_2mo,historic_temps_12mo" in report
    assert report.endswith("1 location(s) compared, 1 failing: MISMATCH")


def _comparison(*outcomes: FeedOutcome) -> LocationComparison:
    return LocationComparison(
        code="nyc",
        at=LOCAL_T,
        feeds=tuple(
            FeedComparison(
                feed=feed_name,
                outcome=outcome,
                shared_count=0,
                differing_count=0,
                largest_difference=None,
                legacy_first=None,
                legacy_last=None,
                bundle_first=None,
                bundle_last=None,
                lag=None,
            )
            for feed_name, outcome in zip(FeedName, outcomes, strict=False)
        ),
        legacy_plots=(),
        bundle_plots=(),
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (FeedOutcome.MATCH, 0),
        (FeedOutcome.EXTRA, 0),
        (FeedOutcome.ABSENT, 0),
        (FeedOutcome.MISSING, 1),
        (FeedOutcome.DISJOINT, 1),
        (FeedOutcome.MISMATCH, 1),
    ],
)
def test_exit_code_passes_only_match_extra_and_absent(
    outcome: FeedOutcome, expected: int
) -> None:
    assert exit_code([_comparison(outcome)]) == expected
    # One failing feed among passing ones still fails the run.
    assert exit_code([_comparison(FeedOutcome.MATCH), _comparison(outcome)]) == expected


def test_configured_feeds_follow_the_location_configuration() -> None:
    assert all(is_configured(TEST_CONFIG_FULL, feed) for feed in FeedName)
    without_currents = TEST_CONFIG_FULL.model_copy(update={"currents_source": None})
    assert not is_configured(without_currents, FeedName.CURRENTS)


def test_missing_read_bucket_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SHALLWESWIM_SNAPSHOT_READ_BUCKET", raising=False)

    with pytest.raises(SystemExit) as error:
        _parse_args([])

    assert error.value.code == 2


def test_at_must_be_a_naive_local_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_SNAPSHOT_READ_BUCKET", "bucket")

    args, bucket = _parse_args(["--at", "2026-06-01T04:00:00"])
    assert args.at == LOCAL_T
    assert bucket == "bucket"

    with pytest.raises(SystemExit) as error:
        _parse_args(["--at", "2026-06-01T04:00:00+00:00"])
    assert error.value.code == 2

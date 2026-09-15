"""One-shot bounded capture job for the durable observation archive.

This module is the job entry point (``uv run python -m shallweswim.update``).
It fetches every archivable feed once in a short-lived process so the existing
capture hook writes the observations to the archive. It is distinct from
``shallweswim.archive.capture``, which is that hook: the hook merges one feed's
observations into the archive, while this module is the host process that makes
the feeds fetch in the first place.

When ``SHALLWESWIM_SNAPSHOT_PUBLISH=1`` the job is also the snapshot publisher:
it runs the full serving cycle of every location through the same
``LocationDataManager`` the web service uses, including tide and prediction
feeds, derived frames, and plots in a process pool, and then publishes one
snapshot generation under ``published/`` in the archive bucket and sweeps the
generations that publication superseded. Archive capture still happens inside
each feed's update, and the sweep never touches archive partitions. The job
never serves traffic or starts FastAPI.

The current generation is also the job's persisted feed schedule: before each
location's cycle the run restores every matching feed's next fetch time from
the current manifest, so a feed that is not yet due is not fetched and its
published entry and plots are carried forward. Each feed therefore keeps its
own interval whatever the job cadence is.

That publishing cycle, ``publish_locations``, has a second host:
``shallweswim.local`` runs it on a timer inside the web app's process against a
local store. It therefore takes its process pool and its store locator as
parameters; this module's ``_run`` supplies the pool it creates and the bucket
its environment names.
"""

import argparse
import asyncio
import datetime
import logging
import os
import sys
import time
import uuid
from concurrent.futures import Executor, ProcessPoolExecutor

import aiohttp

from shallweswim import config as config_lib
from shallweswim import logging_utils
from shallweswim.archive.capture import CaptureResult
from shallweswim.archive.store import object_store
from shallweswim.clients import create_api_clients
from shallweswim.clients.base import BaseApiClient
from shallweswim.core import feeds
from shallweswim.core.manager import (
    PLOT_HARD_TIMEOUT,
    LocationDataManager,
    build_feeds,
)
from shallweswim.snapshot.build import build_location_snapshot, is_held
from shallweswim.snapshot.gc import collect_generations
from shallweswim.snapshot.model import LocationManifest, Manifest, Snapshot
from shallweswim.snapshot.publish import publish
from shallweswim.snapshot.store import SnapshotStore
from shallweswim.types import DataSourceType
from shallweswim.util import utc_now

# Required; fetching without capturing is not a useful run.
ARCHIVE_BUCKET_ENV_VAR = "SHALLWESWIM_ARCHIVE_BUCKET"

# Exactly "1" makes the run publish a snapshot after its capture cycle. The
# job definition sets it; the web service never does.
SNAPSHOT_PUBLISH_ENV_VAR = "SHALLWESWIM_SNAPSHOT_PUBLISH"

# Required when publishing: a snapshot carries the full historical range, and
# the historical feed restores past years from this bucket instead of
# refetching them from the provider.
ARCHIVE_READ_BUCKET_ENV_VAR = "SHALLWESWIM_ARCHIVE_READ_BUCKET"

# Feeds whose observations belong in the archive. Tide feeds are predictions and
# never appear here; currents are included only for observation sources.
ARCHIVABLE_FEED_NAMES = (
    feeds.FEED_LIVE_TEMPS,
    feeds.FEED_HISTORIC_TEMPS,
    feeds.FEED_CURRENTS,
)


def run_outcome(published: int, attempted: int) -> str:
    """Return the bounded summary outcome for one capture run.

    Args:
        published: Number of selected feeds holding data after their update.
        attempted: Number of selected feeds the run tried to update.

    Returns:
        "success" when every attempted feed published, "partial" when only some
        did, and "failed" when none did. A run with nothing to capture is a
        success.
    """
    if published == attempted:
        return "success"
    if published > 0:
        return "partial"
    return "failed"


def _archivable_feeds(
    location_feeds: dict[feeds.FeedName, feeds.Feed | None],
) -> dict[feeds.FeedName, feeds.Feed]:
    """Select the configured feeds whose observations enter the archive.

    Args:
        location_feeds: All feeds built for one location, including None entries
            for unconfigured or disabled sources.

    Returns:
        Archivable feeds keyed by feed name, in fetch order.
    """
    selected: dict[feeds.FeedName, feeds.Feed] = {}
    for feed_name in ARCHIVABLE_FEED_NAMES:
        feed = location_feeds.get(feed_name)
        if feed is None:
            continue
        if feed_name == feeds.FEED_CURRENTS:
            currents_config = feed.feed_config
            assert isinstance(currents_config, config_lib.CurrentsFeedConfig)
            if currents_config.source_type != DataSourceType.OBSERVATION:
                # Prediction currents never enter the observation archive.
                continue
        selected[feed_name] = feed
    return selected


async def _capture_location(
    config: config_lib.LocationConfig,
    clients: dict[str, BaseApiClient],
    *,
    full_history: bool,
) -> tuple[int, int, int, CaptureResult]:
    """Update one location's archivable feeds sequentially.

    A fresh feed is always expired, so one update per feed is the whole cycle.
    Feed failures are isolated here so the location's remaining feeds still run.

    Args:
        config: Location configuration to capture.
        clients: Provider API clients keyed by provider name.
        full_history: Whether to fetch the full configured historical
            temperature year range instead of only the current UTC year.

    Returns:
        Tuple of attempted feed count, published feed count, total published row
        count, and the archive rows this location added and revised.
    """
    location_feeds = build_feeds(
        config,
        clients,
        historic_start_year=None if full_history else utc_now().year,
    )
    selected = _archivable_feeds(location_feeds)
    if not selected:
        logging.info(f"[{config.code}] No archivable feeds configured")
        return 0, 0, 0, CaptureResult(0, 0)

    published = 0
    record_count = 0
    new_count = 0
    revised_count = 0
    for feed_name, feed in selected.items():
        try:
            await feed.update(clients=clients, feed_name=feed_name)
        except Exception:
            # Feed.update already logged this failure at ERROR. Continue so one
            # bad feed does not stop the rest of this location.
            continue
        if feed.has_data:
            published += 1
            record_count += len(feed.values)
            if feed.last_capture is not None:
                new_count += feed.last_capture.new_count
                revised_count += feed.last_capture.revised_count
    return (
        len(selected),
        published,
        record_count,
        CaptureResult(new_count, revised_count),
    )


def _location_counts(
    location_feeds: dict[feeds.FeedName, feeds.Feed | None],
) -> tuple[int, int, int, CaptureResult]:
    """Count one location's feeds after its serving cycle ran.

    A feed the run held because it was not due counts as published: the
    generation carries its entry forward, so its frame is still served. The row
    count covers only the frames this process holds, because a held feed's rows
    were counted by the run that fetched them.

    Args:
        location_feeds: The feeds a manager holds, None for unconfigured ones.

    Returns:
        Tuple of attempted feed count, published feed count, total published row
        count, and the archive rows this location added and revised.
    """
    attempted = 0
    published = 0
    record_count = 0
    new_count = 0
    revised_count = 0
    for feed in location_feeds.values():
        if feed is None:
            continue
        attempted += 1
        if feed.has_data:
            published += 1
            record_count += len(feed.values)
        elif is_held(feed):
            published += 1
        if feed.last_capture is not None:
            new_count += feed.last_capture.new_count
            revised_count += feed.last_capture.revised_count
    return attempted, published, record_count, CaptureResult(new_count, revised_count)


# The cadence the scheduled job runs at, and the design's target. A feed whose
# restored next fetch time falls before the next run would start is fetched
# on this run: restoring it literally would hold a feed whose interval equals
# the cadence on every other run, because its next fetch time lands seconds
# after the next run begins.
JOB_CADENCE = datetime.timedelta(minutes=10)


def restore_schedule(
    manager: LocationDataManager,
    location_manifest: LocationManifest | None,
    *,
    cadence: datetime.timedelta = JOB_CADENCE,
) -> None:
    """Restore each feed's next fetch time from the current generation.

    The published manifest is the job's persisted feed schedule. A fresh feed
    is always due, so without this a bounded run refetches everything on every
    execution; with it, each feed keeps its own expiration interval whatever
    the job cadence is. A feed the manifest does not describe, or describes
    from a different source, keeps its fresh-feed state and fetches.

    Args:
        manager: The location's freshly built data manager.
        location_manifest: The location's entry in the current generation's
            manifest, or None when there is no generation, no entry for this
            location, or the run ignores the schedule.
        cadence: How often runs start. A feed due before the next run stays
            due now rather than waiting a whole cadence past its interval.
    """
    if location_manifest is None:
        return
    next_run = utc_now() + cadence
    for feed_name, feed in manager._feeds.items():
        # The manager exposes no accessor for the feed objects themselves, and
        # scheduling is deliberately private to the feed.
        if feed is None:
            continue
        entry = location_manifest.feeds.get(feed_name)
        if entry is None or entry.source_identity != feed.feed_config.citation_key:
            continue
        if entry.next_fetch_after is None:
            feed._next_fetch_after = None
            continue
        # Feeds keep naive UTC; the manifest states the same instant.
        due_at = entry.next_fetch_after.astimezone(datetime.UTC).replace(tzinfo=None)
        if due_at <= next_run:
            # Due before the next run starts: fetch now (a fresh feed is due).
            continue
        feed._next_fetch_after = due_at


async def _current_manifest(store: SnapshotStore) -> Manifest | None:
    """Return the manifest of the generation that is current right now.

    Args:
        store: The snapshot store the run publishes into.

    Returns:
        The current manifest, or None when no generation is published or the
        pointer names a manifest that is gone.
    """
    current = await store.read_current()
    if current is None:
        return None
    pointer, _version = current
    return await store.read_manifest(pointer.manifest_key)


async def serve_location(manager: LocationDataManager) -> None:
    """Run one location's serving cycle to completion.

    The comparison command (`scripts/compare_snapshot.py`) reuses this so its
    legacy side is exactly the state the publisher publishes from.

    A feed whose update raises has already logged the failure at ERROR and
    scheduled its retry a minute out, so re-entering the cycle skips it and
    updates the feeds after it, as the web loop's next tick would. One entry
    per feed bounds the re-entry.

    Args:
        manager: The location's data manager, freshly built for this run.
    """
    for _ in range(len(manager._feeds)):
        try:
            await manager.update_once()
            return
        except Exception as error:
            # The web loop records an interrupted cycle the same way; a feed
            # failure has also been logged by the feed itself.
            manager.log(f"Error in serving cycle: {error}", level=logging.ERROR)


async def publish_locations(
    clients: dict[str, BaseApiClient],
    run_id: str,
    *,
    pool: Executor,
    locator: str,
    full_history: bool = False,
    cadence: datetime.timedelta = JOB_CADENCE,
) -> tuple[list[tuple[int, int, int, CaptureResult]], str]:
    """Run every location's full serving cycle, then publish one snapshot.

    Locations run concurrently in the same managers the web service uses, so
    archive capture happens inside each feed's update as it does in the
    capture-only path. Each location's schedule is restored from the current
    generation first, so only feeds that are due fetch and the rest keep the
    entries and plots they already published. Plots are generated in the
    caller's process pool and awaited before the snapshot is built. Every
    enabled location is published, whether or not its feeds fetched anything
    this run, so manifest assembly can carry the last published entry of a held
    or failed feed forward. Publication failure is isolated: `publish` has
    already logged its failed event, so the run's outcome and exit code stay
    those of the capture cycle.

    One generation sweep (`snapshot.gc.collect_generations`) follows
    publication, whatever its outcome, and is isolated the same way: it deletes
    the manifests of generations older than the retention window and the
    objects no retained manifest references, logs its own event, and never
    changes this run's outcome.

    The base manifest is read once here for scheduling only. `publish` reads
    the current pointer again for assembly and promotion, so a generation
    another publisher promotes in between cannot make a carried-forward entry
    disagree with the generation promotion is conditioned on.

    The pool and the store locator are parameters because this cycle has two
    hosts: this job, which creates a pool and reads the archive bucket from its
    environment, and `shallweswim.local`, which runs the same cycle inside the
    web app against the app's pool and a local store.

    Args:
        clients: Provider API clients keyed by provider name.
        run_id: Identifier correlating this run with platform execution logs.
        pool: Executor the plots run in; the caller owns its lifetime.
        locator: Store locator the generation is published into, as
            `archive.store.object_store` resolves it.
        full_history: Whether to ignore the published schedule so every feed
            fetches, for backfills and repairs.
        cadence: How often this cycle runs, so a feed due before the next run
            is fetched on this one rather than held for a whole cadence.

    Returns:
        Each location's counts in the order of `_capture_location`, and the
        snapshot publish outcome for the run summary.
    """
    store = SnapshotStore(await asyncio.to_thread(object_store, locator))
    base = None if full_history else await _current_manifest(store)
    managers = [
        LocationDataManager(location_config, clients, pool)
        for location_config in config_lib.CONFIGS.values()
    ]
    for manager in managers:
        restore_schedule(
            manager,
            None if base is None else base.locations.get(manager.config.code),
            cadence=cadence,
        )
    await asyncio.gather(*(serve_location(manager) for manager in managers))
    await asyncio.gather(
        *(manager.wait_for_plots(PLOT_HARD_TIMEOUT) for manager in managers)
    )
    # The feeds themselves carry the capture counts; the manager exposes no
    # other accessor for them.
    results = [_location_counts(manager._feeds) for manager in managers]

    # The publisher skips the run when the assembled manifest can reference
    # nothing at all, so no location is dropped here.
    snapshot = Snapshot(
        locations={
            manager.config.code: build_location_snapshot(manager)
            for manager in managers
        }
    )
    outcome = "failed"
    try:
        result = await publish(
            store,
            snapshot,
            run_id=run_id,
            now=datetime.datetime.now(datetime.UTC),
        )
        outcome = result.outcome
    except Exception:
        # publish() logged the failed event before raising; the capture cycle
        # already ran, so the run keeps its own outcome.
        pass
    # The sweep runs whatever publication did, because a failed publication is
    # exactly when superseded generations are most likely to have piled up. It
    # logs its own event and returns rather than raising, so like publication
    # it cannot change the run's outcome.
    await collect_generations(store, now=datetime.datetime.now(datetime.UTC))
    return results, outcome


def _summary_fields(
    outcome: str,
    started_at: float,
    record_count: int,
    run_id: str,
    archived: CaptureResult,
) -> dict[str, object]:
    """Return bounded fields for the single run summary event.

    Args:
        outcome: Bounded run outcome value.
        started_at: Monotonic timestamp taken when the run started.
        record_count: Total published rows across captured feeds.
        run_id: Identifier correlating this run with platform execution logs.
        archived: Rows this run added to and revised in the archive.

    Returns:
        Approved structured logging fields for the summary event.
    """
    return {
        "component": "updater",
        "operation": "run",
        "outcome": outcome,
        "duration_ms": max(0, round((time.monotonic() - started_at) * 1000)),
        "record_count": record_count,
        "new_count": archived.new_count,
        "revised_count": archived.revised_count,
        "run_id": run_id,
    }


async def _run(*, full_history: bool, publish_snapshot: bool) -> int:
    """Capture every enabled location concurrently and emit one summary event.

    Args:
        full_history: Whether to fetch the full configured historical
            temperature year range instead of only the current UTC year. The
            publishing path always uses the full range, and takes the flag to
            mean "ignore the published schedule", so every feed fetches.
        publish_snapshot: Whether to run the full serving cycle and publish a
            snapshot instead of updating only the archivable feeds.

    Returns:
        Process exit code: 0 for success and partial runs, 1 for failed runs.
    """
    started_at = time.monotonic()
    # Cloud Run sets CLOUD_RUN_EXECUTION on job tasks. This is the job's only
    # platform-specific input, and it exists so run summaries correlate with the
    # platform's execution name.
    run_id = os.environ.get("CLOUD_RUN_EXECUTION") or uuid.uuid4().hex
    attempted = 0
    published = 0
    record_count = 0
    archived = CaptureResult(0, 0)
    publish_note = ""
    try:
        async with aiohttp.ClientSession() as session:
            clients = create_api_clients(session)
            if publish_snapshot:
                pool = ProcessPoolExecutor(max_workers=os.cpu_count())
                try:
                    results, publish_outcome = await publish_locations(
                        clients,
                        run_id,
                        pool=pool,
                        locator=os.environ[ARCHIVE_BUCKET_ENV_VAR],
                        full_history=full_history,
                        # Read at call time so tests can shorten the cadence.
                        cadence=JOB_CADENCE,
                    )
                finally:
                    pool.shutdown(wait=True)
                publish_note = f"; snapshot publish {publish_outcome}"
            else:
                results = await asyncio.gather(
                    *[
                        _capture_location(
                            location_config, clients, full_history=full_history
                        )
                        for location_config in config_lib.CONFIGS.values()
                    ]
                )
        attempted = sum(result[0] for result in results)
        published = sum(result[1] for result in results)
        record_count = sum(result[2] for result in results)
        archived = CaptureResult(
            sum(result[3].new_count for result in results),
            sum(result[3].revised_count for result in results),
        )
        outcome = run_outcome(published, attempted)
    except Exception as error:
        logging.error(
            f"Capture run failed: {error}",
            extra=_summary_fields("failed", started_at, record_count, run_id, archived),
        )
        return 1

    levels = {
        "success": logging.INFO,
        "partial": logging.WARNING,
        "failed": logging.ERROR,
    }
    logging.log(
        levels[outcome],
        f"Capture run {outcome}: {published} of {attempted} feeds published"
        f"{publish_note}",
        extra=_summary_fields(outcome, started_at, record_count, run_id, archived),
    )
    return 0 if outcome != "failed" else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse capture job arguments.

    Args:
        argv: Argument list to parse, or None to read from the command line.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m shallweswim.update",
        description="Fetch archivable feeds once so observations enter the archive.",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help=(
            "Fetch the full configured historical temperature year range "
            "instead of only the current UTC year. A publishing run "
            f"({SNAPSHOT_PUBLISH_ENV_VAR}=1) always fetches the full range, "
            "and ignores the published schedule so every feed fetches."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one bounded capture job.

    Args:
        argv: Argument list to parse, or None to read from the command line.

    Returns:
        Process exit code: 0 for success and partial runs, 1 otherwise.
    """
    args = _parse_args(argv)
    logging_utils.setup_logging()
    if not os.environ.get(ARCHIVE_BUCKET_ENV_VAR):
        logging.error(
            f"{ARCHIVE_BUCKET_ENV_VAR} is required; "
            "the capture job fetches only in order to archive"
        )
        return 1
    publish_snapshot = os.environ.get(SNAPSHOT_PUBLISH_ENV_VAR) == "1"
    if publish_snapshot and not os.environ.get(ARCHIVE_READ_BUCKET_ENV_VAR):
        logging.error(
            f"{ARCHIVE_READ_BUCKET_ENV_VAR} is required when "
            f"{SNAPSHOT_PUBLISH_ENV_VAR}=1; a snapshot carries the full "
            "historical range, which hydrates from the archive"
        )
        return 1
    return asyncio.run(
        _run(full_history=args.full_history, publish_snapshot=publish_snapshot)
    )


if __name__ == "__main__":
    sys.exit(main())

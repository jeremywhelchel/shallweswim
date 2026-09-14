"""Reachability-aware collection of superseded snapshot generations.

Objects are content-addressed, so a generation published minutes ago may still
reference an object written days ago: age alone can never decide what is
deletable. One sweep therefore marks and sweeps. It reads the current pointer,
keeps that generation whatever its age, keeps every generation published inside
`RETAINED_GENERATION_AGE`, deletes the manifests of the rest, and then deletes
only the objects no retained manifest references and that were created more
than `OBJECT_SAFETY_AGE` ago. The safety window is what protects a publisher
that has written a generation's objects but has not promoted it yet.

A manifest it cannot parse is retained and, because its object references are
then unknown, so is every object: the sweep deletes manifests but no objects in
that run rather than risk unlinking the live generation's frames.

The sweep deletes nothing it did not list in the same run, so a manifest or an
object written while it works is out of reach until the next one. It runs
inside the publishing cycle, after publication, and isolates its own failures:
`collect_generations` logs one event per run and returns rather than raising,
because the run's outcome belongs to the capture cycle.
"""

import dataclasses
import datetime
import logging
import time

from shallweswim.snapshot.model import Manifest
from shallweswim.snapshot.store import CURRENT_KEY, SnapshotStore

# The rollback window, and the window a slow instance could still be loading a
# generation from. Every generation published inside it is kept.
RETAINED_GENERATION_AGE = datetime.timedelta(hours=24)

# An unreferenced object younger than this may belong to a generation whose
# publisher has not promoted it yet. Publication takes under two minutes.
OBJECT_SAFETY_AGE = datetime.timedelta(hours=1)


@dataclasses.dataclass(frozen=True)
class GcResult:
    """What one sweep examined and deleted.

    `manifests_examined` and `objects_examined` are the sizes of the two
    listings the sweep made its decisions from; `reason` explains a failed
    outcome.
    """

    outcome: str
    manifests_examined: int
    manifests_deleted: int
    objects_examined: int
    objects_deleted: int
    duration_ms: int
    reason: str | None = None


def _event_fields(result: GcResult) -> dict[str, object]:
    return {
        "component": "snapshot",
        "operation": "gc",
        "outcome": result.outcome,
        "duration_ms": result.duration_ms,
        "record_count": result.objects_deleted,
    }


def _duration_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _referenced_keys(manifest: Manifest) -> set[str]:
    """Return every object key one generation's manifest names."""
    keys: set[str] = set()
    for location in manifest.locations.values():
        keys.update(entry.key for entry in location.feeds.values())
        keys.update(entry.key for entry in location.plots.values())
    return keys


async def collect_generations(
    store: SnapshotStore, *, now: datetime.datetime
) -> GcResult:
    """Delete superseded manifests and the objects nothing retained references.

    Args:
        store: The snapshot store the publishing cycle just published into.
        now: The timezone-aware instant both windows are measured against.

    Returns:
        The sweep's outcome and counts. A store failure or a broken invariant
        returns a `failed` result after logging it at ERROR; the sweep never
        raises, because its caller's outcome is the capture cycle's.
    """
    started_at = time.monotonic()
    manifests_examined = 0
    manifests_deleted = 0
    objects_examined = 0
    objects_deleted = 0
    try:
        current = await store.read_current()
        current_manifest_key = None if current is None else current[0].manifest_key

        referenced: set[str] = set()
        expired_keys: list[str] = []
        unparsable = False
        listed_manifests = await store.list_manifests()
        manifests_examined = len(listed_manifests)
        for listed in listed_manifests:
            if listed.key == CURRENT_KEY:
                # Not reachable through the manifests prefix, and never
                # deletable whatever a future layout change makes reachable.
                continue
            try:
                manifest = await store.read_manifest(listed.key)
            except Exception as error:
                # An unreadable manifest is retained: the sweep cannot tell
                # whether it is the rollback target, and deleting it would lose
                # the only record of what its objects are.
                logging.warning(
                    f"[snapshot] Retaining unparsable manifest {listed.key}: {error}"
                )
                unparsable = True
                continue
            if manifest is None:
                # Deleted between the listing and the read; nothing to do.
                continue
            if (
                listed.key == current_manifest_key
                or now - manifest.published_at <= RETAINED_GENERATION_AGE
            ):
                referenced |= _referenced_keys(manifest)
                continue
            expired_keys.append(listed.key)

        for key in expired_keys:
            await store.delete(key)
            manifests_deleted += 1

        listed_objects = await store.list_objects()
        objects_examined = len(listed_objects)
        for listed in listed_objects:
            # A retained manifest whose references are unknown makes every
            # object potentially reachable, so none is swept this run.
            if unparsable or listed.key in referenced:
                continue
            if listed.created_at >= now - OBJECT_SAFETY_AGE:
                continue
            await store.delete(listed.key)
            objects_deleted += 1

        result = GcResult(
            "success",
            manifests_examined,
            manifests_deleted,
            objects_examined,
            objects_deleted,
            _duration_ms(started_at),
        )
        logging.info(
            f"[snapshot] Collection deleted {manifests_deleted} of "
            f"{manifests_examined} manifests and {objects_deleted} of "
            f"{objects_examined} examined objects",
            extra=_event_fields(result),
        )
        return result
    except Exception as error:
        result = GcResult(
            "failed",
            manifests_examined,
            manifests_deleted,
            objects_examined,
            objects_deleted,
            _duration_ms(started_at),
            reason=str(error),
        )
        logging.error(
            f"[snapshot] Collection failed after deleting {manifests_deleted} of "
            f"{manifests_examined} manifests and {objects_deleted} of "
            f"{objects_examined} examined objects: {error}",
            extra=_event_fields(result),
        )
        return result

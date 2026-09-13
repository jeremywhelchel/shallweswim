"""Snapshot store semantics over the conditional object stores."""

import pytest

from shallweswim.archive.store import (
    FilesystemObjectStore,
    MemoryObjectStore,
    ObjectStore,
    VersionConflictError,
)
from shallweswim.snapshot.model import CurrentPointer
from shallweswim.snapshot.store import (
    CURRENT_KEY,
    PromotionConflictError,
    SnapshotStore,
    object_key,
)


def _object_store(kind: str, tmp_path) -> ObjectStore:
    return MemoryObjectStore() if kind == "memory" else FilesystemObjectStore(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["memory", "filesystem"])
async def test_write_object_is_create_only(store_kind: str, tmp_path) -> None:
    objects = _object_store(store_kind, tmp_path)
    store = SnapshotStore(objects)
    key = object_key(b"frame", "parquet")

    assert await store.write_object(key, b"frame") is True
    assert await store.write_object(key, b"frame") is False
    assert await store.read_object(key) == b"frame"
    assert await store.read_object(object_key(b"absent", "svg")) is None


@pytest.mark.asyncio
async def test_promote_replaces_only_the_observed_version() -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    first = CurrentPointer(manifest_key="published/manifests/a.json", generation_id="a")
    second = CurrentPointer(
        manifest_key="published/manifests/b.json", generation_id="b"
    )
    assert await store.read_current() is None

    version = await store.promote(first, None)
    current = await store.read_current()
    assert current == (first, version)

    with pytest.raises(PromotionConflictError, match="b"):
        await store.promote(second, None)
    with pytest.raises(PromotionConflictError):
        await store.promote(second, "stale")
    await store.promote(second, version)
    stored = await objects.read(CURRENT_KEY)
    assert stored is not None
    assert CurrentPointer.model_validate_json(stored.data) == second


@pytest.mark.asyncio
async def test_manifest_reads_absent_as_none_and_writes_once() -> None:
    from tests.test_snapshot_model import _manifest

    store = SnapshotStore(MemoryObjectStore())
    manifest = _manifest()
    assert await store.read_manifest("published/manifests/missing.json") is None

    key, data = await store.write_manifest(manifest)

    assert key == f"published/manifests/{manifest.generation_id}.json"
    assert await store.read_manifest(key) == manifest
    assert await store.read_object(key) == data
    with pytest.raises(VersionConflictError):
        await store.write_manifest(manifest)

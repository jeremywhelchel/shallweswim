"""Conditional object-store contract tests."""

import asyncio
import datetime
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shallweswim.archive import store as store_module
from shallweswim.archive.store import (
    FilesystemObjectStore,
    GcsObjectStore,
    MemoryObjectStore,
    ObjectStore,
    VersionConflictError,
    object_store,
)


def test_importing_store_does_not_import_google_sdk() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import shallweswim.archive.store; "
                "assert 'google.cloud.storage' not in sys.modules; "
                "assert 'google.api_core.exceptions' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["memory", "filesystem"])
async def test_store_create_replace_and_conflict(store_kind: str, tmp_path) -> None:
    store = (
        MemoryObjectStore()
        if store_kind == "memory"
        else FilesystemObjectStore(tmp_path)
    )

    assert await store.read("archive/temperature/ndbc/12345/2026.parquet") is None
    first_version = await store.compare_and_swap(
        "archive/temperature/ndbc/12345/2026.parquet",
        expected_version=None,
        data=b"first",
    )
    first = await store.read("archive/temperature/ndbc/12345/2026.parquet")
    assert first is not None
    assert first.data == b"first"
    assert first.version == first_version

    with pytest.raises(VersionConflictError):
        await store.compare_and_swap(
            "archive/temperature/ndbc/12345/2026.parquet",
            expected_version=None,
            data=b"second",
        )

    second_version = await store.compare_and_swap(
        "archive/temperature/ndbc/12345/2026.parquet",
        expected_version=first_version,
        data=b"second",
    )
    assert second_version != first_version


@pytest.mark.asyncio
async def test_filesystem_content_version_has_benign_aba(tmp_path) -> None:
    store = FilesystemObjectStore(tmp_path)
    key = "objects/example"
    first_version = await store.compare_and_swap(
        key, expected_version=None, data=b"same"
    )
    changed_version = await store.compare_and_swap(
        key, expected_version=first_version, data=b"changed"
    )
    reverted_version = await store.compare_and_swap(
        key, expected_version=changed_version, data=b"same"
    )

    assert reverted_version == first_version
    await store.compare_and_swap(
        key, expected_version=first_version, data=b"merged from identical state"
    )


@pytest.mark.asyncio
async def test_filesystem_concurrent_cas_has_one_winner(tmp_path) -> None:
    store = FilesystemObjectStore(tmp_path)
    key = "objects/concurrent"
    version = await store.compare_and_swap(key, expected_version=None, data=b"base")

    results = await asyncio.gather(
        store.compare_and_swap(key, expected_version=version, data=b"writer-one"),
        store.compare_and_swap(key, expected_version=version, data=b"writer-two"),
        return_exceptions=True,
    )

    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, VersionConflictError) for result in results) == 1


@pytest.mark.asyncio
async def test_store_rejects_nonportable_keys(tmp_path) -> None:
    store = FilesystemObjectStore(tmp_path)

    for key in ("", "/absolute", "../escape", "folder//double", "folder\\windows"):
        with pytest.raises(ValueError, match="Invalid object key"):
            await store.read(key)


@pytest.mark.asyncio
async def test_gcs_store_maps_generation_reads_and_writes() -> None:
    client = MagicMock()
    bucket = client.bucket.return_value
    existing = MagicMock(generation=17)
    existing.download_as_bytes.return_value = b"existing"
    bucket.get_blob.return_value = existing
    written = MagicMock(generation=18)
    bucket.blob.return_value = written
    store = GcsObjectStore("archive-bucket", client=client)

    result = await store.read("archive/temperature/ndbc/12345/2026.parquet")
    new_version = await store.compare_and_swap(
        "archive/temperature/ndbc/12345/2026.parquet",
        expected_version="17",
        data=b"replacement",
    )

    assert result is not None
    assert result.data == b"existing"
    assert result.version == "17"
    existing.download_as_bytes.assert_called_once_with(if_generation_match=17)
    written.upload_from_string.assert_called_once_with(
        b"replacement", if_generation_match=17
    )
    assert new_version == "18"


@pytest.mark.asyncio
async def test_gcs_store_maps_create_and_precondition_conflict() -> None:
    from google.api_core import exceptions as google_exceptions

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.upload_from_string.side_effect = google_exceptions.PreconditionFailed(
        "conflict"
    )
    store = GcsObjectStore("archive-bucket", client=client)

    with pytest.raises(VersionConflictError):
        await store.compare_and_swap("objects/hash", expected_version=None, data=b"x")

    blob.upload_from_string.assert_called_once_with(b"x", if_generation_match=0)


async def _seed_listing(store: ObjectStore, aged_at: datetime.datetime) -> None:
    """Write two objects under one prefix and one under another."""
    for key in ("published/objects/one", "published/objects/two", "published/other"):
        await store.compare_and_swap(key, expected_version=None, data=key.encode())
    if isinstance(store, MemoryObjectStore):
        store._created["published/objects/one"] = aged_at
    else:
        assert isinstance(store, FilesystemObjectStore)
        path = Path(store._root) / "published/objects/one"
        os.utime(path, (aged_at.timestamp(), aged_at.timestamp()))


@pytest.mark.asyncio
@pytest.mark.parametrize("store_kind", ["memory", "filesystem"])
async def test_store_lists_and_deletes_identically(store_kind: str, tmp_path) -> None:
    """The listing is prefix-scoped, timezone-aware, and delete is idempotent."""
    store: ObjectStore = (
        MemoryObjectStore()
        if store_kind == "memory"
        else FilesystemObjectStore(tmp_path)
    )
    aged_at = datetime.datetime(2026, 6, 1, 12, 0, tzinfo=datetime.UTC)
    await _seed_listing(store, aged_at)

    listed = await store.list("published/objects")

    assert [entry.key for entry in listed] == [
        "published/objects/one",
        "published/objects/two",
    ]
    ages = {entry.key: entry.created_at for entry in listed}
    assert ages["published/objects/one"] == aged_at
    assert ages["published/objects/two"].tzinfo is not None
    assert ages["published/objects/two"] > aged_at

    await store.delete("published/objects/one")
    # Deleting what is already gone is not an error; nor is listing a prefix
    # that never existed.
    await store.delete("published/objects/one")

    assert [entry.key for entry in await store.list("published/objects")] == [
        "published/objects/two"
    ]
    assert await store.read("published/objects/one") is None
    assert await store.read("published/other") is not None
    assert await store.list("published/absent") == []


@pytest.mark.asyncio
async def test_filesystem_listing_skips_locks_and_partial_writes(tmp_path) -> None:
    """Only objects are listed: the lock directory and temporaries are dot-named."""
    store = FilesystemObjectStore(tmp_path)
    await store.compare_and_swap(
        "published/objects/one", expected_version=None, data=b"one"
    )
    partial = tmp_path / "published/objects/.one.tmp12345"
    partial.write_bytes(b"partial")

    assert [entry.key for entry in await store.list("published")] == [
        "published/objects/one"
    ]


@pytest.mark.asyncio
async def test_gcs_store_lists_creation_times_and_deletes() -> None:
    from google.api_core import exceptions as google_exceptions

    client = MagicMock()
    bucket = client.bucket.return_value
    created_at = datetime.datetime(2026, 6, 1, 12, 0, tzinfo=datetime.UTC)
    blob = MagicMock(time_created=created_at)
    # MagicMock resolves `name` through the constructor, not an attribute.
    blob.name = "published/objects/one"
    bucket.list_blobs.return_value = [blob]
    store = GcsObjectStore("archive-bucket", client=client)

    listed = await store.list("published/objects")
    await store.delete("published/objects/one")

    assert [(entry.key, entry.created_at) for entry in listed] == [
        ("published/objects/one", created_at)
    ]
    bucket.list_blobs.assert_called_once_with(prefix="published/objects")
    bucket.blob.return_value.delete.assert_called_once_with()

    # A key another sweep already removed is not an error.
    bucket.blob.return_value.delete.side_effect = google_exceptions.NotFound("gone")
    await store.delete("published/objects/one")


@pytest.mark.asyncio
async def test_list_and_delete_reject_nonportable_keys(tmp_path) -> None:
    for store in (MemoryObjectStore(), FilesystemObjectStore(tmp_path)):
        with pytest.raises(ValueError, match="Invalid object key"):
            await store.list("../escape")
        with pytest.raises(ValueError, match="Invalid object key"):
            await store.delete("/absolute")


def test_object_store_resolves_every_locator_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One locator string names a bucket, a directory, or process memory."""
    constructed: list[str] = []
    monkeypatch.setattr(
        store_module,
        "GcsObjectStore",
        lambda bucket: constructed.append(bucket) or MagicMock(),
    )
    store_module.gcs_store.cache_clear()
    store_module.memory_store.cache_clear()

    filesystem = object_store(str(tmp_path / "local-store"))
    memory = object_store("memory")
    again = object_store("memory")
    object_store("archive-bucket")

    assert isinstance(filesystem, FilesystemObjectStore)
    assert isinstance(memory, MemoryObjectStore)
    # One process-wide memory store, or the local entry point's job half would
    # publish into a store its web half never reads.
    assert again is memory
    assert constructed == ["archive-bucket"]

    store_module.gcs_store.cache_clear()
    store_module.memory_store.cache_clear()

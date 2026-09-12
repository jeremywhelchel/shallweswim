"""Conditional object-store contract tests."""

import asyncio
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from shallweswim.archive.store import (
    FilesystemObjectStore,
    GcsObjectStore,
    MemoryObjectStore,
    VersionConflictError,
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

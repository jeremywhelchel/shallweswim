"""Deployment manifests must keep the capture job the archive's only writer.

The job is also the only snapshot publisher, so the publish switch and the
archive read bucket it needs belong to the job manifest alone. The service
gets one bucket variable, the read-only snapshot bucket that enables shadow
mode, and the job never gets that one.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE_YAML = (ROOT / "service.yaml").read_text()
CAPTURE_JOB_YAML = (ROOT / "capture-job.yaml").read_text()
CLOUDBUILD_YAML = (ROOT / "cloudbuild.yaml").read_text()

ARCHIVE_BUCKET_VAR = "SHALLWESWIM_ARCHIVE_BUCKET"
ARCHIVE_READ_BUCKET_VAR = "SHALLWESWIM_ARCHIVE_READ_BUCKET"
SNAPSHOT_PUBLISH_VAR = "SHALLWESWIM_SNAPSHOT_PUBLISH"
SNAPSHOT_READ_BUCKET_VAR = "SHALLWESWIM_SNAPSHOT_READ_BUCKET"


def _service_account(manifest: str) -> str:
    match = re.search(r"^\s*serviceAccountName:\s*(\S+)\s*$", manifest, re.MULTILINE)
    assert match is not None
    return match.group(1)


def _image(manifest: str) -> str:
    match = re.search(r"^\s*-?\s*image:\s*(\S+)\s*$", manifest, re.MULTILINE)
    assert match is not None
    return match.group(1)


def _env_var_names(manifest: str) -> set[str]:
    """The environment variable names one manifest's containers declare."""
    names = re.findall(r"^\s*-\s*name:\s*(\S+)\s*$", manifest, re.MULTILINE)
    return {name for name in names if name.isupper()}


def test_web_service_never_configures_the_archive_bucket() -> None:
    """Multi-instance web serving must not become a concurrent archive writer."""
    assert ARCHIVE_BUCKET_VAR not in _env_var_names(SERVICE_YAML)


def test_web_service_never_publishes_or_hydrates() -> None:
    """Publication and archive hydration belong to the job, never the service."""
    service_vars = _env_var_names(SERVICE_YAML)
    assert SNAPSHOT_PUBLISH_VAR not in service_vars
    assert ARCHIVE_READ_BUCKET_VAR not in service_vars


def test_web_service_reads_snapshots_from_the_archive_bucket() -> None:
    """Shadow mode reads the bucket the job publishes into, substituted alike."""
    assert SNAPSHOT_READ_BUCKET_VAR in _env_var_names(SERVICE_YAML)
    assert re.search(
        rf"name: {SNAPSHOT_READ_BUCKET_VAR}\s+value: \${{{ARCHIVE_BUCKET_VAR}}}",
        SERVICE_YAML,
    )
    # The service's deploy step substitutes the placeholder, as the job's does.
    assert SERVICE_YAML.count(f"value: ${{{ARCHIVE_BUCKET_VAR}}}") == 1
    assert (
        CLOUDBUILD_YAML.count(f"s|\\$${{{ARCHIVE_BUCKET_VAR}}}|${{_ARCHIVE_BUCKET}}|g")
        == 2
    )
    # Both deploy steps refuse to deploy an empty substitution.
    assert CLOUDBUILD_YAML.count('if [ -z "${_ARCHIVE_BUCKET}" ]; then') == 2


def test_capture_job_never_reads_snapshots() -> None:
    """Only the service loads published generations; the job publishes them."""
    assert SNAPSHOT_READ_BUCKET_VAR not in _env_var_names(CAPTURE_JOB_YAML)


def test_capture_job_configures_the_archive_bucket_placeholder() -> None:
    """The job is the writer, and the bucket name is substituted at deploy time."""
    assert f"name: {ARCHIVE_BUCKET_VAR}" in CAPTURE_JOB_YAML
    assert f"value: ${{{ARCHIVE_BUCKET_VAR}}}" in CAPTURE_JOB_YAML


def test_capture_job_publishes_snapshots_from_the_archive_bucket() -> None:
    """The job publishes, and hydrates history from the bucket it writes to."""
    assert f"name: {SNAPSHOT_PUBLISH_VAR}\n" in CAPTURE_JOB_YAML
    assert re.search(rf"name: {SNAPSHOT_PUBLISH_VAR}\s+value: \"1\"", CAPTURE_JOB_YAML)
    assert re.search(
        rf"name: {ARCHIVE_READ_BUCKET_VAR}\s+value: \${{{ARCHIVE_BUCKET_VAR}}}",
        CAPTURE_JOB_YAML,
    )
    # Cloud Build substitutes every occurrence of the placeholder.
    assert CAPTURE_JOB_YAML.count(f"value: ${{{ARCHIVE_BUCKET_VAR}}}") == 2
    assert f"s|\\$${{{ARCHIVE_BUCKET_VAR}}}|${{_ARCHIVE_BUCKET}}|g" in CLOUDBUILD_YAML


def test_service_and_capture_job_use_distinct_identities() -> None:
    """Only the job identity is bound to the archive bucket, so it must differ."""
    assert _service_account(SERVICE_YAML) != _service_account(CAPTURE_JOB_YAML)


def test_capture_job_runs_the_capture_entry_point() -> None:
    """The job overrides the image command rather than starting the web app."""
    assert 'command: ["python"]' in CAPTURE_JOB_YAML
    assert '"-m", "shallweswim.capture"' in CAPTURE_JOB_YAML


def test_service_and_capture_job_share_one_substituted_image() -> None:
    """A single build tag deploys both, so the job never runs a stale image."""
    service_image = _image(SERVICE_YAML)
    job_image = _image(CAPTURE_JOB_YAML)

    assert service_image.endswith(":${IMAGE_TAG}")
    assert job_image == service_image


def test_cloudbuild_replaces_the_capture_job() -> None:
    """Every service deploy also replaces the job from the same build."""
    assert "capture-job.yaml" in CLOUDBUILD_YAML
    assert "gcloud run jobs replace" in CLOUDBUILD_YAML

#!/usr/bin/env bash

set -euo pipefail

# The capture job writes to an installation-local archive bucket, so the bucket
# name is supplied from the operator environment rather than committed.
if [ -z "${SHALLWESWIM_ARCHIVE_BUCKET:-}" ]; then
  echo "ERROR: SHALLWESWIM_ARCHIVE_BUCKET is unset or empty." >&2
  echo "Set it in .env/.envrc (see .env.example) to the bucket the scheduled" >&2
  echo "capture job writes to, then re-run this script." >&2
  exit 1
fi

gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_ARCHIVE_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET"

#!/usr/bin/env bash

set -euo pipefail

# Run from the repository root: the build context is the whole repository and
# cloudbuild.yaml reads the manifests as infra/service.yaml and
# infra/capture-job.yaml.
if [ ! -f infra/cloudbuild.yaml ]; then
  echo "ERROR: run ./infra/build_and_deploy.sh from the repository root." >&2
  exit 1
fi

# The archive bucket, the project, and the region are installation-local, so
# they come from the operator environment rather than the repository. The
# build runs as the deploy identity, shallweswim-ci, in that project.
for required in SHALLWESWIM_ARCHIVE_BUCKET CLOUDSDK_CORE_PROJECT CLOUDSDK_RUN_REGION; do
  if [ -z "${!required:-}" ]; then
    echo "ERROR: $required is unset or empty." >&2
    echo "Set it in .env/.envrc (see .env.example), then re-run this script." >&2
    exit 1
  fi
done

gcloud builds submit --config infra/cloudbuild.yaml \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --service-account="projects/$CLOUDSDK_CORE_PROJECT/serviceAccounts/shallweswim-ci@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --substitutions=_ARCHIVE_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET",_REGION="$CLOUDSDK_RUN_REGION"

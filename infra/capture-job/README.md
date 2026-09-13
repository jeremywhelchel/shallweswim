# Observation Capture Job

The Cloud Run Job defined by [`capture-job.yaml`](../../capture-job.yaml) runs
`python -m shallweswim.capture` on a schedule. It fetches only archivable feeds
(live temperatures, historical temperatures, and observational currents) and
writes normalized observations to the private archive bucket. It generates no
plots and starts no web server.

**The invariant: this job is the only production writer to the archive.** The
web runtime identity `shallweswim-runtime@shallweswim.iam.gserviceaccount.com`
receives no binding on the archive bucket, and `service.yaml` never sets
`SHALLWESWIM_ARCHIVE_BUCKET`, so the multi-instance web service cannot write
archive objects even accidentally. Keep it that way: enabling capture in the web
service would reintroduce concurrent writers from every serving instance.

Bucket creation and the log-based metrics and dashboard that observe this job
live in [`../monitoring/README.md`](../monitoring/README.md). This file owns the
job's identities, deployment, scheduling, and operation.

Load the operator credential and project through repo-local environment
variables (`GOOGLE_APPLICATION_CREDENTIALS`,
`CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, `CLOUDSDK_CORE_PROJECT`,
`SHALLWESWIM_ARCHIVE_BUCKET`). Do not use `gcloud auth` or modify global
`gcloud` configuration.

## One-time identity setup

Create the archive bucket first (see
[`../monitoring/README.md`](../monitoring/README.md)), then create the two
dedicated service accounts. `shallweswim-capture` runs the job;
`shallweswim-capture-invoker` only triggers it from Cloud Scheduler.

```bash
gcloud iam service-accounts create shallweswim-capture \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --display-name="Shall We Swim observation capture job"

gcloud iam service-accounts create shallweswim-capture-invoker \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --display-name="Shall We Swim capture job scheduler invoker"
```

Grant the job identity object access to the archive bucket only. The
bucket-scoped object role permits the generation-conditional reads and
replacements the merge path needs without granting bucket administration.

```bash
gcloud storage buckets add-iam-policy-binding \
  "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-capture@shallweswim.iam.gserviceaccount.com" \
  --role=roles/storage.objectUser
```

Grant the job identity read access to the USGS API key secret referenced by
`capture-job.yaml`:

```bash
gcloud secrets add-iam-policy-binding waterdata_usgs_gov_api_key \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --member="serviceAccount:shallweswim-capture@shallweswim.iam.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

Let the Cloud Build identity deploy a job that runs as `shallweswim-capture`:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  shallweswim-capture@shallweswim.iam.gserviceaccount.com \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --member="serviceAccount:shallweswim-ci@shallweswim.iam.gserviceaccount.com" \
  --role=roles/iam.serviceAccountUser
```

The Cloud Build identity also needs `roles/run.developer` on the project so
`gcloud run jobs replace` can create and update the job; a service-scoped
deploy binding does not cover jobs. If the project policy contains conditional
bindings, gcloud requires `--condition=None` for an unconditional grant.

```bash
gcloud projects add-iam-policy-binding shallweswim \
  --member="serviceAccount:shallweswim-ci@shallweswim.iam.gserviceaccount.com" \
  --role=roles/run.developer \
  --condition=None
```

Do not grant `shallweswim-capture` anything else, and do not grant the web
runtime identity any archive bucket role.

## First deploy

`build_and_deploy.sh` builds the image, replaces the Cloud Run service, and then
replaces the job from the same image. It refuses to run when
`SHALLWESWIM_ARCHIVE_BUCKET` is unset, and the build step fails if the bucket
substitution reaches Cloud Build empty.

```bash
./build_and_deploy.sh

gcloud run jobs describe shallweswim-capture --region=us-east4
```

Confirm in the description that the image tag matches the build just deployed,
the service account is `shallweswim-capture@shallweswim.iam.gserviceaccount.com`,
`SHALLWESWIM_ARCHIVE_BUCKET` holds the intended bucket name, the task timeout is
1200 seconds, and retries are limited to one.

## Manual runs

Run the scheduled cycle on demand and wait for it to finish:

```bash
gcloud run jobs execute shallweswim-capture --region=us-east4 --wait
```

The one-time historical backfill adds `--full-history`. `gcloud run jobs
execute` accepts `--args` as a per-execution override: the comma-separated list
replaces the container `args` for that execution only, leaving the job
definition (and therefore every scheduled run) unchanged. The `command` stays
`python`, so the full argument vector must be given:

```bash
gcloud run jobs execute shallweswim-capture --region=us-east4 --wait \
  --args="-m,shallweswim.capture,--full-history"
```

A per-execution override is used deliberately in preference to
`gcloud run jobs update --args=...` followed by a revert: an update mutates the
deployed job, and an interrupted session would leave the hourly schedule running
the expensive full-history fetch. A backfill can exceed the 1200-second task
timeout; if it does, add `--task-timeout` to the same command rather than
raising the timeout in `capture-job.yaml`.

`--full-history` is never scheduled. A `partial` run — some feeds published,
some upstream stations unavailable — exits zero by design, so a successful
execution does not by itself mean every feed was captured; check the summary
event described below.

## Schedule

Bind the invoker identity to the job, then create the Cloud Scheduler job. The
hourly minute-7 offset keeps the run clear of the top of the hour.

```bash
gcloud run jobs add-iam-policy-binding shallweswim-capture \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --region=us-east4 \
  --member="serviceAccount:shallweswim-capture-invoker@shallweswim.iam.gserviceaccount.com" \
  --role=roles/run.invoker

gcloud scheduler jobs create http shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location=us-east4 \
  --schedule="7 * * * *" \
  --time-zone=Etc/UTC \
  --uri="https://us-east4-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/shallweswim/jobs/shallweswim-capture:run" \
  --http-method=POST \
  --oauth-service-account-email=shallweswim-capture-invoker@shallweswim.iam.gserviceaccount.com \
  --attempt-deadline=180s
```

The `:run` endpoint returns as soon as the execution is created, so the attempt
deadline covers only that API call and is unrelated to the job's 1200-second
task timeout. Do not raise it to cover the run. (The equivalent v2 endpoint,
`https://run.googleapis.com/v2/projects/PROJECT/locations/us-east4/jobs/shallweswim-capture:run`,
also works; the v1 namespaced form above is the one this deployment uses.)

Creating a scheduler job with `--oauth-service-account-email` requires the
operator to hold `roles/iam.serviceAccountUser` on
`shallweswim-capture-invoker`.

Scheduler policy is what prevents overlapping executions; correctness does not
depend on it, because the archive merge path is conditional read-merge-write and
tolerates an overlap.

## Validation

After the first scheduled runs:

1. `gcloud run jobs executions list --job=shallweswim-capture --region=us-east4`
   shows recent executions and their task completion state.
2. The run summary event is present and reports the expected outcome:

   ```bash
   gcloud logging read \
     'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.component="updater" AND jsonPayload.operation="run"' \
     --limit=10 \
     --format='table(timestamp,severity,jsonPayload.outcome,jsonPayload.duration_ms,jsonPayload.record_count)'
   ```

3. Archive merges are emitted with bounded outcomes:

   ```bash
   gcloud logging read \
     'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.component="archive" AND jsonPayload.operation="merge"' \
     --limit=20 \
     --format='table(timestamp,jsonPayload.source_identity,jsonPayload.outcome)'
   ```

4. Objects exist under the archive prefix:

   ```bash
   gcloud storage ls --recursive "gs://$SHALLWESWIM_ARCHIVE_BUCKET/archive/"
   ```

5. The "Archive merges per 5 minutes by outcome" chart on the
   `Shall We Swim Operations [Terraform]` dashboard shows the job's merges. That
   chart is deliberately not restricted to `cloud_run_revision`, so job series
   appear alongside any service series.

Compare archived row counts with the live feeds for at least a week before
anything reads the archive.

## Pausing

Pausing the schedule stops new captures and leaves every archived object in
place:

```bash
gcloud scheduler jobs pause shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" --location=us-east4

gcloud scheduler jobs resume shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" --location=us-east4
```

Never delete the archive bucket or its objects to stop capture, and do not add a
lifecycle rule to it; normalized observations are retained indefinitely.

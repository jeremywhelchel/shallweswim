# Observation Capture Job

The Cloud Run Job defined by [`capture-job.yaml`](../../capture-job.yaml) runs
`python -m shallweswim.update` on a schedule. It writes normalized
observations to the private archive bucket and, because the job definition sets
`SHALLWESWIM_SNAPSHOT_PUBLISH=1`, it also runs the full serving cycle of every
location (tide and prediction feeds included), generates plots in a process
pool, and publishes one serving snapshot generation under `published/` in the
same bucket. It starts no web server. Without the publish variable the job
fetches only archivable feeds (live temperatures, historical temperatures, and
observational currents) and generates no plots.

A publishing run restores each feed's next fetch time from the generation that
is current when it starts, so it fetches only the feeds that are due and leaves
the rest on the entries and plots they already published. The published
manifest is the job's persisted feed schedule.

The Cloud Run job name (`shallweswim-capture`), the scheduler job
(`shallweswim-capture-hourly`), and the service accounts
(`shallweswim-capture`, `shallweswim-capture-invoker`) all keep their
historical `capture` naming below even though the module the job now runs is
`shallweswim.update`; renaming those deployed resources is a separate,
disruptive action this rename does not take.

Each run also collects the generations its publication superseded. It deletes
manifests under `published/manifests/` older than 24 hours, unless the current
pointer names them, and then objects under `published/objects/` that no
retained manifest references and that were created more than an hour ago.
`published/current.json` is never deleted. **The sweep never deletes anything
under `archive/`: normalized observations are retained indefinitely, and no
lifecycle rule on this bucket deletes them either.** A failed sweep logs one
`snapshot.gc` event at ERROR and leaves the run's outcome and exit code alone.

The job manifest sets three application variables:

- `SHALLWESWIM_ARCHIVE_BUCKET`: the bucket capture writes to and snapshots
  publish into; substituted at deploy time.
- `SHALLWESWIM_SNAPSHOT_PUBLISH`: `"1"` makes every run publish a snapshot
  after its capture cycle.
- `SHALLWESWIM_ARCHIVE_READ_BUCKET`: the same bucket, substituted from the same
  placeholder, so the historical feed restores past years from the archive
  instead of refetching them; a publishing run always uses the full historical
  range. It is required whenever publishing is enabled.

`service.yaml` sets none of them. It sets one bucket variable of its own,
`SHALLWESWIM_SNAPSHOT_READ_BUCKET`, substituted from the same placeholder, which
only reads the generations this job publishes.

**The invariant: this job is the only production writer to the archive.** The
web runtime identity `shallweswim-runtime@shallweswim.iam.gserviceaccount.com`
holds `roles/storage.objectViewer` on the archive bucket and nothing more, and
`service.yaml` never sets `SHALLWESWIM_ARCHIVE_BUCKET`, so the multi-instance
web service can read published generations but cannot write archive objects even
accidentally. Keep it that way: enabling capture in the web service would
reintroduce concurrent writers from every serving instance.

```bash
gcloud storage buckets add-iam-policy-binding \
  "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-runtime@shallweswim.iam.gserviceaccount.com" \
  --role=roles/storage.objectViewer
```

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

Local development hydrates historical temperature years from the archive
through `SHALLWESWIM_ARCHIVE_READ_BUCKET`, which only reads. Grant the local
operator identity read access to the bucket, and no write role; this viewer
grant is the intended steady state for local work.

```bash
gcloud storage buckets add-iam-policy-binding \
  "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-local-operator@shallweswim.iam.gserviceaccount.com" \
  --role=roles/storage.objectViewer
```

The web runtime identity takes the same read-only grant so the service can load
published generations (see the invariant above for the command).

Do not grant `shallweswim-capture` anything else, do not grant the web runtime
identity any role on the archive bucket beyond `roles/storage.objectViewer`, and
do not leave any local identity holding a write role on the bucket.

## Continuous build trigger

The GitHub-triggered Cloud Build on `main` runs the same `cloudbuild.yaml`, so
it must carry the bucket substitution or its job step fails after the service
step has already deployed. Set it once on the trigger (find the trigger id with
`gcloud builds triggers list`):

```bash
gcloud builds triggers update github TRIGGER_ID \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --region=global \
  --update-substitutions=_ARCHIVE_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET"
```

Do this before pushing the capture job definitions to `main`.

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
`SHALLWESWIM_ARCHIVE_BUCKET` and `SHALLWESWIM_ARCHIVE_READ_BUCKET` both hold the
intended bucket name, `SHALLWESWIM_SNAPSHOT_PUBLISH` is `1`, the task timeout is
1200 seconds, and retries are limited to one.

## Manual runs

Run the scheduled cycle on demand and wait for it to finish:

```bash
gcloud run jobs execute shallweswim-capture --region=us-east4 --wait
```

The one-time historical backfill adds `--full-history`. A capture-only run then
fetches every configured year instead of the current one; a publishing run
already fetches the full range, with past years hydrated from the archive, and
takes the flag to mean "ignore the published schedule", so every feed fetches
whether or not it is due. `gcloud run jobs
execute` accepts `--args` as a per-execution override: the comma-separated list
replaces the container `args` for that execution only, leaving the job
definition (and therefore every scheduled run) unchanged. The `command` stays
`python`, so the full argument vector must be given:

```bash
gcloud run jobs execute shallweswim-capture --region=us-east4 --wait \
  --args="-m,shallweswim.update,--full-history"
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

### Cutover: move the cadence to every ten minutes

Cutover makes this job the only process that talks to providers, so the web
service's freshness now depends on how often the job runs. Move the schedule to
every ten minutes (the job name keeps its original `-hourly` suffix; renaming a
scheduler job means deleting and recreating it):

```bash
gcloud scheduler jobs update http shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location=us-east4 \
  --schedule="*/10 * * * *" \
  --time-zone=Etc/UTC
```

This does not multiply provider traffic by six. The run restores each feed's
next fetch time from the current generation's manifest, so a feed that is not
due is not fetched: live temperature every ten minutes, historical temperature
every three hours, tide and current predictions daily — the same rate one web
instance generates today. A run in which no feed is due publishes nothing and
logs `snapshot.publish` with `outcome=unchanged`.

Deploy order: the web service revision that serves from the published
generation goes first. Observe one refresh and the health check on the new
revision, then change the cadence here. Rolling the web service back to a
revision that fetches for itself needs no change to this job or its schedule;
leaving the ten-minute cadence in place is harmless.

`capture-job.yaml` is unchanged by cutover. The cadence is the only operator
action.

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

5. A snapshot generation was published: one `snapshot.publish` event per run
   with a bounded outcome, objects and a manifest under the published prefix,
   and a current pointer whose `manifest_key` exists under
   `published/manifests/`. The run summary message also names the publish
   outcome; a run whose publish failed still exits zero, so check the event
   rather than the execution state.

   ```bash
   gcloud logging read \
     'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.component="snapshot" AND jsonPayload.operation="publish"' \
     --limit=10 \
     --format='table(timestamp,severity,jsonPayload.outcome,jsonPayload.duration_ms,jsonPayload.record_count,jsonPayload.generation_id)'

   gcloud storage ls --recursive "gs://$SHALLWESWIM_ARCHIVE_BUCKET/published/"
   gcloud storage cat "gs://$SHALLWESWIM_ARCHIVE_BUCKET/published/current.json"
   ```

6. Every configured feed reported its freshness: one `snapshot.freshness` event
   per location and feed per run, with `outcome` `success` for a feed fetched
   this run, `held` for one that was not due, so the manifest keeps its entry
   and plots unchanged, `carried` for one whose last published entry the
   manifest carried forward after a failure, and `absent` for one with nothing
   to serve. `held` is the ordinary state of a feed whose interval is longer
   than the ten-minute cadence and is logged at INFO. A growing `age_seconds`
   on a `carried` feed is a feed stuck on a failing source, not a publication
   problem; the publish event stays `success`.

   ```bash
   gcloud logging read \
     'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.component="snapshot" AND jsonPayload.operation="freshness"' \
     --limit=40 \
     --format='table(timestamp,severity,jsonPayload.location,jsonPayload.feed,jsonPayload.outcome,jsonPayload.age_seconds)'
   ```

7. The generations are being collected: one `snapshot.gc` event per publishing
   run, `record_count` being the objects that run deleted. A steady state has
   roughly one day of manifests under `published/manifests/`.

   ```bash
   gcloud logging read \
     'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.component="snapshot" AND jsonPayload.operation="gc"' \
     --limit=10 \
     --format='table(timestamp,severity,jsonPayload.outcome,jsonPayload.duration_ms,jsonPayload.record_count)'

   gcloud storage ls "gs://$SHALLWESWIM_ARCHIVE_BUCKET/published/manifests/"
   ```

8. The "Archive merges per hour by outcome", "Snapshot publishes per hour by
   outcome", and "Snapshot feed age max by feed per hour" charts on the
   `Shall We Swim Operations [Terraform]` dashboard show the job's merges,
   publishes, and published feed ages. Those charts are deliberately not
   restricted to `cloud_run_revision`, so job series appear alongside any
   service series.

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

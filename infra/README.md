# Reference Deployment on Google Cloud

The application is portable: it needs an object store, a scheduled job, and
web servers, and ARCHITECTURE.md "Documentation" states that nothing outside
this directory names a provider. This directory is one deployment of it, on
Google Cloud, and this guide is how to stand that deployment up and operate
it. What the job, the bundle, and the web servers do is in
[DATA_PIPELINE.md](../DATA_PIPELINE.md); the metrics, alert policies,
dashboard, uptime check, and log queries are in [MONITORING.md](MONITORING.md),
applied from [monitoring/](monitoring/README.md).

| Piece | Google Cloud resource | Defined by |
| --- | --- | --- |
| web servers | Cloud Run service `shallweswim` | [`service.yaml`](service.yaml) |
| job | Cloud Run job `shallweswim-capture`, run every ten minutes by Cloud Scheduler job `shallweswim-capture-hourly` | [`capture-job.yaml`](capture-job.yaml) |
| store | one private bucket, `$SHALLWESWIM_ARCHIVE_BUCKET`, holding `archive/` and `published/` | created once, below |
| image | one container image in Artifact Registry, built by Cloud Build | [`cloudbuild.yaml`](cloudbuild.yaml), submitted by [`build_and_deploy.sh`](build_and_deploy.sh) |
| monitoring | log-based metrics, alert policies, dashboard | [`monitoring/`](monitoring/README.md), Terraform |

The job and the scheduler keep their `capture` and `-hourly` names from when
they were introduced; renaming deployed resources is a separate, disruptive
action, and the names are only names.

Every command below takes the project, region, bucket, and credential from
the environment variables in [`.env.example`](../.env.example), so nothing
here names an installation. The resource names that do appear, the service,
the job, the scheduler job, the secret, the custom role, and the service
accounts, are the deployment's own: the commands below create them, and
each is unique only within its project. The bucket is the one globally
unique name, which is why it is a variable. A person normally runs `gcloud` as themselves
after `gcloud auth login`; an agent's sandbox runs as the local operator
identity from a key file through `GOOGLE_APPLICATION_CREDENTIALS` and
`CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and never runs `gcloud auth` or
changes global `gcloud` configuration.

## Identities

| Identity | Purpose | Holds |
| --- | --- | --- |
| `shallweswim-capture` | runs the job | `objectUser` on the archive bucket, read of the USGS key secret |
| `shallweswim-capture-invoker` | triggers the job from Cloud Scheduler | invoker on the job |
| `shallweswim-runtime` | runs the web service | `objectViewer` on the archive bucket |
| `shallweswim-ci` | Cloud Build deploys | `run.developer`, `serviceAccountUser` on the job identity |
| `shallweswim-terraform` | applies `monitoring/` | monitoring and state-bucket roles, see [`monitoring/README.md`](monitoring/README.md) |
| `shallweswim-local-operator` | a person's machine or an agent's sandbox | below |

**The invariant: the job is the only production writer to the archive.** The
web runtime identity holds `objectViewer` on the bucket and nothing more, and
`service.yaml` never sets `SHALLWESWIM_ARCHIVE_BUCKET`, so the multi-instance
web service can read published generations but cannot write archive objects
even accidentally. Keep it that way: enabling capture in the web service
would reintroduce concurrent writers from every serving instance.

The local operator is the identity operators and coding agents act as. It can
read everything: project state, Cloud Run, logs, metrics, alert policies, and
the archive bucket (`viewer`, `logging.viewer`, `iam.securityReviewer`,
`serviceusage.serviceUsageConsumer`, `objectViewer` on the bucket). It
changes the cloud only through named actions, each a command typed on
purpose and never a side effect of running the app or the tests:

| Action | Grant |
| --- | --- |
| deploy, by submitting a build (`build_and_deploy.sh`) | `cloudbuild.builds.editor` |
| execute the capture job by hand, with argument overrides | `run.developer` |
| manage monitoring by impersonating `shallweswim-terraform`: Terraform applies, and `gcloud ... --impersonate-service-account` for the few console-owned monitoring resources | `iam.serviceAccountTokenCreator` on that account |
| open a bucket write window for itself | `bucketPolicyEditor` on the archive bucket, a custom role of `storage.buckets.getIamPolicy` and `storage.buckets.setIamPolicy` |
| grant or revoke the application's own roles on the project, for the application's own identities | `resourcemanager.projectIamAdmin` conditioned with `modifiedGrantsByRole` to a fixed list of roles, below; it can never grant owner, editor, or anything outside the list |

It cannot write the archive by default, so no local run or test can touch
the archive by accident. For a backfill, a repair, or deleting bad objects, a
person opens a write window: a conditional grant that expires on its own, so
nothing has to be revoked afterwards, and the audit log records who opened
it and when.

```bash
gcloud storage buckets add-iam-policy-binding "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-local-operator@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/storage.objectUser \
  --condition="expression=request.time < timestamp('2026-01-01T00:00:00Z'),title=operator-write-window"
```

Set the timestamp a few hours ahead. The expired binding stays listed in the
policy until someone removes it, which is harmless; remove it with the
matching `remove-iam-policy-binding` and the same `--condition` when tidying.
Conditional bindings require uniform bucket-level access, which the bucket
below is created with.

The project-level grant is bounded the same way, by a condition rather than
by trust: the operator may add or remove a binding only for a role in the
list, so it can give the Terraform account a monitoring role or the job
identity a storage role, and nothing else. The list is the roles the
application's identities use:

```text
roles/monitoring.editor  roles/monitoring.viewer  roles/logging.configWriter
roles/logging.viewer  roles/storage.objectViewer  roles/storage.objectUser
roles/run.developer  roles/run.invoker  roles/iam.serviceAccountTokenCreator
```

Granting it is an owner action, once, with the condition from a file
because its expression contains commas:

```bash
cat > operator-grants.yaml <<'EOF'
title: operator-grants-app-roles
description: The local operator may grant or revoke only the application's own roles.
expression: api.getAttribute("iam.googleapis.com/modifiedGrantsByRole", []).hasOnly(["roles/monitoring.editor","roles/monitoring.viewer","roles/logging.configWriter","roles/logging.viewer","roles/storage.objectViewer","roles/storage.objectUser","roles/run.developer","roles/run.invoker","roles/iam.serviceAccountTokenCreator"])
EOF
gcloud projects add-iam-policy-binding "$CLOUDSDK_CORE_PROJECT" \
  --member="serviceAccount:shallweswim-local-operator@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/resourcemanager.projectIamAdmin \
  --condition-from-file=operator-grants.yaml
```

## The bucket

One private bucket holds the observation archive under `archive/` and the
published generations under `published/`. Choose a globally unique name, set
`SHALLWESWIM_ARCHIVE_BUCKET` to it, and create it once:

```bash
gcloud storage buckets create "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location="$CLOUDSDK_RUN_REGION" \
  --uniform-bucket-level-access \
  --public-access-prevention
```

Keep it separate from the Terraform state bucket. Never add a lifecycle rule
that deletes objects: observations under `archive/` are retained
indefinitely, and the job's own sweep is the only thing that removes
superseded generations under `published/`. To stop capture, pause the
schedule (below); archived data is unaffected.

## Identity setup

Create the two dedicated service accounts: `shallweswim-capture` runs the
job, `shallweswim-capture-invoker` only triggers it from Cloud Scheduler.

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
  --member="serviceAccount:shallweswim-capture@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/storage.objectUser
```

Grant the job identity read access to the USGS API key secret referenced by
`capture-job.yaml`:

```bash
gcloud secrets add-iam-policy-binding waterdata_usgs_gov_api_key \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --member="serviceAccount:shallweswim-capture@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

Give the web runtime identity read access to the bucket, and nothing more:

```bash
gcloud storage buckets add-iam-policy-binding \
  "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-runtime@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/storage.objectViewer
```

Let the Cloud Build identity deploy a job that runs as `shallweswim-capture`:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  "shallweswim-capture@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --member="serviceAccount:shallweswim-ci@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/iam.serviceAccountUser
```

The Cloud Build identity also needs `roles/run.developer` on the project so
`gcloud run jobs replace` can create and update the job; a service-scoped
deploy binding does not cover jobs. If the project policy contains conditional
bindings, gcloud requires `--condition=None` for an unconditional grant.

```bash
gcloud projects add-iam-policy-binding "$CLOUDSDK_CORE_PROJECT" \
  --member="serviceAccount:shallweswim-ci@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/run.developer \
  --condition=None
```

Grant the local operator identity read access to the bucket, and the custom
role that lets it open its own expiring write window (see
[Identities](#identities)):

```bash
gcloud storage buckets add-iam-policy-binding \
  "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-local-operator@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/storage.objectViewer

gcloud iam roles create bucketPolicyEditor --project="$CLOUDSDK_CORE_PROJECT" \
  --title="Bucket IAM policy editor" \
  --permissions=storage.buckets.getIamPolicy,storage.buckets.setIamPolicy \
  --stage=GA
gcloud storage buckets add-iam-policy-binding \
  "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --member="serviceAccount:shallweswim-local-operator@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role="projects/$CLOUDSDK_CORE_PROJECT/roles/bucketPolicyEditor"
```

Do not grant `shallweswim-capture` anything else, do not grant the web runtime
identity any role on the archive bucket beyond `roles/storage.objectViewer`, and
never give a local identity an unconditional write role on the bucket: a write
window always carries an expiry.

## Build and deploy

`build_and_deploy.sh` submits the repository to Cloud Build, which builds the
image, replaces the Cloud Run service, and then replaces the job from the
same image, so the job never runs a stale image. Run it from the repository
root: the whole repository is the Docker build context, and
`cloudbuild.yaml` reads the manifests as `infra/service.yaml` and
`infra/capture-job.yaml`.

```bash
./infra/build_and_deploy.sh
```

It refuses to run unless `SHALLWESWIM_ARCHIVE_BUCKET`, `CLOUDSDK_CORE_PROJECT`,
and `CLOUDSDK_RUN_REGION` are set, and each deploy step fails if the bucket or
region substitution reaches Cloud Build empty. The build runs as
`shallweswim-ci`, passed with `--service-account`; the project, region, and
bucket reach the manifests as substitutions, and the image tag is the build
id.

The manifests set the application's variables. The job manifest sets three:

- `SHALLWESWIM_ARCHIVE_BUCKET`: the bucket capture writes to and snapshots
  publish into.
- `SHALLWESWIM_SNAPSHOT_PUBLISH`: `"1"` makes every run publish a snapshot
  after its capture cycle.
- `SHALLWESWIM_ARCHIVE_READ_BUCKET`: the same bucket, so the historical feed
  restores past years from the archive instead of refetching them; required
  whenever publishing is enabled.

`service.yaml` sets none of them. It sets one bucket variable of its own,
`SHALLWESWIM_SNAPSHOT_READ_BUCKET`, the same bucket, which only reads the
generations the job publishes.

After a deploy, confirm the job description shows the image tag of the build
just deployed, the service account `shallweswim-capture`, both archive
variables holding the intended bucket, `SHALLWESWIM_SNAPSHOT_PUBLISH` at `1`,
a task timeout of 1200 seconds, and retries limited to one:

```bash
gcloud run jobs describe shallweswim-capture --region="$CLOUDSDK_RUN_REGION"
```

### Continuous build trigger

A GitHub-triggered Cloud Build on `main` runs the same `cloudbuild.yaml`, so
it must carry the bucket and region substitutions or its deploy steps fail
after the image is built. Set them once on the trigger (find the trigger id
with `gcloud builds triggers list --region=global`):

```bash
gcloud builds triggers update github TRIGGER_ID \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --region=global \
  --update-substitutions=_ARCHIVE_BUCKET="$SHALLWESWIM_ARCHIVE_BUCKET",_REGION="$CLOUDSDK_RUN_REGION"
```

The trigger runs when Cloud Scheduler calls it, and a disabled trigger still
runs when called that way: to stop continuous deployment, pause the scheduler
job that invokes the trigger as well as disabling the trigger. While it is
paused, every deploy is a manual `build_and_deploy.sh`.

## The job and its schedule

The job runs `python -m shallweswim.update` with `SHALLWESWIM_SNAPSHOT_PUBLISH=1`:
one bounded run fetches the feeds that are due, archives what they return,
draws the plots, publishes one generation, and sweeps superseded ones
([DATA_PIPELINE.md](../DATA_PIPELINE.md) "The job's cycle"). It starts no
web server. The current generation's manifest is the job's persisted
schedule, so running every ten minutes does not multiply provider traffic:
live temperature fetches every ten minutes, historical temperature every
three hours, tide and current predictions daily.

Bind the invoker identity to the job, then create the Cloud Scheduler job:

```bash
gcloud run jobs add-iam-policy-binding shallweswim-capture \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --region="$CLOUDSDK_RUN_REGION" \
  --member="serviceAccount:shallweswim-capture-invoker@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --role=roles/run.invoker

gcloud scheduler jobs create http shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location="$CLOUDSDK_RUN_REGION" \
  --schedule="*/10 * * * *" \
  --time-zone=Etc/UTC \
  --uri="https://$CLOUDSDK_RUN_REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$CLOUDSDK_CORE_PROJECT/jobs/shallweswim-capture:run" \
  --http-method=POST \
  --oauth-service-account-email="shallweswim-capture-invoker@$CLOUDSDK_CORE_PROJECT.iam.gserviceaccount.com" \
  --attempt-deadline=180s
```

The `:run` endpoint returns as soon as the execution is created, so the attempt
deadline covers only that API call and is unrelated to the job's 1200-second
task timeout; do not raise it to cover the run. (The equivalent v2 endpoint,
`https://run.googleapis.com/v2/projects/PROJECT/locations/REGION/jobs/shallweswim-capture:run`,
also works; the v1 namespaced form above is the one this deployment uses.)
Creating a scheduler job with `--oauth-service-account-email` requires the
operator to hold `roles/iam.serviceAccountUser` on
`shallweswim-capture-invoker`.

Scheduler policy is what prevents overlapping executions; correctness does not
depend on it, because the archive merge path is a conditional read-merge-write
and tolerates an overlap.

## Manual runs and repairs

Run the scheduled cycle on demand and wait for it to finish:

```bash
gcloud run jobs execute shallweswim-capture --region="$CLOUDSDK_RUN_REGION" --wait
```

`gcloud run jobs execute` accepts `--args` as a per-execution override: the
comma-separated list replaces the container `args` for that execution only,
leaving the job definition, and therefore every scheduled run, unchanged. The
`command` stays `python`, so the full argument vector must be given. A
per-execution override is used deliberately in preference to
`gcloud run jobs update --args=...` followed by a revert: an update mutates
the deployed job, and an interrupted session would leave the schedule running
the expensive variant.

The full-history repair ignores the published schedule so every feed fetches,
and the historical feed serves its whole configured range from the archive:

```bash
gcloud run jobs execute shallweswim-capture --region="$CLOUDSDK_RUN_REGION" --wait \
  --args="-m,shallweswim.update,--full-history"
```

A deep-history backfill (`--backfill-from`, DATA_PIPELINE.md "Deep history")
can run either way. Walking every source takes about ninety minutes, longer
than the task timeout, so as executions it is one location per execution
with `--location`; a single CO-OPS station's decades take about fifteen
minutes and fit. From an operator's machine it runs inside a write window
and needs no timeout. Never raise the timeout in `capture-job.yaml` for it.

A `partial` run, some feeds published and some stations unavailable, exits
zero by design, so a successful execution does not by itself mean every feed
was captured; check the run summary event (MONITORING.md "Log queries").

## Validation

After the first scheduled runs:

1. `gcloud run jobs executions list --job=shallweswim-capture --region="$CLOUDSDK_RUN_REGION"`
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

5. A generation was published: one `snapshot.publish` event per run with a
   bounded outcome, objects and a manifest under the published prefix, and a
   current pointer whose `manifest_key` exists under `published/manifests/`.
   The run summary message also names the publish outcome; a run whose
   publish failed still exits zero, so check the event rather than the
   execution state.

   ```bash
   gcloud logging read \
     'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.component="snapshot" AND jsonPayload.operation="publish"' \
     --limit=10 \
     --format='table(timestamp,severity,jsonPayload.outcome,jsonPayload.duration_ms,jsonPayload.record_count,jsonPayload.generation_id)'

   gcloud storage ls --recursive "gs://$SHALLWESWIM_ARCHIVE_BUCKET/published/"
   gcloud storage cat "gs://$SHALLWESWIM_ARCHIVE_BUCKET/published/current.json"
   ```

6. Every configured feed reported its freshness: one `snapshot.freshness`
   event per location and feed per run, with `outcome` `success` for a feed
   fetched this run, `held` for one that was not due, `carried` for one whose
   last published entry was carried forward after a failure, and `absent` for
   one with nothing to serve. `held` is the ordinary state of a feed whose
   interval is longer than the cadence and is logged at INFO. A growing
   `age_seconds` on a `carried` feed is a feed stuck on a failing source, not
   a publication problem; the publish event stays `success`.

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

8. The web servers load each generation: one `snapshot.load` event per
   instance per generation, within a minute of publication, and the
   dashboard's capture, publish, feed age, and load charts show the run
   (MONITORING.md).

## Pausing

Pausing the schedule stops new runs and leaves every object in place; the web
servers keep serving the last generation.

```bash
gcloud scheduler jobs pause shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" --location="$CLOUDSDK_RUN_REGION"

gcloud scheduler jobs resume shallweswim-capture-hourly \
  --project="$CLOUDSDK_CORE_PROJECT" --location="$CLOUDSDK_RUN_REGION"
```

Never delete the bucket or its objects to stop capture, and do not add a
lifecycle rule to it; observations are retained indefinitely.

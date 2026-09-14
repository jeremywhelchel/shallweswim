# GCP Monitoring Infrastructure

This directory owns Shall We Swim's user-defined log-based metrics, application
operations dashboard, and explicitly marked shadow alert policies. It
deliberately does not own Cloud Run, IAM, notification channels, existing uptime
checks, or pre-Terraform alert policies.
Resources with `[Terraform]` in their display name or `Managed by Terraform` in
their description must not be edited in the GCP console.

The metrics consume the bounded structured events documented in
`OBSERVABILITY_DESIGN.md` and `PERSISTENT_DATA_PIPELINE_DESIGN.md`. Terraform
defines the extraction rules; it does not read or process logs itself. Cloud
Logging creates metric samples from new matching entries after the metrics are
created. Existing log entries are not backfilled.

The metric filters match structured events from both the Cloud Run service and
the `shallweswim-capture` Cloud Run Job, so the capture job's feed-update,
`archive.merge`, `snapshot.publish`, `snapshot.freshness`, and run summary
events feed the same metrics and dashboard as the web service, and the web
service's own `snapshot.load` events feed a matching pair of metrics. The
feed, plot, and snapshot load alert policies stay scoped to the service
resource type; the six capture job policies below are scoped to
`resource.type = "cloud_run_job"` instead.

## Observation archive setup

The observation archive is written by the scheduled Cloud Run capture job, not
by the web service. `SHALLWESWIM_ARCHIVE_BUCKET` (a bucket name without
`gs://`) names the bucket the job writes to; `service.yaml` never sets it. The
managed operations dashboard shows archive merges per hour by outcome, capture
runs per hour by outcome, snapshot publishes per hour by outcome, maximum
published feed age by feed per hour, new and revised observations per hour by
source, and merge duration p95 by source per hour. The capture job runs hourly, so these tiles align on the hour rather than
on five minutes. Failed merges are one colour of the hourly outcome stack
instead of a separate tile; the archive merge failure shadow policy covers that
signal.

The job also publishes a serving snapshot each run and emits one
`snapshot.publish` event per attempt with `outcome` `success`, `unchanged`, or
`failed`. `shallweswim_snapshot_publishes` counts those events by outcome and
`shallweswim_snapshot_publish_duration_ms` records their duration; the
"Snapshot publishes per hour by outcome" tile shows the counter. The event's
`record_count` is the number of objects written, which is exact in the event
and not charted. No alert policy watches publication itself.

Each publish also emits one `snapshot.freshness` event per location and feed,
carrying the age of the frame that generation serves and an `outcome` of
`success` (fetched this run), `carried` (the previous entry carried forward),
or `absent` (configured with nothing to serve, and therefore no age).
`shallweswim_snapshot_feed_age_seconds` is a distribution of those ages by
location, feed, and outcome, charted as "Snapshot feed age max by feed per
hour". It is the signal that a feed is stuck on carried-forward data: a
publish stays `success` while the age grows run after run.

The web service itself emits one `snapshot.load` event per bundle load or
refresh that does work, with `outcome` `success` or `failed`, `duration_ms`,
`record_count` as the number of objects read, `generation_id`, and
`age_seconds` as the age of the loaded generation's publication time at load,
the lag between the job publishing and this instance picking it up. Unlike
the job-side snapshot metrics, these events come from
`resource.type = "cloud_run_revision"` alone. `shallweswim_snapshot_loads`
counts those events by outcome for the "Snapshot loads per hour by outcome"
tile, `shallweswim_snapshot_load_duration_ms` records their duration, and
`shallweswim_snapshot_load_lag_seconds` is a distribution of `age_seconds`
charted as "Snapshot load lag p99 per hour"; the same lag distribution backs
the `snapshot_load_lag` shadow policy, the signal that pages for sustained
refresh failure once the web depends on the bundle.

Merges are value-aware, so a repeated fetch of unchanged readings reports
`outcome=unchanged` and writes nothing. The `new_rows` and `revised_rows`
metrics are distributions of the per-merge row counts, and the
`timeSeriesFilter` widget cannot sum a distribution. Their two tiles therefore
use the Monitoring Query Language, which can: `align delta(1h)` followed by
`group_by [source: metric.source], [rows: sum(sum_from(val()))]`. Those observation
counts are histogram estimates derived from the distribution's bucket counts
rather than exact totals, which is why both tiles are titled `(estimated)`;
exact per-merge counts stay available in the merge events' `new_count` and
`revised_count` fields.

Bucket creation is a one-time operator task, outside this Terraform module.
Load the local-operator credential and project through repo-local environment
variables (`GOOGLE_APPLICATION_CREDENTIALS`,
`CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, `CLOUDSDK_CORE_PROJECT`). Do not use
`gcloud auth` or modify global configuration. Choose a globally unique archive
bucket name and set `SHALLWESWIM_ARCHIVE_BUCKET` in the local environment first.

```bash
gcloud storage buckets create "gs://$SHALLWESWIM_ARCHIVE_BUCKET" \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location=us-east4 \
  --uniform-bucket-level-access \
  --public-access-prevention
```

IAM for this bucket belongs to the job that uses it. Creating the
`shallweswim-capture` and `shallweswim-capture-invoker` identities, binding
`roles/storage.objectUser` on the bucket to `shallweswim-capture` alone,
deploying the job, and scheduling it are documented in
[`../capture-job/README.md`](../capture-job/README.md). The web runtime identity
must not be bound to this bucket: the capture job is the only production writer,
and granting the multi-instance web service write access would reintroduce
concurrent writers.

Keep the archive separate from Terraform state. Do not add a lifecycle rule that
deletes live observation objects; normalized observations are retained
indefinitely. To stop capture, pause the scheduler job as described in the
capture job runbook; archived data is unaffected.

## State bootstrap

Use a dedicated private, uniformly accessed, versioned GCS bucket for state.
The bucket is a one-time prerequisite because Terraform cannot store its own
initial state in a bucket that does not yet exist. Do not reuse application data
or Cloud Build buckets.

Example one-time setup, after choosing a globally unique bucket name:

```bash
gcloud storage buckets create "gs://$SHALLWESWIM_TERRAFORM_STATE_BUCKET" \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location=us-east4 \
  --uniform-bucket-level-access \
  --public-access-prevention

gcloud storage buckets update \
  "gs://$SHALLWESWIM_TERRAFORM_STATE_BUCKET" --versioning
```

The state may contain project metadata and must not be committed. Notification
channels are intentionally outside this module, so their addresses do not enter
this state. Record the chosen bucket in the deployment's uncommitted `.envrc`
as `SHALLWESWIM_TERRAFORM_STATE_BUCKET`; `.env.example` preserves the expected
variable without coupling independent installations to this deployment's name.

Use a dedicated Terraform operator identity rather than a web runtime or build
identity. Grant it only:

- `roles/storage.objectAdmin` on the state bucket
- `roles/logging.configWriter` on the project
- `roles/monitoring.dashboardEditor` on the project
- `roles/monitoring.alertPolicyEditor` on the project
- `roles/serviceusage.serviceUsageConsumer` on the project

Store its credential outside version control and set
`SHALLWESWIM_TERRAFORM_CREDENTIALS` to that file in `.envrc`. The Cloud Run
runtime identity must not have state-bucket access.

## Validate and plan

Terraform uses the dedicated credential explicitly for each command; do not
mutate global `gcloud` authentication or fall back to the application runtime
credential.

```bash
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  terraform -chdir=infra/monitoring fmt -check
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  terraform -chdir=infra/monitoring init \
  -backend-config="bucket=$SHALLWESWIM_TERRAFORM_STATE_BUCKET" \
  -backend-config="prefix=monitoring"
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  terraform -chdir=infra/monitoring validate
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  TF_VAR_project_id="$CLOUDSDK_CORE_PROJECT" \
  terraform -chdir=infra/monitoring plan -out=monitoring.tfplan
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  terraform -chdir=infra/monitoring show monitoring.tfplan
```

For syntax/provider validation without state access, initialize with
`terraform -chdir=infra/monitoring init -backend=false` and then run
`terraform -chdir=infra/monitoring validate`. The native Terraform test uses a
mock Google provider to execute a plan without credentials, state, or GCP:

```bash
terraform -chdir=infra/monitoring test
```

This test pins service and job scoping, bounded label counts, numeric
extractors, the capture job policies' resource scope and heartbeat window, the
snapshot load lag policy's resource scope and threshold, the dashboard
ownership marker, the per-feed snapshot freshness thresholds, and the
dashboard's fourteen tiles, including the two MQL `sum_from` data sets. It
cannot emulate Cloud Logging ingestion.

Review the plan before every apply. The module now owns seventeen log-based
metrics, one dashboard, and eleven `[Terraform][Shadow]` alert policies with no
notification channels. It does not change pre-Terraform monitoring.

An apply that creates a log-based metric and, in the same run, alert policies
that reference it can fail on the policies with "Cannot find metric(s)": the
metric takes up to ten minutes to become visible to Cloud Monitoring after
creation. The metric and everything else still apply. Wait ten minutes, plan
again, and apply the remaining policies; nothing needs changing.

## Capture job shadow policies

Six shadow policies watch the capture job, the first two following the dead-man
switch design in `OBSERVABILITY_DESIGN.md`:

- **Capture job heartbeat**: a metric-absence condition on
  `shallweswim_updater_runs` restricted to `outcome` `success` or `partial`,
  firing after 3 hours without either, which is three missed hourly runs. A
  `partial` run still proves the job executed. Absence conditions evaluate only
  a metric that has produced data, so this policy is trustworthy only once the
  run counter has been populated by real runs.
- **Archive merge failures**: a threshold condition on
  `shallweswim_archive_merges` with `outcome="failed"` over a one-hour
  alignment. A conflicting equally recent claim recovers on the next
  overlapping fetch, so this is a warn candidate.
- **Snapshot feed freshness**, one policy per feed type: a threshold condition
  on `shallweswim_snapshot_feed_age_seconds` restricted to that feed, over a
  one-hour alignment. The thresholds come from the feed health rule, the
  expiration interval plus 15 minutes: 1500s for `live_temps`, 11700s for
  `historic_temps`, and 87300s for `tides` and `currents`. The metric is a
  distribution, which has no maximum aligner, and the hourly job contributes
  one sample per feed and location per hour, so the hour's 99th percentile is
  that hour's maximum age. These thresholds are first guesses to be tuned on
  the baseline.

A seventh shadow policy, **Snapshot load lag**, watches the web service
instead of the job: a threshold condition on
`shallweswim_snapshot_load_lag_seconds` over a one-hour alignment, scoped to
`resource.type = "cloud_run_revision"`. The threshold is 7200 seconds, one
hourly capture job cadence plus the check interval, with margin; it tightens
once the capture job moves to a ten-minute cadence. Like the snapshot
freshness thresholds, this is a first guess to be tuned on the baseline.

Promotion is the same rule as the other shadow policies: review the policy's
production behavior over a real baseline period, choose notification channels
explicitly, and remove the `[Shadow]` marker in a separately reviewed change.

## Apply and integration-test

```bash
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  TF_VAR_project_id="$CLOUDSDK_CORE_PROJECT" \
  terraform -chdir=infra/monitoring apply monitoring.tfplan
```

There is no faithful local emulator for Cloud Logging log-based metrics or
Cloud Monitoring dashboards. Local validation checks HCL, provider schemas, and
the proposed API operations. The GCP integration test is a controlled apply:

1. Confirm the apply creates only the expected resources listed in the plan.
2. Generate or wait for new feed, plot, merge, snapshot publish, snapshot
   freshness, snapshot load, and capture run events. Metrics do not backfill.
3. Verify the seventeen metrics appear with bounded labels and the dashboard
   charts populate after several minutes.
4. Compare metric counts with a Cloud Logging query over the same interval.
5. Confirm shadow policies have no notification channels before applying them.

Shadow policies are enabled for evaluation so incidents appear in Cloud
Monitoring, but they cannot page. Promote a policy only after reviewing its
production behavior, choosing notification channels explicitly, and removing
the `[Shadow]` marker in a separately reviewed change.

Use `terraform plan` afterward to verify an empty plan and detect drift. Remove
test resources only with an explicitly reviewed `terraform destroy` plan.

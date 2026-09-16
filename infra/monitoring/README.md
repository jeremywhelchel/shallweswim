# GCP Monitoring Infrastructure

This directory applies Shall We Swim's user-defined log-based metrics, the
operations dashboard, and the Terraform-managed alert policies. What they
are, what they measure, and which of them notify anyone is in
[MONITORING.md](../MONITORING.md); this README is how to apply them. The
module deliberately does not own Cloud Run, IAM, notification channels, the
uptime check, or the alert policies that predate Terraform. Resources with
`[Terraform]` in their display name or `Managed by Terraform` in their
description must not be edited in the GCP console.

Terraform defines extraction rules; it does not read logs. Cloud Logging
creates metric samples only from entries written after a metric exists, so
nothing is backfilled.

The observation archive bucket, its identities, and the job that writes it
are in [`../README.md`](../README.md), the deployment guide; this module
never touches that bucket. Keep Terraform state in its own bucket, below.

## State bootstrap

Use a dedicated private, uniformly accessed, versioned GCS bucket for state.
The bucket is a one-time prerequisite because Terraform cannot store its own
initial state in a bucket that does not yet exist. Do not reuse application data
or Cloud Build buckets.

Example one-time setup, after choosing a globally unique bucket name:

```bash
gcloud storage buckets create "gs://$SHALLWESWIM_TERRAFORM_STATE_BUCKET" \
  --project="$CLOUDSDK_CORE_PROJECT" \
  --location="$CLOUDSDK_RUN_REGION" \
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
  terraform -chdir=infra/monitoring plan -out=monitoring.tfplan \
  -var="notification_channel_ids=$SHALLWESWIM_ALERT_NOTIFICATION_CHANNELS"
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
dashboard's fifteen tiles, including the two MQL `sum_from` data sets. It
cannot emulate Cloud Logging ingestion.

Review the plan before every apply. The module owns the eighteen log-based
metrics, the dashboard, and the twelve alert policies listed in
MONITORING.md; it does not change pre-Terraform monitoring.

An apply that creates a log-based metric and, in the same run, alert policies
that reference it can fail on the policies with "Cannot find metric(s)": the
metric takes up to ten minutes to become visible to Cloud Monitoring after
creation. The metric and everything else still apply. Wait ten minutes, plan
again, and apply the remaining policies; nothing needs changing.

A log-based metric's label descriptions are immutable: changing one replaces
the metric and erases its history. Change a description only with that cost
accepted.

## Promoting an alert policy

A policy that has not been promoted is enabled with no notification channels,
so it opens incidents in Cloud Monitoring and pages nobody; its display name
carries `[Shadow]` and its `mode=shadow` label. To promote one: review its
incidents over a real baseline period, then in a reviewed change set its
`notification_channels` to `var.notification_channel_ids`, drop the marker,
and switch its labels to `local.paging_alert_labels`.

Notification channels are created in the console and carry the addresses;
this module references them only by resource name, passed on the plan
command line from `SHALLWESWIM_ALERT_NOTIFICATION_CHANNELS` in `.env` and
never committed. The value is a JSON list of channel resource names on one
line, and every plan and apply adds
`-var="notification_channel_ids=$SHALLWESWIM_ALERT_NOTIFICATION_CHANNELS"`.
List the channels with `gcloud beta monitoring channels list`. With the
variable empty, a plan shows every promoted policy losing its channels: do
not apply such a plan.

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
2. Generate or wait for new events of each kind; metrics do not backfill.
3. Verify the eighteen metrics appear with bounded labels and the dashboard
   charts populate after several minutes.
4. Compare metric counts with a Cloud Logging query over the same interval.
5. Confirm the policies have no notification channels before applying them.

Use `terraform plan` afterward to verify an empty plan and detect drift. Remove
test resources only with an explicitly reviewed `terraform destroy` plan.

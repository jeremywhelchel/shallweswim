# GCP Monitoring Infrastructure

This directory owns Shall We Swim's user-defined log-based metrics, application
operations dashboard, and explicitly marked shadow alert policies. It
deliberately does not own Cloud Run, IAM, notification channels, existing uptime
checks, or pre-Terraform alert policies.
Resources with `[Terraform]` in their display name or `Managed by Terraform` in
their description must not be edited in the GCP console.

The metrics consume the bounded structured events documented in
`OBSERVABILITY_DESIGN.md`. Terraform defines the extraction rules; it does not
read or process logs itself. Cloud Logging creates metric samples from new
matching entries after the metrics are created. Existing log entries are not
backfilled.

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

This test pins service scoping, bounded label counts, numeric extractors, and
the dashboard ownership marker. It cannot emulate Cloud Logging ingestion.

Review the plan before every apply. The initial baseline creates five log-based
metrics and one dashboard. The next slice adds only `[Terraform][Shadow]` alert
policies with no notification channels; it does not change existing monitoring.

## Apply and integration-test

```bash
GOOGLE_APPLICATION_CREDENTIALS="$SHALLWESWIM_TERRAFORM_CREDENTIALS" \
  TF_VAR_project_id="$CLOUDSDK_CORE_PROJECT" \
  terraform -chdir=infra/monitoring apply monitoring.tfplan
```

There is no faithful local emulator for Cloud Logging log-based metrics or
Cloud Monitoring dashboards. Local validation checks HCL, provider schemas, and
the proposed API operations. The GCP integration test is a controlled apply:

1. Confirm the apply creates only the six expected resources.
2. Generate or wait for new feed/plot completion events. Metrics do not backfill.
3. Verify the five metrics appear with bounded labels and the dashboard charts
   populate after several minutes.
4. Compare metric counts with a Cloud Logging query over the same interval.
5. Confirm shadow policies have no notification channels before applying them.

Shadow policies are enabled for evaluation so incidents appear in Cloud
Monitoring, but they cannot page. Promote a policy only after reviewing its
production behavior, choosing notification channels explicitly, and removing
the `[Shadow]` marker in a separately reviewed change.

Use `terraform plan` afterward to verify an empty plan and detect drift. Remove
test resources only with an explicitly reviewed `terraform destroy` plan.

# Monitoring

The metrics, alert policies, dashboard, uptime check, and log queries for the
reference GCP deployment. DATA_PIPELINE.md owns what the pipeline's events
mean; this document owns what is built on them. infra/monitoring/README.md
says how to apply the Terraform that creates them.

## Shape

Application code writes structured events to stdout. Cloud Logging turns the
events into log-based metrics by extraction rules that Terraform owns. Cloud
Monitoring evaluates alert policies over those metrics and draws the
dashboard. An external uptime check probes the site. Nothing in the
application calls a monitoring API, carries a provider SDK, or exports
telemetry itself.

Why: the job is a bounded process that exits within a minute, and the web
servers scale to zero. Neither can be scraped, and an in-process exporter
would have to flush before exit or lose its batch. Events on stdout survive
both, cost the application nothing, and keep the application portable: the
same events feed a different backend's extraction rules.

Four questions are kept separate, and each has its own signal:

| Question | Signal |
| --- | --- |
| Can users reach the site? | the uptime check on `/api/healthy` |
| Is the job running? | the capture run heartbeat |
| Is a coherent generation being published and loaded? | publish, freshness, and load events |
| Is each feed current enough for its purpose? | per-feed age thresholds |

## Telemetry contract

Hosted processes set `SHALLWESWIM_LOG_FORMAT=json` (`service.yaml`,
`capture-job.yaml`) so `logging_utils.py` emits one JSON object per line
with `severity`, `message`, `logger`, `source`, and only the approved fields
below. Local runs default to the console format. Under the JSON format the
web server disables Uvicorn access logs, because Cloud Run writes its own
request log for every request under `httpRequest`.

The approved structured fields, `STRUCTURED_FIELDS` in `logging_utils.py`,
are the only ones a log call may pass through `extra`:

```text
component operation location feed provider outcome run_id generation_id
duration_ms record_count age_seconds attempt_count incoming_count new_count
overlap_count revised_count source_identity observed_at
```

Every label is bounded: location codes, feed names, provider families, source
identities from the configured citation keys, and short outcome sets. Station
identifiers from requests, URLs, exception text, timestamps, and generation or
run ids never become labels; they stay in the message or in fields no metric
extracts.

Severity keeps its plain meaning (ARCHITECTURE.md "Logging"): INFO for
expected work, WARNING for handled degradation including a station that has
no data, ERROR for a defect or an exhausted critical operation. An ERROR is
visible but does not by itself page; the policies below decide that.

### Events

One completion event per unit of work, with `component` and `operation`
naming it and `outcome` from a short set. DATA_PIPELINE.md defines what each
pipeline event means; this table is the map from event to metric.

| Event | Emitted by | Outcomes | Fields beyond the pair | Metrics |
| --- | --- | --- | --- | --- |
| `updater` / `feed_update` | every feed update attempt (`core/feeds.py`), in the job and the local entry point | `success`, `unavailable`, `failed` | `location`, `feed`, `provider`, `duration_ms`, `record_count` on success | `feed_updates`, `feed_update_duration_ms`, `feed_records` |
| `plot` / `plot_generation` | each plot harvested from the process pool (`core/manager.py`) | `success`, `failed` | `location`, `feed`, `duration_ms` as submit-to-harvest latency, which includes queueing and CPU starvation | `plot_generations`, `plot_availability_latency_ms` |
| `archive` / `merge` | each partition merge (`archive/merge.py`) | `success`, `unchanged`, `failed` | `source_identity`, `duration_ms`, `attempt_count`, `incoming_count`, `new_count`, `overlap_count`, `revised_count`, `record_count` | `archive_merges`, `archive_merge_duration_ms`, `archive_merge_new_rows`, `archive_merge_revised_rows` |
| `archive` / `hydrate` | the historical feed after reading past years from the archive (`core/feeds.py`) | `success`, `failed` | `location`, `feed`, `record_count` | none |
| `updater` / `run` | the job's run summary (`update.py`) | `success`, `partial`, `failed` | `duration_ms`, `record_count`, `new_count`, `revised_count`, `run_id` | `updater_runs`, `updater_run_duration_ms` |
| `updater` / `backfill` | the backfill walk's summary (`update.py`) | `success`, `partial`, `failed` | as `run` | none, deliberately: a hand-run walk must not look like a scheduled run |
| `snapshot` / `publish` | each publish attempt (`snapshot/publish.py`) | `success`, `unchanged`, `skipped`, `failed` | `generation_id`, `duration_ms`, `record_count` as objects written, `run_id` | `snapshot_publishes`, `snapshot_publish_duration_ms` |
| `snapshot` / `freshness` | one per location and configured feed per publish | `success`, `held`, `carried`, `absent` | `location`, `feed`, `age_seconds` (none when absent) | `snapshot_feed_age_seconds` |
| `snapshot` / `gc` | the sweep after each publish (`snapshot/gc.py`) | `success`, `failed` | `duration_ms`, `record_count` as objects deleted | `snapshot_gcs` |
| `snapshot` / `load` | each web server load that does work (`snapshot/refresh.py`) | `success`, `failed` | `generation_id`, `duration_ms`, `record_count` as objects read, `age_seconds` as publication-to-load lag | `snapshot_loads`, `snapshot_load_duration_ms`, `snapshot_load_lag_seconds` |

`/api/status` reports per-feed health for a person reading it and is not a
metric source: a scraped status would describe one instance's loaded copy,
while the events describe the job's and every instance's work.

## Metrics

Eighteen log-based metrics, all named `shallweswim_<name>` under
`logging.googleapis.com/user`, all `DELTA`, defined in
`infra/monitoring/logging_metrics.tf`. Every filter first restricts to the
Cloud Run service `shallweswim` or the Cloud Run job `shallweswim-capture`,
then to the event's `component` and `operation`. Distributions use twenty
doubling buckets from one (durations, ages) or twelve quadrupling buckets
from one (row counts).

| Metric | Event | Kind, unit | Value | Labels |
| --- | --- | --- | --- | --- |
| `feed_updates` | feed_update | counter | one per event | location, feed, provider, outcome |
| `feed_update_duration_ms` | feed_update | distribution, ms | `duration_ms` | location, feed, provider, outcome |
| `feed_records` | feed_update with outcome success | distribution, records | `record_count` | location, feed, provider |
| `plot_generations` | plot_generation | counter | one per event | location, feed, outcome |
| `plot_availability_latency_ms` | plot_generation | distribution, ms | `duration_ms` | location, feed, outcome |
| `archive_merges` | merge | counter | one per event | source, outcome |
| `archive_merge_duration_ms` | merge | distribution, ms | `duration_ms` | source, outcome |
| `archive_merge_new_rows` | merge | distribution, records | `new_count` | source |
| `archive_merge_revised_rows` | merge | distribution, records | `revised_count` | source |
| `updater_runs` | run | counter | one per event | outcome |
| `updater_run_duration_ms` | run | distribution, ms | `duration_ms` | outcome |
| `snapshot_publishes` | publish | counter | one per event | outcome |
| `snapshot_publish_duration_ms` | publish | distribution, ms | `duration_ms` | outcome |
| `snapshot_feed_age_seconds` | freshness | distribution, s | `age_seconds` | location, feed, outcome |
| `snapshot_gcs` | gc | counter | one per event | outcome |
| `snapshot_loads` | load | counter | one per event | outcome |
| `snapshot_load_duration_ms` | load | distribution, ms | `duration_ms` | outcome |
| `snapshot_load_lag_seconds` | load | distribution, s | `age_seconds` | outcome |

Two facts about log-based metrics shape everything built on them. Samples
exist only for log entries written after the metric was created; nothing is
backfilled. And a label's description is immutable: changing it replaces the
metric and erases its history, which is why `snapshot_feed_age_seconds`
still describes its outcome label as "success or carried" although the
outcome set gained `held`.

## Alert policies

Fifteen policies concern the application. Twelve are Terraform's, in
`infra/monitoring/alert_policies.tf`, enabled, and attached to no
notification channel: they open and close incidents in Cloud Monitoring and
page nobody. Their display names carry `[Terraform][Shadow]` and their
`mode=shadow` label, and they stay that way until each is promoted. Three
predate Terraform and notify the project's two email channels.

Why the twelve notify nobody: each threshold is a first guess. A policy is
promoted only after its incidents have been reviewed against a real baseline,
by choosing its channels and removing the marker in a reviewed change. The
capture job heartbeat is the first to promote, because the web servers serve
only what the job publishes, so it is the pipeline's dead-man switch.

### Terraform policies

| Policy | Resource | Condition | Severity |
| --- | --- | --- | --- |
| Capture job heartbeat | job | no `updater_runs` sample with outcome `success` or `partial` for 30 minutes, three missed runs; a partial run still proves the job ran | CRITICAL |
| Archive merge failures | job | any `archive_merges` with outcome `failed` in a one-hour window, per source | WARNING |
| Snapshot live_temps freshness | job | `snapshot_feed_age_seconds` p99 over an hour above 1500 s, per location | WARNING |
| Snapshot historic_temps freshness | job | the same above 11700 s | WARNING |
| Snapshot tides freshness | job | the same above 87300 s | WARNING |
| Snapshot currents freshness | job | the same above 87300 s | WARNING |
| Snapshot load lag | service | `snapshot_load_lag_seconds` p99 over an hour above 1800 s, three cadences; catches an instance whose refresh path is stuck, not a job that stopped publishing | WARNING |
| Snapshot load failures | service | more than two `snapshot_loads` with outcome `failed` in fifteen minutes; one failure retries a check interval later, repeated ones mean the instance cannot read the store | WARNING |
| Repeated feed failures | job | more than two `feed_updates` with outcome `failed` in ten minutes, per location and feed; `unavailable` is excluded | ERROR |
| Plot generation failure | job | any `plot_generations` with outcome `failed` in five minutes | ERROR |
| Live feed update latency | job | `feed_update_duration_ms` p95 for `live_temps` above 45 s for ten minutes | WARNING |
| Live plot availability latency | job | `plot_availability_latency_ms` p95 for `live_temps` above 45 s for ten minutes | WARNING |

The freshness thresholds are each feed's interval plus fifteen minutes, the
same rule `/api/status` applies. The job publishes every ten minutes, so an
hour holds six samples per feed and location and the hour's 99th percentile
is in effect its maximum. The heartbeat is an absence condition, and an
absence condition evaluates only a metric that has produced data, so its
silence means nothing until `updater_runs` has samples.

Only the two snapshot load policies watch the service resource, because
loading is the web servers' work. Everything else is the job's: it is the
only process that fetches feeds, draws scheduled plots, and writes the
archive, so the feed, plot, archive, and run policies watch the job resource.

### Older policies

| Policy | Condition | Notifies |
| --- | --- | --- |
| Homepage uptime failure | the uptime check below fails for five minutes, with missing data counted as failure | both email channels |
| 5xx error on shallweswim | any request in a one-minute window answered with a `5xx` other than `503`, from Cloud Run's request count | both email channels |
| Error Log | any log entry with `severity=ERROR` whose request status is not `503`, at most one notification an hour | one email channel |

These are the paging surface today. The last two page on a single error,
which is why an unexpected exception can mean an email while the site stays
healthy. They stay until the Terraform policies have been promoted and have
shown equal or better coverage.

The project also holds two YouTube livestream uptime checks and their
policies. They are unrelated to the application and stay separate.

## Dashboard

One dashboard, "Shall We Swim Operations [Terraform]", fifteen tiles from
`infra/monitoring/dashboard.tf`:

| Tiles | What they show |
| --- | --- |
| Feed updates per 5 minutes by outcome; feed update duration p95 by feed; published feed record count p50 by feed | the job's feed events |
| Plot completions per 5 minutes by outcome; plot availability latency p95 by feed | the job's plot events |
| Capture runs per hour by outcome; snapshot publishes per hour by outcome; snapshot collections per hour by outcome | the job's run, publish, and sweep counters |
| Archive merges per hour by outcome; archive merge duration p95 by source per hour | the merge counter and duration |
| New observations per hour by source (estimated); revised observations per hour by source (estimated) | the row-count distributions, summed with the Monitoring Query Language's `sum_from`, because the plain widget cannot sum a distribution; the totals are histogram estimates, and the exact counts are in the merge events |
| Snapshot feed age max by feed per hour | the freshness distribution, the signal that a feed is stuck on carried data |
| Snapshot loads per hour by outcome; snapshot load lag p99 per hour | the web servers' load events |

Request volume, latency, response classes, and instance counts are Cloud
Run's own metrics and are read in its console, not on this dashboard.

## Uptime check

"shallweswim API Healthy Uptime Check" requests
`http://shallweswim.today/api/healthy?uptime` every 60 seconds from Google's
static-IP checkers with a 30-second timeout, sends `X-HealthCheck: uptime`,
and accepts any `2xx`. The request is plain HTTP, so each probe appears twice
in the request log: the redirect and the HTTPS request that follows it. Its
user agent is `GoogleStackdriverMonitoring-UptimeChecks`, which the queries
below filter on. The check and its policy are managed in the console, not by
Terraform.

`/api/healthy` (alias `/api/health`) answers 200 while the instance has a
loaded generation in which any location has data, and 503 otherwise, so one
station outage never fails the check and an instance whose startup load
failed fails it until an elected request loads a generation.

## Log queries

Use `gcloud` with the credentials from `.env` (`CLOUDSDK_CORE_PROJECT`,
`GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`),
never `gcloud auth`.

Cloud Run request logs keep status, URL, latency, and user agent under
`httpRequest`; application events keep their message and structured fields
under `jsonPayload`. `textPayload` is usually empty for both. Cloud Run
reports many `4xx` responses at WARNING, so `severity>=WARNING` alone is noisy
and mostly bot scans for paths such as `/wp`, `/wordpress`, and `/api/health`.
After a deploy, filter to the latest ready revision so older revisions do not
hide current health.

Service status and the serving revision:

```bash
gcloud run services describe shallweswim \
  --region=us-east4 \
  --format='yaml(status.url,status.conditions,status.traffic,status.latestReadyRevisionName,status.latestCreatedRevisionName)'
```

Current revision `5xx` request errors:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND resource.labels.revision_name="REVISION_NAME" AND httpRequest.status>=500' \
  --limit=50 \
  --format='table(timestamp,severity,httpRequest.status,httpRequest.requestMethod,httpRequest.requestUrl,httpRequest.latency,httpRequest.userAgent)'
```

Current revision `4xx` request warnings, often bot noise:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND resource.labels.revision_name="REVISION_NAME" AND httpRequest.status>=400 AND httpRequest.status<500' \
  --limit=30 \
  --format='table(timestamp,severity,httpRequest.status,httpRequest.requestMethod,httpRequest.requestUrl,httpRequest.latency,httpRequest.userAgent)'
```

Uptime check results:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND httpRequest.userAgent:"GoogleStackdriverMonitoring-UptimeChecks"' \
  --limit=20 \
  --format='table(timestamp,severity,httpRequest.status,httpRequest.requestMethod,httpRequest.requestUrl,httpRequest.latency,resource.labels.revision_name)'
```

A time window:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND timestamp>="2026-02-18T20:00:00Z" AND timestamp<="2026-02-18T21:00:00Z"' \
  --limit=100 \
  --format='table(timestamp,severity,httpRequest.status,httpRequest.requestUrl,jsonPayload.message,textPayload)'
```

Full JSON for errors:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND severity>=ERROR' \
  --limit=20 --format=json
```

Application events by message content:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND jsonPayload.message:"san"' \
  --limit=30 --format='table(timestamp,severity,jsonPayload.message)'
```

The job's run summaries and one execution's events:

```bash
gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND jsonPayload.operation="run"' \
  --limit=6 --format='value(timestamp,jsonPayload.outcome,jsonPayload.duration_ms,jsonPayload.message)'

gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="shallweswim-capture" AND labels."run.googleapis.com/execution_name"="EXECUTION_NAME"' \
  --limit=100 --format='value(timestamp,severity,jsonPayload.message)'
```

Which instances loaded the current generation, and how far behind:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="shallweswim" AND jsonPayload.operation="load"' \
  --limit=20 --format='value(timestamp,jsonPayload.outcome,jsonPayload.duration_ms,jsonPayload.age_seconds,jsonPayload.message)'
```

## Applying changes

The metrics, the dashboard, and the twelve Terraform policies are applied
from `infra/monitoring` with a dedicated Terraform identity;
[infra/monitoring/README.md](infra/monitoring/README.md) is the how-to,
including the state bucket, impersonation, the plan and apply commands, the
mock-provider test, the ten-minute wait before a new metric's policies can be
applied, and how a policy is promoted. Everything else here, the uptime check
and the three older policies, is managed in the console.

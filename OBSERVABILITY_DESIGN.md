# Observability Design

**Status:** Proposed; open for review

**Scope:** Operational health, telemetry, dashboards, alerts, and paging for the
current application and the proposed persistent data pipeline. This document
does not define business analytics or user tracking.

## Summary

Shall We Swim should instrument application behavior with provider-neutral
OpenTelemetry APIs and structured status contracts, while using the managed
monitoring and alerting system of the deployment platform. For the current GCP
deployment, Cloud Monitoring remains responsible for uptime checks, dashboards,
alert evaluation, and notification delivery.

The design separates four questions:

1. **Can users reach the web application?**
2. **Is the updater executing successfully?**
3. **Is a coherent serving snapshot being produced and consumed?**
4. **Are individual feeds current enough for their purpose?**

Paging should indicate user impact, persistent stale data, or a pipeline failure
that requires action. An isolated expected upstream outage or arbitrary `ERROR`
log should not automatically page the operator.

## Relationship to the Data Pipeline Design

[Persistent Data Pipeline Design](PERSISTENT_DATA_PIPELINE_DESIGN.md) proposes a
bounded updater, durable observation archive, immutable serving snapshots, and a
read-only web process. This observability design is separate because it:

- Applies to the current architecture before that migration.
- Covers both the web and updater lifecycles.
- Can be rolled out independently.
- Defines operational policy rather than storage and materialization behavior.

The pipeline must nevertheless provide the signals defined here before its
production cutover.

## Current GCP Baseline

The following inventory was read from the live `shallweswim` project with
`gcloud` on August 23, 2026. Notification addresses are intentionally omitted.

### Application Uptime Check

`shallweswim API Healthy Uptime Check` currently:

- Requests `http://shallweswim.today/api/healthy?uptime` every 60 seconds.
- Sends `X-HealthCheck: uptime`.
- Accepts a `2xx` response.
- Uses a 30-second timeout.
- Uses Google static uptime checkers.

Cloud Run request logs show both the HTTP redirect and resulting HTTPS request,
so the final design should consider checking HTTPS directly and validating TLS
explicitly.

The corresponding `Homepage uptime failure` alert is enabled. It uses a
five-minute condition duration, treats missing data as active failure, and
notifies configured channels.

### Application Error Alerts

Two enabled policies cover the application:

1. **Error Log**
   - Matches `severity="ERROR" AND httpRequest.status!=503`.
   - Notifies only when the incident opens.
   - Rate-limits notifications to once per hour.
   - Broadly mixes application defects, internal watchdog messages, and request
     logging behavior.
2. **5xx error on shallweswim**
   - Uses the built-in Cloud Run request-count metric.
   - Excludes HTTP `503`.
   - Evaluates one-minute request-count windows.
   - Can page on a very small number of unexpected `5xx` responses.

This explains the current operational experience: nearly any unexpected error
can become an email page even when the service remains healthy and no action is
required.

### Other Project Monitoring

The project also contains enabled YouTube livestream uptime monitoring. It is
unrelated to Shall We Swim application health and is outside this design's
scope. It should remain logically separate when dashboards and alert ownership
are cleaned up.

### Missing Pieces

The live project currently has:

- No custom Cloud Monitoring dashboards returned by the dashboard inventory.
- No user-defined log-based metrics returned by the logging metric inventory.
- No updater-run, snapshot-publication, or per-feed freshness metrics.
- No dead-man alert for successful data updates.

The notification-channel inventory returned two enabled email channels. Some
alert policies reference additional channel resource IDs that were not returned
by that inventory; those references should be validated before policy cleanup.

## Design Principles

1. Instrument application semantics with OpenTelemetry, not provider SDK calls
   in feed or query code.
2. Use managed platform metrics for infrastructure behavior they already cover.
3. Keep durable operational state in the snapshot/update status, not solely in
   an ephemeral metrics backend.
4. Keep metric attributes low-cardinality and bounded.
5. Separate service availability, pipeline liveness, publication health, and
   feed freshness.
6. Alert on sustained actionable conditions, not individual log lines.
7. Treat expected upstream station unavailability as visible but normally
   non-pageable.
8. Ensure a monitoring-backend outage does not invalidate an otherwise correct
   data publication.
9. Make local development work with a console or no-op telemetry exporter.
10. Store GCP-specific dashboards and policies as deployment configuration,
    separate from provider-neutral instrumentation.

## Telemetry Architecture

```text
Web entry point -------------------+
                                    |
Updater entry point -> force flush  +--> OpenTelemetry SDK
                                    |          |
                                    |          | direct export
                                    |          v
Structured stdout logs ------------+     Managed backend
                                               |
Durable manifest/update status ------------> alerts + dashboards
```

Application modules use the OpenTelemetry API. Runtime entry points configure
the SDK and exporter appropriate to the deployment. The GCP deployment should
initially export directly from each process rather than operate an
OpenTelemetry Collector.

The exact supported GCP exporter and authentication path must be validated in a
prototype. It should use workload-provided credentials and require no global
authentication mutation.

Direct OpenTelemetry export is a preferred path, not a prerequisite for the
first observability improvements. If the supported GCP exporter, authentication,
write-rate constraints, or short-lived-process behavior proves awkward,
structured stdout events plus GCP log-based metrics are the explicit fallback.
Most counters, outcomes, heartbeats, and freshness states can be derived from
bounded JSON event fields with no application-side network transport. In-process
OpenTelemetry can then be reserved for histograms and traces that materially
improve diagnosis.

### Long-Lived Web Process

The web process records measurements during requests. A periodic exporter may
pause while a request-scaled instance is idle; this is acceptable because there
is no new request activity to report. Queued measurements can export when the
next request supplies CPU.

The web process should flush on graceful shutdown where practical, but platform
request count, latency, instance, and response-code metrics remain authoritative
even if an application telemetry batch is lost.

Web-side gauges are traffic-dependent diagnostics. A scaled-to-zero or idle
service legitimately produces no `web.snapshot.age` samples. Freshness paging
must never key solely on missing web telemetry; it uses updater-side signals and
durable pipeline status, with missing web gauge data treated as benign.

### Bounded Updater Process

The updater may finish before a periodic exporter runs. Its entry point must
explicitly flush telemetry in a `finally` path before exiting, using a bounded
timeout. Telemetry flush failure is logged and surfaced operationally but must
not turn a successfully published snapshot into a failed updater result.

The job runner's built-in execution status and duration metrics remain an
independent signal if application telemetry cannot flush.

### Pull Versus Push

When OpenTelemetry export is used, both application processes use direct push
export. A Prometheus-style scrape endpoint is not required:

- A web process could be scraped, but managed platform request metrics already
  cover its basic behavior.
- A short-lived updater cannot be reliably scraped after it exits.
- Explicit updater flush makes push export deterministic enough for semantic
  metrics.

External uptime checks remain pull-based HTTP probes. That is separate from
OpenTelemetry export.

## Durable Operational Status

Metrics are not the only source of truth. Each successful updater execution
should record bounded operational status alongside persisted pipeline state,
including:

- Run identifier, start time, completion time, and outcome
- Whether a new serving generation was published
- Current serving generation and publication time
- Per-feed last successful fetch and latest observation times
- Per-feed failure count and sanitized last failure category
- Archive merge and plot-generation outcomes

The web API can expose a sanitized operational view from the loaded manifest.
Detailed exception messages, credentials, source payloads, and unbounded run
history do not belong in metrics or public status responses.

A dedicated `/api/pipeline-status` endpoint should close the dead-man loop. It
reads current durable updater status from the shared store—not merely the web
instance's possibly stale loaded snapshot—and returns non-`2xx` when the last
semantic success exceeds the configured window. A managed uptime check can
monitor that endpoint without custom absence-metric semantics. It remains
separate from `/api/healthy`, which answers whether the particular web instance
can currently serve requests. Custom heartbeat metrics remain useful for
dashboards and diagnosis but are not the only dead-man implementation.

The endpoint should cache the durable status read in process for roughly 30
seconds. That is negligible relative to a 25–30-minute dead-man window, keeps
normal probes memory-speed, and limits amplification if the endpoint is public.
Cache expiration must still force a new shared-store read; an indefinitely stale
web cache cannot satisfy this check.

## Dead-Man Switches

A normal alert reacts to reported failure. A dead-man switch reacts when an
expected success signal disappears, catching systems too broken to emit an
error.

### Updater Execution Heartbeat

Every semantically successful update cycle emits a success heartbeat, including
a legitimate no-change run. For a ten-minute schedule, an initial alert window
might be 25–30 minutes.

This detects:

- Scheduler stopped invoking the updater
- Job could not start
- Repeated crashes or hangs
- Storage/authentication failure preventing completion
- Semantic update failure despite a container exiting

Platform job-completion metrics and the application heartbeat should be viewed
together. Their disagreement is diagnostic: a platform-successful job without
an application-success heartbeat suggests initialization, instrumentation, or
semantic completion problems.

### Publication and Data Freshness

A successful no-change run intentionally publishes no redundant generation.
Therefore absence of a recent publication is not by itself a dead-man failure.
Monitoring must distinguish:

- Last successful updater execution
- Last actual snapshot publication
- Whether the last run found serving-state changes
- Snapshot age
- Per-feed data and observation age

Feed freshness thresholds are source-specific and should reuse the explicit
freshness budget established by the pipeline design.

## Proposed Metric Catalog

Names are provisional. Units and attribute contracts must be finalized before
implementation.

### Updater Metrics

| Metric | Type | Purpose |
| --- | --- | --- |
| `shallweswim.updater.run` | Counter | Completed runs by bounded outcome |
| `shallweswim.updater.run.duration` | Histogram, seconds | Complete execution time |
| `shallweswim.feed.fetch` | Counter | Fetch attempts by outcome |
| `shallweswim.feed.fetch.duration` | Histogram, seconds | Upstream latency |
| `shallweswim.feed.records` | Histogram or counter | Records received/merged |
| `shallweswim.feed.data.age` | Gauge, seconds | Age of last successful fetch |
| `shallweswim.feed.observation.age` | Gauge, seconds | Age of latest observation |
| `shallweswim.feed.consecutive_failures` | Gauge | Current failure state |
| `shallweswim.archive.merge.duration` | Histogram, seconds | Archive persistence cost |
| `shallweswim.archive.merge` | Counter | Merge outcomes/conflicts |
| `shallweswim.snapshot.gc` | Counter | Published objects examined/deleted and GC outcomes |
| `shallweswim.snapshot.gc.duration` | Histogram, seconds | Published-object mark-and-sweep cost |
| `shallweswim.plot.availability_latency` | Histogram, seconds | Submit-to-harvest latency, including scheduling or CPU starvation |
| `shallweswim.snapshot.publish` | Counter | Changed/no-change/failure outcomes |
| `shallweswim.snapshot.publish.duration` | Histogram, seconds | Publication cost |

### Web Metrics

| Metric | Type | Purpose |
| --- | --- | --- |
| `shallweswim.web.snapshot.age` | Gauge, seconds | Age of loaded generation |
| `shallweswim.web.snapshot.refresh` | Counter | Unchanged/changed/failure outcomes |
| `shallweswim.web.snapshot.refresh.duration` | Histogram, seconds | Elected-request refresh cost |
| `shallweswim.web.snapshot.load.duration` | Histogram, seconds | Readiness/cold-load cost |
| `shallweswim.web.snapshot.changed_objects` | Histogram | Refresh object count |

Built-in platform metrics remain preferred for request count, status code,
latency, CPU, memory, instance count, startup, and job execution state.

### Attribute Policy

Allowed bounded attributes include:

- `location`: configured location code
- `feed`: stable feed name
- `provider`: bounded configured provider name
- `outcome`: a small documented enum
- `operation`: a small documented enum

Do not use generation IDs, station IDs from arbitrary requests, URLs, exception
messages, timestamps, user agents, or run IDs as metric attributes. Those belong
in structured logs or durable status.

During the transitional architecture, each web process is also an updater and
may report a different feed age or failure count. Telemetry must retain the
platform-provided process/task instance as an OpenTelemetry resource identity,
not a domain metric attribute added by feed code. Dashboards aggregate these
multi-writer values deliberately—for example, maximum feed age and maximum
consecutive failures—rather than accepting an arbitrary last writer. Alerts do
not assume a single updater until the bounded-updater cutover.

## Logs and Traces

Production application logging uses a standard stream handler. Hosted
deployments emit JSON lines on stdout for the deployment platform to capture;
local development retains a human-readable console format. The application has
no provider logging SDK or background log transport. Logs include only approved
stable fields such as component, operation, location, feed, provider, bounded
outcome, run ID, and generation ID where applicable.
Uvicorn access logs remain enabled for human-readable local development but are
disabled with JSON logging because Cloud Run already emits richer structured
request logs under `run.googleapis.com/requests`.

The continuous updater emits one completion event for each attempted expired
feed update:

- `component=updater`, `operation=feed_update`
- bounded `location`, semantic `feed`, provider family, and `outcome`
- elapsed `duration_ms`, plus `record_count` on success
- `INFO` for `success`, `WARNING` for handled `unavailable`, and `ERROR` for
  `failed`

Plot harvesting uses the same completion-event pattern with `component=plot`,
`operation=plot_generation`, the plotted feed, submit-to-harvest
`duration_ms`, and a bounded `success` or `failed` outcome. This duration is
plot availability latency, not subprocess render time: it deliberately includes
delayed harvesting, scheduling, and CPU starvation so it detects the production
failure mode that motivated the instrumentation. Fetch/request starts, provider
URLs, station identifiers, and redundant lower-level success messages are DEBUG
diagnostics, not metric dimensions or routine production INFO events.

Structured event schemas should be designed so important counters and states can
be converted into deployment-configured log-based metrics. This supplies the
fallback when direct OpenTelemetry export is unnecessary or unreliable without
coupling domain code to GCP.

Log severity retains its conventional meaning:

- `DEBUG`: detailed diagnostic state
- `INFO`: expected lifecycle and successful operations
- `WARNING`: degraded but handled conditions, including expected station
  unavailability
- `ERROR`: unexpected defect, exhausted critical operation, or failed semantic
  boundary

An `ERROR` remains highly visible but does not automatically imply paging. Alert
policies decide whether the error corresponds to sustained user or pipeline
impact.

Distributed traces are most useful for updater runs and elected snapshot
refreshes. Initial instrumentation should prioritize metrics and structured logs;
traces can be added where they answer questions that timings alone cannot.

## Alert Policy

### Page

- Public HTTPS uptime fails from multiple regions for a sustained window.
- Unexpected web `5xx` rate exceeds both a count and rate threshold for several
  minutes.
- No semantically successful updater execution occurs within the dead-man
  window.
- The job runner reports repeated failed executions.
- No valid serving snapshot exists or all web instances fail readiness.
- Snapshot or critical-feed age exceeds a deliberately pageable threshold.
- Atomic publication or archive integrity repeatedly fails.
- A web loader receives `404` for an object referenced by a valid retained
  manifest; this indicates publication or garbage-collection corruption.

### Warn or Create an Investigation

- One upstream station is unavailable while the rest of the application
  remains useful.
- A noncritical feed exceeds its normal freshness budget.
- Updater duration, archive conflict retries, or snapshot load latency trends
  upward without crossing a failure boundary.
- Telemetry export fails while application work succeeds.
- A source schema/parse error occurs once but succeeds on the next run.

### Record Without Notification

- Expected retryable upstream failures that recover within the run
- Successful no-change updater executions
- Normal scale-to-zero/startup behavior
- Individual handled `404`/`503` responses representing documented data
  unavailability

Exact thresholds require baseline measurements and an explicit decision about
which feeds are critical. Alert policies should link to a concise runbook query
or investigation procedure.

## GCP Reference Implementation

Use managed GCP facilities for the hosted deployment:

- Cloud Run built-in web and Job metrics
- Cloud Monitoring uptime checks
- Cloud Monitoring custom or log-based metrics, with direct OpenTelemetry
  export adopted where the prototype proves it useful
- Cloud Logging for structured stdout logs
- Cloud Trace if tracing is enabled
- Cloud Monitoring dashboards and alert policies
- Existing email notification channels initially

GCP dashboards and alert policies are deployment artifacts. Metric meanings,
attribute contracts, durable statuses, and OpenTelemetry instrumentation remain
provider-neutral.

The initial reference implementation lives in `infra/monitoring`. Terraform
owns only the user-defined log-based metrics and application operations
dashboard in this first slice; it does not own Cloud Run, IAM, notification
channels, uptime checks, or existing alert policies. Native mock-provider tests
validate the plan locally, while a controlled GCP apply validates actual log
ingestion and dashboard population because no faithful local Monitoring
emulator exists.

The current broad `Error Log` and immediate low-volume `5xx` policies should not
be removed until replacement uptime, dead-man, freshness, and sustained-error
alerts have operated successfully in shadow or parallel.

## Dashboard Proposal

One application operations dashboard should show:

### Web

- Uptime and probe latency
- Request volume, p50/p95/p99 latency, and response classes
- Active/idle instance counts and cold starts
- Loaded snapshot age
- Snapshot check/load duration and failure rate

### Updater

- Last platform execution and semantic success
- Duration by phase: fetch, merge, plot, publish
- Changed versus no-change runs
- Publication generation age
- Archive conflicts and failures
- Garbage-collection objects examined/deleted, duration, and failures

### Feeds

- Last success and latest observation age by location/feed
- Consecutive failures
- Fetch duration and record count
- Current expected-unavailable sources separated from unexpected failures

The dashboard should emphasize current state and trends without using unbounded
tables or labels.

## Rollout Plan

### Cross-Design Implementation Order

Observability begins first but does not need to be completed before persistence
work starts. The intended order is:

1. Add a minimal monitoring foundation to the current system: update/feed/plot
   timings and outcomes, data age, JSON stdout events, structured status, and an
   initial dashboard.
2. Begin durable archive capture early because upstream history can disappear;
   instrument archive writes, conflicts, duration, and failures from its first
   deployment.
3. Publish serving snapshots in shadow mode and measure object size,
   publication duration, and no-change behavior.
4. Load snapshots in web shadow mode, compare them with current manager results,
   and measure readiness and elected-request latency.
5. Introduce the bounded updater with explicit telemetry flush, semantic
   heartbeat, publication, and freshness alerts.
6. Cut web serving over only after the replacement operational signals work and
   rollback has been demonstrated.
7. Retire broad "any error" paging only after the new uptime, dead-man,
   freshness, and sustained-error policies demonstrate equivalent or better
   coverage.

This deliberately interleaves the projects: building persistence without
telemetry would make the migration blind, while waiting for a complete
observability platform would unnecessarily delay time-sensitive archive
capture.

### Phase 1: Baseline and Contracts

- Preserve the live GCP inventory in this document.
- Define metric names, units, bounded attributes, and outcome enums.
- Define durable updater and feed status schemas.
- Define JSON stdout event schemas and the stable fields needed for log-based
  metrics.
- Measure existing uptime, request errors, feed age, and update duration where
  logs permit.

### Phase 2: Web Instrumentation

- Replace the provider logging handler with structured JSON stdout, remove the
  provider dependency in the same rollback-safe Cloud Run revision, and then
  validate Cloud Logging severity and field parsing immediately after deploy.
- Prototype direct OpenTelemetry export in the web entry point, retaining
  log-based metrics as the fallback.
- Add snapshot metrics when snapshot serving exists.
- Validate direct export and idle/request-scaled behavior.
- Create an initial managed dashboard without changing paging.

### Phase 3: Current-Updater Instrumentation

- Instrument the existing background update path using the future metric
  contracts where practical.
- Establish feed freshness baselines before the bounded Job migration.
- Preserve per-instance resource identity and validate max/worst-instance
  aggregation while several web processes update independently.

### Phase 4: Bounded-Updater Instrumentation

- Configure OpenTelemetry in the updater entry point.
- Add explicit bounded final flush.
- Correlate platform execution state, semantic heartbeat, publication, and feed
  status.
- Run dead-man and freshness alerts without notifications initially.

### Phase 5: Alert Cutover

- Enable actionable uptime, sustained `5xx`, updater dead-man, publication, and
  freshness notifications.
- Confirm notification channels and remove stale references.
- Observe replacement policies in parallel.
- Retire the broad error-log page only after coverage is demonstrated.

## Testing

- Metric names, units, and bounded attribute enums are asserted.
- High-cardinality values cannot enter metric attributes.
- Web telemetry does not block ordinary requests when export is unavailable.
- Updater telemetry flush is invoked on success and failure with a bounded
  timeout.
- Telemetry export failure does not change a successful publication outcome.
- No-change updater runs emit semantic success without a publication event.
- Dead-man logic distinguishes missing executions from no-change publications.
- Pipeline-status dead-man responses remain distinct from web-serving health.
- Feed freshness calculations match feed-specific budgets.
- Missing traffic-dependent web gauges do not trigger freshness alerts.
- Multi-writer transitional feed gauges aggregate by explicit worst-instance
  semantics.
- Structured stdout events map to expected Cloud Logging severity and fields.
- Log-based metric extraction works without the in-process cloud logging
  transport.
- Garbage collection cannot silently remove a manifest-referenced object; a
  referenced-object `404` is surfaced as a page-worthy integrity error.
- Local development works with a console or no-op exporter.
- Operational status responses omit secrets and unbounded exception content.

## Open Questions

1. Which direct OpenTelemetry exporter is the supported GCP path for metrics and
   traces in both Cloud Run services and Jobs, and which signals should instead
   begin as log-based metrics?
2. What updater cadence and dead-man window will be selected after freshness
   prototyping?
3. Which feeds are critical enough for stale-data paging rather than warning?
4. Should the operational health endpoint be public, authenticated, or split
   into public summary and private detail?
5. What minimum count, error rate, and duration should replace the current
   immediate low-volume `5xx` alert?
6. Which existing notification-channel references are stale?
7. What telemetry volume and custom-metric cost result from the proposed
   per-location/feed series?
8. Should traces be included initially or added only after metrics establish a
   concrete diagnostic need?

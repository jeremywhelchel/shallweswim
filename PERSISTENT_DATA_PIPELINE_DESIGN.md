# Persistent Data Pipeline Design

**Status:** Proposed; open for review
**Motivation:** Separate bounded data materialization from request serving,
share one coherent dataset across processes, and preserve observations that may
later disappear upstream.
**Scope:** Data acquisition, materialization, persistence, plot generation, and
web-serving boundaries. This is not an implementation specification yet.

## Summary

Shall We Swim currently runs acquisition, caching, plot generation, and HTTP
serving together in every application process. This is simple and pleasant to
run locally, but it couples background progress to the lifecycle and resource
allocation of a request-serving process. Multiple processes independently build
ephemeral state, may perform duplicate work, and can observe different data.

The proposed direction separates production data materialization from HTTP
serving while retaining one repository, one application, and preferably one
container image:

```text
Scheduler
      |
      v
Bounded updater execution ---> Upstream NOAA / USGS / NDBC / other APIs
      |
      | atomically publishes
      v
Shared snapshot and observation store
      |
      | loaded and cached in memory
      v
Web service ---> Users
```

The leading storage model is a shared object store, potentially with a small
metadata store if experience shows that objects alone are awkward. The specific
storage product remains open.

## Portability Principle

The application should depend on three provider-neutral capabilities:

1. Run a Docker-compatible container image as an HTTP service.
2. Run the same container image to completion on a schedule.
3. Read and atomically publish named byte objects in shared storage.

The current GCP stack is one implementation of those capabilities.
Provider-specific SDK types, resource names, credentials, lifecycle rules, and
generation semantics should remain in deployment configuration and storage
adapters rather than leaking into clients, feeds, queries, plots, or snapshot
schemas.

The updater entry point must be an ordinary bounded command:

```bash
docker run shallweswim python -m shallweswim.update
```

It should not require a Cloud Run Job API to execute. Scheduling and retries are
the platform's responsibility. Equivalent deployments include:

| Environment | Web process | Scheduled updater | Shared store |
| --- | --- | --- | --- |
| Laptop / single server | Container or Python process | cron/systemd timer | Filesystem |
| Docker Compose | Web container | one-shot container | Shared volume or MinIO |
| GCP | Cloud Run Service | Scheduler + Cloud Run Job | Cloud Storage |
| AWS | ECS/Fargate or App Runner | EventBridge scheduled task | S3 |
| Kubernetes | Deployment | CronJob | Object store or persistent volume |
| Azure | Container Apps | Container Apps Job | Blob Storage |

Snapshot formats should use portable standards such as JSON, Parquet, and SVG.
Object paths and manifests must not contain GCP-only identifiers. A minimal
provider-neutral store interface should cover object reads, immutable writes,
listing or lifecycle discovery where necessary, and conditional promotion of a
current manifest. The filesystem implementation is a real portability target,
not merely a test fake.

The design does accept some deployment-level coupling. Atomic compare-and-swap,
authentication, lifecycle garbage collection, metrics, and job-overlap controls
are expressed differently by each platform. Those differences should be small
adapters and infrastructure configuration rather than reasons to fork the
application.

### GCP Reference Deployment

The initial hosted implementation is expected to use:

- Cloud Run Service for the web entry point
- Cloud Scheduler to trigger a Cloud Run Job for the updater entry point
- Cloud Storage as the shared object-store candidate
- Existing Google Cloud logging, monitoring, and alerting

This mapping is the reference deployment because it matches the project's
current hosting, not because these products define the application architecture.
The provider-neutral entry points, snapshot schema, archive format, and storage
contract remain the design's source of truth.

## Why Revisit the Current Design?

The current design is intentionally compact:

```text
One process -> fetch -> cache in memory -> generate plots -> serve requests
```

It has important advantages:

- One command runs the real system locally.
- There is no database, persistence schema, or distributed coordination.
- Clients, feeds, expiration, retries, queries, and serving live in one
  debuggable application.
- The current hosted deployment remains inexpensive.

It also has production consequences:

- Every application process independently fetches and plots the same data.
- Instances do not share state and may briefly serve different generations.
- Instance termination discards all collected data.
- Background work receives CPU according to whether that specific instance is
  handling a request.
- Monitoring keeps the service warm but does not guarantee CPU to every
  instance.
- Every new process starts with an empty `HistoricalTempsFeed._year_cache` and
  requests each configured year back to 2011 for every applicable location.
  Multiple instances multiply that upstream load despite the process-local
  provider concurrency gates.

### Reference-Deployment Incident: August 22, 2026

A second Cloud Run instance started background updates and submitted four live
temperature plots. During the following nine minutes, all 114 observed HTTP
requests went to the other instance. The idle instance therefore received no
request-allocated CPU after its startup boost. Four normally fast plots crossed
the application's five-minute wall-clock watchdog and logged errors. Cloud Run
later granted CPU during shutdown, and the workers completed.

The same live-plot path takes about 0.11 seconds locally; four concurrent cold
workers complete in about 1.4 seconds. The incident was therefore not a
five-minute plotting computation. It exposed a mismatch between per-instance
background processing, nondeterministic request routing, and request-based CPU
allocation.

Moving to continuously allocated CPU would preserve the current runtime model,
but one continuously warm 4-vCPU/4-GiB instance is estimated at roughly
$200/month. That is not appropriate for this project.

### Smaller Mitigations Considered

Setting the hosted service to a maximum of one serving instance is a cheap
mitigation for the specific idle-secondary-instance incident: ordinary
monitoring and user requests would normally reach the same instance doing
background work. It is not the proposed long-term design because it:

- Reduces availability and removes horizontal scaling.
- Does not eliminate old/new instance overlap during every deployment or
  platform restart.
- Still discards all accumulated observations at process termination.
- Still performs the complete per-year historical bootstrap on every cold
  start.
- Does not create a durable history beyond each provider's retention window.

Session affinity likewise does not guarantee that every background-working
instance receives CPU, and startup boost only provides a bounded initialization
window. These mitigations may reduce alert frequency, but they do not address
the archive, duplicate-fetch, or coherent-publication goals that now justify the
larger design.

## Design Goals

1. Keep the public web service compatible with inexpensive request-scaled
   hosting.
2. Ensure updater and plot work receives CPU for its complete execution.
3. Give every web instance the same coherent, published data generation.
4. Keep end-user requests independent of upstream API availability and latency.
5. Preserve the existing clients, feed validation, concurrency, expiration,
   retry, query, and plotting logic where practical.
6. Preserve the fast local-development experience.
7. Retain last-known-good measurements when an upstream series disappears.
8. Avoid introducing a fleet of microservices or an always-on database.
9. Make paging indicate actionable application failures, not hosting-platform
   scheduling artifacts.
10. Keep operating cost proportionate to a small personal project.
11. Accumulate a longer observation history than upstream APIs with short
    rolling retention windows make available at any single point in time.
12. Ensure a fresh installation can bootstrap from an empty store using only
    configured upstream sources and then build its own independent archive.
13. Keep archive formats portable and exportable without requiring the
    maintainer to operate a public data-distribution service.

## Non-Goals

- Real-time streaming ingestion
- Exactly-once distributed processing
- A general-purpose analytics platform
- One service or job per location or feed
- Replacing the existing upstream clients without a source-specific reason
- Changing station configuration as part of the persistence work
- Keeping every derived web snapshot forever

## Proposed Runtime Components

### 1. Scheduled Updater

A scheduler starts the updater entry point as a bounded execution on a fixed
cadence, initially perhaps every ten minutes. Each execution:

1. Loads the current manifest and persisted feed state.
2. Determines which feeds are expired.
3. Fetches expired feeds concurrently using the existing clients and provider
   concurrency limits.
4. Retries transient errors using the current retry policy.
5. Preserves last-known-good data when a station is temporarily unavailable.
6. Merges newly fetched observations into the durable archive.
7. Recomputes derived data whose inputs changed.
8. Generates affected plots and waits for them to finish.
9. Writes a complete, versioned serving snapshot.
10. Publishes the new generation atomically and exits.

A ten-minute schedule does not imply that every source is fetched every ten
minutes. Existing feed expiration periods remain authoritative. A job may find
little or nothing to update and exit quickly.

The first implementation should remain one job for the whole application.
Separate live, prediction, and historical schedules should be considered only
if measured cost or execution time justifies the additional orchestration.

The bounded updater also centralizes provider traffic. One logical updater
respects one set of NOAA/USGS/provider concurrency limits instead of multiplying
the same fetches and per-process semaphores across serving instances.

### 2. Shared Store

The shared store has two logically different responsibilities:

- **Published snapshots:** disposable, coherent application state optimized for
  web instances to load.
- **Observation archive:** durable normalized measurements and optional raw
  source responses retained independently of current upstream availability.

These responsibilities should have different retention rules even if they use
the same storage product.

### 3. Stateless Web Service

The public web service no longer contacts upstream providers or performs routine
background plotting. It:

1. Loads and validates the latest published generation before becoming ready.
2. Holds decoded feeds, derived data, and plot bytes in process memory.
3. On an incoming request, checks whether the manifest-check interval has
   elapsed and, if so, performs one coalesced freshness check while request CPU
   is available.
4. If a new generation exists, loads changed objects while continuing to serve
   the old generation to other requests.
5. Atomically swaps its in-memory snapshot after validation.
6. Serves normal API requests entirely from memory.

This preserves the current fast request path. A warm user request does not need
to wait for shared storage unless it is the request elected to perform an
overdue manifest check or refresh. A per-process lock prevents concurrent
requests from duplicating the refresh. An instance receiving no requests does no
refresh work, which is correct because it is serving no users.

This request-piggybacked mechanism is deliberate. A free-running background
poller would recreate a smaller version of the current scheduling problem on
platforms that throttle idle request processes. The implementation must keep
the triggering request open until required storage work has completed rather
than assuming an after-response task will continue receiving CPU.

Existing on-demand tide/current detail plots are a separate case. They are
generated as part of an active user request, so request-scaled CPU allocation
does not create the idle-background failure mode. They may remain request-driven
initially. Consequently, the web image still needs Matplotlib/SciPy and an
on-demand process pool; the proposal removes routine feed-driven plotting from
the web lifecycle, not necessarily every plotting code path. Precomputing common
planner times remains independent technical debt rather than a prerequisite for
this migration.

### 4. Shared Image, Separate Entry Points

The agreed direction is one repository, one shared application package, and one
Docker image with separate entry points for each lifecycle:

```bash
uv run python -m shallweswim.web
uv run python -m shallweswim.update
uv run python -m shallweswim.local
```

- `shallweswim.web`: read-only production web service
- `shallweswim.update`: one bounded materialization cycle for a scheduled job
- `shallweswim.local`: updater and web server together, using local or in-memory
  storage

The deployment platform selects the appropriate command from the shared image.
Separate entry points keep production lifecycle wiring, imports, tests, and
shutdown behavior explicit: the web process cannot accidentally start updater
machinery, and the scheduled job does not need to initialize FastAPI. The local
entry point preserves one-command development without making the production
roles conditional modes inside one main function.

Separate images are not currently justified because both roles share clients,
models, feed logic, plotting dependencies, and configuration. They can be
reconsidered if their dependencies or release cadence materially diverge.

## Effect on the Existing Feed Stack

This should be a lifecycle refactor, not a rewrite of source integration.

Expected to remain substantially intact:

- API clients and provider concurrency gates
- Data parsing, normalization, units, and timestamp contracts
- Feed-specific validation
- `StationUnavailableError` and retryable-client behavior
- Feed expiration intervals
- Location configuration
- Core query functions
- Plot generation

Expected to change materially:

- Feed data and update metadata must be serializable and restorable.
- Expiration timestamps must survive process termination.
- The perpetual background loop becomes a bounded `update_once` workflow.
- Retry scheduling after an exhausted attempt becomes "try during the next job"
  rather than an in-process timer.
- Plot work in the updater is awaited before publication rather than submitted
  fire-and-forget.
- The web-facing manager becomes read-only over a loaded snapshot.
- Startup health reflects snapshot availability rather than completion of
  upstream fetching.

`HistoricalTempsFeed` needs special treatment. Its `_year_cache` and
`_year_cache_fetch_timestamp` are the reason completed historical years are not
refetched on every update. Persisted state must retain frames and fetch metadata
at year granularity; serializing only the combined historical DataFrame would
cause every scheduled execution to refetch every year.

The manager's tide and current prediction frames are cheap derived indexes and
should initially be recomputed after snapshot load rather than persisted. Their
current `id(feed._data)` invalidation guard is process-local by definition, so
snapshot-load tests must verify correct reconstruction and invalidation from
source timestamps.

A possible abstraction boundary is:

```python
class SnapshotStore:
    async def load_current(self) -> Snapshot: ...
    async def publish(self, snapshot: Snapshot) -> None: ...
```

Implementations could include memory for tests, filesystem for local
development, and a production shared store. The actual data model should be
designed before treating this sketch as an interface commitment.

## Published Snapshot Model

A generation should be an immutable, self-describing manifest referencing
immutable content-addressed objects. Unchanged feeds and plots are reused across
generations rather than copied into a new timestamp directory. One possible
layout is:

```text
published/
  current.json
  manifests/
    2026-08-22T18-10-00Z-run-a1b2c3.json
  objects/
    sha256-abc123.parquet
    sha256-def456.svg
    sha256-ghi789.svg
```

The updater writes changed content objects first, then the generation manifest,
and updates `current.json` last. A manifest may reference objects first written
by an older generation. Web instances therefore see either the complete
previous generation or the complete new generation, never a partially written
mix.

If no feed data, status, plot, or other serving state changed, the updater
should publish nothing. Operational job-success timestamps belong in monitoring
or separate updater metadata rather than forcing a new serving generation. If
only status or freshness metadata changes, publishing a small manifest without
rewriting unchanged data remains appropriate.

Generation identifiers combine a sortable timestamp with a unique job/run
identifier. Timestamp-only names can collide when overlapping publishers start
within the same second, even though conditional promotion would protect
`current.json`.

The manifest should include at least:

- Generation identifier and publication timestamp
- Schema version
- Per-feed source identity
- Last successful observation and fetch timestamps
- Expiration/freshness state
- Last attempted update and sanitized failure status
- Object checksums or generations
- Plot-to-feed dependency metadata where useful

Serving state and updater state do not need identical granularity:

- **Serving snapshot:** one immutable object per feed per location, with
  historical temperatures represented by the combined frame the web tier
  consumes. Plot SVGs remain separate immutable objects. This keeps cold loads
  coarse and bounded while allowing instances to download only changed feeds.
- **Updater/archive state:** historical temperatures remain partitioned per year
  with per-year fetch timestamps. Completed past-year objects are effectively
  write-once; only the current year normally changes.

The remaining measurement question is whether any combined per-location
historical frame is large enough to justify finer serving partitions. Updater
granularity must not leak into the serving manifest merely for implementation
convenience.

Promotion of `current.json` requires compare-and-swap semantics: publish only if
the current generation still matches the generation observed when the updater
started, and abort rather than moving the pointer backward if a newer publisher
won. On GCS this maps to an `if-generation-match` precondition; other storage
adapters must provide equivalent conditional replacement behavior.

## Durable Observation Archive

The archive protects historical measurements from upstream removal and avoids
re-fetching complete history indefinitely. The Louisville USGS temperature
series is the motivating example: had observations been archived, removal of
parameter `00011` from site `03292494` would stop new measurements without
erasing the history already collected.

Some configured providers expose no historical endpoint, while others retain
only recent months or roughly a year. For those sources, repeated snapshots are
currently discarded when an instance terminates, so the application can never
develop a multi-year record even if it has observed the station continuously.
The archive should accumulate each newly observed interval and gradually create
a history longer than the provider's rolling window.

This is prospective preservation, not historical reconstruction. At migration:

1. Backfill every observation the provider still exposes.
2. Record the earliest and latest available timestamps and any known gaps.
3. Append future observations on every successful update.
4. Never imply that periods before collection began were observed or complete.
5. Allow later approved imports from authoritative archives without confusing
   them with data collected directly by the running application.

Archive completeness should be measurable per source. Useful metadata includes
collection start, earliest observation, latest observation, expected cadence,
known gaps, last successful archive merge, and whether records came from live
collection or a later backfill.

One possible partitioning scheme is:

```text
archive/
  usgs/03292494/00011/2026.parquet
  noaa-coops/8518750/water-temperature/2026.parquet
  ndbc/46237/water-temperature/2026.parquet
```

Normalized observations should retain:

- Observation timestamp
- Normalized value and unit
- Provider, station, parameter, and source type
- Retrieval timestamp
- Available quality/provisional flags
- Schema version and provenance needed to interpret the record

The updater should fetch incrementally with a small overlap window, then merge
and deduplicate by stable source identity and observation timestamp. The overlap
allows providers to revise recent provisional readings.

Archive partition updates require concurrency control. During the transitional
phase, several existing web processes may fetch successfully and attempt to
merge into the same yearly Parquet object. A plain read-merge-write can silently
lose disjoint observations. The initial implementation should use a conditional
read-merge-write retry loop:

1. Read the partition and its object generation/version.
2. Merge and deduplicate the new observations.
3. Replace the partition only if its generation still matches.
4. On conflict, reread, merge again, and retry with a bounded policy.

The filesystem adapter needs equivalent locking and atomic replacement. Unique
append-only staging objects plus later compaction remain an alternative if
contention or partition-rewrite cost proves material, but are not the initial
choice.

Optionally, compressed raw responses can be retained for provenance and future
reprocessing. Raw-response retention may be finite; normalized observations are
expected to be long-lived.

Archived data must not disguise current source availability. The application
should be able to say both "the last archived observation was at time X" and
"the source is currently unavailable."

## Open Source, Independent Bootstrap, and Archive Distribution

The hosted deployment will gradually accumulate observations that a fresh clone
cannot recover. The required principle is:

> An empty shared store is a fully supported production starting state. The
> updater bootstraps from configured upstream sources and incrementally builds a
> complete local archive from that point forward.

This is today's cold-start path. Persistence adds two behaviors around it:

- After bootstrap and subsequent updates, merge observations into a durable
  archive instead of discarding them at process exit.
- On later executions, restore the archive first and fetch only missing or
  expired intervals rather than starting empty again.

Empty-store bootstrap is the tested, documented default, with no dependency on
the maintainer's production resources. If an archive is absent or lost, the
system recovers through the same upstream bootstrap, although uniquely
accumulated historical depth may be lost.

Publishing the maintainer's archive is a separate, undecided policy question.
The design should make export technically possible without assuming that the
project will become a data distributor. Serving derived conditions and plots in
an application is not necessarily the same operational or licensing commitment
as offering bulk source observations for redistribution. Each provider's terms,
required attribution, update expectations, and stewardship burden require a
source-specific review. Growing Parquet archives do not belong in Git. A future
public object prefix or periodic release may be considered, but community
bootstrap and third-party archive exchange are not initial requirements.

## Garbage Collection and Retention

Published generations and durable observations require different policies.

Suggested initial policy:

- Keep the active published generation unconditionally.
- Keep previous complete generations for 7–30 days for rollback and debugging.
- Delete incomplete generation uploads after one day.
- If object versioning is enabled, delete old noncurrent versions after a short
  recovery window.
- Retain normalized observations indefinitely unless a later policy says
  otherwise.
- Retain raw upstream responses for a defined period, such as 30–90 days, or
  omit them initially.

Content-addressed objects complicate age-only lifecycle deletion because a newly
published manifest may still reference an old unchanged object. Manifest cleanup
can use simple age rules, but data-object cleanup requires a reachability-aware
mark-and-sweep process:

1. Read the active and retained generation manifests.
2. Mark every referenced object checksum.
3. Delete only unreferenced objects older than a safety window.

This can run as part of an updater execution or a separate infrequent maintenance
command. Object-store lifecycle policies remain useful for abandoned temporary
uploads and old noncurrent pointer versions, but must never be able to delete an
object referenced by a retained manifest.

## Storage Options

### Cloud Storage

Current leading candidate.

Strengths:

- Serverless with negligible idle cost
- Strongly consistent object reads and writes
- Natural fit for Parquet, JSON manifests, SVGs, and raw payloads
- Immutable generations and atomic publication are straightforward
- Lifecycle-based garbage collection
- Handles historical volume without document-size limits

Weaknesses:

- Requires explicit serialization and whole-object replacement
- Less natural for small independent field mutations or ad hoc queries
- Careless use of many tiny objects can increase complexity and operation count

Despite its name, this can remain a small byte-store dependency rather than a
large data platform. Web instances would normally read it only during cold load
or generation refresh, not once per user request.

### Firestore

Strengths:

- Serverless, structured metadata, transactions, and granular updates
- Natural feed status and freshness documents
- No continuously provisioned instance

Weaknesses:

- 1-MiB document limit complicates historical frames and SVG plots
- DataFrames require chunking or a separate serialization convention
- More application reads and schema surface
- Blob-like plots and archives remain better suited to object storage

A Firestore metadata plus Cloud Storage blob/archive hybrid is viable, but it
adds a second persistence technology and should be justified by a concrete need.

### Memorystore / Redis

This is closest to App Engine memcache semantics.

Strengths:

- Fast shared cache with native expirations
- Convenient independent feed updates
- Familiar in-memory data model

Weaknesses:

- Google Memorystore is continuously provisioned and may cost more than the
  current application.
- Requires network configuration and another running service.
- Cache eviction does not satisfy the new historical-preservation goal.
- Historical DataFrames and SVGs are opaque, relatively large values.

An external serverless Redis provider could reduce idle cost but adds a vendor
and still does not naturally provide a durable observation archive.

### Cloud SQL or Other Database

A relational/time-series database would make observation queries and incremental
writes natural. It also introduces schema migrations, connection management,
availability, and baseline cost that the current scale does not justify. It
should remain an option only if future product requirements demand richer
historical querying than partitioned objects can reasonably provide.

### Local or Instance Filesystem

A request-scaled container's writable filesystem is normally process-local and
ephemeral. It cannot coordinate instances or preserve observations and therefore
is not a production solution in that environment. Filesystem storage remains
appropriate for local development and for a single-server deployment with a
durable mounted volume.

## Request Latency

Warm web requests should remain memory-only and have effectively the same
latency as today.

Expected storage interactions:

- Cold instance: load and validate the current manifest and snapshot before
  becoming ready.
- Warm instance: after the check interval elapses, one incoming request performs
  a coalesced manifest check while other requests continue using memory.
- New generation: that request loads and validates changed objects while other
  requests continue to serve the previous generation, then swaps atomically.

Rough expectations, to be validated with a prototype:

- Manifest check in the same region: tens of milliseconds
- Small snapshot cold load: roughly 100–500 milliseconds
- Multi-megabyte snapshot cold load: potentially around a second
- Warm end-user request: no storage round trip

One elected request per check interval intentionally absorbs the manifest-check
latency, and a request that discovers a new generation may absorb the full
changed-object load—potentially around one second under the estimates above.
This tail-latency cost is accepted to guarantee CPU on request-scaled platforms.
Changed objects should be fetched concurrently with a bounded limit, and the
old snapshot should remain available to other concurrent requests throughout.
The prototype must measure elected-request p95/p99 latency, not only ordinary
warm requests.

Normal HTTP cache headers and ETags can continue to cache plots and API
responses in browsers and intermediaries.

## Freshness Budget

The scheduled model changes refresh precision and must not silently weaken the
headline current-conditions data. Today the one-second manager loop attempts a
live-temperature refresh almost immediately after its ten-minute expiration.
A scheduled updater can discover that expiration as much as one full job cadence
later. The web then observes publication on its next request-driven manifest
check.

Approximate worst-case live-temperature age is:

```text
provider publication lag
  + feed expiration interval
  + scheduler alignment delay
  + updater execution/publication time
  + active web-instance manifest-check delay
```

Using the current assumptions:

| Component | Current loop | 10-minute job | 5-minute job |
| --- | ---: | ---: | ---: |
| Typical upstream observation lag | ~5 min | ~5 min | ~5 min |
| Live feed expiration | 10 min | 10 min | 10 min |
| Maximum scheduler/loop alignment delay | ~1 sec | ~10 min | ~5 min |
| Job and publication | n/a | to measure | to measure |
| Busy-instance manifest detection | n/a | bounded by incoming requests/check interval | bounded by incoming requests/check interval |
| Approximate age before final overhead | ~15 min | ~25 min | ~20 min |

These are conservative planning bounds rather than promised latency. They show
that a ten-minute job may be too coarse for live temperature unless scheduling
uses the persisted due time intelligently or the product accepts the additional
age. The existing health rule considers a ten-minute live feed unhealthy after
an additional 15-minute buffer, so a ten-minute job could sit directly on that
boundary and cause monitoring flaps.

A concrete ten-minute-cadence option avoids stacking a full ten-minute
expiration and a full ten-minute scheduler delay: configure the scheduled live
temperature refresh interval at or below the job cadence and treat it as due on
each execution. Live temperature is then fetched once per location per ten
minutes—the same intended rate as today—and worst-case age becomes roughly
provider lag plus one job cadence plus measured publication/manifest overhead
(approximately 15–17 minutes under current assumptions). This preserves
upstream courtesy while potentially avoiding the cost of a five-minute job.
Expiration and due-time comparison semantics must be tested at exact schedule
boundaries so a job does not accidentally skip every other execution.

The current 1/2/5/10/20/30-minute feed retry ladder is also quantized by job
cadence. With a ten-minute scheduler, the first three retry delays effectively
become "next job run." Persisted `_next_fetch_after` remains authoritative, but
the job cannot act before it starts. Cadence selection must therefore consider
both freshness and retry responsiveness, not just cost. A prototype must measure
complete execution time and validate freshness budgets before production
cutover.

## Cost Expectations for the GCP Reference Deployment

These are planning estimates, not a quote.

Cloud Storage is expected to be the smaller cost:

- Standard regional storage is on the order of cents per GB-month.
- Tens of thousands of manifest reads cost cents.
- A few thousand update publications per month should remain inexpensive if
  object count is controlled.
- Same-region Cloud Run access should avoid material network-transfer cost.

The scheduled Job is likely to dominate incremental cost. Approximate examples
for a job running every ten minutes:

| Job allocation and duration | Approximate monthly compute |
| --- | ---: |
| 1 vCPU for 30 seconds/run | $3 |
| 1 vCPU for 60 seconds/run | $5–6 |
| 4 vCPU for 30 seconds/run | $10 |
| 4 vCPU for 60 seconds/run | $20 |

Actual duration, free-tier treatment, memory, startup, retries, and skipped work
will affect the result. Fetching is largely network-bound and current plot
benchmarks are fast, so 1–2 vCPUs may be sufficient. A prototype must measure
complete update-cycle duration before selecting production resources.

## Failure Semantics and Operations

The redesign should make alerts correspond to actionable boundaries.

Metric contracts, dead-man switches, paging policy, and the GCP reference
implementation are defined separately in
[Observability Design](OBSERVABILITY_DESIGN.md).

Page-worthy examples:

- The updater crashes because of an application defect.
- No valid snapshot has been published beyond an agreed freshness threshold.
- Publication validation or atomic promotion fails.
- The web tier cannot load any valid current or previous snapshot.
- Unexpected web `500` responses occur.

Non-page or warning examples:

- One upstream station is temporarily unavailable while other feeds publish.
- A known source stops returning new observations and archived data remains.
- A job retries a transient `429` or `5xx` successfully.
- A web instance continues serving the previous valid generation during refresh.

Health should report both service health and data freshness without equating one
missing upstream series with total application failure.

Overlapping updater executions should initially be prevented through scheduler
or job-runner policy where convenient, but correctness must not depend on that
configuration. Conditional manifest promotion and the "abort if current changed
since job start" rule make publication safe if an overlap nevertheless occurs.

## Migration Strategy

The migration should be incremental and reversible.
The cross-design order—minimal instrumentation first, then instrumented archive
and snapshot work, followed by monitored cutover—is maintained in
[Observability Design](OBSERVABILITY_DESIGN.md#cross-design-implementation-order).

### Phase 1: Begin Durable Observation Capture

- Define the versioned normalized observation schema and stable source identity.
- Implement filesystem, memory, and initial production archive stores.
- After each successful fetch, let the existing long-running updater merge new
  observations into partitioned Parquet using conditional read-merge-write
  retries, because several serving processes may write concurrently during this
  phase.
- Begin with short-retention temperature sources if implementing every
  observational feed would delay first capture.
- Validate overlap, correction, provenance, and deduplication behavior.

The production web service still uses its current in-memory updater and does not
read the archive. This independent workstream starts preserving time-sensitive
history without waiting for the serving split.

### Phase 2: Define and Publish Snapshots

- Define versioned feed, per-year historical cache, status, and plot
  serialization.
- Export and round-trip snapshots in tests, verifying query/API equivalence.
- Let the existing updater publish content-addressed objects and immutable
  manifests after successful updates.
- Observe object sizes, publication frequency, latency, and cost.

Serving still uses the existing in-process state, so this phase does not put the
new store on the user path.

### Phase 3: Read-Only Web Mode

- Add readiness-blocking snapshot load and request-piggybacked generation
  refresh to the web process.
- Run it in shadow/validation mode against existing manager results.
- Compare API responses, freshness, and plots.
- Switch production serving to snapshot-backed state with rollback available.

### Phase 4: Scheduled Job

- Extract one bounded update cycle from the existing orchestration.
- Deploy the shared image with the updater entry point as a scheduled bounded
  execution (a Cloud Run Job in the GCP reference deployment).
- Disable the web process's background acquisition after successful validation.
- Remove plot watchdog behavior that is no longer relevant to the web tier.

### Phase 5: Use the Archive for Incremental Fetching

- Restore the per-year historical cache and timestamps from archived objects.
- Fetch only missing years, the current year, and configured overlap windows.
- Expand durable capture to any observational feeds deferred in Phase 1.
- Measure and eliminate unnecessary full-history cold-start requests.

Archive writes begin early because delay can permanently lose observations.
Making the archive authoritative for incremental reads remains later because it
requires stronger migration and equivalence validation.

## Testing Strategy

- Serialization round trips preserve indexes, timezones, values, source
  metadata, and missing-data semantics.
- A snapshot produced from existing feed objects yields equivalent API/query
  results after reload.
- Historical per-year cache frames and fetch timestamps survive reload, and
  completed past years are not refetched.
- Tide/current derived prediction frames are reconstructed after load and do not
  depend on stale process-local object identities.
- Publication is atomic under injected failures between object writes.
- A no-change updater execution writes no serving generation, while changed
  manifests reuse unchanged content-addressed objects.
- Web instances keep serving an old generation while a new one is incomplete or
  invalid.
- Manifest checks coalesce under concurrent requests and do not depend on idle
  background CPU.
- Concurrent publishers cannot move `current.json` backward.
- Generation identifiers remain unique when publishers start in the same
  second.
- A publisher may reuse unchanged objects only from the base generation it
  loaded at start; newly written objects remain protected by the GC safety
  window until promotion completes.
- Mark-and-sweep cannot delete objects referenced by the active/retained base of
  an in-flight publisher or by any retained manifest.
- Upstream unavailability retains last-known-good observations and records
  current source status.
- Incremental archive fetches merge revised overlap records without duplicates.
- Lifecycle rules preserve the active generation.
- Local mode retains the current one-command development experience.
- Live integration tests continue validating upstream contracts separately from
  persistence behavior.

## Open Questions

1. Is Cloud Storage alone sufficient, or does mutable metadata justify
   Firestore?
2. What is the measured serialized size of every current feed and plot?
3. Is any combined per-location historical serving frame large enough to need
   finer partitioning than one object per feed?
4. What refresh interval and due-time policy satisfy the explicit freshness and
   retry budget at acceptable job cost?
5. How long should previous published generations and raw responses be retained?
6. Which normalized observations are authoritative when an upstream provider
   revises or deletes records?
7. Should historical archival begin with temperature only or every observational
   feed?
8. What snapshot freshness threshold should page the operator?
9. Can the updater reliably run with 1 vCPU, and what is its measured complete
   execution time?
10. How should schema migrations keep at least one previously published snapshot
    readable during rolling deploys?

## Decision Checkpoints Before Implementation

Before building the production path, align on:

- Persistence product and regional placement
- Snapshot format and object granularity
- Observation-retention and raw-response policy
- Job cadence and CPU/memory allocation
- Freshness and paging policy
- Local runtime behavior
- Migration rollback points

After those decisions, this proposal can be converted into smaller
implementation plans with explicit compatibility and rollout criteria.

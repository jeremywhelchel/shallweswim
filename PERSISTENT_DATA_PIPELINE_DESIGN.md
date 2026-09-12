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

The initial GCS adapter uses the official synchronous `google-cloud-storage`
client with every storage operation offloaded to a worker thread. This keeps
blocking network I/O off the serving event loop during the transitional web
writer phase without adding a separate asynchronous GCS client dependency.

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

Phase 1 capture is implemented for temperature observations end to end;
production validation is still pending deployment of the archive bucket and
runtime access. Observational currents capture is implemented as well: it binds
the same scalar-observation schema, capture path, and
reader code to `velocity` and `kt` under a `currents/` path prefix. Prediction
feeds never enter the observation archive; capture fires only for currents
sources whose configured `source_type` is `OBSERVATION`.

### Temperature Archive Contract

Temperature archive objects use this layout:

```text
archive/
  temperature/
    <provider>/
      <station-or-station-param>/
        <year>.parquet
```

The provider and station path components are derived from the source's
`citation_key` in a deterministic, path-safe form without colons. The path does
not repeat `temperature` in the source component. It does not contain a location:
stations are physical sources, while locations are application configuration.
Two locations that use the same station share one archive partition, and the
source-to-location mapping remains in `config/locations.py`.

The capture hook percent-encodes each path component, preserving the source
identity reversibly. For example, `nwis:temperature:08155500:00010` maps to
`archive/temperature/nwis/08155500%3A00010/<year>.parquet`. Partitions use the UTC
observation year, which can differ from the feed's local calendar year.

`citation_key` is the permanent archive source identity. A golden-list contract
test snapshots every configured source's `citation_key`. Any identity change
must therefore fail CI and require an explicit archive-migration decision rather
than silently creating, abandoning, or conflating archive partitions.

Each temperature Parquet row has exactly these four columns initially:

| Column | Type | Contract |
| --- | --- | --- |
| `observed_at` | timestamp | UTC observation time |
| `value` | float64 | Normalized temperature value |
| `unit` | string | Canonical value `F`; dictionary-encoded in Parquet |
| `retrieved_at` | timestamp | UTC retrieval time for the fetch that supplied this row |

This row shape is the shared scalar-observation contract rather than a
temperature-specific schema. Each supported measurement type declares its feed
value column and canonical archive unit once as named production constants,
pinned by a contract test. Temperature binds the shared mechanism to
`water_temp` and `F`; observational currents bind it to `velocity` and `kt`.
Readers validate rows against the expected unit at the read/write boundary; the
Pandera model itself does not hardcode `F`.

The unit is stored per row because archive files must remain self-describing for
export, while custom Parquet metadata is not reliably preserved by third-party
rewrite tools. Dictionary encoding makes the repeated canonical value
negligible in practice. Retrieval time is also per row because a partition
contains observations from many fetches after its first merge.

Feeds currently store naive location-local timestamps. The archive writer must
convert `observed_at` to UTC at the write boundary using the location's
configured timezone; `retrieved_at` must also be stored as UTC. Daylight-saving
fall-back times require explicit handling: the conversion must never silently
choose one occurrence of an ambiguous local time. If the correct fold cannot be
inferred from the ordered observations, archive capture fails for that update
rather than writing a potentially corrupt deduplication key. Tests must include
a fall-back transition with the repeated 1 AM hour and cover both resolvable and
unresolvable ambiguity. A nonexistent local time in the skipped spring-forward
hour likewise fails archive capture rather than allowing the timezone library to
guess or shift it.

No quality column exists initially because no current client surfaces quality or
provisional flags; an all-null column would preserve no information. When a
client is changed to surface quality flags, add a nullable quality column in the
same change. Old files read as null, which is the honest value.

### Schema Evolution

Archive paths have no version prefix. The archive uses additive schema evolution:

- Archive objects are never rewritten solely for schema migration.
- Every schema change must be additive: a nullable column or a new path prefix.
- If a change cannot be represented that way, existing unmarked prefixes retain
  their original schema, new data goes to an explicitly named new prefix, and
  the archive reader learns to read both.
- All archive reads go through one reader helper. It normalizes frames by adding
  missing newer columns as null and validates the result against the Pandera
  model. Application code must not scatter direct `read_parquet` calls.
- A contract test pins column names, dtypes, and UTC timestamp semantics so every
  schema evolution is a conscious, reviewed diff.

### Merge Semantics

The deduplication key is stable source identity plus `observed_at`. Location is
not part of the key. Within a deduplication key, the row with the newest
`retrieved_at` wins. This makes overlapping fetches, upstream corrections, and
repeated merges deterministic and idempotent. Provider deletion or omission of
an observation never deletes an archived row; preservation is the archive's
purpose.

Any archive-capture failure—including timestamp conversion, exhausted
conditional-write retries, or storage unavailability—is isolated from the
serving update: it emits a structured archive-merge event with `outcome=failed`
but leaves feed publication, serving, and feed scheduling untouched, so a later
overlapping fetch can recover the omitted rows. A conflicting equally recent
claim is an expected, self-recovering anomaly and logs at WARNING; unexpected
archive failures log at ERROR. The failed-outcome metric captures both.

Individually validated historical years can be archived even if the combined
historical publication subsequently fails validation; preserving those valid
observations does not depend on a successful combined publication.

The updater should fetch incrementally with a small overlap window, then merge
using these rules. The overlap allows providers to revise recent readings.

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

Raw responses are not retained in Phase 1. Normalized observations are expected
to be long-lived.

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

Status: implementation complete, including the capture job entry point and
deployment definitions; production deployment pending.

Implemented (schema/stores/merge/capture commits through "Capture
observational currents in the archive"; local commits, not yet pushed):

- Additive normalized observation schema with `citation_key` formalized as the
  permanent archive source identity (golden-list contract test).
- Filesystem, memory, and GCS archive stores with conditional
  read-merge-write retries.
- Capture wired into `Feed.update()`'s success path behind
  `SHALLWESWIM_ARCHIVE_BUCKET` (unset everywhere, so capture is dark), with
  archive failures fully isolated from feed publication and scheduling.
- Temperature and observational-currents capture, UTC conversion with
  daylight-saving handling, overlap/correction/deduplication behavior — all
  covered by unit tests. Prediction feeds never archive.
- The bounded `shallweswim.capture` entry point, the Cloud Run Job and Cloud
  Build definitions, the operator runbook for the dedicated job identities and
  schedule, and job-inclusive log-based metric filters, per the contract below.

**Deployment sequencing revision (2026-09-12):** capture will NOT be enabled in
the multi-instance web service. Instead, the first production writer is an
isolated bounded one-shot capture job — a miniature of the Phase 4 updater —
running on a schedule under a dedicated job identity with access to only the
archive bucket. The web runtime identity gets no archive access, and the web
service never sets `SHALLWESWIM_ARCHIVE_BUCKET`. The capture hook is
host-process-agnostic, so the job reuses it unchanged; the transitional
multi-writer CAS path remains as overlap-safety for job runs and for
local/filesystem use. Validation: monitor `archive.merge` outcomes on the
operations dashboard for at least a week and compare archived row counts with
live feeds before anything reads the archive.

#### Phase 1 Capture Job Contract

The job is an ordinary bounded command with no dependency on a job-runner API:

```bash
uv run python -m shallweswim.capture              # scheduled run
uv run python -m shallweswim.capture --full-history  # one-time backfill
```

It is a temporary entry point. The Phase 4 `shallweswim.update` command absorbs
it once snapshot publication exists; `shallweswim.capture` then retires rather
than becoming a second long-lived updater.

Scope and behavior:

- `SHALLWESWIM_ARCHIVE_BUCKET` is required. An unset variable is a
  configuration error that fails the run before any upstream request, because
  fetching without capturing is the job's only purpose.
- The job fetches only archivable feeds: live temperatures, historical
  temperatures, and currents whose configured source type is observation.
  Tide feeds and prediction currents are never fetched by the job.
- Feed construction is shared with the web manager through one module-level
  builder so both hosts create identical feeds from the same configuration.
  A fresh feed is always expired, so one `update()` per selected feed is the
  whole cycle; expiration intervals stay authoritative only for the web host.
- Historical temperatures default to the current UTC year only. A cold
  process otherwise refetches every configured year (fifteen for NYC) on each
  run, which is write-once data the archive already preserves after its first
  capture. `--full-history` fetches the configured full range for the
  migration backfill and for occasional re-sweeps; it is not scheduled hourly.
- Locations run concurrently and each location's feeds run sequentially,
  matching the web host's per-location tasks and the shared provider gates.
- The job never generates plots, precomputes derived frames, or starts
  FastAPI; it needs no process pool.

Failure semantics:

- Station unavailability is WARNING and unexpected feed errors are ERROR,
  exactly as in the web host; either leaves the remaining feeds running.
- Archive failures stay isolated inside the capture hook and surface only as
  `archive.merge` events with `outcome=failed`.
- The run emits one summary event, `component=updater operation=run`, with
  bounded `outcome` values `success` (every selected feed published),
  `partial` (some feeds published), or `failed` (none published or the run
  crashed), plus `duration_ms` and `record_count`. `run_id` carries the
  platform execution name when present.
- The exit status is non-zero only for `failed`. A `partial` run exits zero so
  a single bad station does not trigger platform retries that refetch every
  other feed; those failures are visible through the feed-update metric and
  the summary event instead. The platform retry bound is one.

Reference deployment (GCP):

- `capture-job.yaml` beside `service.yaml` defines a Cloud Run Job from the
  same image with the command overridden, a single task, a twenty-minute task
  timeout, one retry, `SHALLWESWIM_LOG_FORMAT=json`, the USGS API key secret,
  and `SHALLWESWIM_ARCHIVE_BUCKET` substituted at deploy time from the
  operator environment so the repository never hard-codes an installation's
  bucket name.
- Cloud Build replaces the job after every service deploy so the job never
  runs a stale image. The build fails if the bucket substitution is empty.
- Identities: `shallweswim-capture` runs the job and holds only
  `roles/storage.objectUser` on the archive bucket plus accessor on the USGS
  key secret; `shallweswim-capture-invoker` holds `roles/run.invoker` on the
  job only and is the Cloud Scheduler OIDC identity; the build identity gains
  `roles/iam.serviceAccountUser` on `shallweswim-capture`. The web runtime
  identity receives no bucket binding.
- Cloud Scheduler triggers the job hourly at a fixed minute offset. Overlap is
  prevented by scheduler policy but correctness does not depend on it.
- Log-based metric filters include the job resource type alongside the
  service, and the archive dashboard chart is not restricted to the service,
  so job events are visible. Alert policies remain service-scoped until the
  job has run in production; a job dead-man alert is a follow-up.

The production web service keeps its current in-memory updater and does not
read the archive.

### Phase 1b: Local Development Reads the Archive

Once the capture job has populated the bucket, local development of the web
service may hydrate historical temperature feeds from the archive instead of
performing the full multi-year cold-start refetch. This is Phase 5's restore
path scoped to development first: it makes local startup fast and makes the
developer machine the archive's first read consumer, validating archived data
quality before production depends on it. Guardrails: reads use a separate
`SHALLWESWIM_ARCHIVE_READ_BUCKET` variable so no local configuration can
enable writes, and the local credential holds read-only bucket access.

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
- Temperature archive contract tests pin the four initial columns, dtypes,
  canonical unit, and UTC timestamp semantics.
- A golden-list contract test pins every configured `citation_key` used as an
  archive source identity.
- Every archive read uses the normalizing reader helper; older additive schemas
  load missing newer nullable columns as null.
- Daylight-saving fall-back tests prove that the repeated local 1 AM hour is
  converted without silently conflating observations, and ambiguous input that
  cannot be resolved fails archive capture.
- Lifecycle rules preserve the active generation.
- Local mode retains the current one-command development experience.
- Live integration tests continue validating upstream contracts separately from
  persistence behavior.

## Open Questions

Answered by measurement (`shallweswim.scripts.measure_feed_sizes`,
2026-09-12): the largest serving object is a combined per-location historical
temperature frame at roughly 1.3 MiB (NYC, 15 years hourly), with per-year
slices of 60–90 KiB and a full location's feeds plus plots totaling under
3 MiB. One object per feed per location is comfortably sufficient; no finer
serving partitioning is justified by size.

1. Is Cloud Storage alone sufficient, or does mutable metadata justify
   Firestore?
2. What refresh interval and due-time policy satisfy the explicit freshness and
   retry budget at acceptable job cost?
3. How long should previous published generations be retained?
4. What snapshot freshness threshold should page the operator?
5. Can the updater reliably run with 1 vCPU, and what is its measured complete
   execution time?
6. How should schema migrations keep at least one previously published snapshot
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

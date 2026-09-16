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

Clients return timezone-aware UTC frames and feeds publish naive
location-local frames for serving, so the archive writer receives UTC
instants directly: `observed_at` is the client instant converted to UTC with
no wall-time interpretation, and `retrieved_at` is always stored as UTC. A
naive index reaching the archive is a defect and fails capture rather than
being localized.

Historical temperature capture archives each freshly fetched year at the
provider's native cadence, from the per-year UTC frame before the serving
resample. Resampling to hourly is a serving concern and must not precede
capture.

#### UTC Client Frames

Measured against live 2025 provider data, the transitional naive path still
loses the fall-back hour everywhere: NDBC and NWIS clients sort by wall time
after converting, which interleaves the two folds so inference cannot separate
them, and the CO-OPS local-time products omit one fold upstream. The same
duplicate wall times make a live NDBC or NWIS window that spans a fall-back
day fail the serving model's unique-index rule, so those live feeds go stale
for about a day every November. Both defects have one cause: absolute
instants are discarded before either consumer sees them. The application
therefore standardizes on UTC at the client boundary:

- Every client method that returns a time-indexed frame (CO-OPS tides,
  currents, and temperature; NDBC temperature; NWIS temperature and
  currents; CSPF Sandettie; Irish Lights; Marine Institute tides) and the
  local harmonic tide feed return frames indexed by timezone-aware UTC
  instants, whatever the provider natively speaks. CO-OPS requests use
  `time_zone=gmt` so both folds and the spring-forward hour are exact.
  Clients de-duplicate on the UTC instant, never on wall time. Client
  `timezone` parameters remain only where a request window must be expressed
  in station-local time; they no longer drive output conversion.
- Feeds own the serving derivation in one shared step before validation:
  convert the UTC frame to the location timezone, drop the timezone, and
  collapse any repeated wall time by keeping the first occurrence in instant
  order. Published frames keep today's contract exactly: naive local, unique,
  monotonic. Queries, plots, the API, and the historical resample are
  untouched. A naive frame passes through the step unchanged, because
  composite feeds publish frames their member feeds already converted.
- Capture receives the UTC frame. `normalize_observations` requires an aware
  index and converts it directly; the former fold inference,
  `ambiguous_dropped` event, and naive input path no longer exist. The
  `conflict_dropped` rule for repeated instants stays.
- Historical per-year frames are captured in UTC at native cadence, then
  converted for the serving resample, so both folds reach the archive for
  every source.
- The archive stores provider readings unfiltered. Configured outlier removal
  applies to the serving frame only, so a known-bad reading is absent from
  what the site serves but present in the archive; a future quality column,
  not deletion, is how such readings would be marked.
- After deployment, one `--full-history` execution back-fills the fold rows;
  they appear as `new_count` on the merge metrics, roughly 8 to 12 rows per
  year per source.

Native-cadence provider frames may repeat an instant. After conversion, rows
that share a UTC instant collapse to the first occurrence; the two fall-back
folds are distinct instants and are never collapsed. An identical repeat is
silent. A repeat whose value differs is still dropped but emits the same
WARNING event with `outcome=conflict_dropped` and the dropped count, so a
provider anomaly is visible without failing the partition.

Historical temperature capture archives each freshly fetched year at the
provider's native cadence, from the per-year frame before the serving
resample. Resampling to hourly is a serving concern and collapses the repeated
fall-back hour, so it must not precede capture. Sources whose hourly product
already omits one fold (NOAA CO-OPS hourly history) lose that single row per
year under the drop rule until the client fetches history in UTC, which is a
separate client change.

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
not part of the key. Within a deduplication key, merging is value-aware:

- A key absent from the partition is a **new** row and is added as fetched.
- A key already present with an identical `value` is an **overlapping** row.
  The stored row is kept unchanged, including its original `retrieved_at`, so
  the column records the fetch that first supplied the observation and
  repeated fetches of unchanged data do not rewrite the partition.
- A key already present with a different `value` is a **revised** row. The row
  with the newest `retrieved_at` wins, so upstream corrections replace earlier
  readings deterministically and idempotently.
- Two rows with the same key, the same `retrieved_at`, and different values
  are an integrity conflict and fail the merge.

A merge whose incoming rows are all overlapping leaves the partition
byte-identical and reports `outcome=unchanged`; only new or revised rows
produce a write. Provider deletion or omission of an observation never deletes
an archived row; preservation is the archive's purpose.

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

### Merge Event Contract

Every partition merge emits exactly one completion event, `component=archive`,
`operation=merge`, carrying the source identity, a bounded `outcome`
(`success`, `unchanged`, `failed`), `duration_ms`, `attempt_count` (conditional
write attempts), and these integer row counts:

| Field | Meaning |
| --- | --- |
| `record_count` | Rows in the partition after the merge |
| `incoming_count` | Validated rows the fetch supplied for this partition |
| `new_count` | Incoming rows whose key was absent from the partition |
| `overlap_count` | Incoming rows identical to a stored row |
| `revised_count` | Incoming rows that replaced a stored value |

`incoming_count` equals the sum of the other three on any non-failed merge; a
failed merge reports the counts computed so far and zero for the rest. The
counts are bounded-cardinality numeric fields, never labels.

The run summary event (`component=updater`, `operation=run`) additionally
reports `new_count` and `revised_count` summed across every merge in the run,
so one event per run answers whether the run added anything to the archive.

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

### Generation Garbage Collection Contract

Status: implemented in `shallweswim/snapshot/gc.py`, run by the publishing
cycle. On 2026-09-14, before any collection, the published prefix held 497 MB
in 1,249 objects across 27 hourly generations; at the ten-minute cadence a run
adds a small generation every ten minutes, so object count is the reason to
collect, not cost.

Retention:

- The current generation is kept unconditionally.
- Every generation published within the last 24 hours is kept
  (`RETAINED_GENERATION_AGE`, a named constant). That is the rollback window
  and the window a slow instance could still be loading from.
- Older manifests are deleted. `current.json` is never deleted.
- An object is deleted only when no retained manifest references it and it
  is older than a one-hour safety window (`OBJECT_SAFETY_AGE`). The window
  protects a publisher that has written objects for a generation it has not
  promoted yet; publication takes under two minutes.

Where and when it runs:

- Inside the job's publishing cycle, after publication, every run. It is
  bounded, one listing of manifests and one of objects, and its failure is
  isolated exactly like publication: the run's outcome and exit code are the
  capture cycle's. The local entry point runs it too, so a store directory
  does not grow without bound.
- It reads the current pointer first and treats that generation as retained
  whatever its age, then lists manifests and keeps those inside the retention
  window, then lists objects and deletes those unreferenced by any retained
  manifest and older than the safety window. It never deletes anything it did
  not list in that same run.

Store interface: the object store protocol gains `list(prefix)` returning
each key with its creation time, and `delete(key)`, implemented by the
memory, filesystem, and GCS stores alike so the portability principle holds
(the memory store records insertion time; the filesystem store uses the
file's modification time). A delete of a key that no longer exists is not an
error.

Event: one per run, `component=snapshot operation=gc`, `outcome` in
`success|failed`, `duration_ms`, `record_count` as objects deleted, and the
message naming manifests deleted and objects examined. A log-based counter
by outcome and one dashboard tile follow in Terraform. A failed collection
logs at ERROR: it means the store misbehaved or the sweep's own invariant
failed, and nothing else will notice.

One departure from the sketch above, taken for safety: a manifest the sweep
cannot parse is retained, as stated below, and because its object references
are then unknown that run deletes manifests but no objects at all. Otherwise a
corrupted current manifest — which a publisher would also have failed on —
would make the live generation's objects look unreachable.

Safety, pinned by tests:

- No object referenced by the current generation or any retained manifest is
  ever deleted, including objects the current generation reuses from an older
  one.
- An unreferenced object younger than the safety window survives; the same
  object older than it is deleted once no retained manifest names it.
- A publisher that wrote objects and a manifest but has not promoted yet loses
  nothing to a sweep that runs in between: its manifest is inside the
  retention window and its objects inside the safety window.
- A manifest older than the window that is still current is kept.
- The three store implementations list and delete identically.
- A store failure during the sweep leaves the run outcome unchanged and
  produces one failed gc event.

Out of scope: archive partitions, which are never deleted; lifecycle rules on
the bucket, which stay off; a standalone maintenance command, which is not
needed while the job runs the sweep every ten minutes.

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
- The bounded `shallweswim.update` entry point, the Cloud Run Job and Cloud
  Build definitions, the operator runbook for the dedicated job identities and
  schedule, and job-inclusive log-based metric filters, per the contract below.

**Deployment sequencing revision (2026-09-12):** capture will NOT be enabled in
the multi-instance web service. Instead, the first production writer is an
isolated bounded one-shot capture job — a miniature of the scheduled updater the cutover contract completes —
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
uv run python -m shallweswim.update              # scheduled run
uv run python -m shallweswim.update --full-history  # one-time backfill
```

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

Status: implemented; local validation against the production archive
pending the viewer-only credential.

Once the capture job has populated the bucket, local development of the web
service hydrates historical temperature feeds from the archive instead of
performing the full multi-year cold-start refetch. This is Phase 5's restore
path scoped to development first: it makes local startup fast and makes the
developer machine the archive's first read consumer, validating archived data
quality before production depends on it.

Configuration and guardrails:

- `SHALLWESWIM_ARCHIVE_READ_BUCKET` names the bucket to hydrate from. It is
  independent of `SHALLWESWIM_ARCHIVE_BUCKET`, which remains the only switch
  that enables writes and which local configuration leaves unset. The
  hydration path calls only the store's read operation and never constructs a
  writer, so setting the read variable cannot cause a write. `.env.example`
  documents the read variable; `service.yaml` never sets it, and
  `capture-job.yaml` sets it only because the job publishes snapshots, both
  pinned by the deployment-manifest test.
- The local credential holds `roles/storage.objectViewer` on the bucket, so
  even a misconfiguration that set the write variable locally would fail at
  the bucket. The runbook replaces the temporary local write grant with the
  viewer role.

Behavior:

- When the read variable is set, `HistoricalTempsFeed` hydrates before its
  first provider fetch: for every required year before the current UTC year
  that is not already cached, it reads the partitions that cover that
  station-local year (the UTC year and the following one) through the archive
  reader and keeps the rows whose local time falls inside the year, so a
  hydrated year spans exactly what a provider fetch of that local year would.
  Years hydrate concurrently under a small bound. Rows become a per-year frame in the client shape (a
  timezone-aware UTC index named `time` and the feed's value column) and then
  follow exactly the provider path: serving index derivation, the hourly
  resample, validation, and the year cache with the hydration time as the
  year's fetch timestamp. A served frame built from archived rows is therefore
  identical to one built from a provider fetch of the same year, which a test
  proves against a fixture year.
- Years absent from the archive, and the current year, fetch from the
  provider as today. Hydrated years are never re-captured; capture applies
  only to years fetched from the provider.
- Hydration never fails startup. A read or validation failure for a year is
  logged and that year falls back to the provider fetch.
- The partition key is derived by one shared function from the source
  identity, measurement, and UTC year, used by both capture and hydration.
- One INFO event per hydrated feed (`component=archive`,
  `operation=hydrate`, `location`, `feed`, `outcome=success` or `failed`,
  `record_count` = archived rows loaded) records what the archive supplied;
  a failure event carries the years that fell back.

Scope: temperature history only. Live temperature and observational currents
feeds keep their short provider windows and tide feeds never archive. The
capture job sets the read variable only when it publishes snapshots (Phase
2), because a snapshot carries the full historical range and the builder
restores past years from the archive rather than refetching them; archive
capture alone never hydrates. Broader archive-driven incremental fetching is
Phase 5.

### Phase 2: Define and Publish Snapshots

Status: implemented; the hourly job has published a generation after every
run since 2026-09-13. Observed so far: a bundle of about 21 MiB in 56
objects and a manifest of about 21 KB per generation. Because the job is a
fresh process, every feed refetches on every run and its fetch timestamp
changes, so the `unchanged` outcome is not reachable until the updater
persists feed state (the cutover contract makes the manifest that state);
every run writes a new manifest plus whichever objects changed. The
carry-forward rule below is implemented and deployed.

Phase 2 makes serving state serializable and publishes it from the capture
job so object sizes, publication cost, and manifest semantics are observed in
production before anything reads them. Serving still uses the in-process
state; nothing in this phase is on the user path.

#### Snapshot Contents

A snapshot generation describes every enabled location's serving state:

- One Parquet object per published feed per location (`tides`, `currents`,
  `live_temps`, `historic_temps`), holding the served frame exactly as the
  feed publishes it: the naive station-local `time` index becomes a `time`
  column, value columns keep their dtypes, and categorical columns (tide
  `type`) are restored to their documented categories on load. The frame is
  the only feed state serialized; a loaded frame passes the same Pandera
  model the feed validated on publish.
- One SVG object per plot per location (`live_temps`,
  `historic_temps_2mo`, `historic_temps_12mo`), generated by the publisher in
  its process pool and awaited before publication.
- Per-feed metadata in the manifest, not in objects: source identity
  (`citation_key`), fetch timestamp, next fetch time, expiration interval,
  record count, consecutive failures, last sanitized error, and the
  historical year diagnostics the status endpoint already exposes.

The per-year historical cache is deliberately not serialized. The bundle
builder never depends on a previously produced snapshot for content: every
generation is assembled from the providers and the archive alone, with past
years restored through Phase 1b's hydration. The current generation is
consulted only for bookkeeping, to skip rewriting objects whose
content-addressed key already exists and to refuse to move the current
pointer backward. If every previous generation vanished, the next run would
publish a complete, correct one.

#### Object Layout and Manifest

Objects live in the archive bucket under a separate prefix with its own
retention rules:

```text
published/
  current.json
  manifests/<published_at>-<run_id>.json
  objects/sha256-<digest>.parquet
  objects/sha256-<digest>.svg
```

Objects are content-addressed and immutable: the publisher writes an object
only if its key is absent, and an identical key already present is reuse.
The manifest is written after every referenced object exists, and
`current.json` is promoted last with the store's conditional replacement,
expecting the generation the publisher observed when it started; a newer
generation aborts promotion rather than moving the pointer backward.

`current.json` holds only the current manifest key and generation id. The
manifest holds `schema_version`, `generation_id` (`<published_at>-<run_id>`),
`published_at`, `previous_generation_id`, and per location, per feed and per
plot: the object key, its byte size, and the metadata above. Plots record the
fetch timestamp of the feed they were drawn from.

Publication is skipped, and a `snapshot.publish` event with
`outcome=unchanged` emitted, when every object key and every manifest field
except `published_at` equals the current generation. Otherwise the event
carries `outcome=success` or `failed`, `duration_ms`, and `record_count` as
the number of objects written.

#### Publisher

The capture job is the publisher, because it is the only production writer.
After its capture cycle it builds each location's serving state through the
same `LocationDataManager` machinery the web service uses, generates plots,
and publishes. This adds the process pool and plot generation to the job;
its measured duration and memory are Phase 2 observations. Publication is
enabled by `SHALLWESWIM_SNAPSHOT_PUBLISH=1` in the job definition only, is
isolated from capture like capture is from serving, and never runs in the
web service. When publishing, the job runs the full serving cycle for every
location (all feeds, including tide and prediction feeds, which archive
capture alone skips) with the historical range set to the full configured
years and `SHALLWESWIM_ARCHIVE_READ_BUCKET` set, so past years hydrate from
the archive and only the current year and any missing years reach the
provider. Hourly cadence is inherited from the job; the cutover contract moves it to
ten minutes.

Serialization and loading live in one module with a `Snapshot` model, a
`SnapshotStore` protocol with memory, filesystem, and GCS implementations
over the existing object store, and round-trip tests: a snapshot exported
from live feed objects loads into equivalent frames, plots, and status for
every configured feed type, including the historical frame and categorical
tide types. Garbage collection of old generations is Phase 3 scope; Phase 2
retains everything so its growth is measurable.

#### Carry-Forward of Failed Feeds

Status: implemented. Required before any web server reads the bundle.

Today a generation omits any feed that failed during that run, so one
transient provider failure removes last-known-good data from the bundle even
though the web process would have kept serving its previous frame. The
builder therefore reports every configured feed of every enabled location,
either as a served frame or as a failure. A failure records only what the
run learned: the consecutive failure count, the sanitized last error, and the
scheduled retry. A location is present in the snapshot even when every one of
its feeds failed. Feeds that are not configured for a location are absent,
as today.

Manifest assembly resolves each failed feed against the generation the
publisher observed when it started, the base generation:

- If the base generation has an entry for that feed whose `source_identity`
  equals the feed's configured `citation_key`, the new manifest copies that
  entry. Object key, size, fetch timestamp, record count, timezone,
  expiration, and historical diagnostics stay exactly as published, so the
  last-known-good frame keeps serving. Only the failure fields change:
  `consecutive_failures` becomes the base entry's count plus this run's,
  `last_error` becomes this run's error, and `next_fetch_after` this run's
  scheduled retry. The manifest thereby says both "the served frame was
  fetched at time X" and "the source is failing now", the distinction the
  archive section requires.
- If the base generation has no such entry, because the feed never succeeded
  or its source identity changed, the feed is omitted, as today.
- A plot absent from this run's build is copied from the base generation when
  present there and its source feed is in the assembled manifest, whether that
  feed was carried forward or was refreshed but its plot did not complete in
  time. The copied `feed_fetch_timestamp` states which feed state the plot
  shows. A plot whose feed is no longer published is dropped with it.
- A location or feed that is no longer configured is dropped. Configuration
  is authoritative and carry-forward never resurrects it.
- Carry-forward has no age limit. Retention is separate from alerting and
  from presentation, which the next paragraphs cover; keeping last-known-good
  data is cheap and losing it is irreversible.

Freshness is monitored from the manifest. After assembly, and whether or not
a new generation is promoted, the job emits one event per location and feed,
`component=snapshot operation=freshness` with `location`, `feed`,
`outcome=success|carried|absent`, and `age_seconds` as the age of the served
frame's fetch timestamp (a new approved numeric log field). A feed stuck on
carried-forward data therefore shows a growing age. A log-based metric and a
shadow alert with a per-feed-type threshold follow in Terraform; the
thresholds start from the existing health rule (expiration interval plus 15
minutes) and are tuned on the baseline before any policy notifies.

Whether the site keeps displaying stale data is a presentation policy,
decided per feed type with shadow data rather than guessed here. Two notions
of staleness apply: a prediction feed (tides, prediction currents) is stale
only when the requested time falls outside the fetched window, not by fetch
age; an observation feed (live temperature, observation currents) is stale by
the age of its latest observation. That decision is tracked in `TODO.md` and
is not a precondition for shadow mode.

Carried-forward metadata differs from the base entry, so a repeatedly failing
feed publishes a new manifest every run without writing any object; that is
the small-manifest case the snapshot model already allows. The `unchanged`
rule and conditional promotion are untouched, and assembly uses the same base
the publisher read for promotion, so a concurrent publisher cannot make a
carried-forward entry refer to a generation other than the one promotion is
conditioned on.

The first generation published after this change carries forward from
whatever the current generation holds; nothing needs migrating, and the
schema version stays 1 because no manifest field changes.

### Phase 3: Read-Only Web Mode

Two slices. The first, shadow mode, has its contract below; the second,
cutover, is outlined and receives its own contract once shadow data exists.

End state after both: web servers hold no upstream clients and no feed
objects. Each instance loads the current bundle generation before it is
ready, refreshes it on an elected request, and answers every API request from
memory. The job is the only process that talks to providers.

#### Shadow Mode Contract

Status: implemented and superseded by cutover; shadow mode ran from 15:12 to
19:39 UTC on 2026-09-14.

In shadow mode a web instance keeps fetching and serving exactly as today,
and additionally loads the bundle and keeps it current. Nothing on the user
path changes: every response, health check, and status field still comes
from the in-process managers. Shadow mode exists to prove, in production,
that the refresh mechanism works on request-scaled CPU and to measure how
far behind the job an instance runs. Whether bundle-backed serving is
equivalent to today's serving is answered separately by a local comparison
command (below), because a local process running the legacy stack against
the production bundle holds both sides in memory exactly as a production
instance would. The comparison therefore never runs in production, and the
web service carries no code that cutover would delete.

Configuration (implemented in `main.py`, `service.yaml`, and `cloudbuild.yaml`):

- `SHALLWESWIM_SNAPSHOT_READ_BUCKET` names the bucket whose `published/`
  prefix the web reads. Setting it enables shadow mode; unset, no snapshot
  code path is active and the web behaves as today. It is deliberately
  distinct from `SHALLWESWIM_ARCHIVE_READ_BUCKET` (archive hydration, which
  the web never does) and from `SHALLWESWIM_ARCHIVE_BUCKET` (writes, which
  the web never does). `service.yaml` sets it through the same Cloud Build
  substitution as the job's bucket, and the deployment-manifest test pins
  that the web sets this variable and neither of the other two.
- The web runtime identity gains `roles/storage.objectViewer` on the bucket.
  The runbook invariant that only the job writes is unchanged; the runbook
  gains the read grant. Reads use the existing GCS adapter, every call on a
  worker thread.
- Local development sets the same variable in `.env` to shadow against the
  production bundle with the viewer credential. That is the first validation
  step and happens before the service is deployed, and it is where the
  comparison command runs.

Serving state from a generation (implemented in `snapshot/manager.py` and
`core/serving.py`; built and held by the shadow state, and read by nothing
until cutover):

- A read-only per-location manager (`SnapshotLocationManager`) is
  constructed from a loaded generation: the location's frames, plot bytes,
  and manifest metadata. It computes the derived tide and
  current prediction frames once at construction; there is nothing to
  invalidate because the object is immutable and a new generation constructs
  new managers.
- It exposes the same route-facing surface as `LocationDataManager`:
  `has_data`, `has_feed`, `has_feed_data`, `get_feed_values`, `get_plot`,
  `status`, and the query methods. Routes are typed against one Protocol
  covering that surface, so at cutover the object behind
  `app.state.data_managers` changes and the routes do not. The query
  functions in `core/queries.py` accept any mapping from feed name to an
  object exposing `has_data` and `values`, which a feed and a snapshot feed
  both provide. That is the whole generalization, and this slice is its
  second use.
- Its `status` derives `age_seconds`, `is_expired`, and `is_healthy` from
  the manifest's fetch timestamp, expiration, and retry time with the feed
  rules, so bundle-served data has the same health semantics. In shadow mode
  this status is not exposed through `/api/status`.

Loading and refresh (implemented in `snapshot/load.py`, `snapshot/refresh.py`,
and the `main.py` lifespan and middleware):

- Startup: the lifespan loads the current generation with a bounded timeout
  of 20 seconds. In shadow mode a failure or timeout is logged and the
  instance starts without a generation; readiness is unaffected. At cutover
  the same load gates readiness.
- Refresh is request-piggybacked, as the runtime-components section
  requires. An HTTP middleware runs on every request, health checks
  included: if the check interval (60 seconds) has elapsed since the last
  check and no check is in flight, that request is elected and awaits the
  check before its handler runs. Concurrent requests skip the check and
  proceed. A failed check schedules the next one a full interval later,
  never sooner.
- A check reads `current.json`. Same generation as loaded: nothing else
  happens and nothing is logged above DEBUG. New generation: read its
  manifest, read only the objects whose keys are not already held (objects
  are content-addressed, so a held key is the same bytes) with bounded
  concurrency of 8, validate every frame through its feed model, construct
  the new managers, and swap them in with one assignment. Requests running
  during the swap finish on the generation they started with.
- The web never lists `published/manifests/` and never deletes anything.
- One structured event per load or refresh that does work:
  `component=snapshot operation=load outcome=success|failed`,
  `generation_id`, `duration_ms`, and `record_count` as the number of objects
  read, and `age_seconds` as the age of the loaded generation's
  `published_at` at load time, which is the lag between the job publishing
  and this instance picking it up. Severity follows one rule, ERROR means a
  human needs to look now,
  and the rule is the long-term one from the start: a startup that cannot
  load any generation is ERROR once the web depends on the bundle (nothing
  to serve) and WARNING during shadow, the only interim downgrade; a failed
  refresh while a generation is already loaded is WARNING permanently,
  because a sustained failure pages through the bundle-age alert rather than
  through log lines.

Comparison (a local command, never production; implemented in
`shallweswim/scripts/compare_snapshot.py`, with the pure comparison core unit
tested and the fetch-and-load plumbing exercised only by running it):

- `uv run python -m shallweswim.scripts.compare_snapshot` runs in a process
  that builds the legacy managers exactly as the web service does (fetching
  from the providers, hydrating nothing) and loads the current generation
  from `SHALLWESWIM_SNAPSHOT_READ_BUCKET`, then compares the two for every
  configured location at one location-local instant shared by both sides,
  and prints a report. It exits non-zero when any feed mismatches, so it can
  run in a loop for days and its exit status is the verdict.
- Per configured feed, the report states one outcome and its numbers:
  - `missing`: the legacy manager has data and the bundle has none. Expected
    only while the job's own fetch of that feed fails and nothing is carried
    forward.
  - `extra`: the bundle has data and the legacy manager has none. Expected
    when the local fetch failed.
  - `absent`: neither side has data for a configured feed. Expected only
    for a source that is down on both paths.
  - `disjoint`: both have data and their indexes share no timestamp; the
    bundle is too stale to compare.
  - `mismatch`: on the shared timestamps any value differs (floats compared
    with a relative tolerance of 1e-6, other columns exactly), or a derived
    answer differs: `get_tide_info_at_time(now)` and
    `predict_tide_at_time(now)` for tide feeds, `predict_flow_at_time(now)`
    for prediction current feeds, and `get_current_temperature()` for live
    temperature when both frames end at the same timestamp. The report lists
    the differing rows with both values.
  - `match`: everything above agreed.
  - Alongside the outcome: the number of shared timestamps compared, the
    number that differ and the largest difference, each side's first and
    last timestamp, and the gap between the two latest timestamps, which is
    the bundle's lag for that feed.
- Per location, which plots exist on each side. Plot bytes are not compared,
  because the two sides draw different fetch windows.
- Historical temperature frames should agree on overlap except where a
  provider revised a reading between the two fetches. Every such mismatch is
  investigated once; a recurring source-specific pattern is recorded in
  `TODO.md` before cutover rather than tolerated by loosening the rule. The
  known divergences of the archive-hydration path (NDBC's first year, CO-OPS
  fall-back rows) are expected to show up here and are decided then.

Visibility (implemented): no new route. The load events carry the platform's instance
identity, which answers "is every instance loading the bundle" better than
an endpoint can, because the load balancer sends a request to one arbitrary
instance. At cutover `/api/status` gains the loaded generation id and load
time, since it must describe bundle state then anyway.

Observability: Terraform adds log-based metrics on `snapshot.load` outcomes,
duration, and lag (`age_seconds`), one dashboard tile for outcomes and one
for lag, and a shadow alert on the lag of the loaded generation, which is the
signal that pages for sustained refresh failure after cutover. The existing
shadow alert set is promoted separately.

Out of scope for this slice: any change to what the web serves; removing
fetching from the web; any job change beyond carry-forward; a filesystem
snapshot store for the web, which is the local entry point's concern; and
garbage collection of old generations.

Rollback (implemented): unset the variable and redeploy, or redeploy the
previous revision. Shadow mode cannot change a response.

Measurements so far (2026-09-14, first day of shadow mode, hourly job):

- Cold load on a fresh instance: the 20-second startup bound was exceeded
  while the legacy stack was still fetching fifteen years of history and
  drawing plots on the same CPU; the first elected request two minutes later
  loaded 58 objects (22 MB) in 18.5 seconds. The same load from a
  developer machine takes 2.4 seconds, so the cost is startup contention
  with the fetching stack, which cutover removes.
- Warm refresh: the next generation was picked up 64 seconds after
  publication (60-second check interval), 46 changed objects loaded in 1.2
  seconds, by an elected user request whose total latency was 1.2 seconds.
- The local comparison command matched every feed for NYC and Boston except
  a few historic hourly rows per run that differ by 0.1°F, the provider's
  own rounding granularity, between separate from-scratch fetches. Nothing
  served depends on a from-scratch fetch, so this is noted and not tracked.
  A later run across all eleven locations matched everything except Cork's
  current hour, where Irish Lights serves a running hourly average that
  changes until the hour closes, which the archive's one-revised-row-per-hour
  pattern already showed.

Exit criteria before cutover is deployed (revised 2026-09-14 from a week to
what the evidence actually needs, since the mechanism proved itself on the
first day and the cutover build takes as long as the remaining wait):

- Forty-eight hours of production shadow with every load outcome `success`
  and the load lag alert never firing.
- One local comparison across every configured location with no `mismatch`
  other than the moving 0.1°F historic rows recorded in `TODO.md`, and every
  `missing` explained by a job-side feed failure that predates the bundle.
- Cold load duration, elected-request refresh duration, and the load lag
  distribution measured and recorded in this document (done above).

#### Local Entry Point Contract

Status: implemented. Required before cutover, because cutover removes fetching
from the web service and a fresh clone must keep working with no bucket and no
credentials.

`uv run python -m shallweswim.local` is the clone-and-run command. One
process runs the job's cycle against a local object store and serves the web
app from that same store. A developer gets the providers' full configured
history, plots, and every route, exactly as production serves them; the only
thing missing is the extra history the production archive has accumulated
beyond what the providers still expose.

Store selection:

- One helper resolves every store locator the application reads:
  `object_store(locator)` in `archive/store.py`. A bare name is a GCS bucket,
  as today; a path containing a slash is a `FilesystemObjectStore` rooted
  there; the literal `memory` is one process-wide `MemoryObjectStore`. The
  four places that build a store from an environment variable (capture,
  hydration, the publisher, the web loader) call this helper and change in no
  other way. The GCS branch keeps the per-bucket client cache. This is the
  filesystem store becoming the real portability target the Portability
  Principle promises, and the local entry point is its second use.
- The local entry point sets, in its own process, the three variables
  (`SHALLWESWIM_ARCHIVE_BUCKET`, `SHALLWESWIM_ARCHIVE_READ_BUCKET`,
  `SHALLWESWIM_SNAPSHOT_READ_BUCKET`) to the same locator, so every store the
  job half writes is the store the web half reads. The default locator is
  `memory`; `--store-dir PATH` selects a filesystem store, which persists the
  archive and generations across restarts so the second start hydrates
  history from disk instead of refetching fifteen years. Nothing is read
  from the operator's `.env`: the entry point overrides those three
  variables unconditionally, and never points at a bucket.

Process model:

- The entry point builds the FastAPI app through `start_app` with the same
  command-line options as `shallweswim.web` (host, port, frontend paths),
  and runs it with uvicorn in-process. `--reload` is not supported, because
  the store lives in the process.
- A lifespan task runs the job cycle: the publishing path of
  `shallweswim.update` (every location's serving cycle, capture, plots, and
  one generation published into the store) once at startup and then every
  `--cadence` minutes, default ten, the production cadence the design
  targets. The cycle reuses the job's code unchanged; the only new plumbing
  is calling it with the process's store locator instead of the job's
  environment. Its process pool is the app's pool.
- The web half is exactly the web service in shadow mode: it loads the
  generation the first cycle publishes and refreshes on elected requests.
  Until cutover the legacy fetching managers still serve, so the local
  process fetches twice; that duplication ends with cutover and is accepted
  meanwhile because the entry point must exist first.
- Readiness and health are the web service's. Before the first cycle
  publishes, the legacy managers answer as they do today.

Out of scope: any change to production entry points or manifests. Renaming
the bounded job module — `capture`, under the `shallweswim` package — to
`shallweswim.update` happened later, after cutover.

Tests: the store helper resolves each locator kind and returns one shared
memory store per process; the local cycle publishes a generation into a
memory store from mocked clients and the app loads it; a filesystem store
survives a restart and the second cycle hydrates history from it rather
than fetching; the entry point ignores bucket variables from the
environment.

Documentation: README's local development section makes this the
recommended command and keeps `shallweswim.web` for running the web half
alone; ARCHITECTURE lists the three entry points and the store helper;
`.env.example` notes that the local entry point needs none of its variables.

#### Cutover Contract

Status: implemented and deployed on 2026-09-14 at 19:39 UTC (web) and
19:42 UTC (ten-minute cadence). The first ten-minute run fetched only the
ten live temperature feeds and held the other twenty, published in 22
seconds, and the web service loaded it 28 seconds later in 0.6 seconds; live
temperature age on the site fell from 33 minutes to about 3.5 minutes. The
cold load on the new revision, with no fetching stack competing for CPU, took
1.8 seconds.

End state: the web service serves every request from the loaded generation
and never contacts a provider. The job is the only process that talks to
providers, runs every ten minutes, and fetches each feed no more often than
the feed's own expiration interval. The local entry point runs both halves in
one process, fetching once.

Web service: (implemented)

- The lifespan builds the store from `SHALLWESWIM_SNAPSHOT_READ_BUCKET` and
  runs the initial load. The variable is required: an unset variable is a
  configuration error that fails startup with one clear message, because a
  web process with no store has nothing to serve (`shallweswim.local` is the
  way to run without a bucket). A failed or timed-out initial load logs at
  ERROR, the instance starts anyway, and readiness stays false until an
  elected request loads a generation; the startup probe keeps electing
  itself every check interval, and the platform restarts an instance that
  never becomes ready. The 20-second bound stays: without the fetching
  stack's startup contention the measured cold load is a few seconds.
- Routes resolve a location from the shadow state's current managers on
  every request, never from a dict captured at startup, so a refresh is
  visible to the next request. A location the generation does not carry, or
  one whose manager has no data, answers 503 exactly as today's
  `has_data` check does. `app.state.data_managers` is removed.
- `/api/healthy` returns 200 when a generation is loaded and at least one
  location has data, and 503 otherwise, with the same lenient
  one-station-down semantics as today. `/api/status` keeps its shape: per
  location, per feed, the status derived from the manifest by the snapshot
  manager, plus three additive optional fields on the location status:
  `generation_id`, `published_at`, and `loaded_at`. `/api/locations`'
  `has_data` follows the loaded generation.
- On-demand tide and current detail plots stay request-driven in the web's
  process pool, drawn from the loaded frames; nothing else plots in the web.
- The fetching stack leaves the web runtime path: `initialize_location_data`,
  the manager start and stop calls, the provider clients and HTTP session,
  the background update loop, feed-driven plot generation and its watchdog.
  The modules stay in the package for the job and the local entry point. A
  test asserts the web app's lifespan constructs no `LocationDataManager`
  and opens no client session.
- Severity follows the rule already stated: the readiness-blocking initial
  load failure is ERROR; a failed refresh over a loaded generation stays
  WARNING and pages through the load lag alert.
- Local entry point: the first publishing cycle runs before the app's
  initial load, so the first request already has a generation and the
  process fetches once. Its cycle then continues on the cadence.

Job: (implemented)

- The current generation's manifest is the persisted feed schedule. Before
  updating a feed, the job restores that feed's `next_fetch_after` from the
  current generation when the entry's `source_identity` matches the feed's
  `citation_key`; a feed that would not come due before the next run starts
  is not fetched, and manifest assembly carries its entry forward unchanged,
  plots included, with a freshness outcome of `held`. A feed due before the
  next run fetches on this one: restored literally, a feed whose interval
  equals the cadence lands seconds after the next run begins and is held on
  every other run, which production showed on 2026-09-14 as live temperature
  refreshing every twenty minutes. A feed that is due, has no entry, or changed
  identity fetches as today. This keeps each feed at its own interval under
  any job cadence: live temperature every ten minutes, historical
  temperature every three hours, tide and current predictions daily, which
  is the provider load one web instance generates today.
- Cadence moves to every ten minutes (`*/10 * * * *`), the design's target.
  With the live feed due on every run, worst-case live temperature age is
  provider lag plus one cadence plus publication plus one check interval,
  about seventeen minutes, inside the freshness alert's twenty-five. The
  job timeout stays at twenty minutes and overlap remains safe by conditional
  promotion; a run normally takes under two minutes.
- The `unchanged` publish outcome becomes reachable only when no feed was
  due, which the live feed prevents; a run with nothing new still publishes
  a manifest, as today.
- `--full-history` keeps its meaning: every configured year is fetched
  regardless of schedule, for backfills and repairs.
- This schedule decides whether a feed is fetched, not how much is asked
  for. Fetching only what is new since the last archived observation, with an
  overlap window for revised readings, is the later "incremental fetching"
  step; it builds on the same manifest bookkeeping and is not part of
  cutover. Today past historic years already come from the archive, the
  live feed's 24-hour window is its overlap, and the current historic year
  is the one fetch incremental reading would shrink.

Deployment and rollback:

- `service.yaml` keeps the read bucket variable; `capture-job.yaml` is
  unchanged except the scheduler cadence, which is an operator action in the
  runbook. The web identity keeps read-only access. Memory and CPU limits
  stay until measured on the new path.
- Rollback is traffic back to the previous revision, which still fetches for
  itself, and needs no store change. The job keeps publishing throughout.
- Deploy order: web service first, observe one refresh and the health check
  on the new revision, then the scheduler cadence.

Deferred to follow-up slices, not part of cutover: garbage collection of old
generations, which should follow cutover soon (on 2026-09-14 the published
prefix held 27 generations and 497 MB against a 17 MB archive, growing about
20 MB per generation; keeping the active generation plus a day of
predecessors is roughly 50 MB steady state); the stale-data display policy;
resource limit tuning.

Tests: routes read the current managers after a refresh, not the startup
mapping; health and status reflect the loaded generation and its absence;
the lifespan opens no client session and starts no manager; a missing read
bucket fails startup with the documented message; the job restores the
schedule from the manifest and holds a feed that is not due, carrying its
entry and plots forward with the `held` outcome; a due feed and a changed
identity still fetch; `--full-history` ignores the schedule; the local entry
point serves its first request from its own first generation.

Documentation: README (what the web service does and does not do, the
required variable, the job cadence, and the local command fetching once),
ARCHITECTURE (the request path from the loaded generation, the job as the
only provider client), the capture job runbook (cadence change and rollback),
`infra/monitoring/README.md` if any threshold changes, and this document's
runtime component descriptions where they still describe the fetching web.

### Phase 5: Use the Archive for Incremental Fetching

- Restore the per-year historical cache and timestamps from archived objects.
- Fetch only missing years, the current year, and configured overlap windows.
- Expand durable capture to any observational feeds deferred in Phase 1.
- Measure and eliminate unnecessary full-history cold-start requests.

Archive writes begin early because delay can permanently lose observations.
Making the archive authoritative for incremental reads remains later because it
requires stronger migration and equivalence validation.

## Deep History

Status: implemented, in three slices: the provider client changes, the backfill
command, then the served range and the plot. The archive was backfilled to full
depth on 2026-09-15, and each source's configured `start_year` now follows it.

The archive keeps everything a provider will still give, at every cadence the
provider offers it, once. The served hourly frame then extends back to the
same depth, and the plots draw one line per year exactly as today, with older
years fading. No aggregation: if the result is too busy, aggregation is a
later, separate decision.

Backfilling is a standing step, not a one-off: every new location is
backfilled when it comes online, and the new-location checklist
(`NEW_LOCATION.md`) says so. The command therefore discovers a source's depth
itself rather than taking a range an operator found by hand.

### What the providers hold

Probed on 2026-09-15, read-only, through the providers directly and through
the project's own clients. Every provider signals an empty year cleanly: CO-OPS
answers "No data was found", NDBC has no yearly file, CSPF has no page, and
each client already raises `StationUnavailableError` for it. Nothing was found
hidden behind a multi-year gap; the CO-OPS stations are empty in 1980, 1985,
and 1989.

| Location | Source | Fetched from today | Provider holds from |
| --- | --- | --- | --- |
| bos | NDBC 44013 | 2011 | 1984 |
| san | CO-OPS 9410230 | 2011 | 1993 hourly; six-minute from 1993-10 |
| sea | CO-OPS 9446484 | 2011 | 1996 hourly; six-minute from 1996-07 |
| nyc | CO-OPS 8518750 | 2011 | 1997 hourly; six-minute from 1998-01 |
| dov | CSPF Sandettie | 2011 | 2004-06 |
| sfo | NDBC 46237 | 2011 | 2007-07 |
| aus | NWIS 08155500 | 2011 | 2007-10 (fifteen-minute readings) |
| pbi | CO-OPS 8722670 | 2011 | 2010 hourly; six-minute from 2010-07 |
| chi | NDBC 45198 | 2021 | 2021 |
| cor | Irish Lights | 2024 | 2024-05 |
| sdf | NWIS currents 03292494 | live only | 2013-11 |

Two findings shape the design. CO-OPS's hourly and six-minute products have
different coverage: hourly reaches further back at every station, and hourly
is complete across stretches where six-minute is empty (Tacoma in 1997, The
Battery in October 2011). And the NDBC client cannot read yearly files before
2007: NDBC changed the file layout three times before then (two-digit years,
then four-digit years, then a minutes column), and none of those layouts
carries the header marker the client looks for.

### Provider Clients

- CO-OPS: the request-window split gains a per-interval limit, 31 days for the
  six-minute product and 365 days for hourly, so a caller asks for a year at
  either cadence and the client makes the requests. No caller-side loop.
- NDBC: the historical yearly file parser reads the three pre-2007 layouts as
  well as the current one, verified against real files for 1984, 1998, 2004,
  and 2006, and tolerates a column appended partway through a file, as 44013
  gained TIDE during 2000: a row shorter than the header is missing its
  trailing columns. The frame it returns is unchanged in shape.
- NWIS currents: the client gains a date window like the temperature call has,
  and the currents feed gains a start and end like the temperature feeds. This
  is what lets the observational currents source join the backfill; it is
  ordered last and may follow the temperature backfill.

### Backfill Command

- `python -m shallweswim.update --backfill-from [YEAR] [--location CODE ...]`
  is a capture-only run: it archives what comes back, publishes nothing, and
  hydrates nothing. `YEAR` is the floor of the walk and defaults to 1900
  (`BACKFILL_FLOOR_YEAR`) when omitted; the floor only bounds a source the
  provider really holds that far back. It is rejected together with
  `--full-history` or the publish variable.
- Scope: every historical temperature source of the selected locations, and
  the observational currents source once its client takes a date window. Each
  source walks every year from the current year down to the floor, newest
  first, including the years already configured and archived, so one run
  leaves the archive holding everything the provider offers. Re-archiving a
  year already held changes nothing: the merge treats a row equal to the
  stored one as an overlap.
- Cadence: CO-OPS is fetched as both products per station-year, one hourly
  request and twelve six-minute requests, because neither covers the other.
  Six-minute months the provider lacks are expected and skipped. The merge
  matches rows on their observation instant, so an on-the-hour six-minute row
  overlaps the hourly one. NDBC yearly files, NWIS instantaneous values, CSPF,
  and Irish Lights are one request per year at native cadence, as the
  historical feed makes today. Going forward the live temperature feed keeps
  archiving six-minute CO-OPS readings under the same source identity.
- Empty years: a year is empty when every request in it raised
  `StationUnavailableError`; within a CO-OPS year the hourly request and the
  twelve six-minute requests count together. The walk stops a source after
  five consecutive empty years going backwards (`BACKFILL_EMPTY_YEARS_STOP`).
  Any other error ends that source and is logged at ERROR; the other
  locations continue, as the capture path isolates feed failures today.
- Provider courtesy. Requests within a source run one at a time, newest year
  first, with a short pause between them (`BACKFILL_REQUEST_PAUSE`, one
  second), and locations run one after another rather than concurrently: a
  backfill is run by hand and its duration does not matter, while four CO-OPS
  stations walking at once would quadruple the rate at one provider. A CO-OPS
  station with thirty years is about 400 requests; every source together is
  roughly 1,700, around ninety minutes. It is one pass per source, once per
  location lifetime.
- Rate limiting. The first real walk was answered with HTTP 403 by CO-OPS
  after about 160 requests in eight minutes, and the block lifted within the
  hour. CO-OPS uses 403 for a temporary rate block, which is 429 by the
  letter of the standard. The live client paths keep treating 403 as a
  refusal that fails fast, which is right for the job and the web app. The
  walk alone treats a 403 as a wait: it pauses (`BACKFILL_BLOCK_PAUSE`, five
  minutes) and retries the same request, up to `BACKFILL_BLOCK_RETRIES` (3)
  times, before ending the source as an error. To make that possible the
  client's HTTP error carries its status code as a field rather than only in
  its message. Any other non-transient error still ends the source at once.
- It runs from the operator's machine against the archive bucket, under a
  temporary write grant to the local operator identity that is revoked
  afterwards. Nothing in the code depends on where it runs; the scheduled
  job keeps capturing meanwhile, and the merge's conditional writes handle
  the overlap.
- The run summary is one structured event with a distinct operation name, so
  the capture heartbeat metric and the shadow policies never count a backfill
  as a scheduled capture. It reports, per source, each year's rows added and
  revised or that it was empty, and the earliest year that returned data. The
  run exits zero when every walk completed, even where every year was empty,
  and non-zero only when the run itself fails.
- Serving is unchanged. The archive assumes no cadence: hydration reads
  whatever rows a year holds and resamples to hourly with the first reading
  of each hour, which for CO-OPS is the on-the-hour reading the hourly product
  returns, so a year hydrated from mixed rows serves exactly as one fetched
  hourly did. The current year already holds both cadences today.
- Expected size: thirty years of six-minute readings for one CO-OPS station is
  about 2.5 million rows, a few tens of MB of Parquet across its yearly
  partitions.

### Served Range and Plot

- After a backfill, each source's `start_year` in `config/locations.py` is
  lowered to the earliest archived year, in a reviewed change. From the
  2026-09-15 backfill: bos 1984, san 1993, sea 1996, nyc 1997, dov 2004,
  sfo 2007, aus 2007, pbi 2010; chi stays 2021 and cor 2024. The served
  hourly frame, the bundle object, and the plots then cover that range
  through the existing paths: the job hydrates the extra years from the
  archive on its next historical refresh, nothing is refetched, and every
  generation carries the deeper frame. Cost, accepted: a thirty-year
  station's frame is roughly three times today's 1.3 MB, so a bundle of
  perhaps 40 MB instead of 21 MB, loaded once per generation per instance.
- The local entry point runs the publishing cycle with no archive behind it
  on a fresh start, so it would fetch every configured year from the
  providers, about 230 year requests across the sources, with four CO-OPS
  stations at once: the burst that earned the 403. `shallweswim.local` takes
  `--historic-years N` (default `LOCAL_HISTORIC_YEARS`, 10) and floors the
  historical range at the current year minus N through the floor the feed
  construction already accepts, threaded through the publishing cycle. A
  larger N with `--store-dir` takes the one-time fetch and hydrates after.
  The job and the web service are unaffected: the web never fetches, and the
  job hydrates from the archive and fetches only what it lacks.
- Both historical plots keep one line per year. The current year is fully
  opaque; each earlier year's opacity falls linearly with its age to a floor
  (`FADE_FLOOR`, 0.15) reached at `FADE_YEARS` (10) and held there for all
  older years, so every archived year is drawn and the recent ones dominate.
  The two-month plot was first left unchanged and proved unreadable with
  thirty equal lines, so it takes the same fade. The legend names the
  current year and the years still fading; the years at the floor are drawn
  but left out of the legend, since a reader cannot tell their lines apart,
  and the subtitle states the plot's full year range instead.
  Line width and colour policy are unchanged. Both are named constants in
  the plot module, one treatment for every location.
- Provider-side gaps in old years appear as gaps in the line, as they do
  today.

Out of scope: any averaging or banding across years; smoothing the served
hour to a mean rather than the first reading; per-station backfill for
sources whose provider offers no history; serving anything finer than the
hourly frame.

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
- The builder records a configured feed without data as a failure; assembly
  carries the base entry forward with accumulated failures and this run's
  error, copies an absent plot, and carries nothing when the base lacks the
  feed, its source identity differs, or the feed is no longer configured.
- Every run emits one freshness event per configured feed, with the carried
  outcome and the served frame's age for a carried-forward feed.
- A snapshot manager answers every route-facing query identically to a
  feed-backed manager holding the same frames.
- A refresh reads only objects not already held, a failed check waits a full
  interval, concurrent requests never duplicate a check, and a request in
  flight during a swap finishes on one whole generation.
- Each comparison outcome of the local command is produced by exactly the
  condition that defines it, including the float tolerance and the
  derived-answer checks, and the command exits non-zero on any mismatch.
- The web deployment manifest sets the snapshot read bucket and never the
  archive write or hydration variables.
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
- Daylight-saving fall-back tests prove that two UTC instants sharing a local
  wall time archive as two rows and serve as one, and that a naive index is
  rejected at the archive boundary.
- Historical capture archives the pre-resample per-year frame, so native
  10- and 15-minute cadences and both fall-back folds reach the archive.
- All clients return timezone-aware UTC frames; feeds derive the naive local
  serving index by keeping the first occurrence of a repeated wall
  time, and a live window spanning a fall-back day validates and serves.
- CO-OPS observation history requested in GMT contains 8,761 hourly rows in a
  fall-back year, and the archive holds both folds for every source.
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

# Data Pipeline

The job, the bundle, and the web servers. This document owns the rules for
how observations are fetched, archived, published, and served. README.md says
how to run each piece; ARCHITECTURE.md maps the code; infra/MONITORING.md owns the
metrics and alerts built on the events named here.

## Shape

Two processes and one store. The **job** (`shallweswim.update`) is the only
process that contacts a provider. On each run it fetches the feeds that are
due, archives the observations they return, draws the plots, and publishes one
immutable **generation**, the bundle, into object storage. The **web servers**
(`shallweswim.web`) load the current generation into memory and serve every
request from it. There is no database: the store holds plain content-addressed
objects, and the web servers only read them.

Why: fetching, plotting, and serving used to share one process, so background
progress was tied to the lifecycle and resources of whatever instance was
answering requests, and instances disagreed about the data. Moving all fetching
into a bounded job with a shared store means one coherent dataset for every
instance, no provider latency on a request, and observations that survive a
provider taking them down.

The **local entry point** (`shallweswim.local`) runs the job's publishing
cycle on a timer inside the web app's process against a local store, so one
command still gives a working site. It composes the two halves rather than
adding a mode to either: it wraps the app's lifespan, runs the first cycle and
loads its generation before the server accepts a request, then keeps
publishing on the cadence (`DEFAULT_CADENCE_MINUTES`, 10) and cancels that
task at shutdown. The web app opens no HTTP session, so the local entry point
opens the one its cycles fetch over; the pool they plot in is the app's. Its
`--historic-years N` (`LOCAL_HISTORIC_YEARS`, 10, counting the current year)
floors every historical temperature range, because a fresh local store has no
archive behind it and the configured ranges reach back decades.

### Store locators

Every store comes from `archive/store.py`'s `object_store(locator)`:

| Locator | Store |
| --- | --- |
| a bare name, such as `my-archive-bucket` | a GCS bucket |
| anything containing `/`, such as `./local-store` | a directory on disk |
| the literal `memory` | one in-process store |

The `ObjectStore` protocol is four operations: `read`, `compare_and_swap`,
`list(prefix)` returning each key with its creation time, and `delete(key)`,
which an absent key satisfies. All three stores answer them identically, so a
locator kind is never a code path.

Three environment variables carry locators:

| Variable | Who sets it | Access |
| --- | --- | --- |
| `SHALLWESWIM_ARCHIVE_BUCKET` | the job | writes the archive and the bundle; required by the job, which fetches only in order to archive |
| `SHALLWESWIM_ARCHIVE_READ_BUCKET` | the job when publishing, local development | reads the archive to hydrate historical years |
| `SHALLWESWIM_SNAPSHOT_READ_BUCKET` | the web servers | reads `published/`; required, because a web process with no store has nothing to serve |

The local entry point sets all three to one locator in its own process,
unconditionally, so a local run can never reach the operator's bucket. In the
reference deployment the job identity holds `roles/storage.objectUser` on the
archive bucket and the web runtime and local operator identities hold
`roles/storage.objectViewer`; the local operator opens itself an expiring
write window for a backfill or a repair. infra/README.md "Identities"
defines each identity and what it may do.

## The job's cycle

One run is bounded: it starts, does one cycle, and exits. Production runs it
every ten minutes (`JOB_CADENCE`) with `SHALLWESWIM_SNAPSHOT_PUBLISH=1`.

### Schedule restoration

The current generation's manifest is the job's persisted feed schedule. Before
a location's cycle, `restore_schedule` sets each feed's next fetch time from
the manifest entry that names the same `source_identity`. A feed whose next
fetch time falls before the next run starts (now plus the cadence) is due now;
a feed due later is **held**: it is not fetched, and the new manifest keeps its
entry and its plots exactly as published. A feed the manifest does not
describe, or describes from a different source, fetches as a fresh feed does.
`--full-history` skips restoration, so every feed fetches.

Why: a fresh feed is always due, so without restoration a bounded run would
refetch everything every ten minutes. With it each feed keeps its own
interval whatever the cadence is, and a feed due just after a run would
otherwise wait a whole cadence past its interval.

| Feed | Interval (`EXPIRATION_PERIODS`) |
| --- | --- |
| live temperature | 10 minutes |
| historical temperature | 3 hours |
| tide predictions | 24 hours |
| current predictions | 24 hours |

### The serving cycle

Locations run concurrently, each in the same `LocationDataManager` the web
servers once used: `update_once()` updates the feeds that are due, and
`wait_for_plots()` awaits the plots it submitted to the process pool, bounded
by `PLOT_HARD_TIMEOUT` (300 s). A feed whose update raises has already logged
the failure at ERROR and scheduled its retry; the cycle re-enters and updates
the feeds after it, one re-entry per feed at most.

A failed fetch schedules the next attempt through the shared retry ladder
(`FEED_RETRY_INTERVALS`: 1, 2, 5, 10, 20, 30 minutes) and never later than the
feed's own interval. Because the job cannot act before it starts, the first
three steps of that ladder mean "next run".

### Capture

When a temperature feed or an observational currents feed fetches
successfully, its raw client frame is merged into the archive (see
[The archive](#the-archive)) before the feed's serving frame is derived. The
historical feed captures only the years it fetched this run, never a hydrated
one. Prediction feeds, tides and NOAA current predictions, never enter the
archive. Archive failures are logged as failed merge events and change neither
the feed's success nor its schedule.

### Publishing and the sweep

After the cycle the job builds every location's snapshot, publishes one
generation (see [The bundle](#the-bundle)), and then sweeps superseded
generations (see [Garbage collection](#garbage-collection-and-retention)). A
publishing run requires `SHALLWESWIM_ARCHIVE_READ_BUCKET` as well, because a
generation carries the full historical range, which hydrates from the archive
rather than refetching.
Publication and the sweep each log their own event and isolate their own
failures: the run's outcome and exit code belong to the capture cycle.

### The capture-only run

Without `SHALLWESWIM_SNAPSHOT_PUBLISH=1` a run fetches only the archivable
feeds, live temperatures, historical temperatures, and observational currents,
once each, and archives what they return. It fetches no predictions, draws no
plots, and publishes nothing. The historical feed fetches only the current
year; `--full-history` fetches the whole configured range. Locations run
concurrently and each location's feeds run in sequence.

### Run summary

Each run ends with one structured event, `component=updater`
`operation=run`, whose `outcome` is `success` when every attempted feed holds
data afterwards, `partial` when only some do, and `failed` when none do. A
held feed counts as published, because the generation it was restored from
keeps serving its frame. The event carries `duration_ms`, `record_count`
(served rows), `new_count` and `revised_count` (archive rows), and `run_id`,
which is the platform's execution name when it provides one. A publishing run
names the publish outcome in its message. Exit code is 0 for `success` and
`partial`, 1 for `failed` or a run that raised.

## The bundle

A generation is an immutable manifest referencing immutable content-addressed
objects, written into the store under `published/`:

| Key | Content |
| --- | --- |
| `published/objects/sha256-<digest>.parquet` | one served feed frame |
| `published/objects/sha256-<digest>.svg` | one plot |
| `published/manifests/<generation_id>.json` | one generation's manifest |
| `published/current.json` | the pointer naming the current manifest |

Objects are named by their content, so writing one is create-only and an
identical key already present is reuse. Objects are written first, the
manifest second, and the pointer last, replaced conditionally against the
version the publisher observed when it started, so a reader sees either the
whole previous generation or the whole new one, and two publishers cannot move
the pointer backward. A manifest whose locations equal the current manifest's
is not written at all: the publish outcome is `unchanged`. A run in which no
location holds data publishes nothing and reports `skipped`.

The generation id is `<published_at to the second, UTC>-<run_id>`, sortable
and unique across publishers starting in the same second.

### What a manifest carries

A manifest has a `schema_version` (`SCHEMA_VERSION`, 1), the generation id,
`published_at`, the previous generation id, and one entry per location. A
location entry has one `FeedObject` per served feed and one `PlotObject` per
plot:

| Feed object field | Meaning |
| --- | --- |
| `key`, `size_bytes` | the Parquet object, checked on load |
| `source_identity` | the feed's `citation_key` |
| `fetch_timestamp` | when the served frame was fetched |
| `next_fetch_after`, `expiration_seconds` | the feed's schedule, restored by the next run |
| `record_count` | rows in the served frame |
| `consecutive_failures`, `last_error` | the source's failure state |
| `timezone` | the location's timezone |
| `historical` | the historical feed's year diagnostics, when it is one |

A plot object carries its key, size, the feed it was drawn from, and that
feed's `fetch_timestamp`, which states which feed state the plot shows.

### Carry-forward

Every configured feed of every enabled location is reported. Assembly
resolves each feed against the base generation, the one the publisher
observed when it started:

- A **held** feed keeps the base entry byte for byte, failure fields included.
- A **failed** feed, one that was due and fetched nothing, keeps the base
  entry published for the same source: the object, its fetch timestamp, and
  its record count stay, while the failure count accumulates and the last
  error and next retry become this run's. A transient provider failure never
  drops last-known-good data from a generation.
- A **plot** this run did not produce keeps the base generation's, but only
  while the feed it was drawn from is still in the assembled manifest.
- Nothing is carried for a feed the base generation lacks or published from a
  different source, and a feed or location no longer configured is absent.

Carried objects are referenced by key, never reread or rewritten, so a
repeatedly failing feed publishes a manifest and no object.

### Freshness events

After assembly, whether or not the generation is promoted, the job logs one
`component=snapshot` `operation=freshness` event per location and configured
feed with `outcome` `success` (fetched), `held`, `carried`, or `absent` and
the served frame's `age_seconds`. Fetched and held are INFO; carried and
absent are WARNING. A feed stuck on carried data therefore shows a growing age
even while nothing else changes.

### Publish events

Publication logs one `component=snapshot` `operation=publish` event with
`outcome` `success`, `unchanged`, `skipped`, or `failed`, the generation id,
`record_count` as the objects written, `duration_ms`, and `run_id`. A
promotion conflict, the pointer having moved since the publisher started, is
`failed` and the run keeps its own outcome.

## The web servers

A web server serves only the generation it has loaded. It fetches nothing,
opens no client session, and starts no update loop. Its lifespan builds the
store on a worker thread, runs one startup load bounded by
`INITIAL_LOAD_TIMEOUT_SECONDS` (20), and starts whether or not that load
succeeded. A failed or timed-out startup load is ERROR, because the instance
then has nothing to serve; `/api/healthy` answers 503 until an elected request
loads a generation, and the platform restarts an instance that never becomes
ready. A failed refresh over a loaded generation is WARNING.

### Elected refresh

Refresh rides on requests. An HTTP middleware runs on every request, health
checks included, so an idle instance still notices a new generation: when
`CHECK_INTERVAL_SECONDS` (60) have elapsed since the last check and no check
is in flight, that request is elected and awaits the check before its handler
runs; concurrent requests proceed immediately. A check reads
`published/current.json`. If it names the loaded generation, nothing happens
above DEBUG. If it names a new one, the instance reads that manifest and only
the objects whose keys it does not already hold, `MAX_CONCURRENT_READS` (8) at
a time, checks each against the size and key the manifest recorded, validates
every frame through its feed model, rejects a manifest whose `schema_version`
is not its own, builds the read-only per-location managers, and swaps them in
with one assignment of an immutable mapping, so a request running during the
swap finishes on the generation it started with. A failed check schedules the
next one a full interval later. The web server never lists manifests and never
deletes anything.

Each load that does work logs one `component=snapshot` `operation=load` event
with `outcome` `success` or `failed`, the generation id, `duration_ms`,
`record_count` as objects read, and `age_seconds` as the lag between
publication and this instance picking the generation up. The event carries the
platform's instance identity, which is how "is every instance loading the
bundle" is answered.

### Serving

`SnapshotLocationManager` (`snapshot/manager.py`) is built per location from
the manifest entry, the restored frames, and the plot bytes. It is immutable:
the derived tide and current prediction frames are computed once at
construction, and a new generation constructs new managers. It and
`LocationDataManager` both satisfy `core/serving.py`'s `LocationServing`
protocol, which the API routes are typed against, and the query functions
read any `FeedData`, so bundle-served data answers every route exactly as
fetched data did. Its status applies the feed scheduling and health rules to
the manifest's timestamps: a feed is expired past its interval and unhealthy
past the interval plus `HEALTH_CHECK_BUFFER` (15 minutes).

Routes resolve the location's manager from `app.state.snapshot` on every
request, so the generation an elected request loads is served by the next
one. A location the generation does not carry, or one whose manager holds no
data, answers 503. `/api/healthy` (alias `/api/health`) answers 200 while any
location has data and 503 otherwise. `/api/status` reports the per-feed status
and, on each location, `generation_id`, `published_at`, and `loaded_at`, the
same for every location of one response; `/api/locations` reports `has_data`
for the locations the generation carries.

The only per-request work that is not a lookup is the on-demand tide and
current detail plot (`/api/{location}/plots/current_tide` and `/plots/tide`),
drawn in the web server's process pool from the loaded frames.

## The archive

The archive keeps every observation a provider has returned, at the provider's
native cadence, unfiltered: the frame arrives as the client returned it,
indexed by timezone-aware UTC instants, before the serving conversion and
before configured outlier removal, so both folds of a daylight-saving fall-back
hour are archived as the distinct instants they are.

### Layout and schema

One Parquet object per source, measurement, and UTC year:

```text
archive/<measurement>/<provider>/<station>/<year>.parquet
```

`<measurement>` is `temperature` or `currents`; provider and station are the
two segments of the feed's `citation_key`, percent-encoded, which is the
source identity everywhere in the pipeline. Every row has four columns:

| Column | Meaning |
| --- | --- |
| `observed_at` | timezone-aware UTC instant, unique within a partition |
| `value` | float in the canonical unit |
| `unit` | `F` for temperature, `kt` for currents |
| `retrieved_at` | timezone-aware UTC instant the fetch happened |

Schema evolution is additive: a change is a nullable column or a new path
prefix, objects are never rewritten for a schema change, and every read goes
through `read_observations`, which fills columns absent from older objects
before validating against `ObservationModel`. Contract tests pin the columns,
dtypes, units, and UTC semantics, and a golden list pins every configured
`citation_key` used as a source identity.

### Merge semantics

A capture merges each UTC year of the incoming frame into its partition with
`compare_and_swap`, retrying up to five times with exponential backoff on a
version conflict. Rows are matched on `observed_at`:

| Incoming row | Result |
| --- | --- |
| key absent from the partition | **new**: added |
| same value as stored | **overlap**: partition unchanged, stored `retrieved_at` kept |
| different value, retrieved later than stored | **revised**: replaces the stored row |
| different value, retrieved earlier than stored | **overlap**: the stored row already supersedes it |
| different value, same `retrieved_at` | integrity error: equally authoritative rows disagree |

A merge with nothing new or revised leaves the partition byte-identical, so
re-archiving a year already held changes nothing. Merges from the scheduled
job and a backfill run interleave safely through the conditional write.

### Merge events

Each merge logs one `component=archive` `operation=merge` event with
`source_identity`, `outcome` (`success`, `unchanged`, or `failed`),
`duration_ms`, `record_count` (the partition after the merge),
`attempt_count`, `incoming_count`, `new_count`, `overlap_count`, and
`revised_count`. A capture's new and revised counts sum into the run summary.

### Hydration

When `SHALLWESWIM_ARCHIVE_READ_BUCKET` is set, the historical temperature feed
reads each required past year from the archive before asking the provider,
`ARCHIVE_HYDRATION_CONCURRENCY` (8) reads at a time. A historical year is a
station-local year while partitions are UTC years, so hydration reads the
year's partition and the next one and keeps the rows inside the local year.
The rows then follow exactly the provider path: serving index, resample to
hourly keeping the first reading of each hour, validation. The archive assumes
no cadence, so a year holding six-minute rows, hourly rows, or both serves
identically. The current year always refetches from the provider; a year the
archive lacks, or whose read or validation fails, is left for the provider
fetch; hydrated years are never captured back. Hydration never fails an
update.

## Garbage collection and retention

Objects are content-addressed, so a generation published minutes ago may
reference an object written days ago; age alone cannot decide what is
deletable. Each publishing run therefore marks and sweeps once, after
publication and whatever publication's outcome:

- The generation the current pointer names is kept whatever its age, and so
  is every generation published inside `RETAINED_GENERATION_AGE` (24 hours),
  the rollback window and the window a slow instance could still be loading
  from. The manifests of older generations are deleted. The pointer never is.
- An object is deleted only when no retained manifest references it and it
  was created more than `OBJECT_SAFETY_AGE` (1 hour) ago, which protects a
  publisher that has written a generation's objects but not promoted it yet.
- A manifest the sweep cannot parse is retained, and because its references
  are then unknown, every object is kept that run.
- The sweep deletes nothing it did not list in the same run and never touches
  `archive/`.

It logs one `component=snapshot` `operation=gc` event with `outcome` `success`
or `failed` and the objects deleted as `record_count`, and returns rather than
raising. The local entry point's cycle sweeps too, so a store directory does
not grow without bound.

The archive itself is never deleted or rewritten by anything in the pipeline.

## Freshness budget

Worst-case age of a live temperature reading on the site:

```text
provider publication lag        ~5 minutes
+ one job cadence               10 minutes (the feed's interval equals it and
                                 due-before-next-run fetches it every run)
+ job run and publication       about a minute
+ web check interval            up to 60 seconds
```

About seventeen minutes, against a health rule that calls a live feed
unhealthy after its interval plus the fifteen-minute buffer. Historical
temperature refreshes every three hours and the predictions daily, each on
its own interval through schedule restoration, so the job cadence sets only
how soon a due feed is noticed, not how often a feed is fetched.

## Failure semantics

- One upstream station unavailable, or a station that has stopped returning
  observations, is a WARNING: the feed carries forward, the generation keeps
  serving its last frame, and the archive keeps what was collected.
- A transient provider status (`429`, `5xx`, connection or timeout) is retried
  inside the client; a failure that survives the retries is a feed failure and
  follows the ladder. A `403` fails fast everywhere except the backfill walk.
- A job crash from an application defect, a generation older than the
  freshness threshold, a failed publication or promotion, a web server that
  cannot load any generation, and unexpected `500`s are the page-worthy
  conditions; infra/MONITORING.md owns the policies.
- Health reports service health and data freshness separately, so one missing
  upstream series is never total application failure.
- Overlapping job executions are safe: conditional promotion and the base
  generation observed at start mean a later publisher cannot move the pointer
  backward, and merges are conditional writes.
- Rollback of a bad deploy is routing traffic to the previous revision. The
  store needs no change, and the job keeps publishing throughout.

## Deep history

The archive holds everything each provider still offers, and the served range
follows it.

### What the providers hold

| Location | Source | Holds from |
| --- | --- | --- |
| bos | NDBC 44013 | 1984 |
| san | CO-OPS 9410230 | 1993 hourly; six-minute from 1993-10 |
| sea | CO-OPS 9446484 | 1996 hourly; six-minute from 1996-07 |
| nyc | CO-OPS 8518750 | 1997 hourly; six-minute from 1998-01 |
| dov | CSPF Sandettie | 2004-06 |
| sfo | NDBC 46237 | 2007-07 |
| aus | NWIS 08155500 | 2007-10 (fifteen-minute readings) |
| pbi | CO-OPS 8722670 | 2010 hourly; six-minute from 2010-07 |
| chi | NDBC 45198 | 2021 |
| cor | Irish Lights 992501100 | 2024-05 |
| sdf | NWIS currents 03292494 | 2013-11 |

Every provider signals an empty year cleanly, and each client raises
`StationUnavailableError` for it. CO-OPS's hourly and six-minute products
cover different ranges: hourly reaches further back at every station and is
complete across stretches where six-minute is empty, so neither covers the
other. NDBC yearly files before 2007 use three older layouts (two-digit years
through 1998, four-digit years without minutes through 2004, minutes from
2005) and may gain a column mid-file; the client reads all of them.

### The backfill command

`python -m shallweswim.update --backfill-from [YEAR] [--location CODE ...]`
archives every year a source still holds. It is capture-only: it publishes
nothing and reads nothing back, and it is rejected together with
`--full-history` or the publish variable. `YEAR` is the floor of the walk and
defaults to `BACKFILL_FLOOR_YEAR` (1900); the floor only bounds a source that
really reaches that far back.

- Each selected location's historical temperature source walks every year
  from the current year down to the floor, newest first, including the years
  already archived; re-archiving is an overlap. A year is fetched with the
  same one-year feed the historical feed builds. CO-OPS is fetched as both
  products per year, one hourly request and twelve six-minute months, because
  a month the station lacks would otherwise abort the year's client call.
- A year is empty when every request in it raised `StationUnavailableError`;
  an empty six-minute month inside a year with hourly data is skipped. After
  `BACKFILL_EMPTY_YEARS_STOP` (5) consecutive empty years the source stops;
  a year with data resets the count. Any other error ends the source at ERROR
  and the run is `partial`.
- Locations run one after another and requests within a source one at a
  time, `BACKFILL_REQUEST_PAUSE` (1 s) apart. CO-OPS answers a rate block with
  `403`; the walk waits `BACKFILL_BLOCK_PAUSE` (5 minutes) and retries the
  same request, up to `BACKFILL_BLOCK_RETRIES` (3) times per request, before
  ending the source. Walking every source takes roughly ninety minutes.
- The summary event is `component=updater` `operation=backfill`, so the
  capture-run metrics never count it. It reports per source the years
  archived, the years empty, and the earliest year with data, and exits 0
  whenever every walk completed.

It runs from an operator's machine against the archive bucket inside an
operator write window (infra/README.md "Identities"); the
scheduled job keeps capturing meanwhile. Backfilling is a standing step: every new location is backfilled
when it comes online (NEW_LOCATION.md).

### Served range

Each historical source's `start_year` in `config/locations.py` is the
earliest year the archive holds for it, set in a reviewed change after the
backfill; the summary's earliest year is what to set. The served hourly frame,
the bundle, and the plots then cover that range through the ordinary paths:
the job hydrates the years from the archive and refetches nothing. A
thirty-year station's frame is a few megabytes; the bundle is about 34 MB.

### Plots

Both historical temperature plots draw one line per year. The current year is
fully opaque; each earlier year's opacity falls linearly with its age to
`FADE_FLOOR` (0.15), reached at `FADE_YEARS` (10) and held there for older
years, so every year is drawn and the recent decade dominates. The legend
names only the years still fading, and the subtitle states the plot's full
year range. Line width, style, and colour policy are one treatment for every
location; provider gaps appear as gaps in the line. No averaging or banding
across years.

## Testing

Unit tests run over the memory store and mocked clients; nothing in them
touches a network or a bucket, and the test configuration strips the store
variables from the environment. What the suite pins:

- Serialization round trips preserve indexes, timezones, values, and missing
  data; a snapshot manager answers every route-facing query identically to a
  feed-backed manager holding the same frames.
- Publication writes objects, then the manifest, then the pointer; a
  no-change run writes nothing; concurrent publishers cannot move the pointer
  backward; generation ids stay unique in the same second.
- Carry-forward: a failed feed keeps the base entry with accumulated failures,
  a held feed keeps it unchanged, a plot outlives nothing, and nothing is
  carried across a source change or for an unconfigured feed. One freshness
  event per configured feed per run.
- Loading is incremental, a failed check waits a full interval, concurrent
  requests never duplicate a check, and a request in flight during a swap
  finishes on one whole generation. The web lifespan constructs no fetching
  manager and opens no client session.
- The sweep never deletes the current generation, a retained one, an object
  a retained manifest references, or anything inside the safety window, and
  keeps every object when a manifest will not parse.
- Archive contract tests pin the four columns, dtypes, units, and UTC
  semantics; every read uses the normalizing reader; a fall-back hour archives
  as two rows and serves as one; historical capture archives the native
  cadence frame; merges classify new, overlap, and revised rows and refuse
  equally authoritative conflicts.
- Schedule restoration fetches a feed due before the next run and holds one
  due later; the local entry point floors the historical range and the job
  does not.
- The backfill walks newest first, stops after five empty years, fetches both
  CO-OPS products, waits out a `403`, paces requests, runs locations in
  order, publishes nothing, and names its own operation.

Live integration tests (`--run-integration`) validate the provider clients
against the real services, separately from persistence behaviour, and build
their managers over a two-year historical range so they do not fetch decades.

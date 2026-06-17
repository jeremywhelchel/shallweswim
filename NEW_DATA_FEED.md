# Adding a New Data Feed

Sometimes a great swim spot has a useful data source that Shall We Swim does
not understand yet. Maybe it is a national tide service, a local harbor gauge, a
buoy network, a swim federation page, or a public JSON endpoint hiding in plain
sight.

That can be worth supporting, but it is a bigger step than adding a location.
Once the app depends on a new source, it has to know how to fetch it, parse it,
retry it, test it, explain it, and fail gracefully when the source has a bad
day.

This guide is for that moment: "we found the right data, but the app does not
have a client for it yet."

## Do We Really Need a New Feed?

First, check whether an existing client can already do the job. Reusing an
existing client is easier to maintain and usually safer.

A new feed may be justified when:

- The source is much closer or more locally relevant than anything we already
  support.
- It provides data we cannot get elsewhere, such as local tide predictions,
  historical water temperature, or a specific buoy measurement.
- It is official, public, stable, and likely to keep working.
- The app can clearly explain what the source measures and why it is useful.

A new feed is probably not worth it when:

- The source is fragile scraping with unclear permission.
- The data is regional and not actually useful for the swim spot.
- The source is paid or requires special access we do not have.
- The same thing can be done honestly with an existing client.

The goal is not to collect every possible data source. The goal is to show
swimmers useful conditions without making the app heavy or brittle.

## What to Prove Before Building

Before writing code, answer these in plain language:

- Who publishes the data?
- Is it public, licensed for our use, or permission-based?
- What exact sensor, station, buoy, or model does it represent?
- Where is that sensor relative to the swim spot?
- What does it measure: observation, prediction, modeled value, satellite
  estimate, or something else?
- What are the units, timezone, cadence, history depth, and latency?
- Does it need an API key, token, custom headers, or pagination?
- What happens when the source has no data?

If those answers are fuzzy, keep researching. A source that is mysterious before
implementation will usually be annoying after implementation.

Source IDs, station IDs, buoy IDs, endpoint URLs, API keys, public tokens, and
permission assumptions should be explicitly approved before they are committed.

## Ask an AI Coding Agent for Help

New feeds are a good place to use an AI coding agent, but give it boundaries.
The agent should investigate first and write code second.

Useful prompts:

```text
I found this possible data source for [location]: [URL]. Do not change code yet.
Read NEW_DATA_FEED.md, inspect existing clients, and tell me whether this should
reuse an existing client or needs a new one.
```

```text
Investigate this endpoint as a potential water temperature source. Identify the
actual sensor location, units, history depth, request limits, auth/key needs,
and whether it returns live data, historical data, or both. Do not choose it
until I approve.
```

```text
We approved this source and it needs a new client. Build the smallest
first-party async client. Inspect existing clients and return data in the same
shape they use. Add unit tests, a live integration test, a debug script, and
docs.
```

```text
Review this new data-feed plan as if you will own the production behavior.
Challenge whether the source is good enough and whether a new client is
justified.
```

## Implementation Shape

New source clients belong in `shallweswim/clients/` and should inherit from
`BaseApiClient`. Keep clients narrow: fetch, retry, parse, validate, and return
app-native dataframes. Clients should not know about feed expiration, plotting,
frontend behavior, or location-manager orchestration.

Typical steps:

1. Add a pure async client in `shallweswim/clients/`.
2. Name endpoint URLs, paths, public tokens, pagination limits, provider names,
   and concurrency caps as module-level constants.
3. Use `request_with_retry`, `provider_request_slot`, and the shared timeout and
   retry helpers from `shallweswim.clients.base`.
4. Map expected no-data conditions to `StationUnavailableError`.
5. Map provider schema drift or parse failures to client-specific data errors.
6. Add a source config model in `shallweswim/config/locations.py`.
7. Add a feed wrapper and factory branch in `shallweswim/core/feeds.py`.
8. Register the client in `shallweswim/api/routes.py`.
9. Add a debug script in `shallweswim/scripts/` if this is a new provider.

## Tests and Documentation

New feeds need both unit and live coverage.

Unit tests should cover:

- URL construction and query parameters.
- Unit conversion and timezone normalization.
- Required columns or fields.
- Empty source data.
- Provider error payloads.
- Retryable HTTP/network behavior when practical.
- Feed factory dispatch and missing-client errors.

Live integration tests should hit the real upstream source with a small bounded
request and assert the app-native shape, plausible values, monotonic timestamps,
and local-naive index behavior.

Documentation should update:

- `README.md` when users need to know about the source, debug script, or
  integration-test behavior.
- `ARCHITECTURE.md` when the source adds a new client, source contract, datum
  policy, retry/concurrency behavior, or runtime assumption.
- `shallweswim/scripts/README.md` when adding a debug or investigation script.

## Review Before Commit

Before committing a new feed, ask for a serious second look. The review should
challenge both the implementation and the premise:

- Is this source the right one?
- Were the source IDs, endpoints, keys/tokens, and permission assumptions
  explicitly approved?
- Is a new client justified?
- Is the data quality good enough?
- Are the failure modes handled clearly?
- Is the ongoing maintenance burden acceptable?

It is better to reject a shaky source before it becomes part of the app.

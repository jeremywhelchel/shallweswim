# Adding a New Location

Want your favorite swim spot in Shall We Swim? Great. This project is meant to
grow from real swimmers saying, "I actually swim here, and this data would help."

You do not need to be a software engineer to start a good location proposal. The
most valuable early work is local knowledge: where people swim, what conditions
matter, which nearby sensors locals trust, and which sources are clearly wrong.
Code can come later, and AI coding tools can help with that part.

This guide has two audiences:

- **Swimmers and local contributors** gather location facts and source ideas.
  That is a complete and valuable contribution.
- **Maintainers, technical contributors, and AI coding agents** turn approved
  proposals into config and code.

## What Makes a Good Location?

A good location is not just a dot on a map. It is a swim spot where conditions
are useful enough to check before getting in the water.

Strong candidates usually have:

- A real swim community, event, club, or well-known swim route.
- A clear place name that swimmers recognize.
- Public information about the swim spot.
- At least one useful condition source: water temperature, tide, current,
  webcam, or something else swimmers actually care about.
- Data that is local enough to be honest. A buoy halfway across the sea may be
  interesting, but it may not be useful for deciding whether to swim today.

If the location sounds fun but the data source is vague, that is fine. Put it in
`TODO.md` or open a discussion first. Not every good idea needs to become code
immediately.

## What to Gather First

This is the proposal phase. No code is required.

Start with the human facts:

- Swim spot name.
- City/region/country.
- A one-sentence description.
- A link to a swim club, local group, race, venue, or other page that explains
  the spot.
- Approximate latitude and longitude. A dropped map pin is fine at first.
- Local timezone.
- Whether swimmers usually think in Fahrenheit or Celsius.

Then gather condition-source ideas:

- Where do locals check water temperature?
- Are there nearby tide predictions?
- Are currents important, or is this mostly a tide/temperature location?
- Is there a webcam people use?
- Are the sources official/public, community-maintained, paid, or scraped from a
  web page?

Do not worry if you do not know all of this. A partial proposal with good local
context is still useful.

## Ask an AI Coding Agent for Help

This is a good project for an AI coding agent because adding a location is often
structured research plus a small code change. Give the agent local facts and ask
it to inspect the repo before changing anything.

Useful prompts:

```text
I want to add [swim spot] in [place] to this app. First, do not make code
changes. Read NEW_LOCATION.md and inspect the existing location configs. Tell me
what metadata is missing and what data sources we need to evaluate.
```

```text
Research possible water temperature and tide sources for [swim spot]. Prefer
official/public sources and tell me how close each sensor is to the swim spot.
Do not choose a source or edit config until I approve it.
```

```text
The approved source uses an existing client in this repo. Add the location as a
config-only change, keep the description to one sentence, and do not add tests
that simply duplicate ordinary config values.
```

```text
This source needs a new client. Stop and make a plan using NEW_DATA_FEED.md
before writing code.
```

Important: station IDs, buoy IDs, source endpoints, and data-source choices
should be explicitly approved before they are committed.

## The Big Decision

The key question is:

> Can this location use a data client we already have?

If yes, the change is usually small. It mostly means adding a location config
and pointing it at an existing source type.

If no, pause. A new data client is a bigger project because the app must learn
how to fetch, validate, retry, test, and document a new upstream source. Use
[NEW_DATA_FEED.md](NEW_DATA_FEED.md) for that path.

## If Existing Clients Are Enough

This is the happy path. Reuse an existing source config such as:

- `CoopsTempFeedConfig`
- `CoopsTideFeedConfig`
- `CoopsCurrentsFeedConfig`
- `CspfTempFeedConfig`
- `NdbcTempFeedConfig`
- `NwisCurrentFeedConfig`
- `NwisTempFeedConfig`
- `MarineInstituteTideFeedConfig`
- `LocalHarmonicTideFeedConfig`
- `IrishLightsTempFeedConfig`

The work is usually:

1. Add a `LocationConfig` in `shallweswim/config/locations.py`.
2. Use the approved source IDs and source config types.
3. Check that source citations render clearly.
4. Run the relevant debug script or live integration test for the source.
5. Run pre-commit before committing.

Tests are not required just because a new config value was added. Add or update
tests only when the change introduces new behavior, a new invariant, or a
source relationship that should not accidentally regress.

## If a New Data Feed Is Needed

Do not bury a new upstream API inside a location config change.

A new feed needs its own review because it adds an external dependency and new
failure modes. Before using it in production config, it should have:

- A narrow async client.
- Unit tests.
- A live integration test.
- A debug script if the provider is new.
- Documentation.
- A maintainer second look at both the implementation and whether the source is
  the right choice.

See [NEW_DATA_FEED.md](NEW_DATA_FEED.md).

## Source Quality

The best source is not always the closest source, but closeness matters. Be
honest about what the source measures.

For water temperature, live and historical data should ideally come from the
same physical sensor. If they do not, document the mismatch and review it before
using both.

Before approving a source, ask:

- Where is the actual sensor, buoy, gauge, or station?
- How far is it from the swim spot?
- Is it measuring the same water body swimmers care about?
- What units, timezone, cadence, history depth, and latency does it use?
- Does it require an API key, permission, payment, or fragile scraping?
- Is the source stable enough that the app can depend on it?

Keep source names and citations honest. If a buoy is offshore, regional,
delayed, or a fallback, say so.

## Implementation Details

Location configuration lives in `shallweswim/config/locations.py`.

Every production location needs:

- `code`
- `name`
- `nav_label`
- `swim_location`
- `swim_location_link`
- `location_info_source`
- `latitude`
- `longitude`
- `timezone`
- `default_temperature_unit`
- `description`

Temperature notes:

- `default_temperature_unit` must be explicit.
- `live_temp_source` and `historic_temp_source` are separate on purpose.
- Shared live/historical temperature sources should use `_shared_temp_sources`
  so citations de-duplicate.
- Different live/historical temperature sources are allowed only after review.

Tide notes:

- Know the datum.
- The app displays tide heights in feet.
- Some non-US tide sources provide meters and may need datum offsets.

Presentation notes:

- Keep descriptions to one sentence.
- Prefer authoritative local links.
- Webcam embeds need technical validation and permission/licensing review before
  production use.

## Before Commit

This section is for the person or coding agent making the code change. If you
are proposing a location with local facts and source ideas, you do not need to
run these commands.

For config-only additions, check that:

- The location appears in `/api/locations` and `/api/app/bootstrap`.
- The configured feeds initialize locally.
- Source citations appear in the right place and are readable.
- The relevant debug script or integration test proves the live source still
  works.
- Pre-commit passes.

Ask a maintainer or reviewer for a second look before committing production
location changes. The review should be willing to say "this source is not good
enough" or "this should wait" if that is the honest answer.

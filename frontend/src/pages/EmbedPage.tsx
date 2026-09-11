import { useState } from "react";
import { useParams } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";
import { useLocationConditions } from "../api/conditions";
import type { components } from "../api/generated";
import { PageMessage } from "../components/PageMessage";
import { usePageTitle } from "../hooks/usePageTitle";
import { formatMagnitude, formatTime, formatTimestamp } from "../lib/format";
import { locationPageTitle } from "../lib/pageTitle";
import type { TemperatureUnit } from "../lib/preferences";
import { TemperatureSummary, WindyEmbed } from "./LocationPage";
import "../styles/embed.css";

type Location = components["schemas"]["AppBootstrapLocation"];

export function EmbedPage() {
  const { locationCode = "" } = useParams();
  const bootstrap = useAppBootstrap();
  const location = bootstrap.data?.locations[locationCode];
  usePageTitle(
    location
      ? locationPageTitle(location.metadata.swim_location)
      : "Swimming conditions | shall we swim?",
  );

  return (
    <main className="swim-embed">
      {location ? (
        <EmbedConditions key={locationCode} location={location} />
      ) : (
        <PageMessage
          title={
            bootstrap.isPending
              ? "Loading swimming conditions"
              : bootstrap.isError
                ? "Unable to load swimming conditions"
                : "Location not found"
          }
        />
      )}
      <footer className="text-sm">
        For more info see{" "}
        <a
          className="text-swim-blue underline"
          href={location ? `/${location.metadata.code}` : "/"}
          target="_blank"
          rel="noopener noreferrer"
        >
          shallweswim.today
        </a>
        .
      </footer>
    </main>
  );
}

function EmbedConditions({ location }: { location: Location }) {
  const { metadata } = location;
  const conditions = useLocationConditions(metadata.code);
  const [unit, setUnit] = useState<TemperatureUnit>(
    metadata.default_temperature_unit,
  );
  const { temperature, tides, current } = conditions.data ?? {};
  const features = metadata.features;
  const partial =
    (features.temperature && !temperature) ||
    (features.tides &&
      (!tides?.past.length || (tides?.next.length ?? 0) < 2)) ||
    (features.currents && !current);
  const status = conditions.isPending
    ? "Loading latest conditions…"
    : conditions.isError
      ? conditions.data
        ? "Could not refresh latest conditions. Showing last loaded data."
        : "Unable to load latest conditions. Please try again later."
      : partial
        ? "Some conditions are currently unavailable."
        : "";
  const events = [tides?.past.at(-1), tides?.next[0], tides?.next[1]];

  return (
    <>
      <header>
        <h1 className="font-semibold text-xl leading-tight">
          Swimming conditions at {metadata.swim_location}
        </h1>
      </header>
      <div
        role="status"
        className={status ? "text-sm text-swim-ink" : "sr-only"}
      >
        {status}
      </div>
      <div
        className={`grid items-start gap-3 ${features.temperature && (features.tides || features.currents) ? "md:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]" : ""}`}
      >
        {features.temperature ? (
          conditions.isPending ? (
            <section className="embed-card" aria-busy="true">
              <h2 className="font-semibold">Water Temperature</h2>
              <p className="mt-2 text-sm">Loading water temperature…</p>
            </section>
          ) : (
            <TemperatureSummary
              className="embed-card"
              conditions={conditions.data}
              hasError={conditions.isError && !conditions.data}
              location={location}
              onSetTemperatureUnit={setUnit}
              temperatureUnit={unit}
            />
          )
        ) : null}
        {features.tides || features.currents ? (
          <div className="min-w-0 space-y-3">
            {features.tides ? (
              <section aria-label="Tides" className="embed-card">
                <h2 className="mb-2 font-semibold">Tides</h2>
                <div className="grid grid-cols-3 gap-2">
                  {events.map((event, index) => (
                    <div
                      className="embed-tide"
                      key={["Last", "Next", "Following"][index]}
                    >
                      <h3 className="font-semibold capitalize">
                        {["Last", "Next", "Following"][index]} {event?.type}{" "}
                        tide
                      </h3>
                      {event ? (
                        <time dateTime={event.time} className="mt-2 block">
                          <span className="block">
                            {formatTimestamp(event.time, {
                              weekday: "short",
                              month: "short",
                              day: "numeric",
                            })}
                          </span>
                          <strong className="mt-1 block tabular-nums">
                            {formatTime(event.time)}
                          </strong>
                        </time>
                      ) : (
                        <p className="mt-2">
                          {conditions.isPending ? "Loading…" : "Unavailable"}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            ) : null}
            {features.currents ? (
              <section aria-label="Current estimate" className="embed-card">
                <h2 className="font-semibold">Current estimate</h2>
                {current ? (
                  <>
                    <p className="mt-2 text-base">
                      {current.state_description ??
                        current.direction ??
                        "Observed flow"}
                    </p>
                    <p className="mt-1 text-sm">
                      <strong className="font-mono text-xl">
                        {formatMagnitude(Math.abs(current.magnitude))}
                      </strong>{" "}
                      knots
                    </p>
                  </>
                ) : (
                  <p className="mt-2 text-sm">
                    {conditions.isPending ? "Loading…" : "Unavailable"}
                  </p>
                )}
                {features.water_movement_detail ? (
                  <a
                    className="mt-2 inline-block text-sm text-swim-blue underline"
                    href={`/${metadata.code}?detail=open`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Current details
                  </a>
                ) : null}
              </section>
            ) : null}
          </div>
        ) : null}
      </div>
      <section aria-label="Windy map and forecast" className="embed-card">
        <h2 className="mb-2 font-semibold">Map and forecast</h2>
        <WindyEmbed
          config={location.integrations.windy}
          metadata={metadata}
          temperatureUnit={unit}
        />
      </section>
    </>
  );
}

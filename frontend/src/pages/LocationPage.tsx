import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  Anchor,
  GitHub,
  Shuffle,
  Thermometer,
  Truck,
  Video,
} from "react-feather";
import { Link } from "react-router-dom";
import { useLocationConditions } from "../api/conditions";
import type { components } from "../api/generated";
import { useTransitRoute } from "../api/transit";
import { useDeferredImage } from "../hooks/useDeferredImage";
import {
  formatMagnitude,
  formatStationTimestamp,
  formatTideHeight,
  formatTime,
} from "../lib/format";

type AppBootstrapResponse = components["schemas"]["AppBootstrapResponse"];
type AppBootstrapLocation = components["schemas"]["AppBootstrapLocation"];
type AppLocationMetadata = components["schemas"]["AppLocationMetadata"];
type AppExternalIntegrations = components["schemas"]["AppExternalIntegrations"];
type LocationConditions = components["schemas"]["LocationConditions"];
type CurrentInfo = components["schemas"]["CurrentInfo"];
type TideEntry = components["schemas"]["TideEntry"];
type TideState = components["schemas"]["TideState"];
type TransitRouteConfig = components["schemas"]["TransitRouteConfig"];
type YouTubeLiveConfig = components["schemas"]["YouTubeLiveConfig"];

const METER_SEGMENT_COUNT = 32;

type LocationPageProps = {
  bootstrap: AppBootstrapResponse;
  locationCode: string;
  preserveDefaultUrl?: boolean;
};

declare global {
  interface Window {
    YT?: {
      Player: new (
        elementId: string,
        options: {
          playerVars: Record<string, number>;
          events: {
            onReady: (event: {
              target: { mute: () => void; playVideo: () => void };
            }) => void;
            onError: () => void;
            onStateChange: () => void;
          };
        },
      ) => unknown;
    };
    onYouTubeIframeAPIReady?: () => void;
  }
}

export function LocationPage({
  bootstrap,
  locationCode,
  preserveDefaultUrl = false,
}: LocationPageProps) {
  const location = bootstrap.locations[locationCode];
  const conditions = useLocationConditions(locationCode);
  const firstConditionsSettled = !conditions.isPending;

  if (!location) {
    return (
      <ShellMessage
        tone="warning"
        title="Location not found"
        body={`No app metadata exists for ${locationCode}.`}
      />
    );
  }

  const staleMessage =
    conditions.isRefetchError && conditions.data
      ? "Could not refresh latest conditions. Showing last loaded data."
      : null;
  const unavailableMessage =
    conditions.isError && !conditions.data
      ? "Unable to load latest conditions. Please try again later."
      : null;

  return (
    <div className="space-y-5 sm:space-y-8">
      <header>
        <p className="font-medium text-swim-current text-xs uppercase sm:text-sm">
          {preserveDefaultUrl ? "Default location" : location.metadata.name}
        </p>
        <h1 className="mt-0.5 font-semibold text-2xl text-swim-blue sm:mt-1 sm:text-3xl">
          shall we swim today?
        </h1>
        <p className="mt-1 text-sm text-slate-700 sm:mt-2 sm:text-base">
          ...at{" "}
          <a
            className="text-swim-blue underline"
            href={location.metadata.swim_location_link}
          >
            {location.metadata.swim_location}
          </a>
        </p>
      </header>

      <ConditionsSummary
        conditions={conditions.data}
        hasError={conditions.isError && !conditions.data}
        isLoading={conditions.isPending}
        locationCode={locationCode}
      />
      {staleMessage || unavailableMessage ? (
        <p className="border-swim-alert border-l-4 bg-white px-4 py-3 text-sm text-swim-alert">
          {staleMessage ?? unavailableMessage}
        </p>
      ) : null}

      {location.metadata.features.windy ? (
        <Section title="Forecast">
          <WindyEmbed metadata={location.metadata} />
        </Section>
      ) : null}

      {location.metadata.features.webcam &&
      location.integrations.youtube_live ? (
        <Section title="Live Webcam">
          <YouTubeLiveEmbed config={location.integrations.youtube_live} />
        </Section>
      ) : null}

      {location.metadata.features.temperature ? (
        <Section title="Temperature Trends">
          <TemperaturePlots
            enabled={firstConditionsSettled}
            locationCode={locationCode}
          />
        </Section>
      ) : null}

      {location.metadata.features.transit &&
      location.integrations.transit_routes?.length ? (
        <Section title="Transit Status">
          <TransitStatusSection routes={location.integrations.transit_routes} />
        </Section>
      ) : null}

      <SourcesList bootstrap={bootstrap} location={location} />
    </div>
  );
}

export function ConditionsSummary({
  conditions,
  hasError,
  isLoading,
  locationCode,
}: {
  conditions?: LocationConditions;
  hasError: boolean;
  isLoading: boolean;
  locationCode: string;
}) {
  if (isLoading) {
    return <ShellMessage title="Loading latest conditions" />;
  }

  return (
    <section className="grid gap-0 overflow-hidden rounded border border-swim-line bg-white md:grid-cols-[1fr_2fr] md:items-start md:gap-4 md:overflow-visible md:border-0 md:bg-transparent">
      <TemperatureSummary conditions={conditions} hasError={hasError} />
      <WaterMovementSummary
        current={hasError ? undefined : conditions?.current}
        locationCode={locationCode}
        tides={hasError ? undefined : conditions?.tides}
      />
    </section>
  );
}

function TemperatureSummary({
  conditions,
  hasError,
}: {
  conditions?: LocationConditions;
  hasError: boolean;
}) {
  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <h2 className="font-semibold text-base md:text-lg">Water Temperature</h2>
      <div className="mt-1 flex items-baseline gap-2 md:block">
        <p className="text-sm text-slate-700 md:mt-2 md:text-base">
          The water is currently
        </p>
        <p className="font-mono font-semibold text-2xl text-swim-blue md:mt-1 md:text-3xl">
          {conditions?.temperature && !hasError
            ? `${conditions.temperature.water_temp}°${conditions.temperature.units || "F"}`
            : "Unavailable"}
        </p>
      </div>
      <p className="mt-1 text-xs text-slate-600 md:mt-2 md:text-sm">
        {conditions?.temperature && !hasError ? (
          <>
            at{" "}
            <span>
              {conditions.temperature.station_name ||
                conditions.location.name ||
                "station"}
            </span>
            {formatStationTimestamp(conditions.temperature.timestamp) ? (
              <>
                {" as of "}
                <span className="font-mono">
                  {formatStationTimestamp(conditions.temperature.timestamp)}
                </span>
                .
              </>
            ) : (
              "."
            )}
          </>
        ) : (
          "Current water temperature is unavailable."
        )}
      </p>
    </div>
  );
}

function WaterMovementSummary({
  current,
  locationCode,
  tides,
}: {
  current?: CurrentInfo | null;
  locationCode: string;
  tides?: LocationConditions["tides"];
}) {
  const pastTide = tides?.past?.at(-1);
  const nextTide = tides?.next?.[0];
  const description = describeWaterMovement(tides?.state, current);

  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <h2 className="font-semibold text-base md:text-lg">Water Movement</h2>
      <p className="mt-2 font-semibold text-lg text-swim-current leading-snug md:text-2xl">
        {description}
      </p>
      <TideInstrument
        nextTide={nextTide}
        previousTide={pastTide}
        state={tides?.state}
      />
      <CurrentInstrument current={current} />
      {current?.direction ? (
        <Link
          className="mt-1 inline-block text-xs text-swim-blue underline md:mt-2 md:text-sm"
          to={`/${locationCode}/currents`}
        >
          Current details
        </Link>
      ) : null}
    </div>
  );
}

function describeWaterMovement(
  tideState?: TideState | null,
  current?: CurrentInfo | null,
) {
  if (current) {
    const tidePosition = describeTidePosition(tideState);

    if (isSlackCurrent(current)) {
      if (tidePosition) {
        return `It's ${tidePosition} and calm.`;
      }
      if (current.phase === "slack_before_flood") {
        return "It's calm before the water starts coming in.";
      }
      if (current.phase === "slack_before_ebb") {
        return "It's calm before the water starts going out.";
      }
      return "It's calm right now.";
    }

    const direction = currentDirectionPhrase(current);
    const speed = currentSpeedPhrase(current);
    const trend = currentTrendClause(current);
    const tidePrefix = currentTidePrefix(tideState);

    return `${tidePrefix}the water is ${direction}${speed ? ` ${speed}` : ""}${
      trend ?? ""
    }.`;
  }

  if (tideState?.trend) {
    return `The tide is ${tideState.trend}.`;
  }

  return "Water movement is unavailable right now.";
}

function currentTidePrefix(tideState?: TideState | null) {
  const position = describeTidePosition(tideState);
  if (position) {
    return `${capitalize(position)}, `;
  }

  switch (tideState?.trend) {
    case "rising":
      return "The tide is rising, and ";
    case "falling":
      return "The tide is falling, and ";
    case "steady":
      return "The tide is steady, and ";
    default:
      return "Right now, ";
  }
}

function describeTidePosition(tideState?: TideState | null) {
  if (typeof tideState?.height_pct !== "number") {
    return null;
  }
  if (tideState.height_pct <= 0.15) {
    return "near low tide";
  }
  if (tideState.height_pct >= 0.85) {
    return "near high tide";
  }
  return null;
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function isSlackCurrent(current: CurrentInfo) {
  return (
    current.phase === "slack" ||
    current.phase === "slack_before_flood" ||
    current.phase === "slack_before_ebb"
  );
}

function currentDirectionPhrase(current: CurrentInfo) {
  if (current.phase === "ebb" || current.direction === "ebbing") {
    return "going out";
  }
  if (current.phase === "flood" || current.direction === "flooding") {
    return "coming in";
  }
  return "moving";
}

function currentSpeedPhrase(current: CurrentInfo) {
  switch (current.strength) {
    case "strong":
      return "fast";
    case "moderate":
      return "steadily";
    case "light":
      return "gently";
    default:
      return null;
  }
}

function currentTrendClause(current: CurrentInfo) {
  switch (current.trend) {
    case "building":
      return " and getting stronger";
    case "easing":
      return ", but starting to ease";
    case "steady":
      return " and holding steady";
    default:
      return null;
  }
}

function TideInstrument({
  nextTide,
  previousTide,
  state,
}: {
  nextTide?: TideEntry;
  previousTide?: TideEntry;
  state?: TideState | null;
}) {
  if (!state) {
    return null;
  }

  const percent =
    typeof state.height_pct === "number" && Number.isFinite(state.height_pct)
      ? Math.max(0, Math.min(100, Math.round(state.height_pct * 100)))
      : null;
  const trend = state.trend ?? "steady";
  const tideBounds = getTideBounds(previousTide, nextTide);

  return (
    <DriftBar
      accent="tide"
      label="TIDE"
      leftLabel={
        tideBounds ? (
          <>
            low {formatTideHeight(tideBounds.low.prediction)} {state.units}
            <span className="whitespace-nowrap">
              {" · "}
              {formatTime(tideBounds.low.time)}
            </span>
          </>
        ) : (
          "low"
        )
      }
      percent={percent}
      rightLabel={
        tideBounds ? (
          <>
            high {formatTideHeight(tideBounds.high.prediction)} {state.units}
            <span className="whitespace-nowrap">
              {" · "}
              {formatTime(tideBounds.high.time)}
            </span>
          </>
        ) : (
          "high"
        )
      }
      trend={trend}
      value={`${formatTideHeight(state.estimated_height ?? undefined)} ${state.units}`}
    />
  );
}

function getTideBounds(previousTide?: TideEntry, nextTide?: TideEntry) {
  if (!previousTide || !nextTide || previousTide.type === nextTide.type) {
    return null;
  }

  const tides = [previousTide, nextTide];
  const low = tides.find((tide) => tide.type === "low");
  const high = tides.find((tide) => tide.type === "high");

  return low && high ? { high, low } : null;
}

function CurrentInstrument({ current }: { current?: CurrentInfo | null }) {
  if (current?.magnitude_pct == null) {
    return null;
  }

  const range = current.range;
  const percent = Math.max(
    0,
    Math.min(100, Math.round(current.magnitude_pct * 100)),
  );
  const phase =
    current.phase?.replaceAll("_", " ") ?? current.direction ?? null;
  const trend = current.trend ?? "steady";

  return (
    <DriftBar
      accent="current"
      label="CURRENT"
      leftLabel={
        range ? (
          <>
            slack
            <span className="whitespace-nowrap">
              {" · "}
              {formatTime(range.slack.timestamp)}
            </span>
          </>
        ) : (
          "slack"
        )
      }
      percent={percent}
      phase={phase}
      rightLabel={
        range ? (
          <>
            peak {formatMagnitude(range.peak.magnitude)} {range.peak.units}
            <span className="whitespace-nowrap">
              {" · "}
              {formatTime(range.peak.timestamp)}
            </span>
          </>
        ) : (
          "peak"
        )
      }
      trend={trend}
      value={`${formatMagnitude(current.magnitude)} kt`}
    />
  );
}

const DRIFT_FULL_BLOCK = "█";
const DRIFT_MARKER = "│";

const DRIFT_ACCENT = {
  tide: {
    color: "#5b7f2a",
    soft: "rgba(91, 127, 42, 0.28)",
    cardBg: "bg-[#f6f9f1]",
    labelText: "text-swim-tide",
    arrowText: "text-swim-alert",
  },
  current: {
    color: "#006b8f",
    soft: "rgba(0, 107, 143, 0.28)",
    cardBg: "bg-[#eff7fa]",
    labelText: "text-swim-current",
    arrowText: "text-swim-current",
  },
} as const;

type DriftBarProps = {
  accent: keyof typeof DRIFT_ACCENT;
  label: string;
  leftLabel: ReactNode;
  percent: number | null;
  phase?: string | null;
  rightLabel: ReactNode;
  trend: string;
  value: string;
};

function DriftBar({
  accent,
  label,
  leftLabel,
  percent,
  phase,
  rightLabel,
  trend,
  value,
}: DriftBarProps) {
  if (percent == null) {
    return null;
  }

  const direction = trendDirection(trend);
  const fillChars = Math.max(
    0,
    Math.min(
      METER_SEGMENT_COUNT,
      Math.round((percent / 100) * METER_SEGMENT_COUNT),
    ),
  );
  const emptyChars = METER_SEGMENT_COUNT - fillChars;
  const accentTokens = DRIFT_ACCENT[accent];

  const fillGradient =
    direction === "up"
      ? `linear-gradient(90deg, ${accentTokens.soft} 0%, ${accentTokens.color} 65%, ${accentTokens.color} 100%)`
      : direction === "down"
        ? `linear-gradient(90deg, ${accentTokens.color} 0%, ${accentTokens.color} 55%, ${accentTokens.soft} 100%)`
        : accentTokens.color;

  const pulseClass =
    direction === "down"
      ? "drift-arrow-pulse-left"
      : direction === "up"
        ? "drift-arrow-pulse-right"
        : "";

  return (
    <div
      className={`mt-2 rounded border border-swim-line px-3 py-2 md:mt-3 md:px-4 md:py-2.5 ${accentTokens.cardBg}`}
    >
      <div className="flex items-baseline justify-between gap-2 font-mono text-[11px] font-bold uppercase tracking-[0.1em] md:text-xs">
        <span>
          <span className={accentTokens.labelText}>{label}</span>{" "}
          <span className="font-normal normal-case tracking-normal tabular-nums text-swim-ink">
            {value}
          </span>
        </span>
        <span className="text-swim-ink">
          {phase ? (
            <>
              <span>{phase}</span>
              {" · "}
            </>
          ) : null}
          {direction !== "steady" ? (
            <span className="md:hidden">
              <span>{trend}</span>
              {" · "}
            </span>
          ) : null}
          {percent}%
        </span>
      </div>

      <div
        aria-hidden="true"
        className="my-1.5 flex items-center whitespace-nowrap font-mono text-[clamp(12px,4vw,16px)] leading-none md:text-[clamp(16px,1.4vw,20px)]"
      >
        <span className="font-semibold text-slate-500">[</span>
        <span
          className="-tracking-[0.06em]"
          style={{
            backgroundImage: fillGradient,
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
          }}
        >
          {DRIFT_FULL_BLOCK.repeat(fillChars)}
        </span>
        <span
          className="-mx-px font-black text-swim-ink"
          style={{ fontSize: "1.25em", lineHeight: 0.85 }}
        >
          {DRIFT_MARKER}
        </span>
        <span className="-tracking-[0.06em] text-[#cdd6db]">
          {DRIFT_FULL_BLOCK.repeat(emptyChars)}
        </span>
        <span className="font-semibold text-slate-500">]</span>
        {direction !== "steady" ? (
          <span
            className={`ml-3 font-sans text-[12px] font-extrabold uppercase tracking-wider md:text-[13px] ${accentTokens.arrowText} ${pulseClass}`}
          >
            {direction === "down" ? (
              <>
                ←
                <span className="hidden md:inline">
                  {" "}
                  <span>{trend}</span>
                </span>
              </>
            ) : (
              <>
                <span className="hidden md:inline">
                  <span>{trend}</span>{" "}
                </span>
                →
              </>
            )}
          </span>
        ) : null}
      </div>

      <div className="flex items-baseline justify-between gap-2 text-[11px] tabular-nums text-slate-500 md:text-xs">
        <span className="min-w-0">{leftLabel}</span>
        <span className="min-w-0 text-right">{rightLabel}</span>
      </div>
    </div>
  );
}

function trendDirection(trend: string): "up" | "down" | "steady" {
  if (trend === "rising" || trend === "building") {
    return "up";
  }
  if (trend === "falling" || trend === "easing") {
    return "down";
  }
  return "steady";
}

function WindyEmbed({ metadata }: { metadata: AppLocationMetadata }) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [embedSize, setEmbedSize] = useState({ width: 950, height: 350 });

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) {
      return;
    }

    const updateSize = () => {
      const width = Math.max(280, Math.round(wrapper.clientWidth));
      const height = 350;
      setEmbedSize({ width, height });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(wrapper);

    return () => observer.disconnect();
  }, []);

  const params = new URLSearchParams({
    lat: String(metadata.latitude),
    lon: String(metadata.longitude),
    detailLat: String(metadata.latitude),
    detailLon: String(metadata.longitude),
    width: String(embedSize.width),
    height: String(embedSize.height),
    zoom: "11",
    level: "surface",
    overlay: "waves",
    product: "ecmwfWaves",
    menu: "",
    message: "true",
    marker: "true",
    calendar: "now",
    pressure: "",
    type: "map",
    location: "coordinates",
    detail: "true",
    metricWind: "default",
    metricTemp: "°F",
    radarRange: "-1",
  });

  return (
    <div
      className="h-[350px] min-w-0 max-w-full overflow-hidden rounded border border-swim-line bg-white"
      ref={wrapperRef}
    >
      <iframe
        className="block h-full w-full max-w-full border-0"
        src={`https://embed.windy.com/embed2.html?${params.toString()}`}
        title="Windy forecast"
      />
    </div>
  );
}

function YouTubeLiveEmbed({ config }: { config: YouTubeLiveConfig }) {
  useEffect(() => {
    function createPlayer() {
      if (!window.YT?.Player || !document.getElementById("bbcam_player")) {
        return;
      }

      new window.YT.Player("bbcam_player", {
        playerVars: {
          autoplay: 1,
          playsinline: 1,
          fs: 1,
          controls: 0,
          iv_load_policy: 3,
          rel: 0,
        },
        events: {
          onReady: (event) => {
            event.target.mute();
            event.target.playVideo();
          },
          onError: () => undefined,
          onStateChange: () => undefined,
        },
      });
    }

    if (window.YT?.Player) {
      createPlayer();
      return;
    }

    window.onYouTubeIframeAPIReady = createPlayer;
    if (
      !document.querySelector(
        'script[src="https://www.youtube.com/iframe_api"]',
      )
    ) {
      const script = document.createElement("script");
      script.async = true;
      script.src = "https://www.youtube.com/iframe_api";
      document.head.append(script);
    }
  }, []);

  return (
    <div className="aspect-video overflow-hidden rounded border border-swim-line bg-black">
      <iframe
        allowFullScreen
        className="h-full w-full"
        id="bbcam_player"
        scrolling="no"
        src={config.embed_url}
        title="Live webcam"
      />
    </div>
  );
}

function TemperaturePlots({
  enabled,
  locationCode,
}: {
  enabled: boolean;
  locationCode: string;
}) {
  const [selectedPlot, setSelectedPlot] = useState("12mo");
  const plots = [
    {
      key: "live",
      label: "Live",
      alt: "Live temperature plot",
      src: `/api/${locationCode}/plots/live_temps`,
    },
    {
      key: "2mo",
      label: "2 mo",
      alt: "2 month temperature plot, all years",
      src: `/api/${locationCode}/plots/historic_temps?period=2mo`,
    },
    {
      key: "12mo",
      label: "12 mo",
      alt: "12 month temperature plot, all years",
      src: `/api/${locationCode}/plots/historic_temps?period=12mo`,
    },
  ];
  const plotOptions = [...plots, { key: "all", label: "All" }];

  return (
    <div className="grid gap-3">
      <fieldset className="inline-flex w-fit overflow-hidden rounded border border-swim-line bg-white">
        <legend className="sr-only">Temperature plot range</legend>
        {plotOptions.map((plot) => (
          <button
            className={[
              "min-h-9 px-3 py-1.5 font-mono font-medium text-sm",
              plot.key === selectedPlot
                ? "bg-swim-blue text-white"
                : "bg-white text-swim-blue",
            ].join(" ")}
            key={plot.key}
            onClick={() => setSelectedPlot(plot.key)}
            type="button"
          >
            {plot.label}
          </button>
        ))}
      </fieldset>
      <div className="grid gap-3">
        {plots.map((plot) => (
          <DeferredPlotImage
            alt={plot.alt}
            enabled={enabled}
            isActive={selectedPlot === "all" || plot.key === selectedPlot}
            key={plot.key}
            src={plot.src}
          />
        ))}
      </div>
    </div>
  );
}

function DeferredPlotImage({
  alt,
  enabled,
  isActive,
  src,
}: {
  alt: string;
  enabled: boolean;
  isActive: boolean;
  src: string;
}) {
  const image = useDeferredImage(src, enabled);

  return (
    <div
      className={
        isActive
          ? "block rounded border border-swim-line bg-white p-2"
          : "hidden"
      }
    >
      {image.status === "loaded" && image.src ? (
        <img alt={alt} className="w-full" src={image.src} />
      ) : (
        <div className="flex min-h-28 items-center justify-center text-sm text-slate-600">
          {image.status === "unavailable" ? "Plot unavailable" : "Loading plot"}
        </div>
      )}
    </div>
  );
}

function TransitStatusSection({ routes }: { routes: TransitRouteConfig[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {routes.map((route) => (
        <TransitRouteCard key={route.goodservice_route_id} route={route} />
      ))}
    </div>
  );
}

function TransitRouteCard({ route }: { route: TransitRouteConfig }) {
  const status = useTransitRoute(route);
  const firstLoadUnavailable = status.isError && !status.data;
  const transit = firstLoadUnavailable
    ? { status: "Unavailable", destination: "unavailable" }
    : status.data;

  return (
    <article className="border-swim-line rounded border bg-white p-4">
      <a
        className="flex items-center gap-3 text-swim-ink"
        href={`https://goodservice.io/trains/${route.goodservice_route_id}/S`}
      >
        {route.icon_url ? (
          <img
            alt={`${route.label} train`}
            className="h-12 w-12"
            src={route.icon_url}
          />
        ) : (
          <span className="flex h-12 w-12 items-center justify-center rounded-full bg-swim-blue font-mono font-semibold text-white">
            {route.label}
          </span>
        )}
        <div>
          <h3 className="font-semibold">
            <span className="font-mono">{route.label}</span> train
          </h3>
          <p className="text-sm text-slate-600">
            to{" "}
            <span className="font-mono">{transit?.destination ?? "..."}</span>
          </p>
        </div>
      </a>
      <p
        className={[
          "mt-4 inline-flex rounded px-3 py-1 font-mono font-medium text-sm",
          transitStatusClass(transit?.status),
        ].join(" ")}
      >
        {transit?.status ?? "..."}
      </p>
      {transit?.delay ? (
        <TransitAlert label="Delay" value={transit.delay} />
      ) : null}
      {transit?.serviceChange ? (
        <TransitAlert label="Service Change" value={transit.serviceChange} />
      ) : null}
      {transit?.serviceIrregularity ? (
        <TransitAlert
          label="Service Irregularity"
          value={transit.serviceIrregularity}
        />
      ) : null}
    </article>
  );
}

function TransitAlert({ label, value }: { label: string; value: string }) {
  return (
    <div className="mt-3 border-swim-line border-t pt-3 text-sm">
      <p className="font-medium text-swim-alert">{label}</p>
      <p className="mt-1 font-mono text-slate-700">{value}</p>
    </div>
  );
}

function transitStatusClass(status: string | undefined) {
  switch (status) {
    case "Delay":
    case "No Service":
      return "bg-red-100 text-red-900";
    case "Service Change":
      return "bg-orange-100 text-orange-900";
    case "Slow":
    case "Not Good":
      return "bg-yellow-100 text-yellow-900";
    case "Good Service":
      return "bg-green-100 text-green-900";
    default:
      return "bg-slate-100 text-slate-700";
  }
}

function SourcesList({
  bootstrap,
  location,
}: {
  bootstrap: AppBootstrapResponse;
  location: AppBootstrapLocation;
}) {
  const citations = location.metadata.citations;

  return (
    <Section title="Sources">
      <div className="overflow-hidden rounded border border-swim-line bg-white">
        <SourceHtml
          icon={<Thermometer aria-hidden="true" />}
          label="Temperature"
          html={citations.temperature}
        />
        <SourceHtml
          icon={<Anchor aria-hidden="true" />}
          label="Tides"
          html={citations.tides}
        />
        <SourceHtml
          icon={<Shuffle aria-hidden="true" />}
          label="Currents"
          html={citations.currents}
        />
        <SourceLink
          icon={<Video aria-hidden="true" />}
          includeLabel={false}
          label="Webcam"
          link={location.integrations.webcam_source}
          linkFirst
          secondaryLink={location.integrations.webcam_alternative}
          secondaryPrefix="Alternate:"
        />
        <SourceLink
          icon={<Truck aria-hidden="true" />}
          label="Transit"
          link={location.integrations.transit_source}
        />
        <SourceLink
          icon={<GitHub aria-hidden="true" />}
          label="GitHub"
          link={bootstrap.source_code_link}
        />
      </div>
      <LocationNav bootstrap={bootstrap} currentCode={location.metadata.code} />
    </Section>
  );
}

function SourceHtml({
  icon,
  label,
  html,
}: {
  icon: ReactNode;
  label: string;
  html?: string | null;
}) {
  if (!html) {
    return null;
  }

  return (
    <div className="grid grid-cols-[1.75rem_1fr] gap-3 border-swim-line border-b p-3 text-sm">
      <span
        aria-label={label}
        className="mt-0.5 text-slate-500 [&_svg]:h-4 [&_svg]:w-4 [&_svg]:stroke-[1.8]"
        role="img"
      >
        {icon}
      </span>
      <span
        className="[&_a]:text-swim-blue [&_a]:underline"
        // biome-ignore lint/security/noDangerouslySetInnerHtml: Citation HTML is trusted repository-controlled bootstrap content.
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

function SourceLink({
  icon,
  label,
  link,
  secondaryLink,
  secondaryPrefix,
  includeLabel = true,
  linkFirst = false,
}: {
  icon: ReactNode;
  label: string;
  link?: AppExternalIntegrations["webcam_source"];
  secondaryLink?: AppExternalIntegrations["webcam_source"];
  secondaryPrefix?: string;
  includeLabel?: boolean;
  linkFirst?: boolean;
}) {
  if (!link) {
    return null;
  }

  return (
    <div className="grid grid-cols-[1.75rem_1fr] gap-3 border-swim-line border-b p-3 text-sm">
      <span
        aria-label={label}
        className="mt-0.5 text-slate-500 [&_svg]:h-4 [&_svg]:w-4 [&_svg]:stroke-[1.8]"
        role="img"
      >
        {icon}
      </span>
      <span>
        <span>
          {includeLabel ? `${label}: ` : null}
          {linkFirst ? (
            <>
              <a className="text-swim-blue underline" href={link.url}>
                {link.label}
              </a>
              {link.description ? ` ${link.description}` : null}
            </>
          ) : (
            <>
              {link.description ? `${link.description} ` : null}
              <a className="text-swim-blue underline" href={link.url}>
                {link.label}
              </a>
            </>
          )}
        </span>
        {secondaryLink ? (
          <span className="mt-1 block">
            {secondaryPrefix ? `${secondaryPrefix} ` : null}
            <a className="text-swim-blue underline" href={secondaryLink.url}>
              {secondaryLink.label}
            </a>
            {secondaryLink.description
              ? ` - ${secondaryLink.description}`
              : null}
          </span>
        ) : null}
      </span>
    </div>
  );
}

function LocationNav({
  bootstrap,
  currentCode,
}: {
  bootstrap: AppBootstrapResponse;
  currentCode: string;
}) {
  return (
    <div className="mt-4">
      <h3 className="font-semibold text-lg">Locations</h3>
      <p className="mt-1 text-sm text-slate-600">
        Additional locations are in the works.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {bootstrap.location_order.map((code) => (
          <Link
            className={[
              "rounded border px-3 py-2 text-sm font-medium uppercase",
              code === currentCode
                ? "border-swim-blue bg-swim-blue text-white"
                : "border-swim-line bg-white text-swim-blue",
            ].join(" ")}
            key={code}
            to={`/${code}`}
          >
            {code}
          </Link>
        ))}
        <Link
          className="rounded border border-swim-line bg-white px-3 py-2 font-medium text-sm text-swim-blue"
          to="/locations"
        >
          All
        </Link>
      </div>
    </div>
  );
}

function Section({
  children,
  title,
}: {
  children: React.ReactNode;
  title: string;
}) {
  return (
    <section>
      <h2 className="mb-3 font-semibold text-2xl text-swim-ink">{title}</h2>
      {children}
    </section>
  );
}

function ShellMessage({
  title,
  body,
  tone = "neutral",
}: {
  title: string;
  body?: string;
  tone?: "neutral" | "warning";
}) {
  return (
    <section
      className={[
        "rounded border bg-white p-4",
        tone === "warning" ? "border-swim-alert" : "border-swim-line",
      ].join(" ")}
    >
      <h1 className="font-semibold text-xl">{title}</h1>
      {body ? <p className="mt-2 text-slate-700">{body}</p> : null}
    </section>
  );
}

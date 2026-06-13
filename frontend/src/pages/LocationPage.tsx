import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  Anchor,
  GitHub,
  Map as MapIcon,
  Shuffle,
  Thermometer,
  Truck,
  Video,
} from "react-feather";
import { Link, useSearchParams } from "react-router-dom";
import { useLocationConditions } from "../api/conditions";
import type { components } from "../api/generated";
import { useTransitRoute } from "../api/transit";
import { useDeferredImage } from "../hooks/useDeferredImage";
import { usePageTitle } from "../hooks/usePageTitle";
import {
  formatMagnitude,
  formatStationTimestamp,
  formatTideHeight,
  formatTime,
} from "../lib/format";
import { LOCATION_NOT_FOUND_TITLE, locationPageTitle } from "../lib/pageTitle";
import {
  getTemperatureUnit,
  setTemperatureUnit as persistTemperatureUnit,
  type TemperatureUnit,
} from "../lib/preferences";
import {
  formatWaterTemperature,
  resolveTemperatureUnit,
} from "../lib/temperature";

type AppBootstrapResponse = components["schemas"]["AppBootstrapResponse"];
type AppBootstrapLocation = components["schemas"]["AppBootstrapLocation"];
type AppLocationMetadata = components["schemas"]["AppLocationMetadata"];
type LocationConditions = components["schemas"]["LocationConditions"];
type CurrentInfo = components["schemas"]["CurrentInfo"];
type TideEntry = components["schemas"]["TideEntry"];
type TideState = components["schemas"]["TideState"];
type TransitRouteConfig = components["schemas"]["TransitRouteConfig"];
type AppPresentationLink = components["schemas"]["AppPresentationLink"];
type AppWebcamConfig = components["schemas"]["AppWebcamConfig"];
type AppWindyConfig = components["schemas"]["AppWindyConfig"];

const PLANNER_MIN_MINUTES = -180;
const PLANNER_MAX_MINUTES = 1440;
const PLANNER_STEP_MINUTES = 15;
const AT_TIDE_EDGE_PCT = 0.07;
const NEAR_TIDE_EDGE_PCT = 0.15;
const GENTLE_CURRENT_MAX_KT = 0.4;
const FAST_CURRENT_MIN_KT = 1.0;
const TEMPERATURE_PLOT_IMAGE_SIZE = {
  width: 1200,
  height: 486,
} as const;
const WEBCAM_PRELOAD_MARGIN = "100px";
const NEW_TAB_REL = "noopener noreferrer";

type LocationPageProps = {
  bootstrap: AppBootstrapResponse;
  locationCode: string;
};

function isExternalHref(href: string | null | undefined) {
  return href?.startsWith("http://") || href?.startsWith("https://");
}

function externalLinkProps(href: string | null | undefined) {
  return isExternalHref(href) ? { rel: NEW_TAB_REL, target: "_blank" } : {};
}

function useExternalizedTrustedHtml(html: string | null | undefined) {
  return useMemo(() => {
    if (!html || typeof document === "undefined") {
      return html ?? "";
    }

    const template = document.createElement("template");
    template.innerHTML = html;
    for (const anchor of template.content.querySelectorAll("a[href]")) {
      if (isExternalHref(anchor.getAttribute("href"))) {
        anchor.setAttribute("target", "_blank");
        anchor.setAttribute("rel", NEW_TAB_REL);
      }
    }
    return template.innerHTML;
  }, [html]);
}

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

export function LocationPage({ bootstrap, locationCode }: LocationPageProps) {
  const location = bootstrap.locations[locationCode];
  usePageTitle(
    location
      ? locationPageTitle(location.metadata.swim_location)
      : LOCATION_NOT_FOUND_TITLE,
  );
  const [searchParams, setSearchParams] = useSearchParams();
  const [preferredTemperatureUnit, setPreferredTemperatureUnit] =
    useState<TemperatureUnit | null>(() => getTemperatureUnit());
  const supportsWaterMovementPlanning = Boolean(
    location?.metadata.features.water_movement_planning,
  );
  const supportsWaterMovementDetail = Boolean(
    location?.metadata.features.water_movement_detail,
  );
  const showsWaterMovement = Boolean(
    location?.metadata.features.tides ||
      supportsWaterMovementPlanning ||
      supportsWaterMovementDetail,
  );
  const showsObservedFlow = Boolean(
    location?.metadata.features.currents && !showsWaterMovement,
  );
  const hasWaterMovementControls =
    supportsWaterMovementPlanning || supportsWaterMovementDetail;
  const plannerOpen =
    supportsWaterMovementPlanning && searchParams.get("planner") === "open";
  const detailOpen =
    supportsWaterMovementDetail && searchParams.get("detail") === "open";
  const plannerAt = supportsWaterMovementPlanning
    ? searchParams.get("at")
    : null;
  const conditions = useLocationConditions(locationCode, plannerAt);
  const firstConditionsSettled = !conditions.isPending;
  const temperatureUnit = resolveTemperatureUnit({
    locationDefault: location?.metadata.default_temperature_unit,
    preference: preferredTemperatureUnit,
  });
  const setTemperatureUnit = (unit: TemperatureUnit) => {
    setPreferredTemperatureUnit(unit);
    persistTemperatureUnit(unit);
  };

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
  const setPlannerAt = (at: string | null) => {
    const next = new URLSearchParams(searchParams);
    next.set("planner", "open");
    if (at) {
      next.set("at", at);
    } else {
      next.delete("at");
    }
    setSearchParams(next, { replace: true });
  };
  const resetPlannerAt = () => {
    const next = new URLSearchParams(searchParams);
    next.delete("at");
    setSearchParams(next, { replace: true });
  };
  const openPlanner = () => {
    const next = new URLSearchParams(searchParams);
    next.set("planner", "open");
    setSearchParams(next, { replace: false });
  };
  const closePlanner = () => {
    const next = new URLSearchParams(searchParams);
    next.delete("planner");
    setSearchParams(next, { replace: true });
  };
  const openDetail = () => {
    const next = new URLSearchParams(searchParams);
    next.set("detail", "open");
    setSearchParams(next, { replace: false });
  };
  const closeDetail = () => {
    const next = new URLSearchParams(searchParams);
    next.delete("detail");
    setSearchParams(next, { replace: true });
  };
  const detailPlotUrl =
    detailOpen && supportsWaterMovementDetail
      ? `/api/${locationCode}/plots/current_tide${
          plannerAt ? `?at=${encodeURIComponent(plannerAt)}` : ""
        }`
      : null;

  return (
    <div className="space-y-5 sm:space-y-8">
      <header>
        <p className="font-medium text-swim-current text-xs uppercase sm:text-sm">
          {location.metadata.name}
        </p>
        <h1 className="mt-0.5 font-semibold text-2xl text-swim-blue sm:mt-1 sm:text-3xl">
          shall we swim today?
        </h1>
        <p className="mt-1 text-sm text-slate-700 sm:mt-2 sm:text-base">
          ...at{" "}
          <a
            className="text-swim-blue underline"
            href={location.metadata.swim_location_link}
            {...externalLinkProps(location.metadata.swim_location_link)}
          >
            {location.metadata.swim_location}
          </a>
        </p>
      </header>

      <ConditionsSummary
        conditions={conditions.data}
        hasError={conditions.isError && !conditions.data}
        isLoading={conditions.isPending}
        onSetTemperatureUnit={setTemperatureUnit}
        showObservedFlow={showsObservedFlow}
        showWaterMovement={showsWaterMovement}
        temperatureUnit={temperatureUnit}
        waterMovementControls={
          hasWaterMovementControls
            ? {
                at: plannerAt,
                detailOpen,
                label: plannerAt
                  ? formatPlannerTimeLabel(plannerAt)
                  : "Current conditions",
                location,
                onCloseDetail: closeDetail,
                onClosePlanner: closePlanner,
                onOpenDetail: openDetail,
                onOpenPlanner: openPlanner,
                onResetAt: resetPlannerAt,
                onSetAt: setPlannerAt,
                plannerOpen,
                plotUrl: detailPlotUrl,
                supportsDetail: supportsWaterMovementDetail,
                supportsPlanning: supportsWaterMovementPlanning,
              }
            : undefined
        }
      />
      {staleMessage || unavailableMessage ? (
        <p className="border-swim-alert border-l-4 bg-white px-4 py-3 text-sm text-swim-alert">
          {staleMessage ?? unavailableMessage}
        </p>
      ) : null}

      {location.metadata.features.windy ? (
        <Section title="Forecast">
          <WindyEmbed
            config={location.integrations.windy}
            metadata={location.metadata}
            temperatureUnit={temperatureUnit}
          />
        </Section>
      ) : null}

      {location.metadata.features.webcam && location.integrations.webcam ? (
        <Section title="Live Webcam">
          <WebcamEmbed config={location.integrations.webcam} />
        </Section>
      ) : null}

      {location.metadata.features.temperature ? (
        <Section title="Temperature Trends">
          <TemperaturePlots
            enabled={firstConditionsSettled}
            locationCode={locationCode}
            plotConfig={location.metadata.temperature_plots}
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
  onSetTemperatureUnit,
  showObservedFlow = false,
  showWaterMovement = true,
  temperatureUnit,
  waterMovementControls,
}: {
  conditions?: LocationConditions;
  hasError: boolean;
  isLoading: boolean;
  onSetTemperatureUnit: (unit: TemperatureUnit) => void;
  showObservedFlow?: boolean;
  showWaterMovement?: boolean;
  temperatureUnit: TemperatureUnit;
  waterMovementControls?: WaterMovementControls;
}) {
  const detailMode = Boolean(waterMovementControls?.detailOpen);
  const showMovementPanel = showWaterMovement || showObservedFlow;
  const summaryClassName = detailMode
    ? "grid gap-0 rounded border border-swim-line bg-white md:gap-4 md:border-0 md:bg-transparent"
    : showMovementPanel
      ? "grid gap-0 rounded border border-swim-line bg-white md:grid-cols-[1fr_2fr] md:items-start md:gap-4 md:border-0 md:bg-transparent"
      : "grid gap-0 rounded border border-swim-line bg-white md:border-0 md:bg-transparent";

  if (isLoading) {
    return (
      <section aria-busy="true" className={summaryClassName}>
        <LoadingTemperatureSummary compact={detailMode} />
        {showMovementPanel ? (
          <LoadingWaterMovementSummary
            observed={showObservedFlow && !showWaterMovement}
          />
        ) : null}
      </section>
    );
  }

  return (
    <section className={summaryClassName}>
      <TemperatureSummary
        compact={detailMode}
        conditions={conditions}
        hasError={hasError}
        onSetTemperatureUnit={onSetTemperatureUnit}
        temperatureUnit={temperatureUnit}
      />
      {showWaterMovement ? (
        <WaterMovementSummary
          current={hasError ? undefined : conditions?.current}
          tides={hasError ? undefined : conditions?.tides}
          waterMovementControls={waterMovementControls}
        />
      ) : showObservedFlow ? (
        <ObservedFlowSummary
          current={hasError ? undefined : conditions?.current}
          hasError={hasError}
        />
      ) : null}
    </section>
  );
}

function LoadingTemperatureSummary({ compact = false }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="border-swim-line border-b p-3 md:flex md:items-center md:justify-between md:gap-4 md:rounded md:border md:bg-white">
        <div className="min-w-0 md:flex md:flex-wrap md:items-baseline md:gap-x-3 md:gap-y-1">
          <h2 className="font-semibold text-base md:text-lg">
            Water Temperature
          </h2>
          <div className="mt-1 flex items-baseline gap-2 md:mt-0">
            <p className="text-sm text-slate-700">The water is currently</p>
            <p className="font-mono font-semibold text-2xl text-swim-blue">
              Loading
            </p>
          </div>
        </div>
        <p className="mt-1 min-w-0 text-xs text-slate-600 md:mt-0 md:text-right md:text-sm">
          Loading the latest station reading.
        </p>
      </div>
    );
  }

  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <h2 className="font-semibold text-base md:text-lg">Water Temperature</h2>
      <div className="mt-1 flex items-baseline gap-2 md:block">
        <p className="text-sm text-slate-700 md:mt-2 md:text-base">
          The water is currently
        </p>
        <p className="font-mono font-semibold text-2xl text-swim-blue md:mt-1 md:text-3xl">
          Loading
        </p>
      </div>
      <p className="mt-1 text-xs text-slate-600 md:mt-2 md:text-sm">
        Loading the latest station reading.
      </p>
    </div>
  );
}

function LoadingWaterMovementSummary({ observed }: { observed: boolean }) {
  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <WaterMovementHeader badge={observed ? "Observed" : "Predicted"} />
      <p className="mt-2 font-semibold text-lg text-swim-current leading-snug md:text-2xl">
        Loading water movement
      </p>
      <LoadingDriftBar label={observed ? "FLOW" : "TIDE"} />
      {observed ? null : <LoadingDriftBar label="CURRENT" />}
    </div>
  );
}

function LoadingDriftBar({ label }: { label: string }) {
  return (
    <div className="mt-2 rounded border border-swim-line bg-[#f8fbfc] px-3 py-2 md:mt-3 md:px-4 md:py-2.5">
      <div className="flex items-baseline justify-between gap-2 font-mono text-[11px] font-bold uppercase tracking-[0.1em] md:text-xs">
        <span>
          <span className="text-slate-500">{label}</span>{" "}
          <span className="font-normal normal-case tracking-normal tabular-nums text-swim-ink">
            Loading
          </span>
        </span>
        <span className="text-swim-ink">--%</span>
      </div>
      <div
        aria-hidden="true"
        className="my-2 grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-1"
      >
        <span className="font-mono font-semibold text-xl text-slate-500 leading-none">
          [
        </span>
        <span className="relative h-7 min-w-0 overflow-hidden">
          <span className="absolute inset-x-0 top-1/2 h-4 -translate-y-1/2 bg-[#cdd6db]" />
        </span>
        <span className="font-mono font-semibold text-xl text-slate-500 leading-none">
          ]
        </span>
      </div>
      <div className="flex items-baseline justify-between gap-2 text-[11px] tabular-nums text-slate-500 md:text-xs">
        <span className="min-w-0">loading</span>
        <span className="min-w-0 text-right">loading</span>
      </div>
    </div>
  );
}

function ObservedFlowSummary({
  current,
  hasError,
}: {
  current?: CurrentInfo | null;
  hasError: boolean;
}) {
  const hasCurrent = Boolean(current && !hasError);
  const currentValue =
    hasCurrent && current
      ? `${formatMagnitude(Math.abs(current.magnitude))} kt`
      : "Unavailable";
  const timestamp =
    hasCurrent && current ? formatStationTimestamp(current.timestamp) : null;

  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <WaterMovementHeader badge="Observed" />
      <p className="mt-1 text-sm text-slate-700 md:mt-2 md:text-base">
        The river current is currently
      </p>
      <p className="font-mono font-semibold text-2xl text-swim-current md:mt-1 md:text-3xl">
        {currentValue}
      </p>
      <p className="mt-2 text-sm text-slate-600">
        {hasCurrent ? (
          <>
            Latest observed flow
            {timestamp ? (
              <>
                {" as of "}
                <span className="font-mono">{timestamp}</span>
              </>
            ) : null}
          </>
        ) : (
          "Recent observed flow is unavailable right now."
        )}
      </p>
    </div>
  );
}

function SourceBadge({ label }: { label: "Observed" | "Predicted" }) {
  return (
    <span className="rounded bg-slate-100 px-2 py-1 font-mono text-[11px] text-slate-700 uppercase">
      {label}
    </span>
  );
}

function WaterMovementHeader({
  actions,
  badge,
}: {
  actions?: ReactNode;
  badge: "Observed" | "Predicted";
}) {
  return (
    <div className="flex items-start justify-between gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-semibold text-base md:text-lg">Water Movement</h2>
        <SourceBadge label={badge} />
      </div>
      {actions}
    </div>
  );
}

type WaterMovementControls = {
  at: string | null;
  detailOpen: boolean;
  label: string;
  location: AppBootstrapLocation;
  onCloseDetail: () => void;
  onClosePlanner: () => void;
  onOpenDetail: () => void;
  onOpenPlanner: () => void;
  onResetAt: () => void;
  onSetAt: (at: string | null) => void;
  plannerOpen: boolean;
  plotUrl: string | null;
  supportsDetail: boolean;
  supportsPlanning: boolean;
};

function TemperatureSummary({
  compact = false,
  conditions,
  hasError,
  onSetTemperatureUnit,
  temperatureUnit,
}: {
  compact?: boolean;
  conditions?: LocationConditions;
  hasError: boolean;
  onSetTemperatureUnit: (unit: TemperatureUnit) => void;
  temperatureUnit: TemperatureUnit;
}) {
  const temperatureValue =
    conditions?.temperature && !hasError
      ? formatWaterTemperature(conditions.temperature, temperatureUnit)
      : "Unavailable";
  const stationName =
    conditions?.temperature && !hasError
      ? conditions.temperature.station_name ||
        conditions.location.name ||
        "station"
      : null;
  const stationTimestamp =
    conditions?.temperature && !hasError
      ? formatStationTimestamp(conditions.temperature.timestamp)
      : null;

  if (compact) {
    return (
      <div className="border-swim-line border-b p-3 md:flex md:items-center md:justify-between md:gap-4 md:rounded md:border md:bg-white">
        <div className="min-w-0 md:flex md:flex-wrap md:items-center md:gap-x-3 md:gap-y-1">
          <div className="flex items-center gap-2">
            <h2 className="font-semibold text-base md:text-lg">
              Water Temperature
            </h2>
            <TemperatureUnitToggle
              onChange={onSetTemperatureUnit}
              unit={temperatureUnit}
            />
          </div>
          <div className="mt-1 flex items-baseline gap-2 md:mt-0">
            <p className="text-sm text-slate-700">The water is currently</p>
            <p className="font-mono font-semibold text-2xl text-swim-blue">
              {temperatureValue}
            </p>
          </div>
        </div>
        <p className="mt-1 min-w-0 text-xs text-slate-600 md:mt-0 md:text-right md:text-sm">
          {stationName ? (
            <>
              at <span>{stationName}</span>
              {stationTimestamp ? (
                <>
                  {" as of "}
                  <span className="font-mono">{stationTimestamp}</span>.
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

  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-semibold text-base md:text-lg">
          Water Temperature
        </h2>
        <TemperatureUnitToggle
          onChange={onSetTemperatureUnit}
          unit={temperatureUnit}
        />
      </div>
      <div className="mt-1 flex items-baseline gap-2 md:block">
        <p className="text-sm text-slate-700 md:mt-2 md:text-base">
          The water is currently
        </p>
        <p className="font-mono font-semibold text-2xl text-swim-blue md:mt-1 md:text-3xl">
          {temperatureValue}
        </p>
      </div>
      <p className="mt-1 text-xs text-slate-600 md:mt-2 md:text-sm">
        {stationName ? (
          <>
            at <span>{stationName}</span>
            {stationTimestamp ? (
              <>
                {" as of "}
                <span className="font-mono">{stationTimestamp}</span>.
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

function TemperatureUnitToggle({
  onChange,
  unit,
}: {
  onChange: (unit: TemperatureUnit) => void;
  unit: TemperatureUnit;
}) {
  return (
    <fieldset
      aria-label="Temperature unit"
      className="inline-grid grid-cols-2 overflow-hidden rounded border border-swim-line bg-white text-xs"
    >
      {(["F", "C"] as const).map((option) => (
        <button
          aria-pressed={unit === option}
          className={
            unit === option
              ? "bg-swim-blue px-2 py-1 font-semibold text-white"
              : "px-2 py-1 font-semibold text-slate-600 hover:bg-slate-50"
          }
          key={option}
          onClick={() => onChange(option)}
          type="button"
        >
          °{option}
        </button>
      ))}
    </fieldset>
  );
}

function WaterMovementSummary({
  current,
  tides,
  waterMovementControls,
}: {
  current?: CurrentInfo | null;
  tides?: LocationConditions["tides"];
  waterMovementControls?: WaterMovementControls;
}) {
  const pastTide = tides?.past?.at(-1);
  const nextTide = tides?.next?.[0];
  const description = describeWaterMovement(tides?.state, current);
  const plannedLabel = waterMovementControls?.at
    ? waterMovementControls.label
    : null;
  const detailOpen = Boolean(
    waterMovementControls?.detailOpen && waterMovementControls.plotUrl,
  );
  const isNycDetail =
    waterMovementControls?.location.metadata.code === "nyc" && detailOpen;
  const isNycWaterMovement =
    waterMovementControls?.location.metadata.code === "nyc";
  const nycSwimmerSummary =
    isNycWaterMovement && current ? nycCurrentSwimmerSummary(current) : null;

  return (
    <div className="border-swim-line border-b p-3 md:rounded md:border md:bg-white md:p-4">
      <section
        aria-label="Water movement controls"
        className="sticky top-0 z-20 bg-white/95 pb-2 backdrop-blur"
      >
        <WaterMovementHeader
          actions={
            waterMovementControls ? (
              <WaterMovementActions controls={waterMovementControls} />
            ) : null
          }
          badge="Predicted"
        />
        {plannedLabel ? (
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-600 md:text-sm">
            <button
              className="inline-flex h-6 items-center rounded border border-swim-line bg-white px-2 font-mono text-[11px] text-swim-blue"
              onClick={waterMovementControls?.onResetAt}
              type="button"
            >
              Now
            </button>
            <p>
              Planned for{" "}
              <span className="font-mono text-swim-blue">{plannedLabel}</span>
            </p>
          </div>
        ) : null}
        {waterMovementControls?.plannerOpen ? (
          <PlannerControls
            at={waterMovementControls.at}
            label={waterMovementControls.label}
            location={waterMovementControls.location}
            onSetAt={waterMovementControls.onSetAt}
          />
        ) : null}
      </section>
      {nycSwimmerSummary ? (
        <p className="mt-2 font-semibold text-lg text-swim-current leading-snug md:text-xl">
          {nycSwimmerSummary}
        </p>
      ) : (
        <p className="mt-2 font-semibold text-lg text-swim-current leading-snug md:text-2xl">
          {description}
        </p>
      )}

      {detailOpen && waterMovementControls?.plotUrl ? (
        <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(18rem,0.85fr)_minmax(0,1.35fr)] lg:items-start">
          <div className="order-2 space-y-3 lg:order-1">
            <TideInstrument
              nextTide={nextTide}
              previousTide={pastTide}
              state={tides?.state}
            />
            <CurrentInstrument current={current} />
            {isNycDetail ? (
              <NycWaterMovementGuidance
                current={current}
                essentialsUrl={
                  waterMovementControls.location.metadata.swim_location_link
                }
              />
            ) : null}
          </div>

          <section
            aria-label="Current and tide detail chart"
            className="order-1 rounded border border-swim-line bg-[#f8fbfc] p-3 lg:order-2"
          >
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-medium text-sm text-swim-blue">
                Current and tide detail chart
              </h3>
              <button
                aria-label="Close current and tide detail chart"
                className="rounded border border-swim-line bg-white px-2 py-1 text-xs text-swim-ink"
                onClick={waterMovementControls.onCloseDetail}
                type="button"
              >
                Close
              </button>
            </div>
            <PlannerPlotImage
              alt={`Tide and current plot for ${waterMovementControls.label}`}
              src={waterMovementControls.plotUrl}
            />
            {isNycDetail ? (
              <NycWaterMovementVisuals current={current} tides={tides} />
            ) : null}
          </section>
        </div>
      ) : (
        <>
          <TideInstrument
            nextTide={nextTide}
            previousTide={pastTide}
            state={tides?.state}
          />
          <CurrentInstrument current={current} />
        </>
      )}
    </div>
  );
}

function NycWaterMovementGuidance({
  current,
  essentialsUrl,
}: {
  current?: CurrentInfo | null;
  essentialsUrl: string;
}) {
  const swimAdvice = nycSwimDirectionAdvice(current);

  return (
    <div className="space-y-3">
      <section aria-label="Grimaldo's Chair current guidance">
        <div className="rounded border border-swim-line bg-[#f8fbfc] p-3">
          <h4 className="font-semibold text-sm text-swim-blue">
            Grimaldo&apos;s Chair current guidance
          </h4>
          <p className="mt-1 text-sm text-swim-ink leading-relaxed">
            {swimAdvice}
          </p>
          <dl className="mt-2 grid gap-1 text-xs text-slate-700 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            <div>
              <dt className="font-semibold text-swim-current">Flood current</dt>
              <dd>Water usually pushes east toward Manhattan Beach.</dd>
            </div>
            <div>
              <dt className="font-semibold text-swim-current">Ebb current</dt>
              <dd>Water usually pushes west toward Coney Island Pier.</dd>
            </div>
          </dl>
          <p className="mt-2 text-xs text-slate-600">
            Direction references are relative to Grimaldo&apos;s Chair and
            describe current movement, not a required swim route.
            <a
              className="ml-1 text-swim-blue underline"
              href={essentialsUrl}
              {...externalLinkProps(essentialsUrl)}
            >
              CIBBOWS Essentials
            </a>
          </p>
          <p className="mt-2 text-xs text-slate-600">
            The current summary is centered on Grimaldo&apos;s Chair. Farther
            west toward the Aquarium and Coney Island Pier, the current can
            behave differently and may switch direction; check the map and what
            you see in the water before using this for a longer swim.
          </p>
        </div>
      </section>

      <section aria-label="NYC tide current notes">
        <div className="rounded border border-swim-line bg-[#f8fbfc] p-3">
          <h4 className="font-semibold text-sm text-swim-blue">
            Tide and current timing
          </h4>
          <p className="mt-1 text-sm text-slate-700 leading-relaxed">
            In this swim area, currents lead the tide by about two hours: max
            flood is roughly two hours before high tide, and max ebb is roughly
            two hours before low tide. Positive values in the projection
            indicate flood; negative values indicate ebb.
          </p>
        </div>
      </section>

      <section aria-label="NYC current estimate methodology">
        <div className="rounded border border-swim-line bg-[#f8fbfc] p-3">
          <h4 className="font-semibold text-sm text-swim-blue">
            Estimate source
          </h4>
          <p className="mt-1 text-sm text-slate-700 leading-relaxed">
            The NYC estimate averages nearby NOAA current prediction stations at
            opposite ends of the Coney/Brighton water. That gives a useful
            flood/ebb curve for planning, but it cannot account for wind,
            jetties, or every local eddy.
          </p>
        </div>
      </section>
    </div>
  );
}

function NycWaterMovementVisuals({
  current,
  tides,
}: {
  current?: CurrentInfo | null;
  tides?: LocationConditions["tides"];
}) {
  const currentMap = nycCurrentMap(current);
  const legacyChart = nycLegacyChart(tides, current?.timestamp);

  return (
    <div className="mt-4 space-y-4 border-swim-line border-t pt-4">
      {currentMap ? (
        <section aria-label="Coney Island current map">
          <h4 className="font-semibold text-sm text-swim-blue">
            Local current map
          </h4>
          <div className="overflow-hidden rounded border border-swim-line bg-white">
            <PlannerPlotImage
              alt={currentMap.alt}
              containerClassName="relative"
              src={currentMap.src}
            />
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Arrow size approximates current strength. Direction is interpreted
            for the Grimaldo&apos;s Chair swim area from local knowledge.
          </p>
        </section>
      ) : null}
      {legacyChart ? (
        <section aria-label="Historic New York Harbor current chart">
          <h4 className="font-semibold text-sm text-swim-blue">
            Historic harbor chart
          </h4>
          <p className="mt-1 text-xs text-slate-600">{legacyChart.title}</p>
          <div className="mt-2 overflow-hidden rounded border border-swim-line bg-white">
            <PlannerPlotImage
              alt={`Historic New York Harbor chart: ${legacyChart.title}`}
              containerClassName="relative"
              src={legacyChart.src}
            />
          </div>
        </section>
      ) : null}
    </div>
  );
}

function nycSwimDirectionAdvice(current?: CurrentInfo | null) {
  if (current?.phase === "flood" || current?.direction === "flooding") {
    return "At Grimaldo's, flood usually carries you east toward Manhattan Beach. Start west toward Coney Island Pier if you want the current against you first, or ride it east if you are planning around a tide flip.";
  }
  if (current?.phase === "ebb" || current?.direction === "ebbing") {
    return "At Grimaldo's, ebb usually carries you west toward Coney Island Pier. Start east toward Manhattan Beach if you want the current against you first, then use the ebb on the way back.";
  }
  if (
    current?.phase === "slack" ||
    current?.phase === "slack_before_flood" ||
    current?.phase === "slack_before_ebb"
  ) {
    return "The current is near slack, so direction matters less right now. Watch the next build: flood favors eastward movement, while ebb favors westward movement.";
  }
  return "Use the current direction as a planning input, then confirm with what you see in the water before committing to a longer swim.";
}

function nycCurrentSwimmerSummary(current: CurrentInfo) {
  const speed = nycCurrentSpeedAdjective(current);
  const trend = nycCurrentTrendSentence(current);
  const direction = nycCurrentLandmarkDirection(current);

  if (!direction) {
    if (isSlackCurrent(current)) {
      return "At Grimaldo's, the current is near slack.";
    }
    return "At Grimaldo's, check the current before choosing a direction.";
  }

  return `At Grimaldo's, expect ${speed ? `a ${speed} ` : "a "}push ${direction}.${trend ? ` ${trend}` : ""}`;
}

function nycCurrentLandmarkDirection(current: CurrentInfo) {
  if (current.phase === "flood" || current.direction === "flooding") {
    return "east toward Manhattan Beach";
  }
  if (current.phase === "ebb" || current.direction === "ebbing") {
    return "west toward Coney Island Pier";
  }
  return null;
}

function nycCurrentSpeedAdjective(current: CurrentInfo) {
  switch (currentSpeedPhrase(current)) {
    case "gently":
      return "gentle";
    case "steadily":
      return "steady";
    case "fast":
      return "fast";
    default:
      return null;
  }
}

function nycCurrentTrendSentence(current: CurrentInfo) {
  switch (current.trend) {
    case "building":
      return "The current is getting stronger.";
    case "easing":
      return "The current is starting to ease.";
    case "steady":
      return "The current is holding steady.";
    default:
      return null;
  }
}

function nycCurrentMap(current?: CurrentInfo | null) {
  if (!current?.direction || typeof current.magnitude_pct !== "number") {
    return null;
  }

  const direction =
    current.phase === "flood" || current.direction === "flooding"
      ? "flooding"
      : current.phase === "ebb" || current.direction === "ebbing"
        ? "ebbing"
        : null;
  if (!direction) {
    return null;
  }

  const magnitudeBin = binCurrentMagnitude(current.magnitude_pct);
  return {
    src: `/static/plots/nyc/current_chart_${direction}_${magnitudeBin}.png`,
    alt: `Coney Island ${direction} current map at ${magnitudeBin}% strength`,
  };
}

function binCurrentMagnitude(magnitudePct: number) {
  const bins = [0, 10, 30, 45, 55, 70, 90, 100];
  const pct = clamp(magnitudePct * 100, 0, 100);
  return bins.find((bin) => pct <= bin) ?? 100;
}

function nycLegacyChart(
  tides?: LocationConditions["tides"],
  timestamp?: string,
) {
  const previousTide = tides?.past?.at(-1);
  if (!previousTide || !timestamp) {
    return null;
  }

  const targetDate = parseApiLocalWallDate(timestamp);
  const tideDate = parseApiLocalWallDate(previousTide.time);
  const offsetHours = (targetDate.getTime() - tideDate.getTime()) / 3_600_000;
  if (!Number.isFinite(offsetHours) || offsetHours < 0) {
    return null;
  }

  const tideType =
    previousTide.type === "high" || previousTide.type === "low"
      ? previousTide.type
      : null;
  if (!tideType) {
    return null;
  }

  const chart =
    offsetHours > 5.5
      ? { hours: 0, type: invertTideType(tideType) }
      : { hours: Math.round(offsetHours), type: tideType };
  const tideLabel = `${capitalize(chart.type)} Water at New York`;
  const title =
    chart.hours > 0
      ? `${chart.hours} Hour${chart.hours > 1 ? "s" : ""} after ${tideLabel}`
      : tideLabel;

  return {
    src: `/static/tidecharts/${chart.type}+${chart.hours}.png`,
    title,
  };
}

function invertTideType(type: "high" | "low") {
  return type === "high" ? "low" : "high";
}

function WaterMovementActions({
  controls,
}: {
  controls: WaterMovementControls;
}) {
  if (!controls.supportsDetail && !controls.supportsPlanning) {
    return null;
  }

  return (
    <div className="flex shrink-0 items-center gap-1">
      {controls.supportsDetail ? (
        <button
          aria-pressed={controls.detailOpen}
          className={[
            "rounded border px-2 py-1 text-xs",
            controls.detailOpen
              ? "border-swim-blue bg-swim-blue text-white"
              : "border-swim-line bg-white text-swim-blue",
          ].join(" ")}
          onClick={
            controls.detailOpen ? controls.onCloseDetail : controls.onOpenDetail
          }
          type="button"
        >
          Details
        </button>
      ) : null}
      {controls.supportsPlanning ? (
        <button
          aria-pressed={controls.plannerOpen}
          className={[
            "rounded border px-2 py-1 text-xs",
            controls.plannerOpen
              ? "border-swim-blue bg-swim-blue text-white"
              : "border-swim-line bg-white text-swim-blue",
          ].join(" ")}
          onClick={
            controls.plannerOpen
              ? controls.onClosePlanner
              : controls.onOpenPlanner
          }
          type="button"
        >
          Plan
        </button>
      ) : null}
    </div>
  );
}

function PlannerPlotImage({
  alt,
  containerClassName = "relative mt-3",
  src,
}: {
  alt: string;
  containerClassName?: string;
  src: string;
}) {
  const [displayed, setDisplayed] = useState({ alt, src });
  const [isLoadingNext, setIsLoadingNext] = useState(false);

  useEffect(() => {
    if (src === displayed.src) {
      setDisplayed({ alt, src });
      setIsLoadingNext(false);
      return;
    }

    let cancelled = false;
    const image = new Image();
    setIsLoadingNext(true);
    image.onload = () => {
      if (!cancelled) {
        setDisplayed({ alt, src });
        setIsLoadingNext(false);
      }
    };
    image.onerror = () => {
      if (!cancelled) {
        setIsLoadingNext(false);
      }
    };
    image.src = src;

    return () => {
      cancelled = true;
    };
  }, [alt, displayed.src, src]);

  return (
    <div className={containerClassName}>
      <img alt={displayed.alt} className="w-full" src={displayed.src} />
      {isLoadingNext ? (
        <div className="absolute right-2 top-2 rounded bg-white/90 px-2 py-1 font-mono text-[11px] text-slate-600 shadow-sm">
          Updating
        </div>
      ) : null}
    </div>
  );
}

function PlannerControls({
  at,
  label,
  location,
  onSetAt,
}: {
  at: string | null;
  label: string;
  location: AppBootstrapLocation;
  onSetAt: (at: string | null) => void;
}) {
  const baseAtRef = useRef(formatLocationIso(new Date(), location));
  const baseDate = parseLocationIso(baseAtRef.current);
  const sliderValue = at
    ? clamp(
        Math.round(
          (parseLocationIso(at).getTime() - baseDate.getTime()) /
            60_000 /
            PLANNER_STEP_MINUTES,
        ) * PLANNER_STEP_MINUTES,
        PLANNER_MIN_MINUTES,
        PLANNER_MAX_MINUTES,
      )
    : 0;
  const setMinuteOffset = (minutes: number) => {
    onSetAt(
      minutes === 0 ? null : formatLocalIso(addMinutes(baseDate, minutes)),
    );
  };

  return (
    <section aria-label="Planner mode" className="mt-1">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
        <label className="sr-only" htmlFor="planner-time-slider">
          Planner time
        </label>
        <input
          aria-valuetext={`${label}, ${formatPlannerOffset(sliderValue)}`}
          className="h-8 min-w-0 accent-swim-blue"
          id="planner-time-slider"
          max={PLANNER_MAX_MINUTES}
          min={PLANNER_MIN_MINUTES}
          onChange={(event) =>
            setMinuteOffset(Number(event.currentTarget.value))
          }
          step={PLANNER_STEP_MINUTES}
          type="range"
          value={sliderValue}
        />
        <p className="w-20 whitespace-nowrap text-right font-mono text-xs text-swim-ink tabular-nums">
          {formatPlannerOffset(sliderValue)}
        </p>
      </div>
    </section>
  );
}

function addMinutes(date: Date, minutes: number) {
  return new Date(date.getTime() + minutes * 60_000);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function formatPlannerOffset(minutes: number) {
  if (minutes === 0) {
    return "now";
  }

  const sign = minutes > 0 ? "+" : "-";
  const absolute = Math.abs(minutes);
  const hours = Math.floor(absolute / 60);
  const remainingMinutes = absolute % 60;
  if (remainingMinutes === 0) {
    return `${sign}${hours}h`;
  }
  if (hours === 0) {
    return `${sign}${remainingMinutes}m`;
  }
  return `${sign}${hours}h ${remainingMinutes}m`;
}

function parseLocationIso(at: string) {
  const match = at.match(
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/,
  );
  if (!match) {
    return new Date();
  }

  const [, year, month, day, hour, minute, second = "00"] = match;
  return new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  );
}

function parseApiDate(value: string) {
  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed;
  }

  return parseLocationIso(value);
}

function parseApiLocalWallDate(value: string) {
  const match = value.match(
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/,
  );
  if (!match) {
    return parseApiDate(value);
  }

  const [, year, month, day, hour, minute, second = "00"] = match;
  return new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  );
}

function formatLocalIso(date: Date) {
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function formatLocationIso(date: Date, location: AppBootstrapLocation) {
  const parts = new Intl.DateTimeFormat("en-US", {
    day: "2-digit",
    hour: "2-digit",
    hourCycle: "h23",
    hour12: false,
    minute: "2-digit",
    month: "2-digit",
    second: "2-digit",
    timeZone: location.metadata.timezone,
    year: "numeric",
  }).formatToParts(date);
  const values = Object.fromEntries(
    parts.map((part) => [part.type, part.value]),
  );

  return `${values.year}-${values.month}-${values.day}T${values.hour}:${values.minute}:${values.second}`;
}

function formatPlannerTimeLabel(at: string) {
  const match = at.match(
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::\d{2})?$/,
  );
  if (!match) {
    return at;
  }

  const [, year, month, day, hour, minute] = match;
  const date = new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
  );

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
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
    return describeTideOnlyMovement(tideState);
  }

  return "Water movement is unavailable right now.";
}

function describeTideOnlyMovement(tideState: TideState) {
  const tidePosition = describeTidePosition(tideState);
  const trendClause =
    tideState.trend === "steady" ? "holding steady" : tideState.trend;
  if (tidePosition) {
    return `It's ${tidePosition}, and the tide is ${trendClause}.`;
  }

  switch (tideState.trend) {
    case "rising":
      return "The tide is rising toward high tide.";
    case "falling":
      return "The tide is falling toward low tide.";
    case "steady":
      return "The tide is holding steady.";
    default:
      return `The tide is ${tideState.trend}.`;
  }
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
  if (tideState.height_pct <= AT_TIDE_EDGE_PCT) {
    return "at low tide";
  }
  if (tideState.height_pct <= NEAR_TIDE_EDGE_PCT) {
    return "near low tide";
  }
  if (tideState.height_pct >= 1 - AT_TIDE_EDGE_PCT) {
    return "at high tide";
  }
  if (tideState.height_pct >= 1 - NEAR_TIDE_EDGE_PCT) {
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
  if (Number.isFinite(current.magnitude)) {
    if (current.magnitude < GENTLE_CURRENT_MAX_KT) {
      return "gently";
    }
    if (current.magnitude >= FAST_CURRENT_MIN_KT) {
      return "fast";
    }
    return "steadily";
  }

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
  const accentTokens = DRIFT_ACCENT[accent];
  const fillPercent = Math.max(0, Math.min(100, percent));

  const fillGradient =
    direction === "up"
      ? `linear-gradient(90deg, ${accentTokens.soft} 0%, ${accentTokens.color} 65%, ${accentTokens.color} 100%)`
      : direction === "down"
        ? `linear-gradient(90deg, ${accentTokens.color} 0%, ${accentTokens.color} 55%, ${accentTokens.soft} 100%)`
        : accentTokens.color;

  const markerPulseClass =
    direction === "down"
      ? "drift-marker-pulse-left"
      : direction === "up"
        ? "drift-marker-pulse-right"
        : "";
  const trendIndicator = trendIndicatorText(accent, trend, direction);

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
            <span>
              <span className={accentTokens.arrowText}>{trendIndicator}</span>
              {" · "}
            </span>
          ) : null}
          {percent}%
        </span>
      </div>

      <div
        aria-hidden="true"
        className="my-2 grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-x-1"
      >
        <span className="font-mono font-semibold text-xl text-slate-500 leading-none">
          [
        </span>
        <span className="relative h-7 min-w-0 overflow-hidden">
          <span className="absolute inset-x-0 top-1/2 h-4 -translate-y-1/2 bg-[#cdd6db]" />
          <span
            className="absolute left-0 top-1/2 h-4 -translate-y-1/2"
            style={{
              backgroundImage: fillGradient,
              width: `${fillPercent}%`,
            }}
          />
          <span
            className={`absolute top-1/2 h-7 w-1 -translate-x-1/2 -translate-y-1/2 bg-swim-ink ${markerPulseClass}`}
            style={{ left: `${fillPercent}%` }}
          />
        </span>
        <span className="font-mono font-semibold text-xl text-slate-500 leading-none">
          ]
        </span>
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

function trendIndicatorText(
  accent: keyof typeof DRIFT_ACCENT,
  trend: string,
  direction: "up" | "down" | "steady",
) {
  if (direction === "steady") {
    return null;
  }
  if (accent === "tide") {
    return direction === "up" ? `↑ ${trend}` : `↓ ${trend}`;
  }
  return trend;
}

function WindyEmbed({
  config,
  metadata,
  temperatureUnit,
}: {
  config: AppWindyConfig | null | undefined;
  metadata: AppLocationMetadata;
  temperatureUnit: TemperatureUnit;
}) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [embedSize, setEmbedSize] = useState({ width: 950, height: 350 });
  const windyConfig =
    config ??
    ({
      overlay: "waves",
      product: "ecmwfWaves",
      level: "surface",
      zoom: 11,
      metric_wind: "default",
      metric_temp: "°F",
    } satisfies AppWindyConfig);

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

  // The public iframe URL comes from Windy's embed generator. Keep these
  // configurable parameter names aligned with Windy's Map Forecast docs:
  // https://api4.windy.com/map-forecast/tutorials/parameters
  const params = new URLSearchParams({
    lat: String(metadata.latitude),
    lon: String(metadata.longitude),
    detailLat: String(metadata.latitude),
    detailLon: String(metadata.longitude),
    width: String(embedSize.width),
    height: String(embedSize.height),
    zoom: String(windyConfig.zoom),
    level: windyConfig.level,
    overlay: windyConfig.overlay,
    product: windyConfig.product,
    menu: "",
    message: "true",
    marker: "true",
    calendar: "now",
    pressure: "",
    type: "map",
    location: "coordinates",
    detail: "true",
    metricWind: windyConfig.metric_wind,
    metricTemp: `°${temperatureUnit}`,
    radarRange: "-1",
  });

  return (
    <div
      className="h-[350px] min-w-0 max-w-full overflow-hidden rounded border border-swim-line bg-white"
      ref={wrapperRef}
    >
      <iframe
        className="block h-full w-full max-w-full border-0"
        loading="lazy"
        src={`https://embed.windy.com/embed2.html?${params.toString()}`}
        title="Windy forecast"
      />
    </div>
  );
}

function WebcamEmbed({ config }: { config: AppWebcamConfig }) {
  let embed: ReactNode;
  const lazyLoad = config.provider !== "external_link";

  switch (config.provider) {
    case "youtube_live":
      embed = (
        <DeferredWebcamFrame label={config.label}>
          {(onLoad) => <YouTubeLiveEmbed config={config} onLoad={onLoad} />}
        </DeferredWebcamFrame>
      );
      break;
    case "iframe":
      embed = (
        <DeferredWebcamFrame label={config.label}>
          {(onLoad) => <IframeWebcamEmbed config={config} onLoad={onLoad} />}
        </DeferredWebcamFrame>
      );
      break;
    case "earthcam_embed":
      embed = (
        <DeferredWebcamFrame label={config.label}>
          {(onLoad) => <EarthCamEmbed config={config} onLoad={onLoad} />}
        </DeferredWebcamFrame>
      );
      break;
    case "external_link":
      embed = <ExternalWebcamLink config={config} />;
      break;
    default:
      throw new Error(`Unsupported webcam provider: ${config.provider}`);
  }

  return (
    <div>
      {lazyLoad ? (
        <LazyWebcamMount label={config.label}>{embed}</LazyWebcamMount>
      ) : (
        embed
      )}
      {config.note ? (
        <p className="mt-2 text-sm text-slate-600">{config.note}</p>
      ) : null}
    </div>
  );
}

type WebcamFrameProps = {
  config: AppWebcamConfig;
  onLoad?: () => void;
};

function WebcamPlaceholder({ label }: { label: string }) {
  return (
    <div className="flex aspect-video items-center justify-center overflow-hidden rounded border border-swim-line bg-white text-sm text-slate-600">
      {label} loading
    </div>
  );
}

function LazyWebcamMount({
  children,
  label,
}: {
  children: ReactNode;
  label: string;
}) {
  const [containerRef, shouldLoad] = useNearViewport<HTMLDivElement>(
    WEBCAM_PRELOAD_MARGIN,
  );

  return (
    <div ref={containerRef}>
      {shouldLoad ? children : <WebcamPlaceholder label={label} />}
    </div>
  );
}

function DeferredWebcamFrame({
  children,
  label,
}: {
  children: (onLoad: () => void) => ReactNode;
  label: string;
}) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="relative">
      {children(() => setLoaded(true))}
      {loaded ? null : (
        <div className="absolute inset-0">
          <WebcamPlaceholder label={label} />
        </div>
      )}
    </div>
  );
}

function useNearViewport<T extends Element>(rootMargin: string) {
  const ref = useRef<T | null>(null);
  const [nearViewport, setNearViewport] = useState(false);

  useEffect(() => {
    if (nearViewport) {
      return;
    }

    if (typeof IntersectionObserver === "undefined") {
      setNearViewport(true);
      return;
    }

    const element = ref.current;
    if (!element) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNearViewport(true);
          observer.disconnect();
        }
      },
      { rootMargin },
    );
    observer.observe(element);

    return () => observer.disconnect();
  }, [nearViewport, rootMargin]);

  return [ref, nearViewport] as const;
}

function requireWebcamField(
  value: string | null | undefined,
  provider: AppWebcamConfig["provider"],
  field: keyof AppWebcamConfig,
) {
  if (!value) {
    throw new Error(`${provider} webcam config is missing ${field}`);
  }
  return value;
}

function YouTubeLiveEmbed({ config, onLoad }: WebcamFrameProps) {
  const embedUrl = requireWebcamField(
    config.embed_url,
    config.provider,
    "embed_url",
  );

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
        onLoad={onLoad}
        scrolling="no"
        src={embedUrl}
        title="Live webcam"
      />
    </div>
  );
}

function IframeWebcamEmbed({ config, onLoad }: WebcamFrameProps) {
  const embedUrl = requireWebcamField(
    config.embed_url,
    config.provider,
    "embed_url",
  );

  return (
    <div className="aspect-video overflow-hidden rounded">
      <iframe
        allowFullScreen
        className="block h-full w-full border-0"
        onLoad={onLoad}
        scrolling="no"
        src={embedUrl}
        title={config.label}
      />
    </div>
  );
}

function EarthCamEmbed({ config, onLoad }: WebcamFrameProps) {
  const embedUrl = requireWebcamField(
    config.embed_url,
    config.provider,
    "embed_url",
  );

  return (
    <div
      className="aspect-video overflow-hidden rounded bg-[#ddd]"
      data-earthcam-embed-root=""
    >
      <iframe
        allowFullScreen
        className="block h-full w-full border-0"
        onLoad={onLoad}
        scrolling="no"
        src={embedUrl}
        title={config.label}
      />
    </div>
  );
}

function ExternalWebcamLink({ config }: { config: AppWebcamConfig }) {
  const watchUrl = requireWebcamField(
    config.watch_url,
    config.provider,
    "watch_url",
  );

  return (
    <div className="rounded border border-swim-line bg-white p-4 text-sm">
      <a
        className="text-swim-blue underline"
        href={watchUrl}
        {...externalLinkProps(watchUrl)}
      >
        {config.label}
      </a>
    </div>
  );
}

function TemperaturePlots({
  enabled,
  locationCode,
  plotConfig,
}: {
  enabled: boolean;
  locationCode: string;
  plotConfig: AppLocationMetadata["temperature_plots"];
}) {
  const [selectedPlot, setSelectedPlot] = useState("12mo");
  const plots = [
    ...(plotConfig.live
      ? [
          {
            key: "live",
            label: "Live",
            alt: "Live temperature plot",
            src: `/api/${locationCode}/plots/live_temps`,
          },
        ]
      : []),
    ...(plotConfig.historic
      ? [
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
        ]
      : []),
  ];
  const activePlot = plots.some((plot) => plot.key === selectedPlot)
    ? selectedPlot
    : selectedPlot === "all" && plots.length > 1
      ? "all"
      : (plots[0]?.key ?? null);
  const plotOptions =
    plots.length > 1 ? [...plots, { key: "all", label: "All" }] : plots;

  if (!activePlot) {
    return null;
  }

  return (
    <div className="grid gap-3">
      {plotOptions.length > 1 ? (
        <fieldset className="inline-flex w-fit overflow-hidden rounded border border-swim-line bg-white">
          <legend className="sr-only">Temperature plot range</legend>
          {plotOptions.map((plot) => (
            <button
              className={[
                "min-h-9 px-3 py-1.5 font-mono font-medium text-sm",
                plot.key === activePlot
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
      ) : null}
      <div className="grid gap-3">
        {plots.map((plot) => (
          <DeferredPlotImage
            alt={plot.alt}
            enabled={enabled}
            isActive={activePlot === "all" || plot.key === activePlot}
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
        <img
          alt={alt}
          className="w-full"
          height={TEMPERATURE_PLOT_IMAGE_SIZE.height}
          src={image.src}
          width={TEMPERATURE_PLOT_IMAGE_SIZE.width}
        />
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
        <TransitRouteCard
          key={`${route.goodservice_route_id}-${route.goodservice_direction}`}
          route={route}
        />
      ))}
    </div>
  );
}

function goodServiceDirectionPathSuffix(
  direction: TransitRouteConfig["goodservice_direction"],
): "N" | "S" {
  return direction === "north" ? "N" : "S";
}

function TransitRouteCard({ route }: { route: TransitRouteConfig }) {
  const status = useTransitRoute(route);
  const routeUrl = `https://goodservice.io/trains/${route.goodservice_route_id}/${goodServiceDirectionPathSuffix(route.goodservice_direction)}`;
  const firstLoadUnavailable = status.isError && !status.data;
  const transit = firstLoadUnavailable
    ? { status: "Unavailable", destination: "unavailable" }
    : status.data;

  return (
    <article className="border-swim-line rounded border bg-white p-4">
      <a
        className="flex items-center gap-3 text-swim-ink"
        href={routeUrl}
        {...externalLinkProps(routeUrl)}
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
            {transit?.status === "Not Scheduled" ? (
              <span>No scheduled service now</span>
            ) : (
              <>
                to{" "}
                <span className="font-mono">
                  {transit?.destination ?? "..."}
                </span>
              </>
            )}
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
          icon={<Thermometer aria-hidden="true" />}
          label="Live temperature"
          html={citations.live_temperature}
        />
        <SourceHtml
          icon={<Thermometer aria-hidden="true" />}
          label="Historical temperature"
          html={citations.historical_temperature}
        />
        <SourceHtml
          icon={<Anchor aria-hidden="true" />}
          label="Tides"
          html={citations.tides}
        />
        {location.metadata.code === "nyc" ? (
          <>
            <SourceHtml
              icon={<Shuffle aria-hidden="true" />}
              label="Currents"
              html={
                'Current predictions combine <a href="https://tidesandcurrents.noaa.gov/noaacurrents/Predictions?id=NYH1905_12">NOAA CO-OPS Station NYH1905_12</a> (Rockaway Inlet Entrance) and <a href="https://tidesandcurrents.noaa.gov/noaacurrents/Predictions?id=ACT3876">NOAA CO-OPS Station ACT3876</a> (Coney Island Channel west end).'
              }
            />
            <SourceLink
              icon={<MapIcon aria-hidden="true" />}
              label="Map credit"
              includeLabel={false}
              link={{
                label: "Liam Hartigan",
                url: "http://www.sheahartigan.com",
                description:
                  "Coney Island Brighton Beach Map, Gary Atlas 5000 Edition, by",
              }}
            />
            <SourceLink
              icon={<Anchor aria-hidden="true" />}
              label="Harbor charts"
              includeLabel={false}
              linkFirst
              link={{
                label: "Tidal current charts, New York Harbor",
                url: "https://catalog.hathitrust.org/Record/011421935",
                description:
                  "from U.S. Department of Commerce, Coast and Geodetic Survey, 1946",
              }}
            />
          </>
        ) : (
          <SourceHtml
            icon={<Shuffle aria-hidden="true" />}
            label="Currents"
            html={citations.currents}
          />
        )}
        <SourceLink
          icon={<Video aria-hidden="true" />}
          includeLabel={false}
          label="Webcam"
          link={location.integrations.webcam?.source}
          linkFirst
          secondaryLink={location.integrations.webcam?.alternative}
          secondaryPrefix="Alternate:"
        />
        <SourceLink
          icon={<Truck aria-hidden="true" />}
          includeLabel={false}
          label="Transit"
          link={location.integrations.transit_source}
        />
        <SourceLink
          icon={<GitHub aria-hidden="true" />}
          includeLabel={false}
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
  const externalizedHtml = useExternalizedTrustedHtml(html);

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
        dangerouslySetInnerHTML={{ __html: externalizedHtml }}
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
  link?: AppPresentationLink | null;
  secondaryLink?: AppPresentationLink | null;
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
              <a
                className="text-swim-blue underline"
                href={link.url}
                {...externalLinkProps(link.url)}
              >
                {link.label}
              </a>
              {link.description ? ` ${link.description}` : null}
            </>
          ) : (
            <>
              {link.description ? `${link.description} ` : null}
              <a
                className="text-swim-blue underline"
                href={link.url}
                {...externalLinkProps(link.url)}
              >
                {link.label}
              </a>
            </>
          )}
        </span>
        {secondaryLink ? (
          <span className="mt-1 block">
            {secondaryPrefix ? `${secondaryPrefix} ` : null}
            <a
              className="text-swim-blue underline"
              href={secondaryLink.url}
              {...externalLinkProps(secondaryLink.url)}
            >
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

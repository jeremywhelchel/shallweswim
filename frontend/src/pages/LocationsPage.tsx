import { Link } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";
import { useLocationConditions } from "../api/conditions";
import type { components } from "../api/generated";
import { PageMessage } from "../components/PageMessage";
import { formatStationTimestamp } from "../lib/format";

type AppBootstrapResponse = components["schemas"]["AppBootstrapResponse"];
type AppBootstrapLocation = components["schemas"]["AppBootstrapLocation"];
type AppLocationFeatures =
  components["schemas"]["AppLocationMetadata"]["features"];
type LocationConditions = components["schemas"]["LocationConditions"];

type FeatureChip = {
  label: string;
  enabled: boolean;
};

export function LocationsRoutePage() {
  const bootstrap = useAppBootstrap();

  if (bootstrap.isLoading) {
    return <PageMessage title="Loading locations" />;
  }

  if (bootstrap.isError || !bootstrap.data) {
    return (
      <PageMessage
        body="The React shell loaded, but location metadata could not be fetched."
        title="Locations unavailable"
        tone="warning"
      />
    );
  }

  return <LocationsPage bootstrap={bootstrap.data} />;
}

export function LocationsPage({
  bootstrap,
}: {
  bootstrap: AppBootstrapResponse;
}) {
  const locations = bootstrap.location_order
    .map((code) => bootstrap.locations[code])
    .filter(Boolean);

  return (
    <section className="space-y-5">
      <header>
        <p className="font-medium text-swim-current text-xs uppercase sm:text-sm">
          Locations
        </p>
        <h1 className="mt-0.5 font-semibold text-2xl text-swim-blue sm:mt-1 sm:text-3xl">
          All Swim Locations
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-700 sm:mt-2 sm:text-base">
          Current water temperatures at all swimming locations.
        </p>
      </header>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {locations.map((location) => (
          <LocationCard key={location.metadata.code} location={location} />
        ))}
      </div>
    </section>
  );
}

function LocationCard({ location }: { location: AppBootstrapLocation }) {
  const metadata = location.metadata;
  const featureChips = locationFeatureChips(metadata.features);
  const conditions = useLocationConditions(metadata.code);
  const temperature = temperatureStatus(conditions.data);
  const hasTemperatureError = conditions.isError && !conditions.data;

  return (
    <Link
      className="group flex min-h-full flex-col rounded border border-swim-line bg-white p-4 text-swim-ink transition-colors hover:border-swim-blue focus:outline-none focus:ring-2 focus:ring-swim-blue focus:ring-offset-2"
      to={`/${metadata.code}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-swim-current text-xs uppercase">
            {metadata.nav_label}
          </p>
          <h2 className="mt-1 font-semibold text-lg group-hover:text-swim-blue">
            {metadata.name}
          </h2>
        </div>
        <span className="rounded border border-swim-line px-2 py-1 font-mono text-xs text-slate-600 uppercase">
          {metadata.code}
        </span>
      </div>

      <p className="mt-2 text-sm font-medium text-slate-700">
        {metadata.swim_location}
      </p>
      <div className="mt-4">
        <p className="text-sm text-slate-700">Water temperature</p>
        <p className="mt-1 font-mono font-semibold text-3xl text-swim-blue">
          {conditions.isPending
            ? "..."
            : hasTemperatureError
              ? "Unavailable"
              : temperature.value}
        </p>
        <p className="mt-2 min-h-10 text-xs text-slate-600">
          {conditions.isPending
            ? "Loading latest reading..."
            : hasTemperatureError
              ? "Latest water temperature is unavailable."
              : temperature.detail}
        </p>
      </div>

      <div className="mt-auto flex flex-wrap gap-2 pt-4">
        {featureChips.map((chip) => (
          <span
            className={[
              "rounded px-2 py-1 font-mono text-[11px] uppercase",
              chip.enabled
                ? "bg-swim-mist text-swim-blue"
                : "bg-slate-100 text-slate-400",
            ].join(" ")}
            key={chip.label}
          >
            {chip.label}
          </span>
        ))}
      </div>
    </Link>
  );
}

function temperatureStatus(conditions?: LocationConditions) {
  const temperature = conditions?.temperature;

  if (!temperature) {
    return {
      detail: "Latest water temperature is unavailable.",
      value: "Unavailable",
    };
  }

  const station = temperature.station_name ?? conditions.location.name;
  const timestamp = formatStationTimestamp(temperature.timestamp);

  return {
    detail: `Data from ${station}${timestamp ? ` as of ${timestamp}` : ""}.`,
    value: `${temperature.water_temp}°${temperature.units || "F"}`,
  };
}

function locationFeatureChips(features: AppLocationFeatures): FeatureChip[] {
  return [
    { label: "Temp", enabled: features.temperature },
    { label: "Tides", enabled: features.tides },
    {
      label: waterMovementLabel(features),
      enabled: features.currents,
    },
    { label: "Webcam", enabled: features.webcam },
    { label: "Forecast", enabled: features.windy },
    { label: "Transit", enabled: features.transit },
  ];
}

function waterMovementLabel(features: AppLocationFeatures) {
  if (features.water_movement_detail) {
    return "Current detail";
  }
  if (features.water_movement_planning) {
    return "Planner";
  }
  if (features.currents && !features.tides) {
    return "Observed flow";
  }
  return "Currents";
}

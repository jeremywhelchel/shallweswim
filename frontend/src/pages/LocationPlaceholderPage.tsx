import { Link } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";

type LocationPlaceholderPageProps = {
  locationCode?: string;
  preserveDefaultUrl?: boolean;
};

export function LocationPlaceholderPage({
  locationCode,
  preserveDefaultUrl = false,
}: LocationPlaceholderPageProps) {
  const bootstrap = useAppBootstrap();
  const effectiveCode =
    locationCode ?? bootstrap.data?.default_location_code ?? "nyc";
  const location = bootstrap.data?.locations[effectiveCode]?.metadata;

  if (bootstrap.isLoading) {
    return <ShellMessage title="Loading app shell" />;
  }

  if (bootstrap.isError) {
    return (
      <ShellMessage
        tone="warning"
        title="App metadata is unavailable"
        body="The React shell loaded, but bootstrap metadata could not be fetched."
      />
    );
  }

  if (!location) {
    return (
      <ShellMessage
        tone="warning"
        title="Location not found"
        body={`No app metadata exists for ${effectiveCode}.`}
      />
    );
  }

  return (
    <section className="space-y-5">
      <div>
        <p className="font-medium text-swim-current text-sm uppercase">
          {preserveDefaultUrl ? "Default location" : "Location"}
        </p>
        <h1 className="mt-1 font-semibold text-3xl text-swim-ink">
          {location.name}
        </h1>
        <p className="mt-2 max-w-2xl text-base text-slate-700">
          {location.swim_location}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <StatusTile
          label="Temperature"
          enabled={location.features.temperature}
        />
        <StatusTile label="Tides" enabled={location.features.tides} />
        <StatusTile label="Currents" enabled={location.features.currents} />
      </div>

      <div className="border-swim-line border-l-4 bg-white px-4 py-3">
        <p className="text-sm text-slate-700">
          React app shell route loaded for <strong>{effectiveCode}</strong>.
          Feature parity starts in the next milestone.
        </p>
      </div>

      {location.features.currents ? (
        <Link
          className="inline-flex min-h-11 items-center rounded bg-swim-blue px-4 py-2 font-medium text-white"
          to={`/${effectiveCode}/currents`}
        >
          Currents
        </Link>
      ) : null}
    </section>
  );
}

function StatusTile({ label, enabled }: { label: string; enabled: boolean }) {
  return (
    <div className="border-swim-line rounded border bg-white p-4">
      <p className="font-medium text-sm">{label}</p>
      <p className={enabled ? "text-swim-tide" : "text-slate-500"}>
        {enabled ? "Configured" : "Not configured"}
      </p>
    </div>
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

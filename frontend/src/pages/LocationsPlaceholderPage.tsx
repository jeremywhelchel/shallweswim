import { Link } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";

export function LocationsPlaceholderPage() {
  const bootstrap = useAppBootstrap();

  return (
    <section className="space-y-5">
      <div>
        <p className="font-medium text-swim-current text-sm uppercase">
          Locations
        </p>
        <h1 className="mt-1 font-semibold text-3xl">All Swim Locations</h1>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {(bootstrap.data?.location_order ?? []).map((code) => {
          const location = bootstrap.data?.locations[code]?.metadata;
          return (
            <Link
              className="border-swim-line rounded border bg-white p-4 text-swim-ink"
              key={code}
              to={`/${code}`}
            >
              <p className="font-semibold">{location?.name ?? code}</p>
              <p className="mt-1 text-sm text-slate-700">
                {location?.swim_location ?? "Location metadata unavailable"}
              </p>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

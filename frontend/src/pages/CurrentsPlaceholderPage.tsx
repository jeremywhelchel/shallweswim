import { useSearchParams } from "react-router-dom";

type CurrentsPlaceholderPageProps = {
  locationCode?: string;
};

export function CurrentsPlaceholderPage({
  locationCode = "nyc",
}: CurrentsPlaceholderPageProps) {
  const [params] = useSearchParams();
  const shift = params.get("shift") ?? "0";

  return (
    <section className="space-y-4">
      <div>
        <p className="font-medium text-swim-current text-sm uppercase">
          Current prediction
        </p>
        <h1 className="mt-1 font-semibold text-3xl">Currents</h1>
      </div>
      <div className="border-swim-line rounded border bg-white p-4">
        <p>
          React app shell route loaded for <strong>{locationCode}</strong>.
        </p>
        <p className="mt-2 text-slate-700">Shift: {shift} minutes</p>
      </div>
    </section>
  );
}

export function PageMessage({
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

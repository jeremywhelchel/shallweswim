import { NavLink, Outlet } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";

export function AppShell() {
  const bootstrap = useAppBootstrap();
  const locations = bootstrap.data?.location_order ?? ["nyc"];

  return (
    <div className="min-h-dvh bg-swim-mist text-swim-ink">
      <header className="border-swim-line border-b bg-white">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <a className="font-semibold text-swim-blue text-xl" href="/app">
            Shall We Swim
          </a>
          <nav aria-label="App navigation" className="flex flex-wrap gap-2">
            {locations.map((code) => (
              <NavLink
                className={({ isActive }) =>
                  [
                    "rounded border px-3 py-2 text-sm font-medium",
                    isActive
                      ? "border-swim-blue bg-swim-blue text-white"
                      : "border-swim-line bg-white text-swim-blue",
                  ].join(" ")
                }
                key={code}
                to={`/${code}`}
              >
                {bootstrap.data?.locations[code]?.metadata.nav_label ?? code}
              </NavLink>
            ))}
            <NavLink
              className={({ isActive }) =>
                [
                  "rounded border px-3 py-2 text-sm font-medium",
                  isActive
                    ? "border-swim-blue bg-swim-blue text-white"
                    : "border-swim-line bg-white text-swim-blue",
                ].join(" ")
              }
              to="/locations"
            >
              All
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}

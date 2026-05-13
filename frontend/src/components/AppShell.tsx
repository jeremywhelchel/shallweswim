import { Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="min-h-dvh bg-swim-mist text-swim-ink">
      <main className="mx-auto max-w-5xl px-3 py-3 sm:px-4 sm:py-6">
        <Outlet />
      </main>
    </div>
  );
}

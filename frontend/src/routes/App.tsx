import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, useParams } from "react-router-dom";
import { AppShell } from "../components/AppShell";
import { CurrentsPlaceholderPage } from "../pages/CurrentsPlaceholderPage";
import { LocationPlaceholderPage } from "../pages/LocationPlaceholderPage";
import { LocationsPlaceholderPage } from "../pages/LocationsPlaceholderPage";
import { NotFoundPage } from "../pages/NotFoundPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function DefaultLocationPage() {
  return <LocationPlaceholderPage preserveDefaultUrl />;
}

function LocationRoute() {
  const { locationCode } = useParams();
  return <LocationPlaceholderPage locationCode={locationCode} />;
}

function CurrentsRoute() {
  const { locationCode } = useParams();
  return <CurrentsPlaceholderPage locationCode={locationCode} />;
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<DefaultLocationPage />} />
            <Route path="locations" element={<LocationsPlaceholderPage />} />
            <Route path=":locationCode" element={<LocationRoute />} />
            <Route path=":locationCode/currents" element={<CurrentsRoute />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

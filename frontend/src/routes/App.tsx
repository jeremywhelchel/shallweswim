import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, useParams } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";
import { AppShell } from "../components/AppShell";
import { CurrentsPlaceholderPage } from "../pages/CurrentsPlaceholderPage";
import { LocationPage } from "../pages/LocationPage";
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
  const bootstrap = useAppBootstrap();

  if (bootstrap.isLoading) {
    return <LocationPlaceholderPage preserveDefaultUrl />;
  }

  if (bootstrap.isError || !bootstrap.data) {
    return <LocationPlaceholderPage preserveDefaultUrl />;
  }

  return (
    <LocationPage
      bootstrap={bootstrap.data}
      locationCode={bootstrap.data.default_location_code}
      preserveDefaultUrl
    />
  );
}

function LocationRoute() {
  const { locationCode } = useParams();
  const bootstrap = useAppBootstrap();
  const effectiveCode =
    locationCode ?? bootstrap.data?.default_location_code ?? "nyc";

  if (
    bootstrap.data &&
    effectiveCode === bootstrap.data.default_location_code &&
    bootstrap.data.locations[effectiveCode]
  ) {
    return (
      <LocationPage bootstrap={bootstrap.data} locationCode={effectiveCode} />
    );
  }

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

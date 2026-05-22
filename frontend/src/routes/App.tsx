import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";
import { BrowserRouter, Route, Routes, useParams } from "react-router-dom";
import { useAppBootstrap } from "../api/bootstrap";
import { AppShell } from "../components/AppShell";
import {
  clearLastLocationCode,
  getLastLocationCode,
  recordAppVisit,
  setLastLocationCode,
} from "../lib/preferences";
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
  const savedLocationCode = getLastLocationCode();
  const savedLocationIsValid = Boolean(
    bootstrap.data &&
      savedLocationCode &&
      bootstrap.data.locations[savedLocationCode],
  );

  useEffect(() => {
    if (bootstrap.data && savedLocationCode && !savedLocationIsValid) {
      clearLastLocationCode();
    }
  }, [bootstrap.data, savedLocationCode, savedLocationIsValid]);

  if (bootstrap.isLoading) {
    return <LocationPlaceholderPage preserveDefaultUrl />;
  }

  if (bootstrap.isError || !bootstrap.data) {
    return <LocationPlaceholderPage preserveDefaultUrl />;
  }

  const locationCode =
    savedLocationIsValid && savedLocationCode
      ? savedLocationCode
      : bootstrap.data.default_location_code;

  return (
    <LocationPage bootstrap={bootstrap.data} locationCode={locationCode} />
  );
}

function LocationRoute() {
  const { locationCode } = useParams();
  const bootstrap = useAppBootstrap();
  const effectiveCode =
    locationCode ?? bootstrap.data?.default_location_code ?? "nyc";

  useEffect(() => {
    if (bootstrap.data?.locations[effectiveCode]) {
      setLastLocationCode(effectiveCode);
    }
  }, [bootstrap.data, effectiveCode]);

  if (bootstrap.isLoading || bootstrap.isError || !bootstrap.data) {
    return <LocationPlaceholderPage locationCode={locationCode} />;
  }

  if (bootstrap.data.locations[effectiveCode]) {
    return (
      <LocationPage bootstrap={bootstrap.data} locationCode={effectiveCode} />
    );
  }

  return <NotFoundPage />;
}

export function App() {
  useEffect(() => {
    recordAppVisit();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<DefaultLocationPage />} />
            <Route path="locations" element={<LocationsPlaceholderPage />} />
            <Route path=":locationCode" element={<LocationRoute />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

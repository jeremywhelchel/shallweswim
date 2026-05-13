import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { LocationPlaceholderPage } from "../pages/LocationPlaceholderPage";

const bootstrapPayload = {
  app_name: "Shall We Swim",
  short_name: "Swim",
  default_location_code: "nyc",
  location_order: ["nyc"],
  manifest: {
    name: "Shall We Swim",
    short_name: "Swim",
    start_url: "/app",
    scope: "/app/",
    display: "standalone",
    theme_color: "#000099",
    background_color: "#fcffff",
  },
  locations: {
    nyc: {
      metadata: {
        code: "nyc",
        name: "New York",
        nav_label: "New York",
        swim_location: "Grimaldo's Chair",
        swim_location_link: "https://example.com",
        description:
          "Coney Island Brighton Beach open water swimming conditions",
        latitude: 40.573,
        longitude: -73.954,
        timezone: "US/Eastern",
        features: {
          temperature: true,
          tides: true,
          currents: true,
          webcam: true,
          transit: true,
          windy: true,
        },
        citations: {
          temperature: null,
          tides: null,
          currents: null,
        },
      },
      integrations: {
        youtube_live: null,
        transit_routes: [],
      },
    },
  },
};

function renderLocation() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  queryClient.setQueryData(["app-bootstrap"], bootstrapPayload);

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route
            index
            element={<LocationPlaceholderPage preserveDefaultUrl />}
          />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

test("renders the default location placeholder from bootstrap metadata", async () => {
  renderLocation();

  expect(
    await screen.findByRole("heading", { name: "New York" }),
  ).toBeVisible();
  expect(screen.getByText("Grimaldo's Chair")).toBeVisible();
  expect(screen.getByText(/React app shell route loaded for/)).toBeVisible();
});

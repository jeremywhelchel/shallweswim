import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { components } from "../api/generated";
import { ConditionsSummary, LocationPage } from "../pages/LocationPage";

const bootstrapPayload: components["schemas"]["AppBootstrapResponse"] = {
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
  source_code_link: {
    label: "jeremywhelchel/shallweswim",
    url: "https://github.com/jeremywhelchel/shallweswim",
    description: "Site source on github:",
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
          webcam: false,
          transit: false,
          windy: false,
        },
        citations: {
          temperature: '<a href="https://example.com/temp">Temp source</a>',
          tides: '<a href="https://example.com/tides">Tide source</a>',
          currents: '<a href="https://example.com/currents">Current source</a>',
        },
      },
      integrations: {
        youtube_live: null,
        transit_routes: [],
        webcam_alternative: null,
        webcam_source: null,
        transit_source: null,
      },
    },
  },
};

const conditionsPayload = {
  location: {
    code: "nyc",
    name: "New York",
    swim_location: "Grimaldo's Chair",
  },
  temperature: {
    timestamp: "2026-05-13T07:30:00-04:00",
    water_temp: 61.4,
    units: "F",
    station_name: "Coney Island",
  },
  tides: {
    past: [
      {
        time: "2026-05-13T06:00:00-04:00",
        type: "low",
        prediction: 0.2,
      },
    ],
    next: [
      {
        time: "2026-05-13T12:00:00-04:00",
        type: "high",
        prediction: 4.8,
      },
      {
        time: "2026-05-13T18:15:00-04:00",
        type: "low",
        prediction: 0.4,
      },
    ],
  },
  current: {
    timestamp: "2026-05-13T07:30:00-04:00",
    direction: "ebbing",
    phase: "ebb",
    strength: "moderate",
    trend: "building",
    magnitude: 1.25,
    magnitude_pct: 0.5,
    state_description: "moderate ebb and building",
    source_type: "prediction",
  },
} as const;

function renderLocation() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
    },
  });
  queryClient.setQueryData(["location-conditions", "nyc"], conditionsPayload);

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/"]}>
        <LocationPage
          bootstrap={bootstrapPayload}
          locationCode="nyc"
          preserveDefaultUrl
        />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

test("renders the NYC location page from bootstrap and conditions metadata", async () => {
  renderLocation();

  expect(
    await screen.findByRole("heading", { name: "shall we swim today?" }),
  ).toBeVisible();
  expect(screen.getByText("Grimaldo's Chair")).toBeVisible();
  expect(screen.getByText("61.4°F")).toBeVisible();
  expect(screen.getByText(/moderate ebb and building/)).toBeVisible();
  expect(screen.getByRole("heading", { name: "Sources" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Temp source" })).toBeVisible();
});

test("renders unavailable condition states on first-load failure", () => {
  render(
    <MemoryRouter>
      <ConditionsSummary hasError isLoading={false} locationCode="nyc" />
    </MemoryRouter>,
  );

  expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
  expect(
    screen.getByText("Current water temperature is unavailable."),
  ).toBeVisible();
  expect(screen.getByText("N/A")).toBeVisible();
});

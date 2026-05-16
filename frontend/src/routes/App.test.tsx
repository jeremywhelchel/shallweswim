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

const conditionsPayload: components["schemas"]["LocationConditions"] = {
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
    state: {
      timestamp: "2026-05-13T07:30:00-04:00",
      estimated_height: 1.6,
      units: "ft",
      trend: "rising",
      height_pct: 0.35,
    },
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
    range: {
      slack: {
        timestamp: "2026-05-13T05:45:00",
        magnitude: 0,
        units: "kt",
        phase: null,
      },
      peak: {
        timestamp: "2026-05-13T08:30:00",
        magnitude: 1.8,
        units: "kt",
        phase: "ebb",
      },
    },
    source_type: "prediction",
  },
};

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

function renderConditions(
  overrides: Partial<components["schemas"]["LocationConditions"]> = {},
) {
  return render(
    <MemoryRouter>
      <ConditionsSummary
        conditions={{
          ...conditionsPayload,
          ...overrides,
        }}
        hasError={false}
        isLoading={false}
        locationCode="nyc"
      />
    </MemoryRouter>,
  );
}

function tideState(
  trend: components["schemas"]["TideTrend"],
  heightPct: number,
): components["schemas"]["TideState"] {
  return {
    timestamp: "2026-05-13T07:30:00-04:00",
    estimated_height: 1.6,
    units: "ft",
    trend,
    height_pct: heightPct,
  };
}

function currentState(
  overrides: Partial<components["schemas"]["CurrentInfo"]> = {},
): components["schemas"]["CurrentInfo"] {
  return {
    ...(conditionsPayload.current as components["schemas"]["CurrentInfo"]),
    ...overrides,
  };
}

test("renders the NYC location page from bootstrap and conditions metadata", async () => {
  renderLocation();

  expect(
    await screen.findByRole("heading", { name: "shall we swim today?" }),
  ).toBeVisible();
  expect(screen.getByText("Grimaldo's Chair")).toBeVisible();
  expect(screen.getByText("61.4°F")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Water Movement" })).toBeVisible();
  expect(
    screen.getByText(
      "The tide is rising, and the water is going out steadily and getting stronger.",
    ),
  ).toBeVisible();
  expect(screen.getByText("TIDE")).toBeVisible();
  // Trend word is rendered in two CSS-toggled locations (mobile header /
  // desktop arrow callout); just confirm it's in the DOM at least once.
  expect(screen.getAllByText("rising").length).toBeGreaterThan(0);
  expect(screen.getByText(/low 0.2 ft/)).toBeVisible();
  expect(screen.getByText("1.6 ft")).toBeVisible();
  expect(screen.getByText(/high 4.8 ft/)).toBeVisible();
  expect(screen.getByText("CURRENT")).toBeVisible();
  expect(
    screen.getByText((_, element) => {
      const text = element?.textContent?.replace(/\s+/g, " ").trim();
      return text === "slack · 5:45 AM";
    }),
  ).toBeVisible();
  expect(screen.getByText("1.3 kt")).toBeVisible();
  expect(screen.getByText(/peak 1.8 kt/)).toBeVisible();
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
  expect(
    screen.getByText("Water movement is unavailable right now."),
  ).toBeVisible();
});

test("keeps water movement summary when tide state is unavailable", () => {
  render(
    <MemoryRouter>
      <ConditionsSummary
        conditions={{
          ...conditionsPayload,
          tides: {
            past: conditionsPayload.tides?.past ?? [],
            next: conditionsPayload.tides?.next ?? [],
            state: null,
          },
        }}
        hasError={false}
        isLoading={false}
        locationCode="nyc"
      />
    </MemoryRouter>,
  );

  expect(screen.queryByText("TIDE")).not.toBeInTheDocument();
  expect(
    screen.getByText(
      "Right now, the water is going out steadily and getting stronger.",
    ),
  ).toBeVisible();
  expect(screen.queryByText("Last tide")).not.toBeInTheDocument();
  expect(screen.queryByText("Next tide")).not.toBeInTheDocument();
});

test("describes easing currents near low tide in swimmer language", () => {
  renderConditions({
    tides: {
      past: conditionsPayload.tides?.past ?? [],
      next: conditionsPayload.tides?.next ?? [],
      state: tideState("falling", 0.1),
    },
    current: currentState({
      strength: "strong",
      trend: "easing",
    }),
  });

  expect(
    screen.getByText(
      "Near low tide, the water is going out fast, but starting to ease.",
    ),
  ).toBeVisible();
});

test.each([
  {
    name: "middle of rising tide with ebb building",
    tides: tideState("rising", 0.35),
    current: currentState({
      direction: "ebbing",
      phase: "ebb",
      strength: "moderate",
      trend: "building",
    }),
    expected:
      "The tide is rising, and the water is going out steadily and getting stronger.",
  },
  {
    name: "middle of falling tide with gentle flood holding steady",
    tides: tideState("falling", 0.55),
    current: currentState({
      direction: "flooding",
      phase: "flood",
      strength: "light",
      trend: "steady",
    }),
    expected:
      "The tide is falling, and the water is coming in gently and holding steady.",
  },
  {
    name: "near high tide with strong flood building",
    tides: tideState("rising", 0.9),
    current: currentState({
      direction: "flooding",
      phase: "flood",
      strength: "strong",
      trend: "building",
    }),
    expected:
      "Near high tide, the water is coming in fast and getting stronger.",
  },
  {
    name: "near low tide with strong ebb easing",
    tides: tideState("falling", 0.1),
    current: currentState({
      direction: "ebbing",
      phase: "ebb",
      strength: "strong",
      trend: "easing",
    }),
    expected:
      "Near low tide, the water is going out fast, but starting to ease.",
  },
  {
    name: "near low tide and slack",
    tides: tideState("falling", 0.05),
    current: currentState({
      direction: null,
      phase: "slack",
      strength: null,
      trend: null,
      magnitude_pct: 0,
    }),
    expected: "It's near low tide and calm.",
  },
  {
    name: "slack before flood without tide position",
    tides: tideState("rising", 0.45),
    current: currentState({
      direction: "flooding",
      phase: "slack_before_flood",
      strength: null,
      trend: null,
      magnitude_pct: 0,
    }),
    expected: "It's calm before the water starts coming in.",
  },
  {
    name: "slack before ebb without tide position",
    tides: tideState("falling", 0.45),
    current: currentState({
      direction: "ebbing",
      phase: "slack_before_ebb",
      strength: null,
      trend: null,
      magnitude_pct: 0,
    }),
    expected: "It's calm before the water starts going out.",
  },
])("describes water movement for $name", ({ tides, current, expected }) => {
  renderConditions({
    tides: {
      past: conditionsPayload.tides?.past ?? [],
      next: conditionsPayload.tides?.next ?? [],
      state: tides,
    },
    current,
  });

  expect(screen.getByText(expected)).toBeVisible();
});

test("describes tide-only water movement when currents are unavailable", () => {
  renderConditions({
    tides: {
      past: conditionsPayload.tides?.past ?? [],
      next: conditionsPayload.tides?.next ?? [],
      state: tideState("falling", 0.5),
    },
    current: null,
  });

  expect(screen.getByText("The tide is falling.")).toBeVisible();
});

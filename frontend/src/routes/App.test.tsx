import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { components } from "../api/generated";
import type { TransitStatus } from "../api/transit";
import { ConditionsSummary, LocationPage } from "../pages/LocationPage";
import { LocationsPage } from "../pages/LocationsPage";

const bootstrapPayload: components["schemas"]["AppBootstrapResponse"] = {
  app_name: "shall we swim?",
  short_name: "shallweswim",
  default_location_code: "nyc",
  location_order: ["nyc"],
  manifest: {
    name: "shall we swim?",
    short_name: "shallweswim",
    start_url: "/?source=pwa-react",
    scope: "/",
    display: "standalone",
    theme_color: "#000099",
    background_color: "#000099",
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
        temperature_plots: {
          live: true,
          historic: true,
        },
        citations: {
          temperature: '<a href="https://example.com/temp">Temp source</a>',
          tides: '<a href="https://example.com/tides">Tide source</a>',
          currents: '<a href="https://example.com/currents">Current source</a>',
        },
      },
      integrations: {
        webcam: null,
        transit_routes: [],
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

const shiftedConditionsPayload: components["schemas"]["LocationConditions"] = {
  ...conditionsPayload,
  current: {
    ...(conditionsPayload.current as components["schemas"]["CurrentInfo"]),
    timestamp: "2026-05-13T08:30:00",
    magnitude: 1.4,
    trend: "easing",
  },
  tides: {
    past: conditionsPayload.tides?.past ?? [],
    next: conditionsPayload.tides?.next ?? [],
    state: {
      timestamp: "2026-05-13T08:30:00-04:00",
      estimated_height: 2.2,
      units: "ft",
      trend: "rising",
      height_pct: 0.52,
    },
  },
};

function renderLocation({
  bootstrap = bootstrapPayload,
  conditions = conditionsPayload,
  initialEntry = "/",
  locationCode = "nyc",
  shiftedConditions = shiftedConditionsPayload,
  transitStatuses = {},
}: {
  bootstrap?: components["schemas"]["AppBootstrapResponse"];
  conditions?: components["schemas"]["LocationConditions"];
  initialEntry?: string;
  locationCode?: string;
  shiftedConditions?: components["schemas"]["LocationConditions"];
  transitStatuses?: Record<string, TransitStatus>;
} = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
    },
  });
  queryClient.setQueryData(["location-conditions", locationCode, null], {
    ...conditions,
    location: {
      ...conditions.location,
      code: locationCode,
    },
  });
  queryClient.setQueryData(
    ["location-conditions", locationCode, "2026-05-13T08:30:00"],
    {
      ...shiftedConditions,
      location: {
        ...shiftedConditions.location,
        code: locationCode,
      },
    },
  );
  for (const route of bootstrap.locations[locationCode].integrations
    .transit_routes ?? []) {
    const seededStatus =
      transitStatuses[
        `${route.goodservice_route_id}:${route.goodservice_direction}`
      ];
    if (seededStatus) {
      queryClient.setQueryData(
        [
          "transit-route",
          route.goodservice_route_id,
          route.goodservice_direction,
        ],
        seededStatus,
      );
    }
  }
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <LocationPage bootstrap={bootstrap} locationCode={locationCode} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function renderLocationWithConditions({
  conditions,
  initialEntry = "/nyc?detail=open",
}: {
  conditions: components["schemas"]["LocationConditions"];
  initialEntry?: string;
}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
    },
  });
  const at = new URL(`http://test${initialEntry}`).searchParams.get("at");
  queryClient.setQueryData(["location-conditions", "nyc", at], conditions);

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <LocationPage bootstrap={bootstrapPayload} locationCode="nyc" />
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

function syntheticLocation({
  code,
  features = {},
  integrations = {},
  temperaturePlots = {},
}: {
  code: string;
  features?: Partial<components["schemas"]["AppFeatureFlags"]>;
  integrations?: Partial<components["schemas"]["AppExternalIntegrations"]>;
  temperaturePlots?: Partial<components["schemas"]["AppTemperaturePlotConfig"]>;
}): components["schemas"]["AppBootstrapLocation"] {
  return {
    metadata: {
      ...bootstrapPayload.locations.nyc.metadata,
      code,
      name: `Test ${code.toUpperCase()}`,
      nav_label: `Test ${code.toUpperCase()}`,
      swim_location: "Test Beach",
      swim_location_link: "https://example.com/test-beach",
      description: "Synthetic swimming location for capability tests.",
      latitude: 38,
      longitude: -85,
      timezone: "US/Eastern",
      features: {
        temperature: false,
        tides: false,
        currents: false,
        webcam: false,
        transit: false,
        windy: false,
        ...features,
      },
      temperature_plots: {
        live: false,
        historic: false,
        ...temperaturePlots,
      },
      citations: {
        temperature: '<a href="https://example.com/temp">Temp source</a>',
        tides: features.tides
          ? '<a href="https://example.com/tides">Tide source</a>'
          : null,
        currents: features.currents
          ? '<a href="https://example.com/currents">Current source</a>'
          : null,
      },
    },
    integrations: {
      webcam: null,
      transit_routes: [],
      transit_source: null,
      ...integrations,
    },
  };
}

function syntheticBootstrap(
  location: components["schemas"]["AppBootstrapLocation"],
): components["schemas"]["AppBootstrapResponse"] {
  return {
    ...bootstrapPayload,
    default_location_code: location.metadata.code,
    location_order: [location.metadata.code],
    locations: {
      [location.metadata.code]: location,
    },
  };
}

function syntheticConditions({
  code,
  current = null,
  tides = null,
}: {
  code: string;
  current?: components["schemas"]["CurrentInfo"] | null;
  tides?: components["schemas"]["TideInfo"] | null;
}): components["schemas"]["LocationConditions"] {
  return {
    ...conditionsPayload,
    location: {
      code,
      name: `Test ${code.toUpperCase()}`,
      swim_location: "Test Beach",
    },
    current,
    tides,
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
  expect(screen.getByText("Predicted")).toBeVisible();
  expect(
    screen.getByText(
      "At Grimaldo's, expect a fast push west toward Coney Island Pier. The current is getting stronger.",
    ),
  ).toBeVisible();
  expect(
    screen.queryByText(/Tide height is rising; ebb current/),
  ).not.toBeInTheDocument();
  expect(screen.getByText("TIDE")).toBeVisible();
  // Trend word is rendered in two CSS-toggled locations (mobile header /
  // desktop arrow callout); just confirm it's in the DOM at least once.
  expect(screen.getAllByText(/rising/).length).toBeGreaterThan(0);
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
  expect(
    screen.getByRole("link", { name: "NOAA CO-OPS Station NYH1905_12" }),
  ).toBeVisible();
  expect(
    screen.getByRole("link", { name: "NOAA CO-OPS Station ACT3876" }),
  ).toBeVisible();
  expect(screen.queryByRole("link", { name: "Current source" })).toBeNull();
  expect(
    screen.getByRole("link", {
      name: "Tidal current charts, New York Harbor",
    }),
  ).toBeVisible();
  expect(
    screen.getAllByText((_, element) => {
      const text = element?.textContent?.replace(/\s+/g, " ").trim();
      return Boolean(
        text?.includes("Tidal current charts, New York Harbor from U.S."),
      );
    }).length,
  ).toBeGreaterThan(0);
  expect(screen.getByRole("img", { name: "Map credit" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Liam Hartigan" })).toBeVisible();
  expect(
    screen.getAllByText((_, element) => {
      const text = element?.textContent?.replace(/\s+/g, " ").trim();
      return (
        text ===
        "Coney Island Brighton Beach Map, Gary Atlas 5000 Edition, by Liam Hartigan"
      );
    }).length,
  ).toBeGreaterThan(0);
  expect(screen.getByRole("button", { name: "Details" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Plan" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Now" })).toBeNull();
  expect(screen.queryByRole("region", { name: "Planner mode" })).toBeNull();
  expect(
    screen.queryByRole("img", { name: /^Tide and current plot/ }),
  ).not.toBeInTheDocument();
});

test("renders all configured locations from bootstrap metadata", () => {
  const sdfConditions: components["schemas"]["LocationConditions"] = {
    ...conditionsPayload,
    location: {
      code: "sdf",
      name: "Louisville",
      swim_location: "Community Boathouse",
    },
    temperature: {
      timestamp: "2026-05-13T07:30:00-04:00",
      water_temp: 66.7,
      units: "F",
      station_name: "Ohio River at Water Tower",
    },
    current: {
      ...(conditionsPayload.current as components["schemas"]["CurrentInfo"]),
      source_type: "observation",
    },
    tides: null,
  };
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    location_order: ["nyc", "sdf"],
    locations: {
      ...bootstrapPayload.locations,
      sdf: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          code: "sdf",
          name: "Louisville",
          nav_label: "Louisville",
          swim_location: "Community Boathouse",
          description: "Louisville Kentucky open water swimming conditions",
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            tides: false,
            currents: true,
            transit: false,
          },
        },
        integrations: {
          webcam: null,
          transit_routes: [],
          transit_source: null,
        },
      },
    },
  };
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
    },
  });
  queryClient.setQueryData(
    ["location-conditions", "nyc", null],
    conditionsPayload,
  );
  queryClient.setQueryData(["location-conditions", "sdf", null], sdfConditions);

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <LocationsPage bootstrap={bootstrap} />
      </MemoryRouter>
    </QueryClientProvider>,
  );

  expect(
    screen.getByRole("heading", { name: "All Swim Locations" }),
  ).toBeVisible();
  expect(screen.getByRole("link", { name: /New York/ })).toHaveAttribute(
    "href",
    "/nyc",
  );
  expect(screen.getByRole("link", { name: /Louisville/ })).toHaveAttribute(
    "href",
    "/sdf",
  );
  expect(screen.getByText("61.4°F")).toBeVisible();
  expect(screen.getByText("66.7°F")).toBeVisible();
  expect(screen.getByText(/Data from Coney Island/)).toBeVisible();
  expect(screen.getByText(/Data from Ohio River at Water Tower/)).toBeVisible();
  expect(screen.getByText("Observed flow")).toBeVisible();
  expect(screen.getAllByText("Transit").length).toBeGreaterThan(0);
});

test("renders optional page sections from synthetic feature capabilities", async () => {
  const location = syntheticLocation({
    code: "cap",
    features: {
      temperature: true,
      tides: true,
      webcam: true,
      transit: true,
      windy: true,
    },
    temperaturePlots: {
      live: true,
      historic: true,
    },
    integrations: {
      webcam: {
        provider: "iframe",
        label: "Synthetic webcam",
        embed_url: "https://example.com/synthetic-webcam",
        script_url: null,
        watch_url: null,
        channel_id: null,
        note: null,
        source: {
          label: "Synthetic webcam source",
          url: "https://example.com/synthetic-webcam",
          description: "Synthetic webcam source.",
        },
        alternative: null,
      },
      transit_routes: [
        {
          label: "T",
          goodservice_route_id: "T",
          goodservice_direction: "south",
          icon_url: null,
        },
      ],
      transit_source: {
        label: "Synthetic transit",
        url: "https://example.com/transit",
        description: "Synthetic transit source.",
      },
    },
  });

  renderLocation({
    bootstrap: syntheticBootstrap(location),
    conditions: syntheticConditions({
      code: "cap",
      tides: conditionsPayload.tides,
    }),
    locationCode: "cap",
    transitStatuses: {
      "T:south": {
        status: "Good Service",
        destination: "Test Beach",
      },
    },
  });

  expect(
    await screen.findByRole("heading", { name: "Water Movement" }),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "Plan" })).toBeVisible();
  expect(screen.getByRole("heading", { name: "Forecast" })).toBeVisible();
  expect(screen.getByTitle("Windy forecast")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Live Webcam" })).toBeVisible();
  expect(screen.getByTitle("Synthetic webcam")).toHaveAttribute(
    "src",
    "https://example.com/synthetic-webcam",
  );
  expect(
    screen.getByRole("heading", { name: "Temperature Trends" }),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "12 mo" })).toBeVisible();
  expect(screen.getByRole("heading", { name: "Transit Status" })).toBeVisible();
  expect(screen.getByText("Good Service")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Sources" })).toBeVisible();
  expect(
    screen.getByRole("link", { name: "Synthetic webcam source" }),
  ).toHaveAttribute("href", "https://example.com/synthetic-webcam");
  expect(
    screen.getByRole("link", { name: "Synthetic transit" }),
  ).toHaveAttribute("href", "https://example.com/transit");
});

test("omits optional page sections when synthetic capabilities are disabled", () => {
  const location = syntheticLocation({
    code: "min",
    features: {
      tides: false,
      currents: false,
      webcam: false,
      transit: false,
      windy: false,
    },
  });

  renderLocation({
    bootstrap: syntheticBootstrap(location),
    conditions: syntheticConditions({ code: "min" }),
    locationCode: "min",
  });

  expect(
    screen.queryByRole("heading", { name: "Water Movement" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("heading", { name: "Forecast" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("heading", { name: "Live Webcam" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("heading", { name: "Temperature Trends" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("heading", { name: "Transit Status" }),
  ).not.toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Sources" })).toBeVisible();
});

test("renders only live temperature plot controls when historic plots are disabled", async () => {
  const location = syntheticLocation({
    code: "live",
    features: {
      temperature: true,
    },
    temperaturePlots: {
      live: true,
      historic: false,
    },
  });

  renderLocation({
    bootstrap: syntheticBootstrap(location),
    conditions: syntheticConditions({ code: "live" }),
    locationCode: "live",
  });

  expect(
    await screen.findByRole("heading", { name: "Temperature Trends" }),
  ).toBeVisible();
  expect(screen.queryByRole("button", { name: "Live" })).toBeNull();
  expect(screen.queryByRole("button", { name: "2 mo" })).toBeNull();
  expect(screen.queryByRole("button", { name: "12 mo" })).toBeNull();
  expect(screen.queryByRole("button", { name: "All" })).toBeNull();
  expect(screen.getByText("Loading plot")).toBeVisible();
});

test.each([
  {
    code: "none",
    conditions: syntheticConditions({ code: "none" }),
    features: { tides: false, currents: false },
    planVisible: false,
    sourceBadge: null,
    waterMovementVisible: false,
  },
  {
    code: "tide",
    conditions: syntheticConditions({
      code: "tide",
      tides: conditionsPayload.tides,
    }),
    features: { tides: true, currents: false },
    planVisible: true,
    sourceBadge: "Predicted",
    waterMovementVisible: true,
  },
  {
    code: "flow",
    conditions: syntheticConditions({
      code: "flow",
      current: currentState({
        direction: null,
        phase: null,
        strength: null,
        trend: null,
        magnitude: 0.82,
        magnitude_pct: null,
        state_description: null,
        range: null,
        source_type: "observation",
      }),
    }),
    features: { tides: false, currents: true },
    planVisible: false,
    sourceBadge: "Observed",
    waterMovementVisible: true,
  },
])("renders water movement controls from synthetic $code capabilities", ({
  code,
  conditions,
  features,
  planVisible,
  sourceBadge,
  waterMovementVisible,
}) => {
  const location = syntheticLocation({ code, features });

  renderLocation({
    bootstrap: syntheticBootstrap(location),
    conditions,
    initialEntry: `/${code}?planner=open`,
    locationCode: code,
  });

  const heading = screen.queryByRole("heading", { name: "Water Movement" });
  if (waterMovementVisible) {
    expect(heading).toBeVisible();
  } else {
    expect(heading).not.toBeInTheDocument();
  }

  const planButton = screen.queryByRole("button", { name: "Plan" });
  if (planVisible) {
    expect(planButton).toBeVisible();
    expect(screen.getByRole("region", { name: "Planner mode" })).toBeVisible();
  } else {
    expect(planButton).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Planner mode" }),
    ).not.toBeInTheDocument();
  }

  if (sourceBadge) {
    expect(screen.getByText(sourceBadge)).toBeVisible();
  }
});

test("renders not-scheduled transit without an unavailable destination", async () => {
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    locations: {
      nyc: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            transit: true,
          },
        },
        integrations: {
          ...bootstrapPayload.locations.nyc.integrations,
          transit_routes: [
            {
              label: "B",
              goodservice_route_id: "B",
              goodservice_direction: "south",
              icon_url: "/static/B-train.svg",
            },
          ],
          transit_source: null,
        },
      },
    },
  };

  renderLocation({
    bootstrap,
    transitStatuses: {
      "B:south": {
        status: "Not Scheduled",
        destination: "no scheduled service",
      },
    },
  });

  expect(
    await screen.findByRole("heading", { name: "Transit Status" }),
  ).toBeVisible();
  expect(screen.getByText("No scheduled service now")).toBeVisible();
  expect(screen.getByText("Not Scheduled")).toBeVisible();
  expect(screen.queryByText("unavailable")).not.toBeInTheDocument();
});

test("renders a YouTube live webcam from provider-aware integration config", async () => {
  renderLocation({
    bootstrap: {
      ...bootstrapPayload,
      locations: {
        nyc: {
          ...bootstrapPayload.locations.nyc,
          metadata: {
            ...bootstrapPayload.locations.nyc.metadata,
            features: {
              ...bootstrapPayload.locations.nyc.metadata.features,
              webcam: true,
            },
          },
          integrations: {
            ...bootstrapPayload.locations.nyc.integrations,
            webcam: {
              provider: "youtube_live",
              label: "Live webcam",
              embed_url:
                "https://www.youtube.com/embed/live_stream?channel=abc&enablejsapi=1",
              script_url: null,
              watch_url: "https://www.youtube.com/channel/abc/live",
              channel_id: "abc",
              note: "Looking southwest from Brighton 4th Street.",
              source: {
                label: "Webcam",
                url: "https://www.youtube.com/channel/abc/live",
                description: "thanks to the webcam hosts",
              },
              alternative: {
                label: "Alternate camera",
                url: "https://example.com/alt",
                description: "Another useful view.",
              },
            },
          },
        },
      },
    },
  });

  expect(
    await screen.findByRole("heading", { name: "Live Webcam" }),
  ).toBeVisible();
  expect(screen.getByTitle("Live webcam")).toHaveAttribute(
    "src",
    "https://www.youtube.com/embed/live_stream?channel=abc&enablejsapi=1",
  );
  expect(screen.getByRole("link", { name: "Webcam" })).toHaveAttribute(
    "href",
    "https://www.youtube.com/channel/abc/live",
  );
  expect(
    screen.getByRole("link", { name: "Alternate camera" }),
  ).toHaveAttribute("href", "https://example.com/alt");
  expect(
    screen.getByText("Looking southwest from Brighton 4th Street."),
  ).toBeVisible();
});

test("renders an iframe webcam provider for non-NYC locations", async () => {
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    location_order: ["nyc", "chi"],
    locations: {
      ...bootstrapPayload.locations,
      chi: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          code: "chi",
          name: "Chicago",
          nav_label: "Chicago",
          swim_location: "Ohio Street Beach",
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            currents: false,
            webcam: true,
          },
        },
        integrations: {
          webcam: {
            provider: "iframe",
            label: "Live webcam",
            embed_url: "https://example.com/chicago-webcam",
            script_url: null,
            watch_url: null,
            channel_id: null,
            note: "Ohio Street Beach view from Willis Tower.",
            source: {
              label: "Chicago webcam",
              url: "https://example.com/chicago-webcam",
              description: "Live webcam for Ohio Street Beach conditions.",
            },
            alternative: null,
          },
          transit_routes: [],
          transit_source: null,
        },
      },
    },
  };

  renderLocation({ bootstrap, locationCode: "chi" });

  expect(
    await screen.findByRole("heading", { name: "Live Webcam" }),
  ).toBeVisible();
  expect(screen.getByTitle("Live webcam")).toHaveAttribute(
    "src",
    "https://example.com/chicago-webcam",
  );
  expect(screen.getByRole("link", { name: "Chicago webcam" })).toHaveAttribute(
    "href",
    "https://example.com/chicago-webcam",
  );
  expect(
    screen.getByText("Ohio Street Beach view from Willis Tower."),
  ).toBeVisible();
});

test("renders a named EarthCam provider as a contained iframe", async () => {
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    location_order: ["nyc", "sdf"],
    locations: {
      ...bootstrapPayload.locations,
      sdf: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          code: "sdf",
          name: "Louisville",
          nav_label: "Louisville",
          swim_location: "Community Boathouse",
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            currents: false,
            webcam: true,
          },
        },
        integrations: {
          webcam: {
            provider: "earthcam_embed",
            label: "Live webcam",
            embed_url: "https://share.earthcam.net/test.player",
            script_url: null,
            watch_url: "https://www.earthcam.com/test",
            channel_id: null,
            note: "View overlooking Toehead Island swim channel",
            source: {
              label: "EarthCam Ohio River",
              url: "https://www.earthcam.com/test",
              description: "View overlooking Toehead Island swim channel.",
            },
            alternative: null,
          },
          transit_routes: [],
          transit_source: null,
        },
      },
    },
  };

  renderLocation({ bootstrap, locationCode: "sdf" });

  expect(
    await screen.findByRole("heading", { name: "Live Webcam" }),
  ).toBeVisible();
  const embedRoot = document.querySelector("[data-earthcam-embed-root]");
  expect(embedRoot).not.toBeNull();
  expect(
    within(embedRoot as HTMLElement).getByTitle("Live webcam"),
  ).toHaveAttribute("src", "https://share.earthcam.net/test.player");
  expect(document.querySelector("script.earthcam-embed")).toBeNull();
  expect(
    screen.getByText("View overlooking Toehead Island swim channel"),
  ).toBeVisible();
  expect(
    screen.getByRole("link", { name: "EarthCam Ohio River" }),
  ).toHaveAttribute("href", "https://www.earthcam.com/test");
});

test("planner mode shifts all time-aware water movement elements", async () => {
  renderLocation({
    initialEntry: "/nyc?planner=open&at=2026-05-13T08:30:00",
  });

  const panel = await screen.findByRole("region", {
    name: "Planner mode",
  });
  const controls = screen.getByRole("region", {
    name: "Water movement controls",
  });
  expect(panel).toBeVisible();
  expect(controls).toHaveClass("sticky");
  expect(within(controls).getByText("Water Movement")).toBeVisible();
  expect(within(controls).getByRole("button", { name: "Now" })).toBeVisible();
  expect(screen.getByText("May 13, 2026, 8:30 AM")).toBeVisible();
  expect(
    screen.getByText(/expect a fast push west toward Coney Island Pier/),
  ).toBeVisible();
  expect(screen.getAllByText(/rising/).length).toBeGreaterThan(0);
  expect(screen.getByText("1.4 kt")).toBeVisible();
  expect(within(panel).getByLabelText("Planner time")).toHaveAttribute(
    "max",
    "1440",
  );
  expect(
    screen.queryByRole("img", { name: /^Tide and current plot/ }),
  ).not.toBeInTheDocument();
  expect(screen.getByText("2.2 ft")).toBeVisible();
  expect(screen.queryByText("1.6 ft")).not.toBeInTheDocument();

  fireEvent.change(within(panel).getByLabelText("Planner time"), {
    target: { value: "120" },
  });
  expect(within(panel).getByText("+2h")).toBeVisible();
});

test("detail mode shows the current and tide plot independently", async () => {
  renderLocation({
    initialEntry: "/nyc?detail=open&at=2026-05-13T08:30:00",
  });

  expect(screen.queryByRole("region", { name: "Planner mode" })).toBeNull();
  expect(
    screen.getByText(/expect a fast push west toward Coney Island Pier/),
  ).toBeVisible();
  expect(screen.getByText("1.4 kt")).toBeVisible();
  expect(screen.getByText("2.2 ft")).toBeVisible();
  expect(screen.queryByText("1.6 ft")).not.toBeInTheDocument();
  const shiftedPlot = screen.getByRole("img", {
    name: "Tide and current plot for May 13, 2026, 8:30 AM",
  });
  expect(shiftedPlot).toHaveAttribute(
    "src",
    "/api/nyc/plots/current_tide?at=2026-05-13T08%3A30%3A00",
  );
  expect(
    screen.getByRole("heading", {
      name: "Grimaldo's Chair current guidance",
    }),
  ).toBeVisible();
  expect(screen.getByText(/Start east toward Manhattan Beach/)).toBeVisible();
  expect(screen.getByText("Flood current")).toBeVisible();
  expect(
    screen.getByText("Water usually pushes east toward Manhattan Beach."),
  ).toBeVisible();
  expect(screen.getByText("Ebb current")).toBeVisible();
  expect(
    screen.getByText("Water usually pushes west toward Coney Island Pier."),
  ).toBeVisible();
  expect(
    screen.getByText(/describe current movement, not a required swim route/),
  ).toBeVisible();
  expect(
    screen.getByText(
      /Farther west toward the Aquarium and Coney Island Pier, the current can behave differently and may switch direction/,
    ),
  ).toBeVisible();
  expect(
    screen.getByRole("link", { name: "CIBBOWS Essentials" }),
  ).toHaveAttribute("href", "https://example.com");
  expect(
    screen.getByRole("img", {
      name: "Coney Island ebbing current map at 55% strength",
    }),
  ).toHaveAttribute("src", "/static/plots/nyc/current_chart_ebbing_55.png");
  expect(
    screen.getByRole("img", {
      name: "Historic New York Harbor chart: 3 Hours after Low Water at New York",
    }),
  ).toHaveAttribute("src", "/static/tidecharts/low+3.png");
});

test("NYC local map and harbor chart derive from selected time state", () => {
  const floodConditions: components["schemas"]["LocationConditions"] = {
    ...shiftedConditionsPayload,
    current: currentState({
      timestamp: "2026-05-13T09:30:00",
      direction: "flooding",
      phase: "flood",
      magnitude: 1.7,
      magnitude_pct: 0.91,
      trend: "building",
    }),
    tides: {
      past: [
        {
          time: "2026-05-13T06:00:00-04:00",
          type: "high",
          prediction: 4.8,
        },
      ],
      next: shiftedConditionsPayload.tides?.next ?? [],
      state: shiftedConditionsPayload.tides?.state ?? null,
    },
  };

  renderLocationWithConditions({
    conditions: floodConditions,
    initialEntry: "/nyc?detail=open&at=2026-05-13T09:30:00",
  });

  expect(
    screen.getByRole("img", {
      name: "Coney Island flooding current map at 100% strength",
    }),
  ).toHaveAttribute("src", "/static/plots/nyc/current_chart_flooding_100.png");
  expect(
    screen.getByRole("img", {
      name: "Historic New York Harbor chart: 4 Hours after High Water at New York",
    }),
  ).toHaveAttribute("src", "/static/tidecharts/high+4.png");
});

test("at shifts water movement without opening planner or detail panels", async () => {
  renderLocation({
    initialEntry: "/nyc?at=2026-05-13T08:30:00",
  });

  expect(screen.queryByRole("region", { name: "Planner mode" })).toBeNull();
  expect(
    screen.queryByRole("img", { name: /^Tide and current plot/ }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByText(/expect a fast push west toward Coney Island Pier/),
  ).toBeVisible();
  expect(screen.getByText("1.4 kt")).toBeVisible();
  expect(screen.getByText("2.2 ft")).toBeVisible();
  expect(screen.queryByText("1.6 ft")).not.toBeInTheDocument();
  expect(screen.getByText("May 13, 2026, 8:30 AM")).toBeVisible();

  fireEvent.click(
    within(
      screen.getByRole("region", { name: "Water movement controls" }),
    ).getByRole("button", { name: "Now" }),
  );
  await waitFor(() => {
    expect(screen.getByText("1.6 ft")).toBeVisible();
  });
  expect(screen.queryByText("2.2 ft")).not.toBeInTheDocument();
  expect(screen.queryByText("May 13, 2026, 8:30 AM")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Now" })).toBeNull();
  expect(screen.queryByRole("region", { name: "Planner mode" })).toBeNull();
});

test("omits water movement for locations without tide or current data", () => {
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    location_order: ["nyc", "chi"],
    locations: {
      ...bootstrapPayload.locations,
      chi: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          code: "chi",
          name: "Chicago",
          nav_label: "Chicago",
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            currents: false,
            tides: false,
          },
        },
      },
    },
  };

  renderLocation({
    bootstrap,
    initialEntry: "/chi?planner=open&at=2026-05-13T08:30:00",
    locationCode: "chi",
  });

  expect(
    screen.queryByRole("heading", { name: "Water Movement" }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("region", { name: "Planner mode" }),
  ).not.toBeInTheDocument();
});

test("supports planner mode for tide-only locations without detail controls", () => {
  const tideOnlyConditions: components["schemas"]["LocationConditions"] = {
    ...conditionsPayload,
    current: null,
  };
  const shiftedTideOnlyConditions: components["schemas"]["LocationConditions"] =
    {
      ...shiftedConditionsPayload,
      current: null,
    };
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    location_order: ["nyc", "sfo"],
    locations: {
      ...bootstrapPayload.locations,
      sfo: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          code: "sfo",
          name: "San Francisco",
          nav_label: "San Francisco",
          swim_location: "Aquatic Park",
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            currents: false,
            tides: true,
          },
        },
      },
    },
  };

  renderLocation({
    bootstrap,
    conditions: tideOnlyConditions,
    initialEntry: "/sfo?planner=open&detail=open&at=2026-05-13T08:30:00",
    locationCode: "sfo",
    shiftedConditions: shiftedTideOnlyConditions,
  });

  expect(screen.getByRole("heading", { name: "Water Movement" })).toBeVisible();
  expect(
    screen.getByText("The tide is rising toward high tide."),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "Plan" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
  expect(screen.getByRole("region", { name: "Planner mode" })).toBeVisible();
  expect(screen.getByText("2.2 ft")).toBeVisible();
  expect(screen.queryByText("1.6 ft")).not.toBeInTheDocument();
  expect(
    screen.queryByRole("img", { name: /^Tide and current plot/ }),
  ).not.toBeInTheDocument();
});

test("renders observed flow without tidal water movement for river-current locations", () => {
  const sdfConditions: components["schemas"]["LocationConditions"] = {
    ...conditionsPayload,
    current: {
      timestamp: "2026-05-13T07:30:00-04:00",
      direction: null,
      phase: null,
      strength: null,
      trend: null,
      magnitude: 0.82,
      magnitude_pct: null,
      state_description: null,
      range: null,
      source_type: "observation",
    },
    tides: null,
  };
  const bootstrap: components["schemas"]["AppBootstrapResponse"] = {
    ...bootstrapPayload,
    location_order: ["nyc", "sdf"],
    locations: {
      ...bootstrapPayload.locations,
      sdf: {
        ...bootstrapPayload.locations.nyc,
        metadata: {
          ...bootstrapPayload.locations.nyc.metadata,
          code: "sdf",
          name: "Louisville",
          nav_label: "Louisville",
          swim_location: "Community Boathouse",
          features: {
            ...bootstrapPayload.locations.nyc.metadata.features,
            currents: true,
            tides: false,
          },
        },
      },
    },
  };

  renderLocation({
    bootstrap,
    conditions: sdfConditions,
    initialEntry: "/sdf?planner=open",
    locationCode: "sdf",
  });

  expect(screen.getByRole("heading", { name: "Water Movement" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Plan" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
  expect(screen.getByText("Observed")).toBeVisible();
  expect(screen.getByText("0.8 kt")).toBeVisible();
  expect(screen.queryByText(/not a tide prediction/)).toBeNull();
});

test("renders unavailable condition states on first-load failure", () => {
  render(
    <MemoryRouter>
      <ConditionsSummary hasError isLoading={false} />
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
      />
    </MemoryRouter>,
  );

  expect(screen.queryByText("TIDE")).not.toBeInTheDocument();
  expect(
    screen.getByText(
      "Right now, the water is going out fast and getting stronger.",
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
      magnitude: 0.8,
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
      magnitude: 0.25,
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
      magnitude: 1.1,
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
      magnitude: 1.1,
      trend: "easing",
    }),
    expected:
      "Near low tide, the water is going out fast, but starting to ease.",
  },
  {
    name: "at low tide with slack",
    tides: tideState("falling", 0.02),
    current: currentState({
      direction: null,
      phase: "slack",
      strength: null,
      trend: null,
      magnitude: 0,
      magnitude_pct: 0,
    }),
    expected: "It's at low tide and calm.",
  },
  {
    name: "at high tide with gentle absolute current",
    tides: tideState("rising", 0.99),
    current: currentState({
      direction: "flooding",
      phase: "flood",
      strength: "strong",
      magnitude: 0.35,
      trend: "building",
    }),
    expected:
      "At high tide, the water is coming in gently and getting stronger.",
  },
  {
    name: "moderate relative current but fast absolute current",
    tides: tideState("falling", 0.45),
    current: currentState({
      direction: "ebbing",
      phase: "ebb",
      strength: "moderate",
      magnitude: 1.2,
      trend: "steady",
    }),
    expected:
      "The tide is falling, and the water is going out fast and holding steady.",
  },
  {
    name: "near low tide and slack",
    tides: tideState("falling", 0.1),
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
      state: tideState("falling", 0.96),
    },
    current: null,
  });

  expect(
    screen.getByText("It's at high tide, and the tide is falling."),
  ).toBeVisible();
});

test("describes mid-cycle tide-only water movement with target tide", () => {
  renderConditions({
    tides: {
      past: conditionsPayload.tides?.past ?? [],
      next: conditionsPayload.tides?.next ?? [],
      state: tideState("rising", 0.5),
    },
    current: null,
  });

  expect(
    screen.getByText("The tide is rising toward high tide."),
  ).toBeVisible();
});

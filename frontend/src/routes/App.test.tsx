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
}: {
  bootstrap?: components["schemas"]["AppBootstrapResponse"];
  conditions?: components["schemas"]["LocationConditions"];
  initialEntry?: string;
  locationCode?: string;
  shiftedConditions?: components["schemas"]["LocationConditions"];
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
      "The tide is rising, and the water is going out fast and getting stronger.",
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
              note: null,
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
            note: null,
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
});

test("renders a named EarthCam provider with contained script ownership", async () => {
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
            embed_url: null,
            script_url: "https://share.earthcam.net/embed/test",
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
  await waitFor(() => {
    expect(document.querySelector("script.earthcam-embed")).toHaveAttribute(
      "src",
      "https://share.earthcam.net/embed/test",
    );
  });
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
  expect(screen.getByText(/water is going out fast/)).toBeVisible();
  expect(screen.getAllByText("rising").length).toBeGreaterThan(0);
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
  expect(screen.getByText(/water is going out fast/)).toBeVisible();
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
    screen.getByRole("heading", { name: "Grimaldo's Chair local read" }),
  ).toBeVisible();
  expect(
    screen.getByText(/Start eastbound toward Manhattan Beach/),
  ).toBeVisible();
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
  expect(screen.getByText(/water is going out fast/)).toBeVisible();
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

test("omits tidal water movement for observed-current locations without tides", () => {
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
    initialEntry: "/sdf?planner=open",
    locationCode: "sdf",
  });

  expect(
    screen.queryByRole("heading", { name: "Water Movement" }),
  ).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Plan" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
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

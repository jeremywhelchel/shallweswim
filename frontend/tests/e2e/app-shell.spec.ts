import { expect, type Page, test } from "@playwright/test";

const bootstrapPayload = {
  app_name: "shall we swim?",
  short_name: "shallweswim",
  default_location_code: "nyc",
  location_order: ["nyc", "sfo"],
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
        default_temperature_unit: "F",
        features: {
          temperature: true,
          tides: true,
          currents: true,
          water_movement_planning: true,
          water_movement_detail: true,
          webcam: true,
          transit: true,
          windy: true,
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
        webcam: {
          provider: "youtube_live",
          label: "Live webcam",
          channel_id: "UChh9yX1PSFFreQFmnnIPGuQ",
          embed_url:
            "https://www.youtube.com/embed/live_stream?channel=UChh9yX1PSFFreQFmnnIPGuQ&enablejsapi=1&controls=0&playsinline=1&iv_load_policy=3&rel=0",
          script_url: null,
          watch_url:
            "https://www.youtube.com/channel/UChh9yX1PSFFreQFmnnIPGuQ/live",
          note: null,
          source: {
            label: "Webcam",
            url: "https://www.youtube.com/channel/UChh9yX1PSFFreQFmnnIPGuQ/live",
            description: "thanks to David K and Karol L",
          },
          alternative: {
            label: "Earth Cam Coney Island",
            url: "https://www.earthcam.com/usa/newyork/coneyisland/?cam=coneyisland",
            description: "Great view, including the amusement park.",
          },
        },
        transit_routes: [
          {
            label: "B",
            goodservice_route_id: "B",
            goodservice_direction: "south",
            icon_url: "/static/B-train.svg",
          },
          {
            label: "Q",
            goodservice_route_id: "Q",
            goodservice_direction: "south",
            icon_url: "/static/Q-train.svg",
          },
        ],
        transit_source: {
          label: "goodservice.io",
          url: "https://goodservice.io",
          description: "MTA train status provided by:",
        },
        windy: {
          overlay: "waves",
          product: "ecmwfWaves",
          level: "surface",
          zoom: 11,
          metric_wind: "default",
          metric_temp: "°F",
        },
      },
    },
    sfo: {
      metadata: {
        code: "sfo",
        name: "San Francisco",
        nav_label: "San Francisco",
        swim_location: "Aquatic Park",
        swim_location_link: "https://example.com/sfo",
        description:
          "San Francisco Aquatic Park open water swimming conditions",
        latitude: 37.806,
        longitude: -122.422,
        timezone: "US/Pacific",
        default_temperature_unit: "F",
        features: {
          temperature: true,
          tides: false,
          currents: false,
          water_movement_planning: false,
          water_movement_detail: false,
          webcam: false,
          transit: false,
          windy: false,
        },
        temperature_plots: {
          live: true,
          historic: true,
        },
        citations: {
          temperature: '<a href="https://example.com/sfo-temp">SFO temp</a>',
          tides: null,
          currents: null,
        },
      },
      integrations: {
        webcam: null,
        transit_routes: [],
        transit_source: null,
        windy: null,
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
    water_temp_f: 61.4,
    water_temp_c: 16.3,
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

const sfoConditionsPayload = {
  ...conditionsPayload,
  location: {
    code: "sfo",
    name: "San Francisco",
    swim_location: "Aquatic Park",
  },
  temperature: {
    timestamp: "2026-05-13T07:30:00-07:00",
    water_temp_f: 55.2,
    water_temp_c: 12.9,
    station_name: "San Francisco",
  },
  tides: {
    past: [],
    next: [],
    state: null,
  },
  current: null,
};

const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
  "base64",
);

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    const fixedNow = new Date("2026-05-13T11:30:00Z").valueOf();
    const RealDate = Date;
    class MockDate extends RealDate {
      constructor(...args: ConstructorParameters<DateConstructor>) {
        if (args.length === 0) {
          super(fixedNow);
        } else {
          super(...args);
        }
      }

      static now() {
        return fixedNow;
      }
    }
    window.Date = MockDate as DateConstructor;
  });
  await page.route("**/api/app/bootstrap", async (route) => {
    await route.fulfill({ json: bootstrapPayload });
  });
  await page.route("**/api/*/conditions**", async (route) => {
    const locationCode = new URL(route.request().url()).pathname.split("/")[2];
    const at = new URL(route.request().url()).searchParams.get("at");
    await route.fulfill({
      json:
        locationCode === "sfo"
          ? sfoConditionsPayload
          : at
            ? {
                ...conditionsPayload,
                current: {
                  ...conditionsPayload.current,
                  timestamp: "2026-05-13T08:30:00",
                  magnitude: 1.4,
                  trend: "easing",
                },
                tides: {
                  ...conditionsPayload.tides,
                  state: {
                    timestamp: "2026-05-13T08:30:00-04:00",
                    estimated_height: 2.2,
                    units: "ft",
                    trend: "rising",
                    height_pct: 0.52,
                  },
                },
              }
            : conditionsPayload,
    });
  });
  await page.route("https://goodservice.io/api/routes/**", async (route) => {
    await route.fulfill({
      json: {
        status: "Good Service",
        direction_statuses: { south: "Good Service" },
        destinations: { south: ["Coney Island-Stillwell Av"] },
        delay_summaries: {},
        service_change_summaries: {},
        service_irregularity_summaries: {},
      },
    });
  });
  await page.route("**/api/nyc/plots/**", async (route) => {
    await route.fulfill({ body: onePixelPng, contentType: "image/png" });
  });
  await page.route("**/static/plots/nyc/**", async (route) => {
    await route.fulfill({ body: onePixelPng, contentType: "image/png" });
  });
  await page.route("**/static/tidecharts/**", async (route) => {
    await route.fulfill({ body: onePixelPng, contentType: "image/png" });
  });
  await page.route("https://embed.windy.com/**", async (route) => {
    await route.fulfill({ body: "<html><body>Windy</body></html>" });
  });
  await page.route("https://www.youtube.com/iframe_api", async (route) => {
    await route.fulfill({
      body: `
        window.YT = {
          Player: function (_id, options) {
            options.events.onReady({
              target: { mute: function () {}, playVideo: function () {} }
            });
          }
        };
        window.onYouTubeIframeAPIReady && window.onYouTubeIframeAPIReady();
      `,
      contentType: "application/javascript",
    });
  });
  await page.route("https://www.youtube.com/embed/**", async (route) => {
    await route.fulfill({ body: "<html><body>YouTube</body></html>" });
  });
});

test("boots the default route @smoke", async ({ page }) => {
  await gotoApp(page, "/");

  await expect(page).toHaveURL(/\/$/);
  await expect(
    page.getByRole("heading", { name: "shall we swim today?" }),
  ).toBeVisible();
});

test("renders the default app route without redirecting to /nyc", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "mobile-chromium",
    "Default condition rendering is viewport-independent and covered in mobile Chromium.",
  );

  await gotoApp(page, "/");

  await expect(page).toHaveURL(/\/$/);
  await expect(
    page.getByRole("heading", { name: "shall we swim today?" }),
  ).toBeVisible();
  await expect(page.getByText("61.4°F")).toBeVisible();
});

test("restores the last selected location from local preferences", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "mobile-chromium",
    "Preference persistence is viewport-independent and covered in mobile Chromium.",
  );

  await gotoApp(page, "/sfo");

  await expect(
    page.getByRole("heading", { name: "shall we swim today?" }),
  ).toBeVisible();
  await expect(page.getByText("Aquatic Park")).toBeVisible();
  await expect(page.getByText("55.2°F")).toBeVisible();
  await expect
    .poll(async () =>
      page.evaluate(() =>
        JSON.parse(
          window.localStorage.getItem("shallweswim.appPreferences") ?? "{}",
        ),
      ),
    )
    .toMatchObject({
      version: 1,
      lastLocationCode: "sfo",
      installPrompt: { organicVisitCount: 1 },
    });

  await gotoApp(page, "/");

  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText("Aquatic Park")).toBeVisible();
  await expect(page.getByText("55.2°F")).toBeVisible();

  await page.evaluate(() => {
    window.localStorage.setItem(
      "shallweswim.appPreferences",
      JSON.stringify({
        version: 1,
        lastLocationCode: "missing",
        installPrompt: { organicVisitCount: 4 },
      }),
    );
  });
  await gotoApp(page, "/");

  await expect(page.getByText("Grimaldo's Chair")).toBeVisible();
  await expect(page.getByText("61.4°F")).toBeVisible();
  await expect
    .poll(async () =>
      page.evaluate(() =>
        JSON.parse(
          window.localStorage.getItem("shallweswim.appPreferences") ?? "{}",
        ),
      ),
    )
    .toMatchObject({
      installPrompt: { organicVisitCount: 5 },
    });
  await expect
    .poll(async () =>
      page.evaluate(
        () =>
          JSON.parse(
            window.localStorage.getItem("shallweswim.appPreferences") ?? "{}",
          ).lastLocationCode,
      ),
    )
    .toBeUndefined();
});

test("renders the all locations page", async ({ page }) => {
  test.skip(
    test.info().project.name !== "mobile-chromium",
    "All-locations rendering is viewport-independent and covered in mobile Chromium.",
  );

  await gotoApp(page, "/locations");

  await expect(
    page.getByRole("heading", { name: "All Swim Locations" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /New York/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /San Francisco/ })).toBeVisible();
  await expect(page.getByText("61.4°F")).toBeVisible();
  await expect(page.getByText("55.2°F")).toBeVisible();
  await expect(page.getByText(/Data from Coney Island/)).toBeVisible();
  await expect(page.getByText(/Data from San Francisco/)).toBeVisible();
  await expect(page.getByText("Temp").first()).toBeVisible();
  await expect(page.getByText("Webcam").first()).toBeVisible();
});

test("renders the NYC location vertical slice", async ({ page }) => {
  test.skip(
    test.info().project.name !== "mobile-chromium",
    "Full vertical slice is covered in mobile Chromium; desktop interaction coverage is handled by planner mode.",
  );

  await gotoApp(page, "/nyc");

  await expect(page.getByRole("heading", { name: "Forecast" })).toBeVisible();
  await expect(page.getByTitle("Windy forecast")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Live Webcam" }),
  ).toBeVisible();
  await expect(
    page
      .getByTitle("Live webcam")
      .or(page.getByText("Live webcam loading"))
      .first(),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Temperature Trends" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "12 month temperature plot, all years" }),
  ).toBeVisible();
  await page.getByRole("button", { exact: true, name: "2 mo" }).click();
  await expect(
    page.getByRole("img", {
      exact: true,
      name: "2 month temperature plot, all years",
    }),
  ).toBeVisible();
  await page.getByRole("button", { exact: true, name: "Live" }).click();
  await expect(
    page.getByRole("img", { name: "Live temperature plot" }),
  ).toBeVisible();
  await page.getByRole("button", { exact: true, name: "All" }).click();
  await expect(
    page.getByRole("img", { name: "Live temperature plot" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      exact: true,
      name: "2 month temperature plot, all years",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "12 month temperature plot, all years" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Transit Status" }),
  ).toBeVisible();
  await expect(page.getByText("Coney Island-Stillwell Av")).toHaveCount(2);
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Water Movement" }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "At Grimaldo's, expect a fast push west toward Coney Island Pier. The current is getting stronger.",
    ),
  ).toBeVisible();
  await expect(page.getByText("TIDE", { exact: true })).toBeVisible();
  await expect(page.getByText(/low 0.2 ft/)).toBeVisible();
  await expect(page.getByText("1.6 ft", { exact: true })).toBeVisible();
  await expect(page.getByText(/high 4.8 ft/)).toBeVisible();
  await expect(
    page.getByText("slack · 5:45 AM", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("1.3 kt", { exact: true })).toBeVisible();
  await expect(page.getByText(/peak 1.8 kt/)).toBeVisible();
  await expect(
    page.getByRole("link", { name: "goodservice.io" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Earth Cam Coney Island" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "NOAA CO-OPS Station NYH1905_12" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "NOAA CO-OPS Station ACT3876" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", {
      name: "Tidal current charts, New York Harbor",
    }),
  ).toBeVisible();
  await expect(
    page.getByText(/Tidal current charts, New York Harbor from U\.S\./),
  ).toBeVisible();
  await expect(page.getByRole("img", { name: "Map credit" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Liam Hartigan" })).toBeVisible();
  await expect(
    page.getByText(
      /Coney Island Brighton Beach Map, Gary Atlas 5000 Edition, by Liam Hartigan/,
    ),
  ).toBeVisible();
  await expect(page.getByText("Alternate:")).toBeVisible();
  await expect(page.getByText("Alternative option:")).toHaveCount(0);

  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(hasHorizontalOverflow).toBe(false);
});

test("opens mobile detail mode from the condition stack", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "mobile-chromium",
    "Mobile detail interaction is covered once in mobile Chromium.",
  );

  await gotoApp(page, "/nyc");

  await expect(page.getByText("61.4°F")).toBeVisible();
  await page.getByRole("button", { name: "Details" }).click();

  await expect(
    page.getByRole("region", { name: "Current and tide detail chart" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /^Tide and current plot for / }),
  ).toBeVisible();
  await expect(page.getByText("61.4°F")).toBeVisible();
  await expect(page.getByText("TIDE", { exact: true })).toBeVisible();
});

test("planner mode shifts dashboard water movement from URL state @desktop", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-chromium",
    "Planner URL-state behavior is viewport-independent and covered in desktop Chromium.",
  );

  await gotoApp(page, "/nyc?planner=open&at=2026-05-13T08:30:00");

  const panel = page.getByRole("region", { name: "Planner mode" });
  const controls = page.getByRole("region", {
    name: "Water movement controls",
  });
  await expect(panel).toBeVisible();
  await expect(controls).toHaveCSS("position", "sticky");
  await expect(controls.getByText("Water Movement")).toBeVisible();
  await expect(
    page.getByText("May 13, 2026, 8:30 AM", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText(/expect a fast push west toward Coney Island Pier/).first(),
  ).toBeVisible();
  await expect(page.getByText(/rising/)).toHaveCount(1);
  await expect(panel.getByLabel("Planner time")).toHaveAttribute("max", "1440");
  await expect(page.locator('img[alt^="Tide and current plot"]')).toHaveCount(
    0,
  );

  await gotoApp(page, "/nyc?detail=open&at=2026-05-13T08:30:00");
  await expect(page.getByRole("region", { name: "Planner mode" })).toHaveCount(
    0,
  );
  await expect(
    page.getByText(/expect a fast push west toward Coney Island Pier/).first(),
  ).toBeVisible();
  await expect(page.getByText("2.2 ft", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Tide and current plot for May 13, 2026, 8:30 AM",
    }),
  ).toHaveAttribute(
    "src",
    "/api/nyc/plots/current_tide?at=2026-05-13T08%3A30%3A00",
  );
  await expect(
    page.getByRole("heading", {
      name: "Grimaldo's Chair current guidance",
    }),
  ).toBeVisible();
  await expect(
    page.getByText(/Start east toward Manhattan Beach/),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Coney Island ebbing current map at 55% strength",
    }),
  ).toHaveAttribute("src", "/static/plots/nyc/current_chart_ebbing_55.png");
  await expect(
    page.getByRole("img", {
      name: "Historic New York Harbor chart: 3 Hours after Low Water at New York",
    }),
  ).toHaveAttribute("src", "/static/tidecharts/low+3.png");

  await gotoApp(page, "/nyc?at=2026-05-13T08:30:00");
  await expect(
    page.getByText(/expect a fast push west toward Coney Island Pier/).first(),
  ).toBeVisible();
  await expect(page.getByText("2.2 ft", { exact: true })).toBeVisible();
  await expect(
    page.getByText("May 13, 2026, 8:30 AM", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("region", { name: "Water movement controls" })
      .getByRole("button", { name: "Now" }),
  ).toBeVisible();
  await expect(page.locator('img[alt^="Tide and current plot"]')).toHaveCount(
    0,
  );
  await Promise.all([
    page.waitForResponse((response) => {
      const url = new URL(response.url());
      return (
        url.pathname === "/api/nyc/conditions" && !url.searchParams.has("at")
      );
    }),
    page
      .getByRole("region", { name: "Water movement controls" })
      .getByRole("button", { name: "Now" })
      .click(),
  ]);
  await expect(page.getByText("1.6 ft", { exact: true })).toBeVisible();
  await expect(page.getByText("2.2 ft", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Now" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Planner mode" })).toHaveCount(
    0,
  );
});

async function gotoApp(page: Page, url: string) {
  await page.goto(url, { waitUntil: "domcontentloaded" });
}

import { expect, test } from "@playwright/test";

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
          webcam: true,
          transit: true,
          windy: true,
        },
        citations: {
          temperature: '<a href="https://example.com/temp">Temp source</a>',
          tides: '<a href="https://example.com/tides">Tide source</a>',
          currents: '<a href="https://example.com/currents">Current source</a>',
        },
      },
      integrations: {
        youtube_live: {
          channel_id: "UChh9yX1PSFFreQFmnnIPGuQ",
          embed_url:
            "https://www.youtube.com/embed/live_stream?channel=UChh9yX1PSFFreQFmnnIPGuQ&enablejsapi=1&controls=0&playsinline=1&iv_load_policy=3&rel=0",
          watch_url:
            "https://www.youtube.com/channel/UChh9yX1PSFFreQFmnnIPGuQ/live",
        },
        transit_routes: [
          {
            label: "B",
            goodservice_route_id: "B",
            icon_url: "/static/B-train.svg",
          },
          {
            label: "Q",
            goodservice_route_id: "Q",
            icon_url: "/static/Q-train.svg",
          },
        ],
        webcam_alternative: {
          label: "Earth Cam Coney Island",
          url: "https://www.earthcam.com/usa/newyork/coneyisland/?cam=coneyisland",
          description: "Great view, including the amusement park.",
        },
        webcam_source: {
          label: "Webcam",
          url: "https://www.youtube.com/channel/UChh9yX1PSFFreQFmnnIPGuQ/live",
          description: "thanks to David K and Karol L",
        },
        transit_source: {
          label: "goodservice.io",
          url: "https://goodservice.io",
          description: "MTA train status provided by:",
        },
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

const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
  "base64",
);

test.beforeEach(async ({ page }) => {
  await page.route("**/api/app/bootstrap", async (route) => {
    await route.fulfill({ json: bootstrapPayload });
  });
  await page.route("**/api/nyc/conditions", async (route) => {
    await route.fulfill({ json: conditionsPayload });
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

test("renders the default app route without redirecting to /app/nyc", async ({
  page,
}) => {
  await page.goto("/app/");

  await expect(page).toHaveURL(/\/app\/$/);
  await expect(
    page.getByRole("heading", { name: "shall we swim today?" }),
  ).toBeVisible();
  await expect(page.getByText("61.4°F")).toBeVisible();
});

test("renders the NYC location vertical slice", async ({ page }) => {
  await page.goto("/app/nyc");

  await expect(page.getByRole("heading", { name: "Forecast" })).toBeVisible();
  await expect(page.getByTitle("Windy forecast")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Live Webcam" }),
  ).toBeVisible();
  await expect(page.getByTitle("Live webcam")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Temperature Trends" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "12 month temperature plot, all years" }),
  ).toBeVisible();
  await page.getByRole("button", { exact: true, name: "2 mo" }).click();
  await expect(
    page.getByRole("img", { name: "2 month temperature plot, all years" }),
  ).toBeVisible();
  await page.getByRole("button", { exact: true, name: "Live" }).click();
  await expect(
    page.getByRole("img", { name: "Live temperature plot" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Transit Status" }),
  ).toBeVisible();
  await expect(page.getByText("Coney Island-Stillwell Av")).toHaveCount(2);
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(page.getByText("TIDE", { exact: true })).toBeVisible();
  await expect(page.getByText(/low 0.2 ft/)).toBeVisible();
  await expect(page.getByText("now 1.6 ft")).toBeVisible();
  await expect(page.getByText(/high 4.8 ft/)).toBeVisible();
  await expect(page.getByText("slack 5:45 AM", { exact: true })).toBeVisible();
  await expect(page.getByText("now 1.3 kt")).toBeVisible();
  await expect(page.getByText(/peak 1.8 kt/)).toBeVisible();
  await expect(
    page.getByRole("link", { name: "goodservice.io" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Earth Cam Coney Island" }),
  ).toBeVisible();
  await expect(page.getByText("Alternate:")).toBeVisible();
  await expect(page.getByText("Alternative option:")).toHaveCount(0);

  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(hasHorizontalOverflow).toBe(false);
});

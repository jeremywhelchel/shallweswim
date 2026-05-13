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

test.beforeEach(async ({ page }) => {
  await page.route("**/api/app/bootstrap", async (route) => {
    await route.fulfill({ json: bootstrapPayload });
  });
});

test("renders the default app route without redirecting to /app/nyc", async ({
  page,
}) => {
  await page.goto("/app/");

  await expect(page).toHaveURL(/\/app\/$/);
  await expect(page.getByRole("heading", { name: "New York" })).toBeVisible();
  await expect(page.getByText("Default location")).toBeVisible();
});

test("renders placeholder nested app routes", async ({ page }) => {
  await page.goto("/app/nyc/currents?shift=60");

  await expect(page.getByRole("heading", { name: "Currents" })).toBeVisible();
  await expect(page.getByText("Shift: 60 minutes")).toBeVisible();
});

import {
  clearLastLocationCode,
  getLastLocationCode,
  readAppPreferences,
  recordAppVisit,
  resetPreferenceRuntimeForTests,
  setLastLocationCode,
  writeAppPreferences,
} from "./preferences";

const STORAGE_KEY = "shallweswim.appPreferences";

beforeEach(() => {
  window.localStorage.clear();
  resetPreferenceRuntimeForTests();
});

test("stores and reads the last selected location", () => {
  setLastLocationCode("sfo");

  expect(getLastLocationCode()).toBe("sfo");
  expect(readAppPreferences()).toEqual({
    version: 1,
    lastLocationCode: "sfo",
  });
});

test("clears the last selected location without dropping other preferences", () => {
  writeAppPreferences({
    version: 1,
    lastLocationCode: "missing",
    installPrompt: { organicVisitCount: 2 },
  });

  clearLastLocationCode();

  expect(readAppPreferences()).toEqual({
    version: 1,
    installPrompt: { organicVisitCount: 2 },
  });
});

test("keeps install prompt state when updating the last location", () => {
  writeAppPreferences({
    version: 1,
    installPrompt: { organicVisitCount: 2, dismissedAt: "2026-05-22T00:00Z" },
  });

  setLastLocationCode("nyc");

  expect(readAppPreferences()).toEqual({
    version: 1,
    lastLocationCode: "nyc",
    installPrompt: {
      organicVisitCount: 2,
      dismissedAt: "2026-05-22T00:00Z",
    },
  });
});

test("resets invalid or stale stored preferences", () => {
  window.localStorage.setItem(STORAGE_KEY, "{bad json");
  expect(readAppPreferences()).toEqual({ version: 1 });

  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ version: 0, lastLocationCode: "nyc" }),
  );
  expect(readAppPreferences()).toEqual({ version: 1 });
});

test("records one app visit per page session", () => {
  expect(recordAppVisit().installPrompt?.organicVisitCount).toBe(1);
  expect(recordAppVisit().installPrompt?.organicVisitCount).toBe(1);

  resetPreferenceRuntimeForTests();
  expect(recordAppVisit().installPrompt?.organicVisitCount).toBe(2);
});

test("treats browser storage failures as non-fatal", () => {
  const getItem = vi
    .spyOn(Storage.prototype, "getItem")
    .mockImplementation(() => {
      throw new Error("storage unavailable");
    });
  const setItem = vi
    .spyOn(Storage.prototype, "setItem")
    .mockImplementation(() => {
      throw new Error("storage unavailable");
    });

  expect(readAppPreferences()).toEqual({ version: 1 });
  expect(() => setLastLocationCode("nyc")).not.toThrow();
  expect(recordAppVisit()).toEqual({
    version: 1,
    installPrompt: { organicVisitCount: 1 },
  });

  getItem.mockRestore();
  setItem.mockRestore();
});

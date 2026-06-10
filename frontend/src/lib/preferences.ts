const PREFERENCES_KEY = "shallweswim.appPreferences";
const PREFERENCES_VERSION = 1;

export type TemperatureUnit = "F" | "C";

type InstallPromptPreferences = {
  dismissedAt?: string;
  lastSeenAt?: string;
  organicVisitCount: number;
};

export type AppPreferences = {
  installPrompt?: InstallPromptPreferences;
  lastLocationCode?: string;
  temperatureUnit?: TemperatureUnit;
  version: typeof PREFERENCES_VERSION;
};

let appVisitRecordedForThisPage = false;

export function readAppPreferences(): AppPreferences {
  try {
    const raw = window.localStorage.getItem(PREFERENCES_KEY);
    if (!raw) {
      return defaultPreferences();
    }

    return normalizePreferences(JSON.parse(raw));
  } catch {
    return defaultPreferences();
  }
}

export function writeAppPreferences(preferences: AppPreferences) {
  try {
    window.localStorage.setItem(
      PREFERENCES_KEY,
      JSON.stringify(normalizePreferences(preferences)),
    );
  } catch {
    // Preference persistence is best-effort; storage can be unavailable.
  }
}

export function updateAppPreferences(
  update: (preferences: AppPreferences) => AppPreferences,
) {
  writeAppPreferences(update(readAppPreferences()));
}

export function getLastLocationCode() {
  return readAppPreferences().lastLocationCode ?? null;
}

export function setLastLocationCode(locationCode: string) {
  updateAppPreferences((preferences) => ({
    ...preferences,
    lastLocationCode: locationCode,
  }));
}

export function clearLastLocationCode() {
  updateAppPreferences((preferences) => {
    const { lastLocationCode: _lastLocationCode, ...nextPreferences } =
      preferences;
    return nextPreferences;
  });
}

export function getTemperatureUnit() {
  return readAppPreferences().temperatureUnit ?? null;
}

export function setTemperatureUnit(unit: TemperatureUnit) {
  updateAppPreferences((preferences) => ({
    ...preferences,
    temperatureUnit: unit,
  }));
}

export function recordAppVisit() {
  if (appVisitRecordedForThisPage) {
    return readAppPreferences();
  }

  appVisitRecordedForThisPage = true;
  const preferences = readAppPreferences();
  const installPrompt = preferences.installPrompt ?? { organicVisitCount: 0 };
  const nextPreferences = {
    ...preferences,
    installPrompt: {
      ...installPrompt,
      organicVisitCount: installPrompt.organicVisitCount + 1,
    },
  };

  writeAppPreferences(nextPreferences);
  return nextPreferences;
}

export function resetPreferenceRuntimeForTests() {
  appVisitRecordedForThisPage = false;
}

function normalizePreferences(value: unknown): AppPreferences {
  if (!isObject(value) || value.version !== PREFERENCES_VERSION) {
    return defaultPreferences();
  }

  const preferences: AppPreferences = {
    version: PREFERENCES_VERSION,
  };

  if (typeof value.lastLocationCode === "string" && value.lastLocationCode) {
    preferences.lastLocationCode = value.lastLocationCode;
  }

  if (value.temperatureUnit === "F" || value.temperatureUnit === "C") {
    preferences.temperatureUnit = value.temperatureUnit;
  }

  if (isObject(value.installPrompt)) {
    const count = value.installPrompt.organicVisitCount;
    preferences.installPrompt = {
      organicVisitCount:
        typeof count === "number" && Number.isFinite(count) && count >= 0
          ? Math.floor(count)
          : 0,
    };

    if (typeof value.installPrompt.dismissedAt === "string") {
      preferences.installPrompt.dismissedAt = value.installPrompt.dismissedAt;
    }
    if (typeof value.installPrompt.lastSeenAt === "string") {
      preferences.installPrompt.lastSeenAt = value.installPrompt.lastSeenAt;
    }
  }

  return preferences;
}

function defaultPreferences(): AppPreferences {
  return { version: PREFERENCES_VERSION };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

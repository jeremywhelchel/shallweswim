import type { components } from "../api/generated";
import type { TemperatureUnit } from "./preferences";

type TemperatureInfo = components["schemas"]["TemperatureInfo"];

export function resolveTemperatureUnit({
  locationDefault,
  preference,
}: {
  locationDefault?: TemperatureUnit;
  preference?: TemperatureUnit | null;
}): TemperatureUnit {
  return preference ?? locationDefault ?? "F";
}

export function formatWaterTemperature(
  temperature: TemperatureInfo,
  unit: TemperatureUnit,
) {
  const value =
    unit === "C" ? temperature.water_temp_c : temperature.water_temp_f;

  return `${formatTemperatureValue(value)}°${unit}`;
}

function formatTemperatureValue(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "N/A";
}

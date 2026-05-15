export function formatTimestamp(
  isoString: string | undefined,
  options: Intl.DateTimeFormatOptions,
) {
  if (!isoString) {
    return "";
  }

  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleString("en-US", options);
}

export function formatTideDate(isoString: string | undefined) {
  return formatTimestamp(isoString, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

export function formatTime(isoString: string | undefined) {
  return formatTimestamp(isoString, {
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  });
}

export function formatStationTimestamp(isoString: string | undefined) {
  return formatTimestamp(isoString, {
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  });
}

export function formatMagnitude(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "N/A";
}

export function formatTideHeight(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "N/A";
}

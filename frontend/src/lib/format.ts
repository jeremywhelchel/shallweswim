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

/**
 * Render an observation's age, as the server reports it, for display.
 *
 * The server decides freshness and the age; the page never does clock
 * arithmetic of its own, so this only words a number of seconds.
 */
export function formatAge(ageSeconds: number) {
  const minutes = Math.floor(ageSeconds / 60);
  if (minutes < 60) {
    return minutes === 1 ? "1 minute ago" : `${minutes} minutes ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 48) {
    return hours === 1 ? "1 hour ago" : `${hours} hours ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days} days ago`;
}

import {
  formatAge,
  formatMagnitude,
  formatStationTimestamp,
  formatTideDate,
  formatTime,
} from "./format";

test("formats tide and station timestamps for display", () => {
  expect(formatTideDate("2026-05-13T12:00:00-04:00")).toMatch(
    /Wednesday, May 13/,
  );
  expect(formatTime("2026-05-13T12:00:00-04:00")).toMatch(/\d{1,2}:00 [AP]M/);
  expect(formatStationTimestamp("2026-05-13T07:30:00-04:00")).toMatch(/May 13/);
});

test("formats current magnitude consistently", () => {
  expect(formatMagnitude(1.25)).toBe("1.3");
  expect(formatMagnitude(undefined)).toBe("N/A");
});

test("words an observation's age from the server's seconds", () => {
  expect(formatAge(59)).toBe("0 minutes ago");
  expect(formatAge(60)).toBe("1 minute ago");
  expect(formatAge(45 * 60)).toBe("45 minutes ago");
  expect(formatAge(3 * 3600 + 59 * 60)).toBe("3 hours ago");
  expect(formatAge(47 * 3600)).toBe("47 hours ago");
  expect(formatAge(48 * 3600)).toBe("2 days ago");
  expect(formatAge(12 * 86400 + 3600)).toBe("12 days ago");
});

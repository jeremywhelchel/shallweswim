import {
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

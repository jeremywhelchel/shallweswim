import { describe, expect, it } from "vitest";
import { parseTransitStatus } from "./transit";

describe("parseTransitStatus", () => {
  it("uses the configured GoodService direction", () => {
    const status = parseTransitStatus(
      {
        direction_statuses: {
          north: "Delays",
          south: "Good Service",
        },
        destinations: {
          north: ["Manhattan"],
          south: ["Coney Island-Stillwell Av"],
        },
        delay_summaries: {
          north: ["Signal problems"],
        },
        service_change_summaries: {
          both: "Trains run local. ",
          north: "Some northbound stops skipped.",
        },
        service_irregularity_summaries: {
          north: ["Expect crowding"],
        },
      },
      "north",
    );

    expect(status).toEqual({
      status: "Delays",
      destination: "Manhattan",
      delay: "Signal problems",
      serviceChange: "Trains run local. Some northbound stops skipped.",
      serviceIrregularity: "Expect crowding",
    });
  });

  it("keeps route-level not-scheduled status direction independent", () => {
    expect(
      parseTransitStatus(
        {
          status: "Not Scheduled",
          direction_statuses: {
            south: "Good Service",
          },
        },
        "south",
      ),
    ).toEqual({
      status: "Not Scheduled",
      destination: "no scheduled service",
    });
  });
});

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { components } from "./generated";

export type TransitRouteConfig = components["schemas"]["TransitRouteConfig"];
type GoodServiceDirection = TransitRouteConfig["goodservice_direction"];

type GoodServiceResponse = {
  status?: string;
  direction_statuses?: Partial<Record<GoodServiceDirection, string>>;
  destinations?: Partial<Record<GoodServiceDirection, string[]>>;
  delay_summaries?: Partial<Record<GoodServiceDirection, string | string[]>>;
  service_change_summaries?: {
    both?: string;
  } & Partial<Record<GoodServiceDirection, string>>;
  service_irregularity_summaries?: Partial<
    Record<GoodServiceDirection, string | string[]>
  >;
};

export type TransitStatus = {
  status: string;
  destination: string;
  delay?: string;
  serviceChange?: string;
  serviceIrregularity?: string;
};

const REFRESH_INTERVAL_MS = 60_000;

function textValue(value: string | string[] | undefined): string | undefined {
  if (Array.isArray(value)) {
    const joined = value.filter(Boolean).join(" ");
    return joined || undefined;
  }

  return value || undefined;
}

export function parseTransitStatus(
  data: GoodServiceResponse,
  direction: GoodServiceDirection,
): TransitStatus {
  if (data.status === "Not Scheduled") {
    return {
      status: "Not Scheduled",
      destination: "no scheduled service",
    };
  }

  return {
    status: data.direction_statuses?.[direction] || "No Data",
    destination: data.destinations?.[direction]?.[0] || "unknown",
    delay: textValue(data.delay_summaries?.[direction]),
    serviceChange:
      [
        data.service_change_summaries?.both,
        data.service_change_summaries?.[direction],
      ]
        .filter(Boolean)
        .join("") || undefined,
    serviceIrregularity: textValue(
      data.service_irregularity_summaries?.[direction],
    ),
  };
}

export function useTransitRoute(routeConfig: TransitRouteConfig) {
  return useQuery({
    queryKey: [
      "transit-route",
      routeConfig.goodservice_route_id,
      routeConfig.goodservice_direction,
    ],
    queryFn: async () => {
      const response = await fetch(
        `https://goodservice.io/api/routes/${routeConfig.goodservice_route_id}`,
      );

      if (!response.ok) {
        throw new Error(
          `Transit request failed with status ${response.status}`,
        );
      }

      return parseTransitStatus(
        (await response.json()) as GoodServiceResponse,
        routeConfig.goodservice_direction,
      );
    },
    placeholderData: keepPreviousData,
    refetchInterval: REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
    retry: 1,
  });
}

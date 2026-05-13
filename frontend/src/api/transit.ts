import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { components } from "./generated";

export type TransitRouteConfig = components["schemas"]["TransitRouteConfig"];

type GoodServiceResponse = {
  status?: string;
  direction_statuses?: {
    south?: string;
  };
  destinations?: {
    south?: string[];
  };
  delay_summaries?: {
    south?: string | string[];
  };
  service_change_summaries?: {
    both?: string;
    south?: string;
  };
  service_irregularity_summaries?: {
    south?: string | string[];
  };
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

function parseTransitStatus(data: GoodServiceResponse): TransitStatus {
  if (data.status === "Not Scheduled") {
    return {
      status: "Not Scheduled",
      destination: "unavailable",
    };
  }

  return {
    status: data.direction_statuses?.south || "No Data",
    destination: data.destinations?.south?.[0] || "unknown",
    delay: textValue(data.delay_summaries?.south),
    serviceChange:
      [
        data.service_change_summaries?.both,
        data.service_change_summaries?.south,
      ]
        .filter(Boolean)
        .join("") || undefined,
    serviceIrregularity: textValue(data.service_irregularity_summaries?.south),
  };
}

export function useTransitRoute(routeConfig: TransitRouteConfig) {
  return useQuery({
    queryKey: ["transit-route", routeConfig.goodservice_route_id],
    queryFn: async () => {
      const response = await fetch(
        `https://goodservice.io/api/routes/${routeConfig.goodservice_route_id}`,
      );

      if (!response.ok) {
        throw new Error(
          `Transit request failed with status ${response.status}`,
        );
      }

      return parseTransitStatus((await response.json()) as GoodServiceResponse);
    },
    placeholderData: keepPreviousData,
    refetchInterval: REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
    retry: 1,
  });
}

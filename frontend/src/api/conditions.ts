import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

const REFRESH_INTERVAL_MS = 60_000;

export function useLocationConditions(locationCode: string) {
  return useQuery({
    queryKey: ["location-conditions", locationCode],
    queryFn: async () => {
      const { data, error } = await apiClient.GET(
        "/api/{location}/conditions",
        {
          params: { path: { location: locationCode } },
        },
      );

      if (error) {
        throw new Error("Conditions request failed");
      }

      return data;
    },
    enabled: Boolean(locationCode),
    placeholderData: keepPreviousData,
    refetchInterval: REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
  });
}

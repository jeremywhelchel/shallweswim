import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

const REFRESH_INTERVAL_MS = 60_000;

export function useLocationCurrents(
  locationCode: string,
  at?: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ["location-currents", locationCode, at ?? null],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/api/{location}/currents", {
        params: {
          path: { location: locationCode },
          query: at ? { at } : undefined,
        },
      });

      if (error) {
        throw new Error("Currents request failed");
      }

      return data;
    },
    enabled: Boolean(locationCode) && enabled,
    placeholderData: keepPreviousData,
    refetchInterval: at ? false : REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
  });
}

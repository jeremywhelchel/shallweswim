import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";

export function useAppBootstrap() {
  return useQuery({
    queryKey: ["app-bootstrap"],
    queryFn: async () => {
      const { data, error } = await apiClient.GET("/api/app/bootstrap");

      if (error) {
        throw new Error("Bootstrap request failed");
      }

      return data;
    },
  });
}

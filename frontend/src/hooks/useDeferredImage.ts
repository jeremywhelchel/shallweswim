import { useEffect, useState } from "react";

const RETRY_DELAYS_MS = [1000, 3000, 7000];

type DeferredImageState = {
  status: "idle" | "loading" | "loaded" | "unavailable";
  src?: string;
};

export function useDeferredImage(src: string, enabled: boolean) {
  const [state, setState] = useState<DeferredImageState>({ status: "idle" });

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let cancelled = false;
    let retryTimer: number | undefined;
    let probe: HTMLImageElement | undefined;

    function load(attempt: number) {
      setState({ status: "loading" });
      probe = new Image();
      probe.onload = () => {
        if (!cancelled) {
          setState({ status: "loaded", src });
        }
      };
      probe.onerror = () => {
        if (cancelled) {
          return;
        }

        const retryDelay = RETRY_DELAYS_MS[attempt];
        if (retryDelay === undefined) {
          setState({ status: "unavailable" });
          return;
        }

        retryTimer = window.setTimeout(() => load(attempt + 1), retryDelay);
      };
      probe.src = src;
    }

    load(0);

    return () => {
      cancelled = true;
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
      }
      if (probe) {
        probe.onload = null;
        probe.onerror = null;
      }
    };
  }, [enabled, src]);

  return state;
}

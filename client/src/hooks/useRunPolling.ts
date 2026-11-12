import React from "react";
import { fetchRun } from "../api";
import type { AnalysisRun } from "../types";
import { createRequestSequence } from "./requestSequence";

type Handlers = {
  onUpdate(run: AnalysisRun): void;
  onSettled(): Promise<unknown> | void;
  onError(message: string): void;
};

/**
 * Polls a running batch. User actions (pause, resume…) must go through the returned
 * `invalidate` so that a poll already in flight cannot overwrite their newer state.
 */
export function useRunPolling(run: AnalysisRun | null, handlers: Handlers) {
  const sequence = React.useRef(createRequestSequence()).current;
  const latest = React.useRef(handlers);
  latest.current = handlers;
  const runId = run?.runId;
  const status = run?.status;

  React.useEffect(() => {
    if (!runId || status !== "running") return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | null = null;
    async function poll() {
      const token = sequence.take();
      controller = new AbortController();
      try {
        const next = await fetchRun(runId!, controller.signal);
        if (stopped || !sequence.isCurrent(token)) {
          if (!stopped) timer = setTimeout(() => void poll(), 2000);
          return;
        }
        latest.current.onUpdate(next);
        if (next.status !== "running") {
          await latest.current.onSettled();
          return;
        }
      } catch (e) {
        if (!stopped && sequence.isCurrent(token))
          latest.current.onError(e instanceof Error ? e.message : "Lecture de progression impossible.");
      }
      if (!stopped) timer = setTimeout(() => void poll(), 2000);
    }
    timer = setTimeout(() => void poll(), 1000);
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, [runId, status, sequence]);

  return React.useCallback(() => sequence.bump(), [sequence]);
}

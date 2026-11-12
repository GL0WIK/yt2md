/**
 * Orders asynchronous reads against user actions: a token taken before a request is only
 * current while no newer action has bumped the sequence, so late responses can be dropped.
 */
export type RequestSequence = {
  /** Token for a request about to start. */
  take(): number;
  /** Invalidates every token taken so far. */
  bump(): void;
  isCurrent(token: number): boolean;
};

export function createRequestSequence(): RequestSequence {
  let generation = 0;
  return {
    take: () => generation,
    bump: () => {
      generation++;
    },
    isCurrent: (token) => token === generation,
  };
}

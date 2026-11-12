import { describe, expect, it } from "vitest";
import { createRequestSequence } from "./requestSequence";

describe("request sequence", () => {
  it("drops a poll that was in flight when a user action happened", () => {
    const sequence = createRequestSequence();
    const poll = sequence.take();
    expect(sequence.isCurrent(poll)).toBe(true);
    sequence.bump(); // e.g. Pause clicked
    expect(sequence.isCurrent(poll)).toBe(false);
    const nextPoll = sequence.take();
    expect(sequence.isCurrent(nextPoll)).toBe(true);
  });
  it("keeps independent sequences apart", () => {
    const a = createRequestSequence();
    const b = createRequestSequence();
    const token = b.take();
    a.bump();
    expect(b.isCurrent(token)).toBe(true);
  });
});

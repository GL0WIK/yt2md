import { describe, expect, it } from "vitest";
import {
  deleteConfirmation,
  deleteFailureMessage,
  exportFilenames,
  pruneDeletedResults,
  selectionExports,
} from "./exports";
import { fixtureVideo } from "../../server/test/fixtures";
import type { AnalyzeVideoResult } from "./types";

const usage = {
  model: "",
  promptTokenCount: null,
  candidatesTokenCount: null,
  totalTokenCount: null,
  elapsedMs: 0,
  cacheHits: 0,
  visionGroups: 0,
};
function success(id: string): AnalyzeVideoResult {
  return {
    status: "success",
    videoId: id,
    title: id,
    markdownPath: "",
    downloadUrl: `/api/outputs/${id}.md`,
    jsonDownloadUrl: `/api/outputs/${id}.json`,
    provenanceUrl: `/api/outputs/${id}.local.json`,
    warnings: [],
    usage,
  };
}
const failure: AnalyzeVideoResult = { status: "error", videoId: "b", title: "b", error: "boom" };
const videos = ["a", "b", "c"].map((id) => ({ ...fixtureVideo, id }));

describe("exports selection", () => {
  it("only targets selected videos that have exports", () => {
    const results = { a: success("a"), b: failure };
    expect(exportFilenames(failure)).toEqual([]);
    expect(selectionExports(videos, results)).toEqual({ videoIds: ["a"], filenames: ["a.md", "a.json"] });
  });
  it("clears deleted results but keeps errors and failed deletions", () => {
    const results = { a: success("a"), b: failure, c: success("c") };
    const next = pruneDeletedResults(
      results,
      ["a", "c"],
      [{ filename: "c.json", videoId: "c", error: "EBUSY" }],
    );
    expect(Object.keys(next).sort()).toEqual(["b", "c"]);
    expect(pruneDeletedResults(results, ["a"])).not.toHaveProperty("a");
  });
  it("states the count and lists failures", () => {
    expect(deleteConfirmation(1)).toContain("1 vidéo ");
    expect(deleteConfirmation(3)).toContain("3 vidéos");
    expect(deleteFailureMessage(undefined)).toBe("");
    expect(deleteFailureMessage([{ filename: "c.json", videoId: "c", error: "EBUSY" }])).toContain(
      "c.json (EBUSY)",
    );
  });
});

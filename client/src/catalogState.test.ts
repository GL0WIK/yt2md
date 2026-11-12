import { describe, expect, it } from "vitest";
import { addSelection, reattachExports } from "./catalogState";
import { fixtureVideo } from "../../server/test/fixtures";

describe("local catalog", () => {
  it("selects a batch of 400 and respects the 1000 video cap without mutating prior selection", () => {
    const before = new Set(["kept"]);
    const result = addSelection(
      before,
      Array.from({ length: 1100 }, (_, i) => String(i)),
    );
    expect(result.size).toBe(1000);
    expect(result.has("kept")).toBe(true);
    expect(before.size).toBe(1);
    expect(
      addSelection(
        new Set(),
        Array.from({ length: 400 }, (_, i) => String(i)),
      ).size,
    ).toBe(400);
  });
  it("relinks canonical exports without fabricating model usage", () => {
    const restored = reattachExports(
      [fixtureVideo],
      [{ videoId: fixtureVideo.id, markdown: "video.md", json: "video.json", local: "video.local.json" }],
      {},
    )[fixtureVideo.id];
    expect(restored?.status === "success" && restored.jsonDownloadUrl).toBe("/api/outputs/video.json");
    expect(restored?.status === "success" && restored.usage.totalTokenCount).toBeNull();
  });
  it("lets an on-disk export replace a stale error but never a success", () => {
    const entry = { videoId: fixtureVideo.id, markdown: "video.md", json: "video.json", local: null };
    const failed = {
      status: "error" as const,
      videoId: fixtureVideo.id,
      title: fixtureVideo.title,
      error: "boom",
    };
    const relinked = reattachExports([fixtureVideo], [entry], { [fixtureVideo.id]: failed })[fixtureVideo.id];
    expect(relinked?.status).toBe("success");
    expect(relinked?.status === "success" && relinked.restored).toBe(true);
    const live = reattachExports(
      [fixtureVideo],
      [{ videoId: fixtureVideo.id, markdown: "video.md", json: null, local: null }],
      {},
    )[fixtureVideo.id]!;
    const kept = reattachExports([fixtureVideo], [entry], { [fixtureVideo.id]: live })[fixtureVideo.id];
    expect(kept).toBe(live);
    // An error without any file on disk stays visible.
    expect(reattachExports([fixtureVideo], [], { [fixtureVideo.id]: failed })[fixtureVideo.id]).toBe(failed);
  });
});

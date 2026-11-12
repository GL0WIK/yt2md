import { describe, expect, it } from "vitest";
import { buildCollections, buildResults, videoState, type Library } from "./library";
import { detectLink } from "./api";
import { parseRoute, routeHref } from "./router";
import { fixtureVideo } from "../../server/test/fixtures";
import type { AnalysisRun } from "./types";

const video = (id: string, channel: string | null, channelId: string | null = null) => ({
  ...fixtureVideo,
  id,
  url: `https://www.youtube.com/watch?v=${id}`,
  channel,
  channelId,
});
const exported = (videoId: string) => ({
  videoId,
  markdown: `video--${videoId}.md`,
  json: null,
  local: null,
});
const stored = (id: string, name: string, videoIds: string[]) => ({
  id,
  kind: "channel" as const,
  name,
  url: "https://www.youtube.com/@x/videos",
  videoIds,
  nextOffset: 50,
  complete: false,
  updatedAt: "2026-10-01T00:00:00.000Z",
});

describe("library by channel", () => {
  it("groups processed videos under their channel: listing, then id, then (longer) name", () => {
    const library: Library = {
      videos: [
        video("aaaaaaaaaaa", "Autre nom", null),
        video("bbbbbbbbbbb", "Les Freres Poulain, French DIY and Tech Explorers"),
        video("ccccccccccc", "Scilabus", "UCsci"),
        video("ddddddddddd", "Inconnue"),
      ],
      exports: ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"].map(exported),
      processedAt: { ddddddddddd: "2026-10-14T00:00:00.000Z" },
      collections: [
        stored("UCpoulain", "Les Freres Poulain", ["aaaaaaaaaaa", "eeeeeeeeeee"]),
        stored("UCsci", "Scilabus", []),
      ],
    };
    const results = buildResults(library.videos, library.exports, null);
    const views = buildCollections(library, results);
    const byId = Object.fromEntries(views.map((view) => [view.id, view]));
    expect(byId.UCpoulain.videoIds).toEqual(["aaaaaaaaaaa", "eeeeeeeeeee", "bbbbbbbbbbb"]);
    expect(byId.UCpoulain.processed).toBe(2);
    expect(byId.UCsci.videoIds).toEqual(["ccccccccccc"]);
    // A channel known only from its processed videos still gets a page, most recent first.
    expect(views[0]).toMatchObject({ id: "name:Inconnue", stored: false, processed: 1 });
  });

  it("follows a video through the queue, the processing and its result", () => {
    const run = {
      status: "running",
      videos: [video("aaaaaaaaaaa", "x"), video("bbbbbbbbbbb", "x")],
      results: [{ status: "error", videoId: "ccccccccccc", title: "t", error: "boom" }],
      progress: { videoId: "aaaaaaaaaaa" },
    } as unknown as AnalysisRun;
    const results = buildResults([video("ccccccccccc", "x")], [], run);
    expect(videoState("aaaaaaaaaaa", results, run)).toBe("running");
    expect(videoState("bbbbbbbbbbb", results, run)).toBe("queued");
    expect(videoState("ccccccccccc", results, run)).toBe("error");
    expect(videoState("ddddddddddd", results, run)).toBe("todo");
  });

  it("recognises pasted links and routes", () => {
    expect(detectLink("https://youtu.be/28xvyGOgM00")).toEqual({
      kind: "video",
      url: "https://www.youtube.com/watch?v=28xvyGOgM00",
    });
    expect(detectLink("www.youtube.com/@scilabus")?.kind).toBe("channel");
    expect(detectLink("https://www.youtube.com/playlist?list=PL0123456789abcdef")?.kind).toBe("playlist");
    // A video shared from a playlist is still that video.
    expect(detectLink("https://www.youtube.com/watch?v=28xvyGOgM00&list=PL0123456789abcdef")?.kind).toBe(
      "video",
    );
    expect(detectLink("bonjour")).toBeNull();
    expect(parseRoute(routeHref({ page: "collection", id: "name:Les Freres" }))).toEqual({
      page: "collection",
      id: "name:Les Freres",
    });
    expect(parseRoute("#/traitement")).toEqual({ page: "run" });
    expect(parseRoute("#/nimporte")).toEqual({ page: "library" });
  });
});

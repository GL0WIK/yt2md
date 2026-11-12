import { describe, expect, it, vi } from "vitest";
import {
  createYtDlpCatalogService,
  isExcludedVideo,
  isWithinInclusiveDateRange,
  toCatalogVideo,
  toFlatCatalogCandidate,
} from "../src/catalog/ytDlp.js";

describe("catalog filters", () => {
  it("excludes Shorts by URL path", () => {
    expect(isExcludedVideo({ webpage_url: "https://www.youtube.com/shorts/abc123def45" })).toBe(true);
  });

  it("excludes live and replay statuses", () => {
    expect(isExcludedVideo({ is_live: true })).toBe(true);
    expect(isExcludedVideo({ was_live: true })).toBe(true);
    expect(isExcludedVideo({ live_status: "is_upcoming" })).toBe(true);
    expect(isExcludedVideo({ live_status: "post_live" })).toBe(true);
  });

  it("keeps ordinary direct URLs with missing stats", () => {
    const video = toCatalogVideo(
      {
        id: "abc123def45",
        title: "Test",
        webpage_url: "https://www.youtube.com/watch?v=abc123def45",
        upload_date: "20260501",
        duration: 42,
      },
      "direct",
    );

    expect(isExcludedVideo({ webpage_url: video.url, live_status: "not_live" })).toBe(false);
    expect(video.publicStats.views).toBeNull();
    expect(video.publishedAt).toBe("2026-05-01");
  });

  it("filters publication dates inclusively in app code", () => {
    expect(isWithinInclusiveDateRange("2025-04-01", "2025-04-01", "2025-05-28")).toBe(true);
    expect(isWithinInclusiveDateRange("2025-05-28", "2025-04-01", "2025-05-28")).toBe(true);
    expect(isWithinInclusiveDateRange("2025-03-31", "2025-04-01", "2025-05-28")).toBe(false);
    expect(isWithinInclusiveDateRange("2025-05-29", "2025-04-01", "2025-05-28")).toBe(false);
    expect(isWithinInclusiveDateRange(null, "2025-04-01", "2025-05-28")).toBe(false);
  });

  it("builds flat channel candidates from lightweight playlist entries", () => {
    expect(
      toFlatCatalogCandidate({
        id: "abc123def45",
        url: "https://www.youtube.com/watch?v=abc123def45",
        upload_date: "20250528",
      }),
    ).toEqual({
      id: "abc123def45",
      url: "https://www.youtube.com/watch?v=abc123def45",
      publishedAt: "2025-05-28",
    });
  });

  it("uses timestamp dates for flat entries when upload_date is missing", () => {
    expect(
      toFlatCatalogCandidate({
        id: "abc123def45",
        timestamp: 1748390400,
      }),
    ).toEqual({
      id: "abc123def45",
      url: "https://www.youtube.com/watch?v=abc123def45",
      publishedAt: "2025-05-28",
    });
  });

  it("keeps flat entries without a usable publication date so detailed metadata can be fetched", () => {
    expect(toFlatCatalogCandidate({ id: "abc123def45" })).toEqual({
      id: "abc123def45",
      url: "https://www.youtube.com/watch?v=abc123def45",
      publishedAt: null,
    });
  });
});

describe("paged catalog", () => {
  const channelUrl = "https://youtube.com/@test";
  const playlistUrl = "https://youtube.com/playlist?list=PL1234567890";
  const row = (id: string) => ({ id, title: id, duration: 42.3 });
  const response = (value: unknown) => ({ stdout: JSON.stringify(value), stderr: "" });

  it("reads one lightweight page, coalesces requests and never opens its videos", async () => {
    const runner = vi
      .fn()
      .mockResolvedValue(response({ entries: [row("old123def45"), row("new123def45"), row("end123def45")] }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    const [first, same] = await Promise.all([
      service.getChannelVideos({ channelUrl, pageSize: 2 }),
      service.getChannelVideos({ channelUrl: channelUrl + "/videos", pageSize: 2 }),
    ]);
    expect(first).toEqual(same);
    expect(first.videos.map((video) => video.id)).toEqual(["old123def45", "new123def45"]);
    expect(first.videos[0]).toMatchObject({
      durationSeconds: 43,
      publishedAt: null,
      metadataStatus: "preview",
    });
    expect(first.nextOffset).toBe(2);
    expect(runner).toHaveBeenCalledTimes(1);
    expect(runner.mock.calls[0][1]).toEqual(
      expect.arrayContaining(["--flat-playlist", "--lazy-playlist", "--playlist-items", "1:3"]),
    );
  });

  it("requests preferred-language metadata so YouTube does not return auto-translated titles", async () => {
    const runner = vi.fn().mockResolvedValue(response({ entries: [row("old123def45")] }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    await service.getChannelVideos({ channelUrl, pageSize: 1 });
    const args = runner.mock.calls[0][1] as string[];
    expect(args).toEqual(expect.arrayContaining(["--extractor-args", "youtube:lang=fr"]));
  });

  it("uses the configured metadata language for listings", async () => {
    const runner = vi.fn().mockResolvedValue(response({ entries: [row("old123def45")] }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "en-US", runner);
    await service.getChannelVideos({ channelUrl, pageSize: 1 });
    expect(runner.mock.calls[0][1]).toEqual(
      expect.arrayContaining(["--extractor-args", "youtube:lang=en-US"]),
    );
  });

  it("keeps the page's default language on full video metadata so like counts stay exact", async () => {
    const runner = vi.fn().mockResolvedValue(response({ ...row("old123def45"), upload_date: "20250528" }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    const page = await service.getDirectVideos(["https://youtu.be/old123def45"]);
    expect(page.videos).toHaveLength(1);
    const args = runner.mock.calls[0][1] as string[];
    expect(args).toContain("--no-playlist");
    expect(args).not.toContain("--extractor-args");
    expect(args.join(" ")).not.toContain("youtube:lang");
  });

  it("advances by raw positions even for duplicates and unavailable videos", async () => {
    const runner = vi
      .fn()
      .mockResolvedValueOnce(
        response({
          entries: [
            row("old123def45"),
            row("old123def45"),
            { ...row("bad123def45"), title: "[Deleted video]" },
            row("new123def45"),
          ],
        }),
      )
      .mockResolvedValueOnce(response({ entries: [row("new123def45")] }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    const first = await service.getPlaylistVideos({ playlistUrl, pageSize: 3 });
    expect(first.videos).toHaveLength(1);
    expect(first.warnings).toHaveLength(1);
    expect(first.nextOffset).toBe(3);
    const next = await service.getPlaylistVideos({ playlistUrl, pageSize: 3, offset: first.nextOffset! });
    expect(next.videos[0].id).toBe("new123def45");
    expect(next.nextOffset).toBeNull();
    expect(runner.mock.calls[1][1]).toContain("4:7");
  });

  it("returns a usable catalogue and explicit notice at the configured limit", async () => {
    const runner = vi.fn().mockResolvedValue(response({ entries: [row("old123def45"), row("new123def45")] }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 1, "fr", runner);
    const page = await service.getChannelVideos({ channelUrl });
    expect(page.videos).toHaveLength(1);
    expect(page.limitReached).toBe(true);
    expect(page.nextOffset).toBeNull();
    expect(page.warnings[0]).toContain("Limite");
  });

  it("filters exact dates only when requested, regardless of playlist ordering, and caches metadata", async () => {
    const runner = vi.fn(async (_binary: string, args: string[]) => {
      if (args.includes("--flat-playlist"))
        return response({ entries: [row("old123def45"), row("new123def45")] });
      const id = new URL(args.at(-1)!).searchParams.get("v")!;
      return response({ ...row(id), upload_date: id === "old123def45" ? "20240101" : "20260501" });
    });
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    const page = await service.getChannelVideos({
      channelUrl,
      startDate: "2026-01-01",
      endDate: "2026-12-31",
    });
    expect(page.videos.map((video) => video.id)).toEqual(["new123def45"]);
    expect(page.videos[0].metadataStatus).toBe("complete");
    await service.getDirectVideos([
      "https://youtu.be/new123def45",
      "https://youtube.com/watch?v=new123def45",
    ]);
    expect(runner).toHaveBeenCalledTimes(3);
  });

  it("keeps successful direct videos when another is unavailable and retries failures", async () => {
    const runner = vi.fn(async (_binary: string, args: string[]) => {
      const id = new URL(args.at(-1)!).searchParams.get("v")!;
      return response(id === "bad123def45" ? { ...row(id), availability: "private" } : row(id));
    });
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    const page = await service.getDirectVideos([
      "https://youtu.be/bad123def45",
      "https://youtu.be/new123def45",
    ]);
    expect(page.videos.map((video) => video.id)).toEqual(["new123def45"]);
    expect(page.warnings[0]).toContain("bad123def45");
    await service.getDirectVideos(["https://youtu.be/bad123def45", "https://youtu.be/new123def45"]);
    expect(runner).toHaveBeenCalledTimes(3);
  });

  it("bounds subprocess concurrency across independent browser requests", async () => {
    let active = 0,
      peak = 0;
    const runner = vi.fn(async () => {
      peak = Math.max(peak, ++active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active--;
      return response({ entries: [] });
    });
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    await Promise.all(
      Array.from({ length: 8 }, (_, offset) => service.getChannelVideos({ channelUrl, offset })),
    );
    expect(peak).toBe(2);
  });

  it("does not cache failed playlist reads", async () => {
    const runner = vi
      .fn()
      .mockResolvedValueOnce({ stdout: "invalid", stderr: "" })
      .mockResolvedValueOnce(response({ entries: [] }));
    const service = createYtDlpCatalogService("yt-dlp", 1000, 100, "fr", runner);
    await expect(service.getChannelVideos({ channelUrl })).rejects.toThrow("illisibles");
    expect((await service.getChannelVideos({ channelUrl })).videos).toEqual([]);
    expect(runner).toHaveBeenCalledTimes(2);
  });

  it("handles invalid dates and extreme timestamps without crashing", () => {
    expect(
      toFlatCatalogCandidate({
        id: "abc123def45",
        upload_date: "20260230",
        timestamp: Number.MAX_SAFE_INTEGER,
      })?.publishedAt,
    ).toBeNull();
  });
});

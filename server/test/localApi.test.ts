import { describe, expect, it, afterEach, vi } from "vitest";
import request from "supertest";
import fs from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { createApp } from "../src/app.js";
import { getConfig, modelLabel } from "../src/config.js";
import type { LocalWorker, WorkerAnalysis } from "../src/local/worker.js";
import { createAnalysisService } from "../src/api/analysisService.js";
import type { YtDlpCatalogService } from "../src/catalog/ytDlp.js";
import type { AnalyzeVideoResult } from "../src/shared/schemas.js";
import { fixtureAnalysis, fixtureVideo } from "./fixtures.js";
import { temporaryDirectory } from "./helpers.js";
import { YOUTUBE_BLOCKED_MESSAGE } from "../src/shared/errors.js";

const apps: Array<ReturnType<typeof createApp>> = [];
afterEach(async () => {
  for (const app of apps.splice(0)) await app.locals.close();
});
async function setup(block = false, invalid = false) {
  const directory = await temporaryDirectory();
  const config = getConfig({ OUTPUT_DIR: "outputs", CACHE_DIR: "cache", MODELS_DIR: "models" }, directory);
  const seen: string[] = [];
  const cleanup = vi.fn(async () => {});
  let ready = true;
  const worker: LocalWorker = {
    busy: false,
    health: async () => ({ ready, issues: ready ? [] : ["Poids locaux absents."] }),
    async analyze(video, signal, progress): Promise<WorkerAnalysis> {
      seen.push(video.id);
      progress({ phase: "audio", chunk: 1, totalChunks: 1, detail: "Qwen3-ASR" });
      if (block && seen.length === 1) {
        await new Promise<void>((resolve) =>
          signal.addEventListener("abort", () => resolve(), { once: true }),
        );
        signal.throwIfAborted();
      }
      return {
        durationSeconds: 120,
        model: modelLabel(config),
        promptVersion: "local-observed-video-v1",
        chunkSeconds: 7,
        transcript: [
          {
            speaker: "Indéterminé",
            timestamp: "00:01",
            endTimestamp: "00:03",
            text: "bonjour qwen",
            chunkIndex: 1,
          },
        ],
        visualSegments: invalid
          ? [{ ...fixtureAnalysis.visualSegments[0], endTimestamp: "03:00" }]
          : fixtureAnalysis.visualSegments,
        warnings: [],
        localProvenance: {
          transcriptionSource: "qwen3_asr_forced_alignment",
          words: [{ text: "bonjour", startMs: 1250, endMs: 1600 }],
        },
        usage: {
          model: config.visionModel,
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15,
          elapsedMs: 100,
          cacheHits: 0,
          visionGroups: 8,
        },
      };
    },
    cleanup,
    close: async () => {},
  };
  const catalog = {
    getVideoMetadata: vi.fn(async (video: typeof fixtureVideo) => ({
      ...video,
      title: "Titre yt-dlp",
      description: "Description yt-dlp",
      metadataStatus: "complete" as const,
    })),
    getChannelVideos: vi.fn(),
    getPlaylistVideos: vi.fn(),
    getDirectVideos: vi.fn(async () => ({
      videos: [fixtureVideo],
      warnings: [],
      nextOffset: null,
      scannedCount: 1,
      limitReached: true,
    })),
  };
  const app = createApp({ config, catalog, worker });
  apps.push(app);
  return {
    app,
    config,
    seen,
    cleanup,
    catalog,
    worker,
    setReady: (value: boolean) => {
      ready = value;
    },
  };
}

describe("local workflow API", () => {
  it("serves the same journal after the server is recreated", async () => {
    const { app, config, worker, catalog } = await setup();
    const started = await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo] })
      .expect(202);
    await app.locals.runs.settle();
    const before = await request(app).get(`/api/runs/${started.body.runId}/logs?cursor=0`).expect(200);
    await app.locals.close();
    const restarted = createApp({ config, worker, catalog });
    apps.push(restarted);
    const after = await request(restarted).get(`/api/runs/${started.body.runId}/logs?cursor=0`).expect(200);
    expect(after.body).toEqual(before.body);
  });
  it("writes original-format JSON, deterministic MD and separate native provenance", async () => {
    const { app, config, cleanup } = await setup();
    const started = await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo], requestId: randomUUID() })
      .expect(202);
    await app.locals.runs.settle();
    const run = await request(app).get(`/api/runs/${started.body.runId}`).expect(200);
    expect(run.body.status).toBe("completed");
    expect(run.body.results[0].status).toBe("success");
    const json = JSON.parse(
      await fs.readFile(path.join(config.outputDir, "video--abc123def45.json"), "utf8"),
    );
    expect(json.source.title).toBe("Titre yt-dlp");
    expect(json.youtubeDescription).toBe("Description yt-dlp");
    expect(json.transcript[0].text).toBe("bonjour qwen");
    expect(json.transcript[0].speakerId).toBeNull();
    expect(json.schemaVersion).toBe("2.0");
    expect(json).not.toHaveProperty("words");
    const local = JSON.parse(
      await fs.readFile(path.join(config.outputDir, "video--abc123def45.local.json"), "utf8"),
    );
    expect(local.words[0].startMs).toBe(1250);
    expect(local.contentHash).toBe(json.contentHash);
    expect(local.transcriptionSource).toBe("qwen3_asr_forced_alignment");
    const md = await fs.readFile(path.join(config.outputDir, "video--abc123def45.md"), "utf8");
    expect(md).toContain("bonjour qwen");
    expect(md).not.toContain("## Article");
    expect(cleanup).toHaveBeenCalledWith("abc123def45");
    const download = await request(app).get("/api/outputs/video--abc123def45.json").expect(200);
    expect(download.body).toEqual(json);
    const corpus = await request(app).get("/api/corpus/export").expect(200);
    expect(corpus.text).toContain("bonjour qwen");
    await request(app)
      .post("/api/outputs/export-zip")
      .send({ filenames: ["video--abc123def45.md", "video--abc123def45.json"] })
      .expect(200)
      .expect("Content-Type", /zip/);
    expect((await request(app).get("/api/outputs/index")).body.exports).toHaveLength(1);
    // The interface lists every processed video from the exports alone, without any run.
    const library = await request(app).get("/api/library").expect(200);
    expect(library.body.exports).toHaveLength(1);
    expect(library.body.videos).toEqual([
      expect.objectContaining({
        id: "abc123def45",
        title: "Titre yt-dlp",
        url: "https://www.youtube.com/watch?v=abc123def45",
        description: "Description yt-dlp",
        metadataStatus: "complete",
      }),
    ]);
    const logs = await request(app).get(`/api/runs/${started.body.runId}/logs?cursor=0`).expect(200);
    expect(logs.body.entries.map((entry: { event: string }) => entry.event)).toEqual(
      expect.arrayContaining([
        "local.video.start",
        "local.export.complete",
        "local.video.complete",
        "local.run.completed",
      ]),
    );
    expect(logs.body.entries.every((entry: { runId: string }) => entry.runId === started.body.runId)).toBe(
      true,
    );
    const tail = await request(app)
      .get(`/api/runs/${started.body.runId}/logs?cursor=${logs.body.nextCursor}`)
      .expect(200);
    expect(tail.body.entries).toEqual([]);
    const journal = await request(app).get(logs.body.downloadUrl).expect(200);
    expect(journal.headers["content-disposition"]).toContain(".jsonl");
    await request(app).get(`/api/runs/${started.body.runId}/logs?cursor=-1`).expect(400);
    await request(app).get(`/api/runs/${started.body.runId}/logs?limit=501`).expect(400);
    await request(app).get(`/api/runs/${randomUUID()}/logs`).expect(404);
  });
  it("pauses, rejects competing work/deletion, and resumes an interrupted chunk", async () => {
    const { app, seen } = await setup(true);
    const requestId = randomUUID();
    await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo], requestId })
      .expect(202);
    // Metadata and the worker run begin asynchronously after the acceptance response.
    await vi.waitFor(() => expect(seen).toHaveLength(1));
    await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo] })
      .expect(409);
    await request(app)
      .post("/api/outputs/delete")
      .send({ videoIds: [fixtureVideo.id] })
      .expect(409);
    const duplicate = await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo], requestId })
      .expect(202);
    expect(duplicate.body.runId).toBe(requestId);
    await request(app).post(`/api/runs/${requestId}/pause`).send({}).expect(200);
    expect((await request(app).get(`/api/runs/${requestId}`)).body.status).toBe("interrupted");
    await request(app).post(`/api/runs/${requestId}/resume`).send({}).expect(202);
    await app.locals.runs.settle();
    expect(seen).toEqual([fixtureVideo.id, fixtureVideo.id]);
    expect((await request(app).get(`/api/runs/${requestId}`)).body.completed).toBe(1);
    const logs = await request(app).get(`/api/runs/${requestId}/logs?cursor=0`).expect(200);
    expect(logs.body.entries.map((entry: { event: string }) => entry.event)).toEqual(
      expect.arrayContaining([
        "local.run.pause.requested",
        "local.video.interrupted",
        "local.run.interrupted",
        "local.run.completed",
      ]),
    );
    await request(app).post(`/api/runs/${requestId}/resume`).send({}).expect(202);
    expect(seen).toHaveLength(2);
  });
  it("accepts 400 videos and refuses articles, wrong models, missing runtime and cross-origin requests", async () => {
    const { app, setReady } = await setup();
    const videos = Array.from({ length: 400 }, (_, i) => ({
      ...fixtureVideo,
      id: String(i).padStart(11, "0"),
      url: "https://www.youtube.com/watch?v=" + String(i).padStart(11, "0"),
    }));
    setReady(false);
    await request(app).post("/api/runs").send({ videos }).expect(503); // schema accepts the complete batch
    await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo], mode: "article" })
      .expect(400);
    await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo], model: "gemini-3-flash" })
      .expect(400);
    await request(app)
      .post("/api/catalog/videos")
      .set("Origin", "https://attacker.example")
      .send({ videoUrls: [fixtureVideo.url] })
      .expect(403);
    await request(app).get("/api/outputs/%2e%2e%2f.env").expect(400);
  });
  it("keeps a failed video's media and reports its error while the batch finishes", async () => {
    const { app, config, cleanup } = await setup(false, true);
    // An invalid worker result must be rejected by canonical validation before cleanup.
    await fs.mkdir(path.join(config.cacheDir, fixtureVideo.id), { recursive: true });
    await fs.writeFile(
      path.join(config.cacheDir, fixtureVideo.id, "metadata.json"),
      JSON.stringify({ ...fixtureVideo, durationSeconds: 120 }),
    );
    const run = await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo] })
      .expect(202);
    await app.locals.runs.settle();
    expect((await request(app).get(`/api/runs/${run.body.runId}`)).body.status).toBe("completed");
    expect(cleanup).not.toHaveBeenCalled();
    expect((await request(app).get(`/api/runs/${run.body.runId}`)).body.results[0].status).toBe("error");
    const logs = await request(app).get(`/api/runs/${run.body.runId}/logs`).expect(200);
    expect(
      logs.body.entries.find((entry: { event: string }) => entry.event === "local.video.failed"),
    ).toMatchObject({
      level: "error",
      videoId: fixtureVideo.id,
    });
    await expect(
      fs.access(path.join(config.cacheDir, fixtureVideo.id, "metadata.json")),
    ).resolves.toBeUndefined();
  });
  it("pauses the run when YouTube blocks this machine, keeping the remaining videos pending", async () => {
    const { app, catalog, seen } = await setup();
    catalog.getVideoMetadata.mockRejectedValue(
      new Error("ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies-from-browser"),
    );
    const second = { ...fixtureVideo, id: "bbbbbbbbbbb", url: "https://www.youtube.com/watch?v=bbbbbbbbbbb" };
    const run = await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo, second] })
      .expect(202);
    await app.locals.runs.settle();
    const body = (await request(app).get(`/api/runs/${run.body.runId}`)).body;
    expect(body).toMatchObject({ status: "interrupted", completed: 0 });
    expect(body.error).toContain(YOUTUBE_BLOCKED_MESSAGE);
    expect(body.results).toEqual([]);
    // The next video is not attempted: it would fail the same way.
    expect(catalog.getVideoMetadata).toHaveBeenCalledTimes(1);
    expect(seen).toEqual([]);
  });
  it("remembers a listed channel and its videos for the library", async () => {
    const { app, catalog } = await setup();
    catalog.getChannelVideos.mockResolvedValue({
      videos: [fixtureVideo],
      warnings: [],
      nextOffset: 50,
      scannedCount: 50,
      limitReached: false,
      collection: {
        id: "UCtest",
        kind: "channel",
        name: "Chaîne test",
        url: "https://www.youtube.com/@test/videos",
        avatarUrl: null,
        bannerUrl: null,
        followers: null,
      },
    });
    await request(app)
      .post("/api/catalog/channel")
      .send({ channelUrl: "https://www.youtube.com/@test" })
      .expect(200);
    const library = await request(app).get("/api/library").expect(200);
    expect(library.body.collections).toMatchObject([
      { id: "UCtest", name: "Chaîne test", videoIds: [fixtureVideo.id], nextOffset: 50, complete: false },
    ]);
    const collection = await request(app).get("/api/collections/UCtest").expect(200);
    expect(collection.body.videos[0].title).toBe(fixtureVideo.title);
    await request(app).post("/api/collections/remove").send({ id: "UCtest" }).expect(200);
    await request(app).get("/api/collections/UCtest").expect(404);
  });
  it("refuses traversal and malformed names for downloads and ZIP exports", async () => {
    const { app } = await setup();
    await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo] })
      .expect(202);
    await app.locals.runs.settle();
    for (const name of [
      "..%2Fpackage.json",
      "video--abc123def45.json.bak",
      "video--abc123def45%2F..%2F..%2Fpackage.json",
      "video--abc12.json",
      "%2e%2e",
    ])
      await request(app).get(`/api/outputs/${name}`).expect(400);
    await request(app).get("/api/outputs/video--zzzzzzzzzzz.json").expect(404);
    for (const filenames of [
      ["../.env"],
      ["video--abc123def45.json", "..\\..\\package.json"],
      ["video--zzzzzzzzzzz.md"],
    ])
      await request(app).post("/api/outputs/export-zip").send({ filenames }).expect(404);
    await request(app).post("/api/outputs/export-zip").send({ filenames: [] }).expect(400);
  });
  it("keeps deleting after a locked file and only prunes fully deleted videos", async () => {
    const { app, config } = await setup();
    const other = { ...fixtureVideo, id: "zyx987wvu65", url: "https://www.youtube.com/watch?v=zyx987wvu65" };
    const run = await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo, other] })
      .expect(202);
    await app.locals.runs.settle();
    const rm = fs.rm;
    const spy = vi.spyOn(fs, "rm").mockImplementation(async (target, options) => {
      if (String(target).endsWith("video--abc123def45.md"))
        throw Object.assign(new Error("EBUSY: resource busy or locked"), { code: "EBUSY" });
      return rm(target, options);
    });
    try {
      const response = await request(app)
        .post("/api/outputs/delete")
        .send({ videoIds: [fixtureVideo.id, other.id] })
        .expect(500);
      expect(response.body.error).toContain("video--abc123def45.md");
      expect(response.body.failed).toEqual([
        { filename: "video--abc123def45.md", videoId: fixtureVideo.id, error: "EBUSY" },
      ]);
      expect(response.body.deleted).toEqual(
        expect.arrayContaining([
          "video--abc123def45.json",
          "video--zyx987wvu65.md",
          "video--zyx987wvu65.json",
        ]),
      );
      expect(response.body.prunedRuns).toBe(1);
    } finally {
      spy.mockRestore();
    }
    const stored = (await request(app).get(`/api/runs/${run.body.runId}`)).body;
    expect(stored.videos.map((video: { id: string }) => video.id)).toEqual([fixtureVideo.id]);
    await expect(fs.access(path.join(config.outputDir, "video--abc123def45.md"))).resolves.toBeUndefined();
    const retry = await request(app)
      .post("/api/outputs/delete")
      .send({ videoIds: [fixtureVideo.id] })
      .expect(200);
    expect(retry.body).toMatchObject({ deleted: ["video--abc123def45.md"], failed: [], prunedRuns: 1 });
  });
  it("exports a complete, consistent corpus while a run is busy", async () => {
    const { app, worker, seen } = await setup();
    await request(app)
      .post("/api/runs")
      .send({ videos: [fixtureVideo] })
      .expect(202);
    await app.locals.runs.settle();
    const before = await request(app).get("/api/corpus/export").expect(200);
    const analyze = worker.analyze.bind(worker);
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    worker.analyze = async (video, signal, progress) => {
      seen.push(video.id);
      await gate;
      return analyze(video, signal, progress);
    };
    const other = { ...fixtureVideo, id: "zyx987wvu65", url: "https://www.youtube.com/watch?v=zyx987wvu65" };
    await request(app)
      .post("/api/runs")
      .send({ videos: [other] })
      .expect(202);
    await vi.waitFor(() => expect(seen).toContain(other.id));
    expect(app.locals.runs.busy).toBe(true);
    const during = await request(app).get("/api/corpus/export").expect(200);
    expect(during.text).toBe(before.text);
    expect(
      during.text
        .trim()
        .split("\n")
        .every((line: string) => JSON.parse(line).videoId === fixtureVideo.id),
    ).toBe(true);
    release();
    await app.locals.runs.settle();
    const after = await request(app).get("/api/corpus/export").expect(200);
    expect(after.text).toContain(other.id);
  });
  it("records a video whose exports were written even when the pause lands afterwards", async () => {
    const { config, worker, catalog, cleanup } = await setup();
    const service = createAnalysisService(config, catalog as unknown as YtDlpCatalogService, worker);
    const abort = new AbortController();
    const results: AnalyzeVideoResult[] = [];
    await service.analyzeVideos([fixtureVideo], randomUUID(), abort.signal, {
      onProgress: (progress) => {
        if (progress.phase === "export") abort.abort();
      },
      onResult: async (result) => {
        results.push(result);
      },
    });
    expect(results).toHaveLength(1);
    expect(results[0].status).toBe("success");
    expect(cleanup).not.toHaveBeenCalled();
    await expect(fs.access(path.join(config.outputDir, "video--abc123def45.md"))).resolves.toBeUndefined();
  });
});

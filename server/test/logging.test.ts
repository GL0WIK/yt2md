import { afterEach, describe, expect, it, vi } from "vitest";
import { observeStage } from "../src/shared/logging.js";
import fs from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { FileLogger, logEvent, readRunLog, resolveRunLog, withLogContext } from "../src/shared/logging.js";
import { temporaryDirectory } from "./helpers.js";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("persistent local journals", () => {
  it("continues to log in the terminal if persistence fails", async () => {
    const log = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    const root = await temporaryDirectory();
    const file = path.join(root, "blocked");
    await fs.writeFile(file, "A file cannot be a logging directory");
    const logger = new FileLogger(file);
    withLogContext(
      {},
      () => {
        logEvent("video.complete", { message: "Vidéo terminée" });
        logEvent("other.complete", { message: "Autre vidéo terminée" });
      },
      logger,
    );
    await expect(logger.flush()).resolves.toBeUndefined();
    expect(log.mock.calls.filter(([text]) => String(text).includes("terminée"))).toHaveLength(2);
    expect(warning).toHaveBeenCalledTimes(1);
  });
  it("keeps concurrent contexts separate, redacts credentials and reads incrementally", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    const root = await temporaryDirectory();
    const logger = new FileLogger(root);
    const first = randomUUID();
    const second = randomUUID();
    await Promise.all(
      [first, second].map((runId) =>
        withLogContext(
          { runId },
          async () => {
            logEvent("video.start", {
              message: "Début",
              videoId: runId === first ? "abc123def45" : "def456ghi78",
            });
            await Promise.resolve();
            logEvent("video.complete", {
              message:
                "url=https://user:password@example.test/video?signature=PRIVATE token hf_abcdefghijklmnopqrstuvwxyz",
              apiKey: "PRIVATE",
              durationMs: 12,
            });
          },
          logger,
        ),
      ),
    );
    await logger.flush();
    for (const runId of [first, second]) {
      const page = await readRunLog(root, runId, 0, 1);
      expect(page.entries).toHaveLength(1);
      expect(page.hasMore).toBe(true);
      expect(page.entries[0]).toMatchObject({ event: "video.start", runId });
      const next = await readRunLog(root, runId, page.nextCursor);
      expect(next.entries).toHaveLength(1);
      expect(next.hasMore).toBe(false);
      const text = JSON.stringify(next.entries);
      expect(text).not.toMatch(/PRIVATE|password|hf_abcdefghijklmnopqrstuvwxyz/);
      expect(text).toContain("https://example.test/video");
      expect(next.entries[0].apiKey).toBe("[redacted]");
    }
  });
  it("retains the complete batch journal when the bounded server journal rotates", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    const root = await temporaryDirectory();
    const logger = new FileLogger(root, "info", 800);
    const runId = randomUUID();
    withLogContext(
      { runId },
      () => {
        for (let i = 0; i < 20; i++) logEvent("chunk.complete", { message: "Morceau terminé", chunk: i });
        logEvent("ignored.debug", {}, "debug");
      },
      logger,
    );
    await logger.flush();
    expect((await readRunLog(root, runId, 0)).entries).toHaveLength(20);
    expect((await fs.readdir(root)).filter((name) => name.startsWith("server.jsonl"))).toHaveLength(4);
    expect((await fs.stat(path.join(root, "server.jsonl"))).size).toBeLessThanOrEqual(800);
  });
  it("waits for a complete UTF-8 line and returns only recent entries on initial connection", async () => {
    const root = await temporaryDirectory();
    await fs.mkdir(path.join(root, "runs"));
    const runId = randomUUID();
    const filename = path.join(root, "runs", runId + ".jsonl");
    const record = Buffer.from(
      JSON.stringify({
        timestamp: new Date().toISOString(),
        level: "info",
        event: "test",
        message: "écran 🇫🇷",
      }) + "\n",
    );
    const split = record.indexOf(Buffer.from("🇫")) + 1;
    await fs.writeFile(filename, record.subarray(0, split));
    expect(await readRunLog(root, runId, 0)).toEqual({ entries: [], nextCursor: 0, hasMore: true });
    await fs.appendFile(filename, record.subarray(split));
    expect((await readRunLog(root, runId, 0)).entries[0].message).toBe("écran 🇫🇷");
    await fs.appendFile(filename, Buffer.concat(Array.from({ length: 600 }, () => record)));
    const tail = await readRunLog(root, runId, undefined, 10);
    expect(tail.entries).toHaveLength(10);
    expect(tail.nextCursor).toBe((await fs.stat(filename)).size);
    expect(tail.hasMore).toBe(false);
  });
  it("skips a line longer than the read window instead of re-polling it forever", async () => {
    const root = await temporaryDirectory();
    await fs.mkdir(path.join(root, "runs"));
    const runId = randomUUID();
    const filename = path.join(root, "runs", runId + ".jsonl");
    const small = JSON.stringify({ timestamp: "t", level: "info", event: "after", message: "suite" }) + "\n";
    const huge = JSON.stringify({ event: "huge", message: "x".repeat(700 * 1024) }) + "\n";
    await fs.writeFile(filename, huge + small);
    const first = await readRunLog(root, runId, 0);
    expect(first.entries).toHaveLength(1);
    expect(first.entries[0]).toMatchObject({ event: "log.line.skipped", level: "warn" });
    expect(first.nextCursor).toBe(Buffer.byteLength(huge));
    const second = await readRunLog(root, runId, first.nextCursor);
    expect(second.entries.map((entry) => entry.event)).toEqual(["after"]);
    expect(second.hasMore).toBe(false);
    // A huge line still being written (no newline yet) keeps the cursor so it is read once complete.
    await fs.writeFile(filename, huge.slice(0, -1));
    expect(await readRunLog(root, runId, 0)).toMatchObject({ entries: [], nextCursor: 0, hasMore: true });
  });
  it("refuses journals reached through a junction escaping the log directory", async () => {
    const root = await temporaryDirectory();
    const outside = await temporaryDirectory();
    const runId = randomUUID();
    await fs.writeFile(path.join(outside, runId + ".jsonl"), "private");
    await fs.symlink(outside, path.join(root, "runs"), "junction");
    await expect(resolveRunLog(root, runId)).rejects.toThrow("Chemin");
    await expect(resolveRunLog(root, "../.env")).rejects.toThrow("Identifiant");
  });
});
describe("analysis stage logging", () => {
  it("reports a pending operation then stops heartbeat after completion", async () => {
    vi.useFakeTimers();
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    let finish!: (value: string) => void;
    const pending = observeStage(
      "gemini.request",
      { runId: "run", videoId: "video", chunk: 1 },
      () =>
        new Promise<string>((resolve) => {
          finish = resolve;
        }),
    );
    await vi.advanceTimersByTimeAsync(30000);
    expect(log.mock.calls.map(([line]) => JSON.parse(line).event)).toEqual([
      "gemini.request.start",
      "gemini.request.waiting",
    ]);
    finish("ok");
    await expect(pending).resolves.toBe("ok");
    await vi.advanceTimersByTimeAsync(60000);
    expect(log.mock.calls.map(([line]) => JSON.parse(line).event)).toEqual([
      "gemini.request.start",
      "gemini.request.waiting",
      "gemini.request.complete",
    ]);
    expect(JSON.parse(log.mock.calls[1][0])).toMatchObject({ runId: "run", videoId: "video", chunk: 1 });
    expect(vi.getTimerCount()).toBe(0);
  });
  it("preserves the error without logging provider content and clears its timer", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "log").mockImplementation(() => {});
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const error = Object.assign(new Error("secret-key and transcript content"), {
      status: 429,
      body: "private response",
    });
    await expect(
      observeStage("gemini.request", { videoId: "video" }, async () => {
        throw error;
      }),
    ).rejects.toBe(error);
    const fields = JSON.parse(log.mock.calls[0][0]);
    expect(fields).toMatchObject({ event: "gemini.request.failed", httpStatus: 429, errorType: "Error" });
    expect(log.mock.calls[0][0]).not.toMatch(/secret-key|transcript content|private response/);
    expect(vi.getTimerCount()).toBe(0);
  });
});

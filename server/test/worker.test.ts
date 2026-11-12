import { describe, expect, it, vi } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import { PythonWorker, WorkerAnalysisSchema } from "../src/local/worker.js";
import { getConfig } from "../src/config.js";
import { fixtureVideo } from "./fixtures.js";
import { temporaryDirectory } from "./helpers.js";
import { randomUUID } from "node:crypto";
import { FileLogger, readRunLog, withLogContext } from "../src/shared/logging.js";

async function bridge() {
  const directory = await temporaryDirectory();
  const script = path.join(directory, "fake_worker.py");
  await fs.writeFile(
    script,
    `import sys, json, time, os
for line in sys.stdin:
    request = json.loads(line)
    action = request["action"]
    def emit(value):
        print(json.dumps({"id": request["id"], **value}), flush=True)
    if action == "health":
        emit({"ok": True, "result": {"ready": True, "issues": [], "pid": os.getpid()}})
    elif action == "analyze":
        print("progress log on stderr", file=sys.stderr, flush=True)
        emit({"event": "log", "name": "vision.generate.waiting", "level": "info", "message": "Qwen actif", "fields": {"vramFreeBytes": 1000000000}})
        emit({"event": "progress", "phase": "audio", "chunk": 0, "totalChunks": 1})
        time.sleep(600)
    elif action == "cleanup":
        emit({"ok": False, "error": {"message": "cleanup rejected for test"}})
    elif action == "shutdown":
        emit({"ok": True, "result": {"stopped": True}})
        break
`,
    "utf8",
  );
  return new PythonWorker({ ...getConfig({}), workerScript: script });
}
describe("Python process bridge", () => {
  it("serves real JSON-lines requests and closes an idle Python process normally", async () => {
    const worker = await bridge();
    try {
      const state = await worker.health();
      expect(state.ready).toBe(true);
      await expect(worker.cleanup(fixtureVideo.id)).rejects.toThrow("cleanup rejected for test");
      expect(worker.busy).toBe(false);
    } finally {
      await worker.close();
    }
  });
  it("aborts a blocking Python inference, rejects the request and starts a fresh worker", async () => {
    const worker = await bridge();
    const signal = new AbortController();
    const directory = await temporaryDirectory();
    const logger = new FileLogger(directory);
    const runId = randomUUID();
    try {
      await worker.health();
      const work = withLogContext(
        { runId, videoId: fixtureVideo.id },
        () => worker.analyze(fixtureVideo, signal.signal, () => signal.abort()),
        logger,
      );
      await expect(work).rejects.toMatchObject({ name: "AbortError" });
      await logger.flush();
      const journal = await readRunLog(directory, runId, 0);
      expect(journal.entries.find((entry) => entry.event === "vision.generate.waiting")).toMatchObject({
        message: "Qwen actif",
        vramFreeBytes: 1000000000,
        runId,
        videoId: fixtureVideo.id,
        source: "python",
      });
      expect(worker.busy).toBe(false);
      expect((await worker.health()).ready).toBe(true);
    } finally {
      await worker.close();
    }
  });
  it("refuses invalid worker payloads instead of exporting partial speech", () => {
    expect(WorkerAnalysisSchema.safeParse({ durationSeconds: 20, transcript: [] }).success).toBe(false);
  });
});
async function scriptedWorker(source: string, shutdownGraceMs?: number) {
  const directory = await temporaryDirectory();
  const script = path.join(directory, "scripted_worker.py");
  await fs.writeFile(script, source, "utf8");
  return new PythonWorker({ ...getConfig({}), workerScript: script }, { shutdownGraceMs });
}
describe("Python process bridge robustness", () => {
  it("ignores stray native stdout and invalid events instead of killing the worker", async () => {
    const worker = await scriptedWorker(`import sys, json, os
for line in sys.stdin:
    request = json.loads(line)
    print("ggml_cuda_init: found 1 CUDA devices", flush=True)
    print("[1, 2, 3]", flush=True)
    print(json.dumps({"id": request["id"], "event": "progress", "phase": "nope"}), flush=True)
    print(json.dumps({"id": request["id"], "event": "log", "name": "BAD NAME", "level": "info", "message": "x"}), flush=True)
    if request["action"] == "shutdown":
        print(json.dumps({"id": request["id"], "ok": True, "result": {}}), flush=True)
        break
    print(json.dumps({"id": request["id"], "ok": True, "result": {"ready": True, "issues": [], "pid": os.getpid()}}), flush=True)
`);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const first = (await worker.health()) as { ready: boolean; pid?: number };
      expect(first.ready).toBe(true);
      expect(warn.mock.calls.some(([line]) => String(line).includes("worker.stdout.ignored"))).toBe(true);
      // Same process: the stray output did not restart the worker.
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(worker.busy).toBe(false);
    } finally {
      await worker.close();
      warn.mockRestore();
    }
  });
  it("forces the stop of a worker that acknowledges shutdown but never exits", async () => {
    const worker = await scriptedWorker(
      `import sys, json, time
for line in sys.stdin:
    request = json.loads(line)
    if request["action"] == "shutdown":
        print(json.dumps({"id": request["id"], "ok": True, "result": {"stopped": True}}), flush=True)
        time.sleep(600)
    print(json.dumps({"id": request["id"], "ok": True, "result": {"ready": True, "issues": []}}), flush=True)
`,
      200,
    );
    expect((await worker.health()).ready).toBe(true);
    const started = performance.now();
    await worker.close();
    expect(performance.now() - started).toBeLessThan(10000);
    expect(worker.busy).toBe(false);
  }, 20000);
});

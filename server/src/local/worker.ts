import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { z } from "zod";
import type { AppConfig } from "../config.js";
import { logEvent, captureLogContext, withCapturedLogContext } from "../shared/logging.js";
import { killProcessTree } from "../shared/process.js";
import {
  TranscriptEntrySchema,
  VisualSegmentSchema,
  type CatalogVideo,
  type AnalysisProgress,
} from "../shared/schemas.js";

export const WorkerAnalysisSchema = z.object({
  durationSeconds: z.number().int().positive(),
  model: z.string().min(1),
  promptVersion: z.string().min(1),
  chunkSeconds: z.number().positive(),
  transcript: z.array(TranscriptEntrySchema),
  visualSegments: z.array(VisualSegmentSchema).min(1),
  warnings: z.array(z.string()),
  localProvenance: z.record(z.unknown()),
  usage: z.object({
    model: z.string(),
    promptTokenCount: z.number().int().nonnegative().nullable(),
    candidatesTokenCount: z.number().int().nonnegative().nullable(),
    totalTokenCount: z.number().int().nonnegative().nullable(),
    elapsedMs: z.number().nonnegative(),
    cacheHits: z.number().int().nonnegative(),
    visionGroups: z.number().int().positive(),
  }),
});
export type WorkerAnalysis = z.infer<typeof WorkerAnalysisSchema>;
export type WorkerHealth = {
  ready: boolean;
  python?: string;
  gpu?: { name: string; totalVramBytes: number; freeVramBytes: number; torchCuda: string } | null;
  packages?: Record<string, string>;
  models?: unknown;
  issues: string[];
};
export interface LocalWorker {
  readonly busy: boolean;
  health(): Promise<WorkerHealth>;
  analyze(
    video: CatalogVideo,
    signal: AbortSignal,
    progress: (event: Omit<AnalysisProgress, "videoId" | "videoTitle">) => void,
  ): Promise<WorkerAnalysis>;
  cleanup(videoId: string): Promise<void>;
  close(): Promise<void>;
}

export function workerConfig(config: AppConfig) {
  const {
    projectRoot: _root,
    pythonBinary: _python,
    workerScript: _script,
    outputDir: _output,
    port: _port,
    host: _host,
    logsDir: _logs,
    logLevel: _logLevel,
    workerTimeoutMs: _timeout,
    ytDlpTimeoutMs: _catalogTimeout,
    catalogMaxVideos: _max,
    ...values
  } = config;
  return values;
}

type Pending = {
  id: string;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  progress?: (event: Omit<AnalysisProgress, "videoId" | "videoTitle">) => void;
  cleanup: () => void;
  context: ReturnType<typeof captureLogContext>;
  action: string;
  started: number;
};
type RequestOptions = {
  signal?: AbortSignal;
  progress?: Pending["progress"];
  timeoutMs?: number;
};
const WorkerLogSchema = z.object({
  name: z.string().regex(/^[a-z0-9_.-]{1,100}$/),
  level: z.enum(["debug", "info", "warn", "error"]),
  message: z.string().max(8000),
  fields: z.record(z.union([z.string().max(8000), z.number().finite(), z.boolean(), z.null()])).default({}),
});
const MAX_STDOUT_LINE_BYTES = 32 * 1024 * 1024;
const ProgressSchema = z.object({
  phase: z.enum(["metadata", "download", "extract", "audio", "vision", "export"]),
  chunk: z.number().int().nonnegative(),
  totalChunks: z.number().int().nonnegative(),
  detail: z.string().optional(),
});

export class PythonWorker implements LocalWorker {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending: Pending | null = null;
  private stopping: Promise<void> | null = null;
  private exitReason: Error | null = null;
  private stderr = "";
  private lastHealth: WorkerHealth | null = null;
  private healthTime = 0;
  // The worker handles one request at a time: later requests wait instead of failing.
  private queue: Promise<unknown> = Promise.resolve();
  get busy() {
    return this.pending !== null;
  }

  constructor(
    private readonly config: AppConfig,
    private readonly options: { shutdownGraceMs?: number } = {},
  ) {}

  private launch() {
    if (this.child) return this.child;
    this.stderr = "";
    this.exitReason = null;
    const child = spawn(this.config.pythonBinary, ["-u", this.config.workerScript, "--stdio"], {
      cwd: this.config.projectRoot,
      shell: false,
      windowsHide: true,
      detached: process.platform !== "win32",
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONUNBUFFERED: "1",
        HF_HUB_DISABLE_TELEMETRY: "1",
        HF_HOME: path.join(this.config.modelsDir, ".hub"),
      },
      stdio: "pipe",
    });
    this.child = child;
    const launchContext = captureLogContext();
    logEvent("worker.started", { message: "Worker Python démarré.", pid: child.pid, source: "python" });
    const ignoreLine = (reason: string, line: string) =>
      withCapturedLogContext(this.pending?.context ?? launchContext, () =>
        logEvent(
          "worker.stdout.ignored",
          {
            message: reason + " : " + line.slice(0, 500),
            bytes: Buffer.byteLength(line),
            source: "python.stdout",
          },
          "warn",
        ),
      );
    const handleLine = (line: string) => {
      if (!line.trim()) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        // Native libraries (CUDA, llama.cpp, tqdm) sometimes write to stdout: never fatal.
        ignoreLine("Sortie non JSON du worker ignorée", line);
        return;
      }
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        ignoreLine("Message worker sans objet JSON ignoré", line);
        return;
      }
      const message = parsed as Record<string, unknown>;
      const pending = this.pending;
      if (!pending || message.id !== pending.id) return;
      if (message.event === "log") {
        const entry = WorkerLogSchema.safeParse(message);
        if (!entry.success) return ignoreLine("Journal worker invalide ignoré", line);
        withCapturedLogContext(pending.context, () =>
          logEvent(
            entry.data.name,
            { ...entry.data.fields, message: entry.data.message, source: "python" },
            entry.data.level,
          ),
        );
        return;
      }
      if (message.event === "progress") {
        const event = ProgressSchema.safeParse(message);
        if (!event.success) return ignoreLine("Progression worker invalide ignorée", line);
        pending.progress?.(event.data);
        return;
      }
      this.pending = null;
      pending.cleanup();
      withCapturedLogContext(pending.context, () =>
        logEvent(
          "worker.request." + (message.ok === true ? "complete" : "failed"),
          {
            message:
              message.ok === true
                ? "Requête Python terminée."
                : String((message.error as { message?: unknown })?.message ?? "Échec du worker."),
            action: pending.action,
            durationMs: Math.round(performance.now() - pending.started),
            source: "python",
          },
          message.ok === true ? (pending.action === "health" ? "debug" : "info") : "error",
        ),
      );
      if (message.ok === true) pending.resolve(message.result);
      else {
        const details = message.error as { message?: unknown } | undefined;
        pending.reject(
          new Error(typeof details?.message === "string" ? details.message : "Échec du worker local."),
        );
      }
    };
    const dispatch = (line: string) => {
      try {
        handleLine(line);
      } catch (error) {
        void this.stop(error instanceof Error ? error : new Error("Protocole worker invalide."));
      }
    };
    // Byte-level splitter: unlike readline, the line buffer is bounded before it grows past the limit.
    let parts: Buffer[] = [];
    let buffered = 0;
    let overflowed = false;
    child.stdout.on("data", (chunk: Buffer) => {
      if (overflowed) return;
      let start = 0;
      while (start < chunk.length) {
        const end = chunk.indexOf(10, start);
        const piece = chunk.subarray(start, end < 0 ? chunk.length : end);
        buffered += piece.length;
        if (buffered > MAX_STDOUT_LINE_BYTES) {
          overflowed = true;
          parts = [];
          void this.stop(new Error("Réponse worker trop volumineuse."));
          return;
        }
        parts.push(piece);
        if (end < 0) return;
        const line = Buffer.concat(parts).toString("utf8").replace(/\r$/, "");
        parts = [];
        buffered = 0;
        dispatch(line);
        start = end + 1;
      }
    });
    child.stdout.on("end", () => {
      if (overflowed || !parts.length) return;
      const line = Buffer.concat(parts).toString("utf8");
      parts = [];
      buffered = 0;
      dispatch(line);
    });
    child.stderr.setEncoding("utf8");
    let stderrLine = "";
    const flushStderr = (line: string) => {
      const clean = line.trim();
      if (!clean) return;
      withCapturedLogContext(this.pending?.context ?? launchContext, () =>
        logEvent(
          "worker.output",
          { message: clean, source: "python.stderr" },
          /traceback|error|exception/i.test(clean) ? "warn" : "info",
        ),
      );
    };
    child.stderr.on("data", (chunk: string) => {
      this.stderr = (this.stderr + chunk).slice(-8000);
      stderrLine += chunk;
      const lines = stderrLine.split(/\r\n|\r|\n/);
      stderrLine = lines.pop() ?? "";
      for (const line of lines) flushStderr(line);
      if (stderrLine.length > 8000) {
        flushStderr(stderrLine.slice(0, 8000));
        stderrLine = "";
      }
    });
    child.stdin.on("error", (error) => {
      void this.stop(error);
    });
    child.on("error", (error) => {
      const pending = this.pending;
      this.pending = null;
      pending?.cleanup();
      pending?.reject(
        new Error("Python local indisponible. Exécuter npm run python:setup. " + error.message),
      );
    });
    child.on("close", (code, signal) => {
      flushStderr(stderrLine);
      withCapturedLogContext(this.pending?.context ?? launchContext, () =>
        logEvent(
          "worker.stopped",
          {
            message: "Worker Python arrêté.",
            pid: child.pid,
            exitCode: code,
            signal,
            source: "python",
          },
          code && !this.exitReason ? "warn" : "info",
        ),
      );
      if (this.child !== child) return;
      this.child = null;
      const pending = this.pending;
      this.pending = null;
      pending?.cleanup();
      pending?.reject(
        this.exitReason ?? new Error("Le worker Python s’est arrêté. " + this.stderr.slice(-1200)),
      );
    });
    return child;
  }

  private async stop(reason: Error): Promise<void> {
    if (this.stopping) return this.stopping;
    const child = this.child;
    if (!child) return;
    this.exitReason = reason;
    withCapturedLogContext(this.pending?.context ?? captureLogContext(), () =>
      logEvent(
        "worker.stop.requested",
        {
          message: reason.message,
          source: "python",
          pid: child.pid,
        },
        reason.name === "AbortError" ? "info" : "error",
      ),
    );
    this.lastHealth = null;
    const work = new Promise<void>((resolve) => {
      child.once("close", () => resolve());
      killProcessTree(child, "SIGTERM", { group: true });
      const force = setTimeout(() => {
        if (this.child === child) killProcessTree(child, "SIGKILL", { group: true });
      }, 2000);
      force.unref();
      child.once("close", () => clearTimeout(force));
    });
    this.stopping = work;
    try {
      await work;
    } finally {
      this.stopping = null;
    }
  }

  private request(
    action: string,
    payload: Record<string, unknown> = {},
    options?: RequestOptions,
  ): Promise<unknown> {
    // A health check between two videos must never make the next analysis fail as "already running".
    const result = this.queue.then(() => this.send(action, payload, options));
    this.queue = result.catch(() => undefined);
    return result;
  }

  private async send(
    action: string,
    payload: Record<string, unknown>,
    options?: RequestOptions,
  ): Promise<unknown> {
    if (this.stopping) await this.stopping;
    options?.signal?.throwIfAborted();
    if (this.pending) throw new Error("Une inférence locale est déjà en cours.");
    const child = this.launch();
    return new Promise((resolve, reject) => {
      const id = randomUUID();
      const logContext = captureLogContext();
      logEvent(
        "worker.request.start",
        { message: "Requête envoyée au worker Python.", action, source: "python" },
        action === "health" ? "debug" : "info",
      );
      const abort = () => {
        void this.stop(new DOMException("Analyse mise en pause.", "AbortError"));
      };
      const timer = setTimeout(() => {
        void this.stop(
          new Error("Délai du worker dépassé. La reprise utilisera les morceaux déjà sauvegardés."),
        );
      }, options?.timeoutMs ?? this.config.workerTimeoutMs);
      timer.unref();
      const heartbeat = setInterval(
        () =>
          logEvent("worker.request.waiting", {
            message: "Worker actif ; attente du résultat.",
            action,
            source: "python",
            elapsedMs: Math.round(performance.now() - this.pending!.started),
          }),
        30000,
      );
      heartbeat.unref();
      options?.signal?.addEventListener("abort", abort, { once: true });
      this.pending = {
        id,
        resolve,
        reject,
        context: logContext,
        action,
        started: performance.now(),
        progress: options?.progress,
        cleanup: () => {
          clearTimeout(timer);
          clearInterval(heartbeat);
          options?.signal?.removeEventListener("abort", abort);
        },
      };
      child.stdin.write(JSON.stringify({ id, action, config: workerConfig(this.config), ...payload }) + "\n");
    });
  }

  async health(): Promise<WorkerHealth> {
    if (this.busy)
      return this.lastHealth ?? { ready: false, issues: ["Initialisation du moteur local en cours."] };
    if (this.lastHealth && Date.now() - this.healthTime < 30000) return this.lastHealth;
    try {
      const value = await this.request("health", {}, { timeoutMs: 60000 });
      this.lastHealth = z
        .object({ ready: z.boolean(), issues: z.array(z.string()) })
        .passthrough()
        .parse(value) as WorkerHealth;
      this.healthTime = Date.now();
      return this.lastHealth;
    } catch (error) {
      // Not cached: once Python is fixed, the next check must see it immediately.
      this.lastHealth = null;
      return { ready: false, issues: [error instanceof Error ? error.message : "Worker indisponible."] };
    }
  }
  async analyze(video: CatalogVideo, signal: AbortSignal, progress: Pending["progress"]) {
    return WorkerAnalysisSchema.parse(await this.request("analyze", { video }, { signal, progress }));
  }
  async cleanup(videoId: string) {
    await this.request("cleanup", { videoId });
  }
  async close() {
    const child = this.child;
    if (!child) return;
    if (this.pending || this.stopping) {
      await this.stop(new DOMException("Arrêt du serveur local.", "AbortError"));
      return;
    }
    // An idle worker can release its models and exit normally; no platform-specific kill needed.
    const closed = new Promise<void>((resolve) => child.once("close", () => resolve()));
    try {
      await this.request("shutdown", {}, { timeoutMs: 10000 });
    } catch {
      await this.stop(new DOMException("Arrêt du serveur local.", "AbortError"));
    }
    // Python may acknowledge the shutdown and then hang while a native library unloads.
    let grace: NodeJS.Timeout | undefined;
    const exited = await Promise.race([
      closed.then(() => true),
      new Promise<boolean>((resolve) => {
        grace = setTimeout(() => resolve(false), this.options.shutdownGraceMs ?? 5000);
      }),
    ]);
    clearTimeout(grace);
    if (exited) return;
    logEvent(
      "worker.shutdown.timeout",
      { message: "Le worker Python ne s’arrête pas ; arrêt forcé.", pid: child.pid, source: "python" },
      "warn",
    );
    await this.stop(new DOMException("Arrêt du serveur local.", "AbortError"));
    await closed;
  }
}

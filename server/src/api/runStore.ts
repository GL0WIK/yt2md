import fs from "node:fs/promises";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { modelLabel, type AppConfig } from "../config.js";
import type { AnalysisRun, CatalogVideo } from "../shared/schemas.js";
import { atomicWrite } from "../shared/files.js";
import { HttpError } from "../shared/errors.js";
import { logEvent, errorFields, withLogContext } from "../shared/logging.js";
import type { AnalysisService } from "./analysisService.js";

type StoredRun = AnalysisRun & { requestHash: string; runtimeHash: string };
const RUN_FILE = /^[a-f0-9-]{36}\.json$/;

/** Internal hashes never leave the server. */
export function publicRun({ requestHash: _request, runtimeHash: _runtime, ...run }: StoredRun): AnalysisRun {
  return run;
}
function requestHash(videos: CatalogVideo[], model: string, runtimeHash: string) {
  return createHash("sha256").update(JSON.stringify({ videos, model, runtimeHash })).digest("hex");
}

export function createRunStore(config: AppConfig, service: AnalysisService) {
  const directory = path.join(config.outputDir, ".runs");
  const model = modelLabel(config);
  const runtimeHash = createHash("sha256")
    .update(
      JSON.stringify({
        asr: config.asrModel,
        asrLanguage: config.asrLanguage,
        vision: config.visionModel,
        visionBackend: config.visionBackend,
        audioChunkSeconds: config.audioChunkSeconds,
        audioOverlapSeconds: config.audioOverlapSeconds,
        fps: config.fps,
        groupFrames: config.groupFrames,
        frameMaxSide: config.frameMaxSide,
        visionMaxPixels: config.visionMaxPixels,
        visionMaxNewTokens: config.visionMaxNewTokens,
        visionQuantization: config.visionQuantization,
      }),
    )
    .digest("hex");
  let active: StoredRun | null = null;
  let controller: AbortController | null = null;
  let task: Promise<void> | null = null;
  let starting = false;
  let maintaining = false;
  let writes: Promise<void> = Promise.resolve();
  const file = (id: string) => {
    if (!/^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/.test(id))
      throw new HttpError(400, "Identifiant d’analyse invalide.");
    return path.join(directory, id + ".json");
  };
  const save = (run: StoredRun) => {
    run.updatedAt = new Date().toISOString();
    const snapshot = JSON.stringify(run);
    const next = writes.then(() => atomicWrite(file(run.runId), snapshot));
    writes = next.catch(() => {});
    return next;
  };
  async function read(id: string): Promise<StoredRun | null> {
    const filename = file(id);
    if (active?.runId === id) return structuredClone(active);
    try {
      const run = JSON.parse(await fs.readFile(filename, "utf8")) as StoredRun;
      if (run.runId !== id || !Array.isArray(run.videos) || !Array.isArray(run.results))
        throw new Error("Historique local invalide.");
      if (run.status === "running") {
        run.status = "interrupted";
        run.error = "Le serveur a redémarré. Reprendre utilisera les morceaux déjà sauvegardés.";
      }
      return run;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw error;
    }
  }
  const runFiles = async () =>
    (
      await fs.readdir(directory).catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return [] as string[];
        throw error;
      })
    ).filter((name) => RUN_FILE.test(name));
  function launch(run: StoredRun, videos: CatalogVideo[]) {
    active = run;
    const abort = new AbortController();
    controller = abort;
    let lastProgressSave = 0;
    let persistenceError: Error | null = null;
    let blocked: string | null = null;
    task = withLogContext({ runId: run.runId }, async () => {
      const started = performance.now();
      logEvent("local.run.start", {
        message: "Démarrage du traitement local.",
        totalVideos: run.total,
        pendingVideos: videos.length,
        completedVideos: run.completed,
      });
      try {
        await service.analyzeVideos(videos, run.runId, abort.signal, {
          onProgress(progress) {
            run.progress = progress;
            logEvent("local.progress", {
              message: progress.detail ?? progress.phase,
              videoId: progress.videoId,
              phase: progress.phase,
              chunk: progress.chunk,
              totalChunks: progress.totalChunks,
            });
            if (Date.now() - lastProgressSave < 1000) return;
            lastProgressSave = Date.now();
            void save(run).catch((error: Error) => {
              persistenceError = error;
              abort.abort(error);
            });
          },
          async onResult(result) {
            run.results = run.results.filter((old) => old.videoId !== result.videoId);
            run.results.push(result);
            run.completed = run.results.length;
            await save(run);
            logEvent("local.result.saved", {
              message: "Résultat de la vidéo enregistré.",
              runId: run.runId,
              videoId: result.videoId,
              status: result.status,
            });
          },
          onBlocked(message) {
            blocked = message;
            abort.abort();
          },
        });
        if (persistenceError) throw persistenceError;
        // A pause during the last export still finishes every video: the run is complete, not resumable.
        const paused = abort.signal.aborted && run.completed < run.total;
        run.status = paused ? "interrupted" : "completed";
        if (paused) run.error = blocked ?? "Analyse mise en pause. Les morceaux terminés sont conservés.";
        logEvent("local.run." + run.status, {
          message: paused ? "Traitement en pause ; checkpoints conservés." : "Traitement terminé.",
          durationMs: Math.round(performance.now() - started),
          completedVideos: run.completed,
          successCount: run.results.filter((result) => result.status === "success").length,
          errorCount: run.results.filter((result) => result.status === "error").length,
        });
      } catch (error) {
        run.status = "failed";
        run.error = error instanceof Error ? error.message : "Échec de l’analyse locale.";
        logEvent("local.run.failed", { message: run.error, ...errorFields(error) }, "error");
      } finally {
        delete run.progress;
        try {
          await save(run);
        } catch (error) {
          logEvent("local.history.write.failed", errorFields(error), "error");
        }
        if (active === run) {
          active = null;
          controller = null;
        }
      }
    });
  }
  const assertAvailable = () => {
    if (active || starting || maintaining || service.busy)
      throw new HttpError(409, "Une analyse ou une opération sur les exports est déjà en cours.");
  };
  return {
    read,
    get busy() {
      return Boolean(active || starting || maintaining || service.busy);
    },
    async start(videos: CatalogVideo[], id: string = randomUUID()): Promise<AnalysisRun> {
      if (starting || maintaining) throw new HttpError(409, "Une opération démarre déjà.");
      starting = true;
      try {
        const hash = requestHash(videos, model, runtimeHash);
        const existing = await read(id);
        if (existing) {
          if (existing.requestHash !== hash)
            throw new HttpError(409, "Cette requête correspond à une autre sélection.");
          return publicRun(existing);
        }
        if (active || service.busy) throw new HttpError(409, "Une analyse est déjà en cours.");
        const now = new Date().toISOString();
        const run: StoredRun = {
          runId: id,
          requestHash: hash,
          runtimeHash,
          model,
          videos,
          status: "running",
          total: videos.length,
          completed: 0,
          createdAt: now,
          updatedAt: now,
          results: [],
        };
        await save(run);
        launch(run, videos);
        return publicRun(structuredClone(run));
      } finally {
        starting = false;
      }
    },
    async resume(id: string): Promise<AnalysisRun> {
      assertAvailable();
      starting = true;
      try {
        const run = await read(id);
        if (!run) throw new HttpError(404, "Analyse introuvable.");
        if (run.model !== model || run.runtimeHash !== runtimeHash)
          throw new HttpError(
            409,
            "Les modèles ou paramètres ont changé. Sélectionnez de nouveau ces vidéos puis lancez la transcription.",
          );
        const done = new Set<string>();
        for (const result of run.results) {
          if (result.status !== "success") continue;
          const id = result.videoId;
          if (!/^[A-Za-z0-9_-]{11}$/.test(id)) continue;
          try {
            await Promise.all(
              ["json", "md", "local.json"].map((ext) =>
                fs.access(path.join(config.outputDir, `video--${id}.${ext}`)),
              ),
            );
            done.add(id);
          } catch {
            /* Missing export must be regenerated from the worker cache. */
          }
        }
        const pending = run.videos.filter((video) => !done.has(video.id));
        if (!pending.length) {
          // Every export exists: close a run left "interrupted" so it is no longer offered for resume.
          if (run.status !== "completed") {
            run.status = "completed";
            delete run.error;
            await save(run);
          }
          return publicRun(run);
        }
        run.results = run.results.filter((result) => result.status === "success" && done.has(result.videoId));
        run.completed = run.results.length;
        run.status = "running";
        delete run.error;
        await save(run);
        launch(run, pending);
        return publicRun(structuredClone(run));
      } finally {
        starting = false;
      }
    },
    async pause(id: string) {
      file(id);
      if (active?.runId !== id) throw new HttpError(409, "Cette analyse n’est pas en cours.");
      logEvent("local.run.pause.requested", {
        runId: id,
        message: "Pause demandée : arrêt de l’inférence en cours.",
      });
      controller?.abort();
      await task;
      const run = await read(id);
      return run && publicRun(run);
    },
    async list() {
      const dated = await Promise.all(
        (await runFiles()).map(async (name) => ({
          name,
          time: (await fs.stat(path.join(directory, name))).mtimeMs,
        })),
      );
      const recent = dated.sort((a, b) => b.time - a.time).slice(0, 50);
      const runs = await Promise.all(recent.map(({ name }) => read(name.slice(0, -5)).catch(() => null)));
      return runs.filter((run): run is StoredRun => run !== null);
    },
    async removeVideos(ids: string[]) {
      if (active || starting)
        throw new HttpError(409, "Mettez le traitement en pause avant de supprimer ses exports.");
      const wanted = new Set(ids);
      let changed = 0;
      for (const name of await runFiles()) {
        const run = await read(name.slice(0, -5));
        if (!run || !run.videos.some((video) => wanted.has(video.id))) continue;
        run.videos = run.videos.filter((video) => !wanted.has(video.id));
        run.results = run.results.filter((result) => !wanted.has(result.videoId));
        run.total = run.videos.length;
        run.completed = run.results.length;
        run.requestHash = requestHash(run.videos, run.model, run.runtimeHash);
        await save(run);
        changed++;
      }
      return changed;
    },
    async maintenance<T>(work: () => Promise<T>): Promise<T> {
      assertAvailable();
      maintaining = true;
      try {
        return await work();
      } finally {
        maintaining = false;
      }
    },
    async settle() {
      await task;
      await writes;
    },
    async close() {
      controller?.abort();
      await task;
      await writes;
    },
  };
}
export type RunStore = ReturnType<typeof createRunStore>;

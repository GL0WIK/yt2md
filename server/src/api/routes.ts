import { Router } from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { modelLabel, type AppConfig } from "../config.js";
import type { YtDlpCatalogService } from "../catalog/ytDlp.js";
import type { LocalWorker } from "../local/worker.js";
import { publicRun, type RunStore } from "./runStore.js";
import { createCollectionStore } from "./collections.js";
import type { CatalogPage } from "../shared/schemas.js";
import {
  AnalyzeRequestSchema,
  ChannelCatalogRequestSchema,
  PlaylistCatalogRequestSchema,
  DirectVideosRequestSchema,
  DeleteVideosRequestSchema,
} from "../shared/schemas.js";
import { HttpError } from "../shared/errors.js";
import { errorFields, logEvent, readRunLog, resolveRunLog, type FileLogger } from "../shared/logging.js";
import { buildCorpusExport } from "../corpus/storage.js";
import {
  exportFilePattern,
  readExportIndex,
  readLibrary,
  resolveOutputPath,
  streamZip,
  videoIdFromFilename,
} from "./outputFiles.js";

export function createApiRouter(input: {
  config: AppConfig;
  catalog: YtDlpCatalogService;
  worker: LocalWorker;
  runs: RunStore;
  logger: FileLogger;
}) {
  const router = Router();
  const { config, catalog, worker, runs } = input;
  const model = modelLabel(config);
  const requireRuntime = async () => {
    if (runs.busy) throw new HttpError(409, "Une analyse est déjà en cours.");
    const health = await worker.health();
    if (!health.ready) throw new HttpError(503, health.issues.join(" "));
  };
  router.get("/health", async (_req, res) =>
    res.json({
      status: "ok",
      local: true,
      model,
      models: { asr: config.asrModel, vision: config.visionModel },
      asrLanguage: config.asrLanguage,
      busy: runs.busy,
      fps: config.fps,
      groupFrames: config.groupFrames,
      runtime: await worker.health(),
    }),
  );
  const collections = createCollectionStore(config.outputDir);
  /** Every listed page enriches its collection, so the library knows the channel and its videos. */
  async function remember(page: CatalogPage, input: { offset: number; startDate?: string }) {
    // A date filter hides videos: such a page does not describe the channel's list.
    if (page.collection && !input.startDate)
      await collections.record(
        page.collection,
        page.videos,
        input.offset,
        page.limitReached ? null : page.nextOffset,
      );
    return page;
  }
  router.post("/catalog/channel", async (req, res) => {
    const input = ChannelCatalogRequestSchema.parse(req.body);
    res.json(await remember(await catalog.getChannelVideos(input), input));
  });
  router.post("/catalog/playlist", async (req, res) => {
    const input = PlaylistCatalogRequestSchema.parse(req.body);
    res.json(await remember(await catalog.getPlaylistVideos(input), input));
  });
  router.get("/collections/:id", async (req, res) => {
    const collection = await collections.get(String(req.params.id));
    if (!collection) throw new HttpError(404, "Chaîne ou playlist inconnue.");
    res.json(collection);
  });
  router.post("/collections/remove", async (req, res) => {
    const { id } = z.object({ id: z.string().min(1).max(140) }).parse(req.body);
    res.json({ removed: await collections.remove(id) });
  });
  router.post("/catalog/videos", async (req, res) =>
    res.json(await catalog.getDirectVideos(DirectVideosRequestSchema.parse(req.body).videoUrls)),
  );
  router.post("/runs", async (req, res) => {
    const body = AnalyzeRequestSchema.parse(req.body);
    if (body.model && body.model !== config.visionModel)
      throw new HttpError(400, "Le modèle se configure dans le fichier .env.");
    // A duplicate request returns its existing run, even if the GPU is busy with it.
    const existing = body.requestId ? await runs.read(body.requestId) : null;
    if (!existing) await requireRuntime();
    res.status(202).json(await runs.start(body.videos, body.requestId));
  });
  router.get("/runs", async (_req, res) =>
    res.json({
      runs: (await runs.list()).map((stored) => {
        const { results, ...run } = publicRun(stored);
        return {
          ...run,
          videos: [],
          results: [],
          successCount: results.filter((r) => r.status === "success").length,
          errorCount: results.filter((r) => r.status === "error").length,
        };
      }),
    }),
  );
  router.get("/runs/:id", async (req, res) => {
    const run = await runs.read(String(req.params.id));
    if (!run) throw new HttpError(404, "Analyse introuvable.");
    res.json(publicRun(run));
  });
  router.post("/runs/:id/pause", async (req, res) => res.json(await runs.pause(String(req.params.id))));
  router.post("/runs/:id/resume", async (req, res) => {
    await requireRuntime();
    res.status(202).json(await runs.resume(String(req.params.id)));
  });
  router.get("/runs/:id/logs", async (req, res) => {
    const id = String(req.params.id);
    if (!(await runs.read(id))) throw new HttpError(404, "Analyse introuvable.");
    const query = z
      .object({
        cursor: z.coerce.number().int().min(0).max(Number.MAX_SAFE_INTEGER).optional(),
        limit: z.coerce.number().int().min(1).max(500).default(250),
      })
      .parse(req.query);
    await input.logger.flush();
    const page = await readRunLog(config.logsDir, id, query.cursor, query.limit);
    // Lets the web journal decide between replaying a small journal from its start and reading its tail.
    const filename = await resolveRunLog(config.logsDir, id);
    const size = filename ? (await fs.stat(filename)).size : 0;
    res.json({ ...page, size, downloadUrl: `/api/runs/${id}/logs/download` });
  });
  router.get("/runs/:id/logs/download", async (req, res) => {
    const id = String(req.params.id);
    if (!(await runs.read(id))) throw new HttpError(404, "Analyse introuvable.");
    await input.logger.flush();
    const filename = await resolveRunLog(config.logsDir, id);
    if (!filename) throw new HttpError(404, "Ce traitement ne possède pas encore de journal.");
    await new Promise<void>((resolve, reject) =>
      res.download(filename, `run-${id}.jsonl`, { dotfiles: "allow" }, (error) =>
        error ? reject(error) : resolve(),
      ),
    );
  });
  router.get("/outputs/index", async (_req, res) =>
    res.json({ exports: await readExportIndex(config.outputDir) }),
  );
  router.get("/library", async (_req, res) =>
    res.json({
      ...(await readLibrary(config.outputDir, config.cacheDir)),
      collections: await collections.list(),
    }),
  );
  router.get("/outputs/:filename", async (req, res) => {
    const filename = String(req.params.filename);
    if (!exportFilePattern.test(filename)) throw new HttpError(400, "Nom de fichier invalide.");
    const root = await outputRoot(config.outputDir);
    const target = root && (await resolveOutputPath(root, filename));
    if (!target) throw new HttpError(404, "Fichier introuvable.");
    await new Promise<void>((resolve, reject) =>
      res.download(target, filename, { dotfiles: "allow" }, (error) => (error ? reject(error) : resolve())),
    );
  });
  router.post("/outputs/export-zip", async (req, res) => {
    const { filenames } = z
      .object({ filenames: z.array(z.string().min(1)).min(1).max(3000) })
      .parse(req.body);
    const root = await outputRoot(config.outputDir);
    if (!root) throw new HttpError(404, "Aucun fichier à télécharger.");
    await streamZip(root, filenames, res);
  });
  router.post("/outputs/delete", async (req, res) => {
    const { videoIds } = DeleteVideosRequestSchema.parse(req.body);
    const outcome = await runs.maintenance(async () => {
      const root = await outputRoot(config.outputDir);
      const wanted = new Set(videoIds);
      const deleted: string[] = [];
      const failed: Array<{ filename: string; videoId: string; error: string }> = [];
      if (root) {
        for (const entry of await fs.readdir(root, { withFileTypes: true })) {
          const videoId = videoIdFromFilename(entry.name);
          if (!entry.isFile() || !videoId || !wanted.has(videoId)) continue;
          try {
            await fs.rm(path.join(root, entry.name), { force: true });
            deleted.push(entry.name);
          } catch (error) {
            // A locked file (antivirus, open editor on Windows) must not stop the other deletions.
            const code = (error as NodeJS.ErrnoException).code;
            failed.push({ filename: entry.name, videoId, error: code ?? "Suppression impossible." });
            logEvent(
              "outputs.delete.failed",
              { message: "Suppression d’un export impossible.", filename: entry.name, ...errorFields(error) },
              "warn",
            );
          }
        }
      }
      // Only videos whose exports are all gone leave the run history; the others stay resumable.
      const incomplete = new Set(failed.map((item) => item.videoId));
      const prunedRuns = await runs.removeVideos(videoIds.filter((id) => !incomplete.has(id)));
      return { deleted, prunedRuns, failed };
    });
    if (!outcome.failed.length) return res.json(outcome);
    res.status(500).json({
      error:
        `${outcome.failed.length} fichier(s) n’ont pas pu être supprimé(s) : ` +
        outcome.failed.map((item) => item.filename).join(", ") +
        ". Fermez les programmes qui les utilisent puis réessayez.",
      ...outcome,
    });
  });
  router.get("/corpus/export", async (_req, res) => {
    let body: string;
    try {
      // The whole JSONL is built from one validated read of each document before the 200 headers:
      // a concurrent export write or deletion can neither truncate nor mix a "successful" download.
      body = await buildCorpusExport(config.outputDir);
    } catch (error) {
      throw new HttpError(
        (error as NodeJS.ErrnoException).code === "ENOENT" ? 404 : 422,
        "Aucun corpus valide disponible : " +
          (error instanceof Error ? error.message : "lancez une analyse."),
      );
    }
    if (!body) throw new HttpError(404, "Aucun segment à exporter.");
    res.attachment("corpus-v2.jsonl").type("application/x-ndjson").send(body);
  });
  return router;
}
async function outputRoot(directory: string): Promise<string | null> {
  try {
    return await fs.realpath(directory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

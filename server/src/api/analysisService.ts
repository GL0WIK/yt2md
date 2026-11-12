import fs from "node:fs/promises";
import path from "node:path";
import type { AppConfig } from "../config.js";
import type { YtDlpCatalogService } from "../catalog/ytDlp.js";
import {
  CatalogVideoSchema,
  type CatalogVideo,
  type AnalysisProgress,
  type AnalyzeVideoResult,
} from "../shared/schemas.js";
import { atomicWrite } from "../shared/files.js";
import { buildCorpusDocument } from "../corpus/document.js";
import { writeCorpusDocument } from "../corpus/storage.js";
import { renderVideoAnalysisMarkdown } from "../markdown/render.js";
import { writeMarkdownOutput } from "../markdown/write.js";
import { validateAndCanonicalizeAnalysis } from "../validation/analysis.js";
import type { LocalWorker } from "../local/worker.js";
import { logEvent, errorFields, observeStage, withLogContext } from "../shared/logging.js";
import { YOUTUBE_BLOCKED_MESSAGE, isYoutubeBlocked } from "../shared/errors.js";

export type AnalysisService = ReturnType<typeof createAnalysisService>;
export function createAnalysisService(config: AppConfig, catalog: YtDlpCatalogService, worker: LocalWorker) {
  let busy = false;
  return {
    get busy() {
      return busy;
    },
    async analyzeVideos(
      videos: CatalogVideo[],
      runId: string,
      signal: AbortSignal,
      callbacks: {
        onProgress: (progress: AnalysisProgress) => void;
        onResult: (result: AnalyzeVideoResult) => Promise<void>;
        /** YouTube refuses this machine: the run must pause and keep this video pending. */
        onBlocked?: (message: string) => void;
      },
    ) {
      if (busy) throw new Error("Une analyse locale est déjà en cours.");
      busy = true;
      try {
        // Sequential GPU ownership also prevents two runs from invalidating each other's caches.
        for (const requested of videos) {
          if (signal.aborted) break;
          await withLogContext({ runId, videoId: requested.id }, async () => {
            const started = performance.now();
            logEvent("local.video.start", { message: "Traitement de la vidéo.", title: requested.title });
            const progress = (
              phase: AnalysisProgress["phase"],
              chunk = 0,
              totalChunks = 1,
              detail?: string,
            ) =>
              callbacks.onProgress({
                videoId: requested.id,
                videoTitle: requested.title,
                phase,
                chunk,
                totalChunks,
                detail,
              });
            let result: AnalyzeVideoResult;
            // Set as soon as every export file is on disk: a later pause must still record it,
            // otherwise resume would regenerate a video that is already complete.
            let exported: AnalyzeVideoResult | null = null;
            try {
              progress("metadata");
              const metadataFile = path.join(config.cacheDir, requested.id, "metadata.json");
              let video: CatalogVideo | undefined;
              try {
                const cached = CatalogVideoSchema.parse(JSON.parse(await fs.readFile(metadataFile, "utf8")));
                if (cached.id === requested.id && cached.url === requested.url) {
                  video = cached;
                  logEvent("local.metadata.cache_hit", { message: "Métadonnées retrouvées dans le cache." });
                }
              } catch (error) {
                if (
                  (error as NodeJS.ErrnoException).code !== "ENOENT" &&
                  error instanceof Error &&
                  error.name !== "SyntaxError" &&
                  error.name !== "ZodError"
                )
                  throw error;
              }
              if (!video) {
                video = await observeStage(
                  "local.metadata",
                  { message: "Lecture des métadonnées YouTube avec yt-dlp." },
                  () => catalog.getVideoMetadata(requested),
                );
                await atomicWrite(metadataFile, JSON.stringify(video));
              }
              signal.throwIfAborted();
              const raw = await worker.analyze(video, signal, (event) => {
                callbacks.onProgress({ ...event, videoId: video!.id, videoTitle: video!.title });
              });
              const warnings = [...raw.warnings];
              // The catalog rounds down and the media probe rounds up: only a real gap is reported.
              if (video.durationSeconds !== null && Math.abs(video.durationSeconds - raw.durationSeconds) > 2)
                warnings.push(
                  "La durée a été vérifiée sur le média téléchargé et remplace celle du catalogue YouTube.",
                );
              video = { ...video, durationSeconds: raw.durationSeconds };
              const settings = {
                model: raw.model,
                fps: config.fps,
                mediaResolution: `video_${config.fps}fps_${config.groupFrames}frames_max${config.frameMaxSide}_${config.visionMaxPixels}px`,
              };
              const checked = validateAndCanonicalizeAnalysis(
                { transcript: raw.transcript, visualSegments: raw.visualSegments },
                video,
                settings,
              );
              if (!checked.ok) throw new Error("Analyse locale invalide : " + checked.issues.join("; "));
              logEvent("local.validation.complete", {
                message: "Transcription et observations validées avant export.",
                utterances: raw.transcript.length,
                visionGroups: raw.visualSegments.length,
                warningCount: warnings.length + checked.warnings.length,
              });
              progress("export");
              logEvent("local.export.start", {
                message: "Écriture du JSON original, de la provenance locale et du Markdown.",
              });
              const document = buildCorpusDocument(checked.analysis, {
                videoId: video.id,
                runId,
                promptVersion: raw.promptVersion,
                chunkSeconds: raw.chunkSeconds,
                warnings: [...warnings, ...checked.warnings],
              });
              const jsonPath = await writeCorpusDocument(config.outputDir, document);
              const provenanceFile = `video--${video.id}.local.json`;
              await atomicWrite(
                path.join(config.outputDir, provenanceFile),
                JSON.stringify(
                  {
                    ...raw.localProvenance,
                    videoId: video.id,
                    runId,
                    contentHash: document.contentHash,
                    generatedAt: document.generatedAt,
                    legacyCompatibility: {
                      schemaVersion: "2.0",
                      transcriptionSource: "gemini_video",
                      explanation:
                        "Constante du schéma d’origine conservée pour compatibilité ; aucun appel Gemini. La source effective figure ci-dessus.",
                    },
                  },
                  null,
                  2,
                ) + "\n",
              );
              const markdownPath = await writeMarkdownOutput({
                outputDir: config.outputDir,
                video,
                markdown: renderVideoAnalysisMarkdown(checked.analysis),
              });
              logEvent("local.export.complete", {
                message: "Exports écrits.",
                jsonFile: path.basename(jsonPath),
                markdownFile: path.basename(markdownPath),
                provenanceFile,
              });
              for (const warning of document.quality.warnings)
                logEvent("local.video.warning", { message: warning }, "warn");
              result = {
                status: "success",
                videoId: video.id,
                title: video.title,
                jsonPath,
                markdownPath,
                jsonDownloadUrl: "/api/outputs/" + path.basename(jsonPath),
                downloadUrl: "/api/outputs/" + path.basename(markdownPath),
                provenanceUrl: "/api/outputs/" + provenanceFile,
                usage: raw.usage,
                warnings: document.quality.warnings,
              };
              exported = result;
              if (!signal.aborted) {
                try {
                  await worker.cleanup(video.id);
                } catch (error) {
                  logEvent(
                    "local.cleanup.failed",
                    {
                      message: error instanceof Error ? error.message : "Nettoyage des médias impossible.",
                      ...errorFields(error),
                    },
                    "warn",
                  );
                  result.warnings.push(
                    "Les médias temporaires sont conservés ; le prochain passage pourra les nettoyer.",
                  );
                }
              }
            } catch (error) {
              if (exported) {
                logEvent(
                  "local.video.post_export_failed",
                  {
                    message: "Exports écrits ; une étape finale a échoué et sera ignorée.",
                    ...errorFields(error),
                  },
                  "warn",
                );
                result = exported;
              } else if (signal.aborted) {
                logEvent("local.video.interrupted", {
                  message: "Vidéo interrompue ; les morceaux terminés sont conservés.",
                  durationMs: Math.round(performance.now() - started),
                });
                return;
              } else if (callbacks.onBlocked && error instanceof Error && isYoutubeBlocked(error.message)) {
                logEvent(
                  "local.youtube.blocked",
                  { message: YOUTUBE_BLOCKED_MESSAGE, detail: error.message.slice(-600) },
                  "error",
                );
                callbacks.onBlocked(
                  YOUTUBE_BLOCKED_MESSAGE +
                    " Le traitement est en pause : « Reprendre » relancera les vidéos restantes.",
                );
                return;
              } else {
                logEvent(
                  "local.video.failed",
                  {
                    message: error instanceof Error ? error.message : "Échec de la vidéo.",
                    durationMs: Math.round(performance.now() - started),
                    ...errorFields(error),
                  },
                  "error",
                );
                result = {
                  status: "error",
                  videoId: requested.id,
                  title: requested.title,
                  error: error instanceof Error ? error.message : "Échec de l’analyse locale.",
                };
              }
            }
            if (result.status === "success")
              logEvent("local.video.complete", {
                message: "Vidéo terminée.",
                durationMs: Math.round(performance.now() - started),
                cacheHits: result.usage.cacheHits,
                promptTokens: result.usage.promptTokenCount,
                outputTokens: result.usage.candidatesTokenCount,
              });
            await callbacks.onResult(result);
          });
        }
      } finally {
        busy = false;
      }
    },
  };
}

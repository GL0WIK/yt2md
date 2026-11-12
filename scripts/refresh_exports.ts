/**
 * Re-applies the current export rules to videos already in outputs/, without any model call:
 * drops "no visible text" from the observation limits and the two warnings that were noise
 * (one-language audio track never marked "original", 1-second catalogue rounding).
 * Usage: npm run exports:refresh [-- --dry-run]
 */
import fs from "node:fs/promises";
import path from "node:path";
import { getConfig } from "../server/src/config.js";
import { assessQuality, buildCorpusDocument, CorpusDocumentSchema } from "../server/src/corpus/document.js";
import { writeCorpusDocument } from "../server/src/corpus/storage.js";
import { renderVideoAnalysisMarkdown } from "../server/src/markdown/render.js";
import { writeMarkdownOutput } from "../server/src/markdown/write.js";
import { atomicWrite } from "../server/src/shared/files.js";
import type { VideoAnalysis } from "../server/src/shared/schemas.js";
import { observationLimits } from "../server/src/validation/analysis.js";
import { createHash } from "node:crypto";

const AUDIO_WARNING = /n’est pas explicitement marquée « original »/;
const DURATION_WARNING = /remplace celle du catalogue YouTube/;
const dryRun = process.argv.includes("--dry-run");
const config = getConfig();

async function readJson(file: string): Promise<Record<string, unknown> | null> {
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/** Only drops a warning that the cache proves was noise; without its cache, a warning is kept. */
async function noise(videoId: string, durationSeconds: number | null) {
  const cache = path.join(config.cacheDir, videoId);
  const receipt = await readJson(path.join(cache, "source.json"));
  const formats = Array.isArray(receipt?.audioFormats)
    ? (receipt.audioFormats as Array<{ format_id?: string }>)
    : null;
  // yt-dlp suffixes audio formats with "-N" only when YouTube offers several languages.
  const singleLanguage = formats !== null && formats.every((f) => !/-\d+$/.test(String(f.format_id ?? "")));
  const catalog = await readJson(path.join(cache, "metadata.json"));
  const listed = typeof catalog?.durationSeconds === "number" ? catalog.durationSeconds : null;
  const rounding = listed !== null && durationSeconds !== null && Math.abs(listed - durationSeconds) <= 2;
  return (warning: string) =>
    (singleLanguage && AUDIO_WARNING.test(warning)) || (rounding && DURATION_WARNING.test(warning));
}

let changed = 0;
let removedLimits = 0;
let removedWarnings = 0;
const names = (await fs.readdir(config.outputDir)).filter((name) =>
  /^video--[A-Za-z0-9_-]{11}\.json$/.test(name),
);
for (const name of names) {
  const file = path.join(config.outputDir, name);
  const document = CorpusDocumentSchema.parse(JSON.parse(await fs.readFile(file, "utf8")));
  const analysis: VideoAnalysis = {
    metadata: document.source,
    publicStats: document.publicStats,
    youtubeDescription: document.youtubeDescription,
    analysisSettings: {
      model: document.provenance.model,
      fps: document.provenance.fps,
      mediaResolution: document.provenance.mediaResolution,
    },
    transcript: document.transcript.map(
      ({ id: _id, startSeconds: _start, endSeconds: _end, speakerId: _speaker, ...entry }) => entry,
    ),
    visualSegments: document.visualSegments.map(
      ({ id: _id, startSeconds: _start, endSeconds: _end, ...segment }) => segment,
    ),
  };
  if (createHash("sha256").update(JSON.stringify(analysis)).digest("hex") !== document.contentHash) {
    console.warn(`${document.videoId} : contenu reconstruit différent de l’export, ignoré.`);
    continue;
  }
  const refreshed: VideoAnalysis = {
    ...analysis,
    visualSegments: analysis.visualSegments.map((segment) =>
      segment.uncertainties === undefined
        ? segment
        : { ...segment, uncertainties: observationLimits(segment.uncertainties) as string[] },
    ),
  };
  // Warnings computed from the content are recomputed; the others come from the pipeline.
  const computed = new Set(assessQuality(analysis).warnings);
  const isNoise = await noise(document.videoId, document.source.durationSeconds);
  const pipeline = document.quality.warnings.filter((warning) => !computed.has(warning));
  const kept = pipeline.filter((warning) => !isNoise(warning));
  const next = buildCorpusDocument(refreshed, {
    videoId: document.videoId,
    runId: document.provenance.runId,
    generatedAt: document.generatedAt,
    promptVersion: document.provenance.promptVersion,
    chunkSeconds: document.provenance.chunkSeconds ?? undefined,
    warnings: kept,
  });
  if (JSON.stringify(next) === JSON.stringify(document)) continue;
  changed++;
  removedLimits += document.quality.visualUncertainties - next.quality.visualUncertainties;
  removedWarnings += pipeline.length - kept.length;
  console.log(
    `${document.videoId} : ${document.quality.visualUncertainties} → ${next.quality.visualUncertainties} limites, ` +
      `${document.quality.warnings.length} → ${next.quality.warnings.length} avertissements`,
  );
  if (dryRun) continue;
  await writeCorpusDocument(config.outputDir, next);
  await writeMarkdownOutput({
    outputDir: config.outputDir,
    video: { id: document.videoId } as Parameters<typeof writeMarkdownOutput>[0]["video"],
    markdown: renderVideoAnalysisMarkdown(refreshed),
  });
  const provenanceFile = path.join(config.outputDir, `video--${document.videoId}.local.json`);
  const provenance = await readJson(provenanceFile);
  if (provenance)
    await atomicWrite(
      provenanceFile,
      JSON.stringify({ ...provenance, contentHash: next.contentHash }, null, 2) + "\n",
    );
}
console.log(
  `${changed} export(s) sur ${names.length} ${dryRun ? "à mettre à jour" : "mis à jour"} : ` +
    `${removedLimits} « aucun texte visible » et ${removedWarnings} avertissement(s) sans objet retirés.`,
);

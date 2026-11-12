import { createHash } from "node:crypto";
import { z } from "zod";
import {
  AnalysisMetadataSchema,
  PublicStatsSchema,
  TranscriptEntrySchema,
  VisualSegmentSchema,
  VideoAnalysis,
} from "../shared/schemas.js";
import { parseTimestampToSeconds } from "../shared/time.js";

export const CORPUS_VERSION = "2.0";
const seconds = z.number().int().nonnegative();
export const CorpusDocumentSchema = z.object({
  schemaVersion: z.literal(CORPUS_VERSION),
  videoId: z.string().regex(/^[A-Za-z0-9_-]{11}$/),
  contentHash: z.string().regex(/^[a-f0-9]{64}$/),
  generatedAt: z.string().datetime(),
  provenance: z.object({
    runId: z.string(),
    model: z.string(),
    promptVersion: z.string(),
    fps: z.number().positive(),
    mediaResolution: z.string(),
    chunkSeconds: z.number().positive().nullable(),
    transcriptionSource: z.literal("gemini_video"),
    timestampAccuracy: z.literal("model_estimated_seconds"),
    speakerScope: z.literal("chunk"),
  }),
  source: AnalysisMetadataSchema,
  publicStats: PublicStatsSchema,
  youtubeDescription: z.string(),
  transcript: z.array(
    TranscriptEntrySchema.extend({
      id: z.string(),
      startSeconds: seconds,
      endSeconds: seconds.nullable(),
      speakerId: z.string().nullable(),
    }),
  ),
  visualSegments: z.array(
    VisualSegmentSchema.extend({
      id: z.string(),
      startSeconds: seconds,
      endSeconds: seconds,
    }),
  ),
  quality: z.object({
    reviewRequired: z.boolean(),
    warnings: z.array(z.string()),
    visualCoverageRatio: z.number().min(0).max(1).nullable(),
    visualGaps: z.array(z.object({ startSeconds: seconds, endSeconds: seconds })),
    inaudibleEntries: z.number().int().nonnegative(),
    missingSpeechEndTimes: z.number().int().nonnegative(),
    visualUncertainties: z.number().int().nonnegative(),
  }),
});
export type CorpusDocument = z.infer<typeof CorpusDocumentSchema>;

export function assessQuality(analysis: VideoAnalysis, warnings: string[] = []) {
  const duration = analysis.metadata.durationSeconds;
  const gaps: Array<{ startSeconds: number; endSeconds: number }> = [];
  let cursor = 0,
    covered = 0;
  for (const segment of analysis.visualSegments) {
    const start = parseTimestampToSeconds(segment.timestamp)!;
    const end = parseTimestampToSeconds(segment.endTimestamp)!;
    if (start > cursor) gaps.push({ startSeconds: cursor, endSeconds: start });
    covered += Math.max(0, end - Math.max(cursor, start));
    cursor = Math.max(cursor, end);
  }
  if (duration !== null && cursor < duration) gaps.push({ startSeconds: cursor, endSeconds: duration });
  const messages = [...warnings];
  if (gaps.length)
    messages.push(
      "Couverture visuelle : " +
        gaps.length +
        " intervalle(s) sans observation. Consulter visualGaps dans le JSON.",
    );
  const missingSpeechEndTimes = analysis.transcript.filter((entry) => !entry.endTimestamp).length;
  if (missingSpeechEndTimes)
    messages.push("Certaines paroles n’ont pas d’horodatage de fin ; leur durée reste inconnue.");

  const pointSpeech = analysis.transcript.filter(
    (entry) =>
      entry.endTimestamp !== undefined &&
      parseTimestampToSeconds(entry.timestamp) === parseTimestampToSeconds(entry.endTimestamp),
  ).length;
  if (pointSpeech)
    messages.push(
      pointSpeech +
        " parole(s) ont un début et une fin identiques à la seconde près ; durée précise inconnue, horodatages conservés sans allongement.",
    );
  const inaudibleEntries = analysis.transcript.filter((entry) => /\[inaudible\]/i.test(entry.text)).length;
  if (inaudibleEntries) messages.push(inaudibleEntries + " passage(s) contiennent des paroles inaudibles.");
  const visualUncertainties = analysis.visualSegments.reduce(
    (total, segment) => total + (segment.uncertainties?.length ?? 0),
    0,
  );
  if (visualUncertainties)
    messages.push(visualUncertainties + " limite(s) d’observation signalée(s) dans les segments visuels.");
  return {
    reviewRequired: messages.length > 0,
    warnings: [...new Set(messages)],
    visualCoverageRatio:
      duration && duration > 0 ? Math.min(1, Math.round((covered / duration) * 10000) / 10000) : null,
    visualGaps: gaps,
    inaudibleEntries,
    missingSpeechEndTimes,
    visualUncertainties,
  };
}

export function buildCorpusDocument(
  analysis: VideoAnalysis,
  options: {
    videoId: string;
    runId: string;
    generatedAt?: string;
    promptVersion: string;
    chunkSeconds?: number;
    warnings?: string[];
  },
): CorpusDocument {
  const voices = new Map<string, string>();
  const transcript = analysis.transcript.map((entry, index) => {
    const scope = entry.chunkIndex ?? 1;
    const key = scope + ":" + entry.speaker;
    if (!voices.has(key)) voices.set(key, options.videoId + ":c" + scope + ":voice" + (voices.size + 1));
    return {
      ...entry,
      id: options.videoId + ":t" + String(index + 1).padStart(5, "0"),
      startSeconds: parseTimestampToSeconds(entry.timestamp),
      endSeconds: entry.endTimestamp ? parseTimestampToSeconds(entry.endTimestamp) : null,
      speakerId: !entry.chunkIndex || /indéterminé|unknown/i.test(entry.speaker) ? null : voices.get(key),
    };
  });
  return CorpusDocumentSchema.parse({
    schemaVersion: CORPUS_VERSION,
    videoId: options.videoId,
    contentHash: createHash("sha256").update(JSON.stringify(analysis)).digest("hex"),
    generatedAt: options.generatedAt ?? new Date().toISOString(),
    provenance: {
      runId: options.runId,
      model: analysis.analysisSettings.model,
      promptVersion: options.promptVersion,
      fps: analysis.analysisSettings.fps,
      mediaResolution: analysis.analysisSettings.mediaResolution,
      chunkSeconds: options.chunkSeconds ?? null,
      transcriptionSource: "gemini_video",
      timestampAccuracy: "model_estimated_seconds",
      speakerScope: "chunk",
    },
    source: analysis.metadata,
    publicStats: analysis.publicStats,
    youtubeDescription: analysis.youtubeDescription,
    transcript,
    visualSegments: analysis.visualSegments.map((segment, index) => ({
      ...segment,
      id: options.videoId + ":v" + String(index + 1).padStart(5, "0"),
      startSeconds: parseTimestampToSeconds(segment.timestamp),
      endSeconds: parseTimestampToSeconds(segment.endTimestamp),
    })),
    quality: assessQuality(analysis, options.warnings),
  });
}

/** Deterministic temporal alignment, no extra model calls or paraphrasing. */
export function* buildRetrievalChunks(document: CorpusDocument, windowSeconds = 60) {
  if (!Number.isSafeInteger(windowSeconds) || windowSeconds < 1) throw new Error("Invalid retrieval window.");
  const duration =
    document.source.durationSeconds ??
    Math.max(
      0,
      ...document.visualSegments.map((s) => s.endSeconds),
      ...document.transcript.map((t) => Math.max(t.endSeconds ?? t.startSeconds, t.startSeconds + 1)),
    );
  for (let start = 0; start < duration; start += windowSeconds) {
    const end = Math.min(duration, start + windowSeconds);
    const transcript = document.transcript.filter(
      (entry) =>
        entry.startSeconds < end &&
        (entry.endSeconds === null || entry.endSeconds === entry.startSeconds
          ? entry.startSeconds >= start
          : entry.endSeconds > start),
    );
    const visuals = document.visualSegments.filter(
      (entry) => entry.startSeconds < end && entry.endSeconds > start,
    );
    if (!transcript.length && !visuals.length) continue;
    const text = [
      "Titre : " + document.source.title,
      ...transcript.map((entry) => "Paroles [" + entry.timestamp + "] " + entry.speaker + " : " + entry.text),
      ...visuals.flatMap((entry) => [
        "Visuel [" +
          entry.timestamp +
          "–" +
          entry.endTimestamp +
          "] " +
          entry.title +
          " : " +
          entry.visualDescription,
        ...entry.screenText.map((text) => "Texte à l’écran : " + text),
        ...(entry.uncertainties ?? []).map((text) => "Limite d’observation : " + text),
      ]),
    ].join("\n");
    yield {
      schemaVersion: CORPUS_VERSION,
      id: document.videoId + ":w" + start,
      videoId: document.videoId,
      contentHash: document.contentHash,
      startSeconds: start,
      endSeconds: end,
      url: document.source.url + "&t=" + start + "s",
      transcriptIds: transcript.map((entry) => entry.id),
      visualSegmentIds: visuals.map((entry) => entry.id),
      reviewRequired: document.quality.reviewRequired,
      text,
    };
  }
}

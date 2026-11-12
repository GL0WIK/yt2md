import { AnalysisSettings, CatalogVideo, VideoAnalysis, VideoAnalysisSchema } from "../shared/schemas.js";
import { formatSecondsAsTimestamp, parseTimestampToSeconds } from "../shared/time.js";

const MAX_VISUAL_SEGMENT_GAP_SECONDS = 15;

export type AnalysisValidationResult =
  | {
      ok: true;
      analysis: VideoAnalysis;
      warnings: string[];
    }
  | {
      ok: false;
      issues: string[];
    };

export function validateAndCanonicalizeAnalysis(
  value: unknown,
  video: CatalogVideo,
  settings: AnalysisSettings,
): AnalysisValidationResult {
  const parsed = VideoAnalysisSchema.safeParse(normalizeModelAnalysis(value, video, settings));
  if (!parsed.success) {
    return {
      ok: false,
      issues: parsed.error.issues.map((issue) => `${issue.path.join(".") || "root"}: ${issue.message}`),
    };
  }

  const warnings: string[] = [];
  if (parsed.data.metadata.url !== video.url) {
    warnings.push("Local metadata.url differed from yt-dlp URL and was canonicalized.");
  }
  if (parsed.data.metadata.title !== video.title) {
    warnings.push("Local metadata.title differed from yt-dlp title and was canonicalized.");
  }

  const canonical: VideoAnalysis = {
    ...parsed.data,
    metadata: {
      title: video.title,
      url: video.url,
      channel: video.channel,
      publicationDate: video.publishedAt,
      durationSeconds: video.durationSeconds,
    },
    publicStats: video.publicStats,
    youtubeDescription: video.description,
    analysisSettings: settings,
    visualSegments: parsed.data.visualSegments,
  };

  const businessValidation = validateBusinessRules(canonical);
  if (businessValidation.issues.length > 0) {
    return {
      ok: false,
      issues: businessValidation.issues,
    };
  }
  warnings.push(...businessValidation.warnings);

  return {
    ok: true,
    analysis: canonical,
    warnings,
  };
}

function normalizeModelAnalysis(value: unknown, video: CatalogVideo, settings: AnalysisSettings): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const input = value as Record<string, unknown>;

  return {
    metadata: normalizeMetadata(input.metadata, video),
    publicStats: normalizePublicStats(input.publicStats, video),
    youtubeDescription:
      typeof input.youtubeDescription === "string" ? input.youtubeDescription : video.description,
    analysisSettings: normalizeAnalysisSettings(input.analysisSettings, settings),
    transcript: normalizeTranscriptEntries(input.transcript),
    visualSegments: normalizeVisualSegments(input.visualSegments),
  };
}

function normalizeMetadata(value: unknown, video: CatalogVideo): VideoAnalysis["metadata"] {
  const metadata =
    value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  return {
    title: stringValue(metadata.title) ?? video.title,
    url: stringValue(metadata.url) ?? video.url,
    channel: stringValue(metadata.channel) ?? video.channel,
    publicationDate: stringValue(metadata.publicationDate) ?? video.publishedAt,
    durationSeconds: numberValue(metadata.durationSeconds) ?? video.durationSeconds,
  };
}

function normalizePublicStats(value: unknown, video: CatalogVideo): VideoAnalysis["publicStats"] {
  const publicStats =
    value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  return {
    views: numberValue(publicStats.views) ?? video.publicStats.views,
    likes: numberValue(publicStats.likes) ?? video.publicStats.likes,
    comments: numberValue(publicStats.comments) ?? video.publicStats.comments,
  };
}

function normalizeAnalysisSettings(value: unknown, settings: AnalysisSettings): AnalysisSettings {
  const analysisSettings =
    value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  return {
    model: stringValue(analysisSettings.model) ?? settings.model,
    fps: numberValue(analysisSettings.fps) ?? settings.fps,
    mediaResolution: stringValue(analysisSettings.mediaResolution) ?? settings.mediaResolution,
  };
}

function normalizeTranscriptEntries(value: unknown): unknown[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  return value.map(normalizeTranscriptEntry);
}

function normalizeTranscriptEntry(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const entry = value as Record<string, unknown>;
  return {
    speaker: stringValue(entry.speaker),
    timestamp: normalizeTimestamp(entry.timestamp),
    text: stringValue(entry.text),
    ...(entry.endTimestamp !== undefined
      ? { endTimestamp: normalizeTimestamp(entry.endTimestamp) ?? entry.endTimestamp }
      : {}),
    ...(entry.chunkIndex !== undefined ? { chunkIndex: entry.chunkIndex } : {}),
  };
}

function normalizeVisualSegments(value: unknown): unknown[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  return value.map(normalizeVisualSegment);
}

function normalizeVisualSegment(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const entry = value as Record<string, unknown>;
  const timestamp = normalizeTimestamp(entry.timestamp);
  const screenText = Array.isArray(entry.screenText)
    ? entry.screenText.filter((item): item is string => typeof item === "string")
    : undefined;

  return {
    timestamp,
    endTimestamp: normalizeTimestamp(entry.endTimestamp),
    title: stringValue(entry.title),
    visualDescription: stringValue(entry.visualDescription),
    screenText,
    ...(entry.uncertainties !== undefined ? { uncertainties: observationLimits(entry.uncertainties) } : {}),
    ...(entry.chunkIndex !== undefined ? { chunkIndex: entry.chunkIndex } : {}),
  };
}

// "No visible text" is an observation, not a limit: counting it would flag almost every video.
const NO_TEXT = /^(?:il n['’]y a )?aucun(?:e)? (?:texte|inscription)\b.*\b(?:visible|lisible)s?\b/i;
export function observationLimits(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  return value.filter(
    (item) => typeof item !== "string" || !NO_TEXT.test(item.trim()) || /partiel|au-delà/i.test(item),
  );
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }

  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function validateBusinessRules(analysis: VideoAnalysis): { issues: string[]; warnings: string[] } {
  const issues: string[] = [];
  const warnings: string[] = [];

  if (analysis.transcript.length === 0) {
    warnings.push("Aucune parole détectée. Vérifiez que la vidéo ne contient pas de discours omis.");
  }

  let previousTranscriptSeconds = -1;
  for (const [index, entry] of analysis.transcript.entries()) {
    const seconds = parseTimestampToSeconds(entry.timestamp);
    if (seconds === null) {
      issues.push(`transcript.${index}.timestamp is not a valid MM:SS or HH:MM:SS timestamp`);
      continue;
    }

    if (seconds < previousTranscriptSeconds) {
      issues.push(`transcript.${index}.timestamp is earlier than the previous transcript entry`);
    }
    previousTranscriptSeconds = seconds;
    if (entry.endTimestamp !== undefined) {
      const end = parseTimestampToSeconds(entry.endTimestamp);
      if (
        end === null ||
        end < seconds ||
        (analysis.metadata.durationSeconds !== null && end > analysis.metadata.durationSeconds)
      )
        issues.push(
          "transcript." + index + ".endTimestamp must not precede start and must be within the video",
        );
    }

    if (analysis.metadata.durationSeconds !== null && seconds > analysis.metadata.durationSeconds) {
      issues.push(`transcript.${index}.timestamp exceeds the known video duration`);
    }
  }

  if (analysis.visualSegments.length === 0) {
    issues.push("visualSegments must not be empty");
  }

  const lastVisualSeconds = validateSegments(
    "visualSegments",
    analysis.visualSegments,
    analysis.metadata.durationSeconds,
    issues,
    warnings,
  );

  if (analysis.metadata.durationSeconds !== null && analysis.metadata.durationSeconds >= 120) {
    validateCoverage("visualSegments", lastVisualSeconds, analysis.metadata.durationSeconds, warnings);
  }

  return { issues, warnings };
}

function validateSegments(
  label: "visualSegments",
  segments: Array<{ timestamp: string; endTimestamp: string }>,
  durationSeconds: number | null,
  issues: string[],
  warnings: string[],
): number | null {
  let previousSeconds = -1;
  let previousEndSeconds: number | null = null;
  let lastValidEndSeconds: number | null = null;

  for (const [index, entry] of segments.entries()) {
    const seconds = parseTimestampToSeconds(entry.timestamp);
    const endSeconds = parseTimestampToSeconds(entry.endTimestamp);
    if (seconds === null) {
      issues.push(`${label}.${index}.timestamp is not a valid MM:SS or HH:MM:SS timestamp`);
      continue;
    }

    if (endSeconds === null) {
      issues.push(`${label}.${index}.endTimestamp is not a valid MM:SS or HH:MM:SS timestamp`);
    } else if (endSeconds <= seconds) {
      issues.push(`${label}.${index}.endTimestamp must be after timestamp`);
    }

    if (seconds < previousSeconds) {
      issues.push(`${label}.${index}.timestamp is earlier than the previous segment`);
    }
    if (previousEndSeconds !== null && seconds - previousEndSeconds > MAX_VISUAL_SEGMENT_GAP_SECONDS) {
      warnings.push(
        `${label}.${index} starts too far after the previous segment: ${seconds - previousEndSeconds}s gap exceeds ${MAX_VISUAL_SEGMENT_GAP_SECONDS}s`,
      );
    }
    if (previousEndSeconds !== null && seconds < previousEndSeconds)
      issues.push(`${label}.${index} overlaps the previous segment`);
    if (index === 0 && seconds > MAX_VISUAL_SEGMENT_GAP_SECONDS)
      warnings.push("Le début de la vidéo n’est pas couvert par les segments visuels.");
    previousSeconds = seconds;
    lastValidEndSeconds = endSeconds ?? seconds;
    previousEndSeconds = endSeconds ?? seconds;

    if (durationSeconds !== null && seconds > durationSeconds) {
      issues.push(`${label}.${index}.timestamp exceeds the known video duration`);
    }
    if (durationSeconds !== null && endSeconds !== null && endSeconds > durationSeconds) {
      issues.push(`${label}.${index}.endTimestamp exceeds the known video duration`);
    }
  }

  return lastValidEndSeconds;
}

function validateCoverage(
  label: string,
  lastValidSeconds: number | null,
  durationSeconds: number,
  warnings: string[],
): void {
  if (lastValidSeconds === null) {
    return;
  }

  const requiredCoverageSeconds = Math.max(0, durationSeconds - Math.max(60, durationSeconds * 0.15));
  if (lastValidSeconds < requiredCoverageSeconds) {
    warnings.push(
      `${label} coverage is incomplete: last timestamp ${lastValidSeconds}s is too far from known duration ${durationSeconds}s`,
    );
  }
}

function normalizeTimestamp(value: unknown): string | undefined {
  const text = stringValue(value);
  if (!text) return undefined;
  const seconds = parseTimestampToSeconds(text);
  return seconds === null ? text : formatSecondsAsTimestamp(seconds);
}

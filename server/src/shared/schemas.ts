import { z } from "zod";
import { isIsoDate, normalizeChannelUrl, normalizePlaylistUrl, normalizeVideoUrl } from "./youtube.js";

const youtubeUrl = (normalize: (value: string) => string) =>
  z
    .string()
    .trim()
    .max(2048)
    .transform((value, ctx) => {
      try {
        return normalize(value);
      } catch (error) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: (error as Error).message });
        return z.NEVER;
      }
    });
const IsoDateSchema = z.string().refine(isIsoDate, "Date invalide, format attendu : AAAA-MM-JJ.");

const NullablePublicCountSchema = z.number().int().nonnegative().nullable();

export const PublicStatsSchema = z.object({
  views: NullablePublicCountSchema,
  likes: NullablePublicCountSchema,
  comments: NullablePublicCountSchema,
});

export const VideoIdSchema = z.string().regex(/^[A-Za-z0-9_-]{11}$/);

export const CatalogVideoSchema = z
  .object({
    id: VideoIdSchema,
    title: z.string().trim().min(1).max(1000),
    url: youtubeUrl(normalizeVideoUrl),
    channel: z.string().nullable(),
    /** YouTube channel identifier (UC…) and page: they group the library by channel. */
    channelId: z.string().max(100).nullable().optional(),
    channelUrl: z.string().url().max(500).nullable().optional(),
    language: z.string().max(35).nullable().optional(),
    publishedAt: IsoDateSchema.nullable(),
    durationSeconds: z.number().int().nonnegative().nullable(),
    description: z.string().max(50000),
    publicStats: PublicStatsSchema,
    source: z.enum(["channel", "playlist", "direct"]).optional(),
    metadataStatus: z.enum(["preview", "complete"]).optional(),
  })
  .refine(
    (video) => video.url === `https://www.youtube.com/watch?v=${video.id}`,
    "L’identifiant et l’URL de la vidéo ne correspondent pas.",
  );

export const AnalysisSettingsSchema = z.object({
  model: z.string().min(1),
  fps: z.number().positive(),
  mediaResolution: z.string().min(1),
});

export const AnalysisMetadataSchema = z.object({
  title: z.string().min(1),
  url: z.string().url(),
  channel: z.string().nullable(),
  publicationDate: z.string().nullable(),
  durationSeconds: z.number().int().nonnegative().nullable(),
});

export const UtteranceSchema = z.object({
  speaker: z.string().min(1),
  timestamp: z.string().min(1),
  text: z.string().min(1),
  endTimestamp: z.string().min(1).optional(),
  chunkIndex: z.number().int().positive().optional(),
});

export const TranscriptEntrySchema = UtteranceSchema;

export const VisualSegmentSchema = z.object({
  timestamp: z.string().min(1),
  endTimestamp: z.string().min(1),
  title: z.string().min(1),
  visualDescription: z.string().min(1),
  screenText: z.array(z.string()),
  uncertainties: z.array(z.string()).optional(),
  chunkIndex: z.number().int().positive().optional(),
});

export const VideoAnalysisSchema = z.object({
  metadata: AnalysisMetadataSchema,
  publicStats: PublicStatsSchema,
  youtubeDescription: z.string(),
  analysisSettings: AnalysisSettingsSchema,
  transcript: z.array(TranscriptEntrySchema),
  visualSegments: z.array(VisualSegmentSchema),
});

const CatalogPagingSchema = z.object({
  offset: z.number().int().min(0).max(9999).default(0),
  pageSize: z.number().int().min(1).max(100).default(50),
  startDate: IsoDateSchema.optional(),
  endDate: IsoDateSchema.optional(),
});
const validRange = (value: { startDate?: string; endDate?: string }) =>
  (!value.startDate && !value.endDate) ||
  Boolean(value.startDate && value.endDate && value.startDate <= value.endDate);
const rangeMessage = "Indiquez les deux dates, avec la date de début avant la date de fin.";
export const ChannelCatalogRequestSchema = CatalogPagingSchema.extend({
  channelUrl: youtubeUrl(normalizeChannelUrl),
}).refine(validRange, rangeMessage);
export const PlaylistCatalogRequestSchema = CatalogPagingSchema.extend({
  playlistUrl: youtubeUrl(normalizePlaylistUrl),
}).refine(validRange, rangeMessage);
export type CatalogPageInput = z.infer<typeof CatalogPagingSchema>;
/** A channel or playlist as the library shows it: one page per collection. */
export const CollectionSchema = z.object({
  id: z.string().min(1).max(140),
  kind: z.enum(["channel", "playlist"]),
  name: z.string().min(1).max(500),
  url: z.string().url().max(500),
  avatarUrl: z.string().url().max(2000).nullable().optional(),
  bannerUrl: z.string().url().max(2000).nullable().optional(),
  followers: z.number().int().nonnegative().nullable().optional(),
  /** Every video listed so far, in YouTube order; `complete` once the end of the list was reached. */
  videos: z.array(CatalogVideoSchema).max(10000),
  /** Raw YouTube position to continue the listing from. */
  nextOffset: z.number().int().nonnegative().nullable(),
  complete: z.boolean(),
  updatedAt: z.string(),
});
export type Collection = z.infer<typeof CollectionSchema>;
export type CollectionInfo = Omit<Collection, "videos" | "nextOffset" | "complete" | "updatedAt">;
export type CollectionSummary = Omit<Collection, "videos"> & { videoIds: string[] };

export type CatalogPage = {
  videos: CatalogVideo[];
  nextOffset: number | null;
  scannedCount: number;
  limitReached: boolean;
  warnings: string[];
  title?: string;
  /** The channel or playlist this page belongs to. */
  collection?: CollectionInfo;
};

export const DirectVideosRequestSchema = z.object({
  videoUrls: z
    .array(youtubeUrl(normalizeVideoUrl))
    .min(1)
    .max(1000)
    .transform((urls) => [...new Set(urls)]),
});

export const DeleteVideosRequestSchema = z.object({
  videoIds: z
    .array(VideoIdSchema)
    .min(1)
    .max(1000)
    .transform((ids) => [...new Set(ids)]),
});

export const AnalysisModeSchema = z.literal("rag");
export type AnalysisMode = z.infer<typeof AnalysisModeSchema>;
export const AnalyzeRequestSchema = z
  .object({
    requestId: z.string().uuid().optional(),
    mode: AnalysisModeSchema.default("rag"),
    model: z
      .string()
      .min(1)
      .max(160)
      .regex(/^[A-Za-z0-9_./-]+$/)
      .optional(),
    videos: z
      .array(CatalogVideoSchema)
      .min(1)
      .max(1000)
      .refine(
        (videos) => new Set(videos.map((video) => video.id)).size === videos.length,
        "Chaque vidéo doit être unique.",
      ),
  })
  .strict();

export type PublicStats = z.infer<typeof PublicStatsSchema>;
export type CatalogVideo = z.infer<typeof CatalogVideoSchema>;
export type AnalysisSettings = z.infer<typeof AnalysisSettingsSchema>;
export type VideoAnalysis = z.infer<typeof VideoAnalysisSchema>;
export type TranscriptEntry = z.infer<typeof TranscriptEntrySchema>;
export type VisualSegment = z.infer<typeof VisualSegmentSchema>;

export type LocalUsage = {
  model: string;
  promptTokenCount: number | null;
  candidatesTokenCount: number | null;
  totalTokenCount: number | null;
  elapsedMs: number;
  cacheHits: number;
  visionGroups: number;
};
export type AnalyzeVideoSuccess = {
  status: "success";
  videoId: string;
  title: string;
  markdownPath: string;
  jsonPath?: string;
  downloadUrl?: string;
  jsonDownloadUrl?: string;
  provenanceUrl?: string;
  restored?: boolean;
  usage: LocalUsage;
  warnings: string[];
};
export type AnalyzeVideoFailure = {
  status: "error";
  videoId: string;
  title: string;
  error: string;
};
export type AnalyzeVideoResult = AnalyzeVideoSuccess | AnalyzeVideoFailure;
export type AnalysisProgress = {
  videoId: string;
  videoTitle: string;
  chunk: number;
  totalChunks: number;
  phase: "metadata" | "download" | "extract" | "audio" | "vision" | "export";
  detail?: string;
};
export type AnalysisRun = {
  runId: string;
  model: string;
  videos: CatalogVideo[];
  progress?: AnalysisProgress;
  status: "running" | "completed" | "interrupted" | "failed";
  total: number;
  completed: number;
  createdAt: string;
  updatedAt: string;
  results: AnalyzeVideoResult[];
  error?: string;
};

import { HttpError, YOUTUBE_BLOCKED_MESSAGE, isYoutubeBlocked } from "../shared/errors.js";
import { logEvent, observeStage } from "../shared/logging.js";
import { spawn } from "node:child_process";
import { killProcessTree } from "../shared/process.js";
import {
  CatalogPage,
  CatalogPageInput,
  CatalogVideo,
  CatalogVideoSchema,
  type CollectionInfo,
} from "../shared/schemas.js";
import {
  isIsoDate,
  normalizeChannelUrl,
  normalizePlaylistUrl,
  normalizeVideoUrl,
} from "../shared/youtube.js";

type RawVideo = Record<string, unknown>;
type Source = NonNullable<CatalogVideo["source"]>;
export type YtDlpCatalogService = {
  getChannelVideos(input: Partial<CatalogPageInput> & { channelUrl: string }): Promise<CatalogPage>;
  getPlaylistVideos(input: Partial<CatalogPageInput> & { playlistUrl: string }): Promise<CatalogPage>;
  getDirectVideos(videoUrls: string[]): Promise<CatalogPage>;
  getVideoMetadata(video: CatalogVideo): Promise<CatalogVideo>;
};

export class YtDlpError extends Error {
  constructor(
    message: string,
    public readonly details: {
      command: string;
      args: string[];
      stderr?: string;
      exitCode?: number | null;
      cause?: unknown;
    },
  ) {
    super(message);
    this.name = "YtDlpError";
  }
}
/** YouTube refuses this machine for now: retrying the next videos would fail the same way. */
export class YoutubeBlockedError extends YtDlpError {
  constructor(details: YtDlpError["details"]) {
    super(YOUTUBE_BLOCKED_MESSAGE, details);
    this.name = "YoutubeBlockedError";
  }
}
type Runner = (
  binary: string,
  args: string[],
  timeoutMs: number,
) => Promise<{ stdout: string; stderr: string }>;

export function createYtDlpCatalogService(
  binary = "yt-dlp",
  timeoutMs = 120000,
  maxVideos = 1000,
  metadataLang = "fr",
  runner: Runner = runProcess,
  /** --cookies or --cookies-from-browser, so YouTube sees a signed-in browser. */
  authArgs: string[] = [],
): YtDlpCatalogService {
  const metadataCache = new Map<string, { expires: number; value: Promise<CatalogVideo> }>();
  const pageCache = new Map<string, { expires: number; value: Promise<RawVideo> }>();
  // Bound subprocesses across all browsers and analysis workers sharing this service.
  let active = 0;
  const waiting: Array<() => void> = [];
  async function limited<T>(work: () => Promise<T>): Promise<T> {
    if (active >= 2) {
      if (waiting.length >= 200)
        throw new HttpError(429, "Trop de lectures YouTube en attente. Réessayez dans un instant.");
      await new Promise<void>((resolve) => waiting.push(resolve));
    } else active++;
    try {
      return await work();
    } finally {
      const next = waiting.shift();
      if (next) next();
      else active--;
    }
  }
  function cached<T>(
    cache: Map<string, { expires: number; value: Promise<T> }>,
    key: string,
    limit: number,
    work: () => Promise<T>,
  ): Promise<T> {
    let entry = cache.get(key);
    if (!entry || entry.expires < Date.now()) {
      const value = work();
      entry = { expires: Date.now() + 300000, value };
      cache.set(key, entry);
      void value.catch(() => {
        if (cache.get(key)?.value === value) cache.delete(key);
      });
      if (cache.size > limit) cache.delete(cache.keys().next().value!);
    }
    return entry.value;
  }
  async function json(args: string[], includeMetadataLanguage = true): Promise<RawVideo> {
    // Listings need the preferred language to avoid English auto-translated titles.
    // Full single-video metadata must keep the page's default language: the localized page
    // abbreviates like counts ("4,2 k") and yt-dlp then stores only the leading integer.
    const languageArgs =
      includeMetadataLanguage && metadataLang ? ["--extractor-args", "youtube:lang=" + metadataLang] : [];
    const target = args.at(-1) ?? "";
    const output = await observeStage(
      "catalog.ytdlp",
      {
        message: "Lecture des métadonnées YouTube avec yt-dlp.",
        source: "yt-dlp",
        kind: args.includes("--flat-playlist") ? "collection" : "video",
        videoId: new URL(target).searchParams.get("v") ?? undefined,
      },
      () =>
        limited(() =>
          runner(
            binary,
            [
              "--ignore-config",
              "--no-cache-dir",
              ...languageArgs,
              ...authArgs,
              "--js-runtimes",
              "node:" + process.execPath,
              "--skip-download",
              "--no-warnings",
              "--socket-timeout",
              "20",
              "--retries",
              "2",
              "--extractor-retries",
              "2",
              ...args,
            ],
            timeoutMs,
          ),
        ),
    ).catch((error) => {
      if (error instanceof YtDlpError)
        logEvent(
          "catalog.ytdlp.error",
          {
            message: error.message,
            stderr: error.details.stderr,
            exitCode: error.details.exitCode,
          },
          "error",
        );
      throw error;
    });
    try {
      const value: unknown = JSON.parse(output.stdout);
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid object");
      return value as RawVideo;
    } catch (cause) {
      throw new YtDlpError("yt-dlp a renvoyé des métadonnées illisibles.", { command: binary, args, cause });
    }
  }
  const metadata = async (url: string, source: Source): Promise<CatalogVideo> => {
    const key = normalizeVideoUrl(url);
    const video = await cached(metadataCache, key, 1000, async () => {
      const row = await json(["--dump-single-json", "--no-playlist", "--", key], false);
      if (isExcludedVideo(row))
        throw new HttpError(422, "Vidéo indisponible : les directs, rediffusions et Shorts sont exclus.");
      const result = toCatalogVideo(row, "direct");
      if (result.url !== key)
        throw new HttpError(422, "YouTube a renvoyé une autre vidéo que celle demandée.");
      return { ...result, metadataStatus: "complete" as const };
    });
    return { ...video, source };
  };
  async function resolveVideos(
    items: Array<{ url: string; source: Source }>,
  ): Promise<{ videos: CatalogVideo[]; warnings: string[] }> {
    const videos: CatalogVideo[] = [];
    const warnings: string[] = [];
    for (let i = 0; i < items.length; i += 2) {
      const batch = items.slice(i, i + 2);
      const results = await Promise.allSettled(batch.map(({ url, source }) => metadata(url, source)));
      results.forEach((result, index) => {
        if (result.status === "fulfilled") videos.push(result.value);
        else
          warnings.push(
            batch[index].url +
              " : " +
              (result.reason instanceof HttpError || result.reason instanceof YtDlpError
                ? result.reason.message
                : "Lecture des métadonnées impossible. Réessayez cette vidéo séparément."),
          );
      });
    }
    return { videos, warnings };
  }
  async function collection(
    url: string,
    source: Source,
    input: Partial<CatalogPageInput>,
  ): Promise<CatalogPage> {
    const offset = input.offset ?? 0;
    const size = Math.min(input.pageSize ?? 50, maxVideos - offset);
    if (size <= 0)
      throw new HttpError(422, "Limite du catalogue atteinte. Augmentez CATALOG_MAX_VIDEOS pour poursuivre.");
    const end = offset + size;
    const data = await cached(pageCache, JSON.stringify([url, offset, size]), 64, () =>
      json([
        "--flat-playlist",
        "--lazy-playlist",
        "--dump-single-json",
        "--playlist-items",
        offset + 1 + ":" + (end + 1),
        "--",
        url,
      ]),
    );
    if (!Array.isArray(data.entries))
      throw new YtDlpError("YouTube n’a pas renvoyé de liste de vidéos.", { command: binary, args: [url] });
    // Probe one extra raw position; the next cursor must not depend on filtering/deduplication.
    const entries = data.entries as Array<RawVideo | null>;
    const hasMore = entries.length > size;
    const page = entries.slice(0, size);
    const unique = new Map<string, CatalogVideo>();
    let skipped = 0;
    for (const row of page) {
      if (!row || isExcludedVideo(row) || !toFlatCatalogCandidate(row)) {
        skipped++;
        continue;
      }
      try {
        const video = { ...toCatalogVideo(row, source), metadataStatus: "preview" as const };
        if (!unique.has(video.id)) unique.set(video.id, video);
      } catch {
        skipped++;
      }
    }
    const info = collectionInfo(data, source, url);
    // Flat channel entries carry no channel: they all belong to the listed one.
    let videos = [...unique.values()].map((video) =>
      source === "channel" && !video.channel
        ? { ...video, channel: info.name, channelId: info.id, channelUrl: info.url }
        : video,
    );
    const warnings = skipped
      ? [skipped + " entrée(s) indisponible(s) ou exclue(s) ignorée(s) sur cette page."]
      : [];
    if (input.startDate && input.endDate) {
      const details = await resolveVideos(videos.map(({ url }) => ({ url, source })));
      videos = details.videos.filter((video) =>
        isWithinInclusiveDateRange(video.publishedAt, input.startDate!, input.endDate!),
      );
      warnings.push(...details.warnings);
      const unknownDates = details.videos.filter((video) => !video.publishedAt).length;
      if (unknownDates) warnings.push(unknownDates + " vidéo(s) sans date vérifiable exclue(s) du filtre.");
    }
    const limitReached = hasMore && end >= maxVideos;
    if (limitReached)
      warnings.push(
        "Limite de " +
          maxVideos +
          " entrées atteinte. La liste reste utilisable ; augmentez CATALOG_MAX_VIDEOS pour aller plus loin.",
      );
    logEvent("catalog.page.complete", {
      message: "Page du catalogue chargée.",
      source,
      offset,
      scannedEntries: page.length,
      videos: videos.length,
      excludedEntries: skipped,
      warningCount: warnings.length,
    });
    return {
      videos,
      warnings,
      title: stringField(data, "title") ?? undefined,
      collection: info,
      scannedCount: offset + page.length,
      nextOffset: hasMore && !limitReached ? end : null,
      limitReached,
    };
  }
  return {
    getChannelVideos: (input) => collection(normalizeChannelUrl(input.channelUrl), "channel", input),
    getPlaylistVideos: (input) => collection(normalizePlaylistUrl(input.playlistUrl), "playlist", input),
    async getDirectVideos(urls) {
      const unique = [...new Set(urls.map(normalizeVideoUrl))];
      const result = await resolveVideos(unique.map((url) => ({ url, source: "direct" })));
      logEvent("catalog.direct.complete", {
        message: "Catalogue des liens directs chargé.",
        requestedVideos: unique.length,
        loadedVideos: result.videos.length,
        warningCount: result.warnings.length,
      });
      return { ...result, nextOffset: null, scannedCount: unique.length, limitReached: false };
    },
    getVideoMetadata: (video) => metadata(video.url, video.source ?? "direct"),
  };
}

/** Channel pictures come full size (over 1 MB); Google image URLs take the wanted size as suffix. */
function thumbnail(data: RawVideo, id: string, size: string): string | null {
  const list = Array.isArray(data.thumbnails) ? (data.thumbnails as RawVideo[]) : [];
  const found = list.find((item) => item.id === id);
  const url = found && stringField(found, "url");
  if (!url || !/^https:\/\//.test(url)) return null;
  return /googleusercontent\.com\/.+=s0$/.test(url) ? url.replace(/=s0$/, "=" + size) : url;
}

/** Name, page and pictures of the listed channel or playlist, from the listing itself. */
export function collectionInfo(data: RawVideo, source: Source, url: string): CollectionInfo {
  if (source === "playlist") {
    const playlistId = stringField(data, "id") ?? new URL(url).searchParams.get("list") ?? url;
    const pictures = Array.isArray(data.thumbnails) ? (data.thumbnails as RawVideo[]) : [];
    const last = pictures.at(-1);
    return {
      id: "playlist:" + playlistId,
      kind: "playlist",
      name: stringField(data, "title") ?? "Playlist",
      url,
      avatarUrl: (last && stringField(last, "url")) || null,
      bannerUrl: null,
      followers: null,
    };
  }
  const handle = stringField(data, "uploader_url");
  let page = url;
  try {
    page = normalizeChannelUrl(handle ?? stringField(data, "channel_url") ?? url);
  } catch {
    /* The requested URL already is a valid channel page. */
  }
  return {
    id: stringField(data, "channel_id") ?? page,
    kind: "channel",
    name: stringField(data, "channel") ?? stringField(data, "uploader") ?? stringField(data, "title") ?? page,
    url: page,
    avatarUrl: thumbnail(data, "avatar_uncropped", "s176"),
    bannerUrl: thumbnail(data, "banner_uncropped", "w1280"),
    followers: count(data.channel_follower_count),
  };
}

export function toFlatCatalogCandidate(
  video: RawVideo,
): { id: string; url: string; publishedAt: string | null } | null {
  try {
    const id = stringField(video, "id");
    const url = normalizeVideoUrl(
      id ? `https://www.youtube.com/watch?v=${id}` : String(video.webpage_url ?? video.url ?? ""),
    );
    return { id: new URL(url).searchParams.get("v")!, url, publishedAt: publicationDate(video) };
  } catch {
    return null;
  }
}

export function isWithinInclusiveDateRange(date: string | null, start: string, end: string): boolean {
  return date !== null && date >= start && date <= end;
}

export function isExcludedVideo(video: RawVideo): boolean {
  return (
    ["private", "premium_only", "subscriber_only", "needs_auth"].includes(String(video.availability)) ||
    ["[Private video]", "[Deleted video]"].includes(String(video.title)) ||
    ["webpage_url", "original_url", "url"].some((key) => stringField(video, key)?.includes("/shorts/")) ||
    video.is_live === true ||
    video.was_live === true ||
    ["is_live", "is_upcoming", "post_live", "was_live"].includes(String(video.live_status))
  );
}

export function toCatalogVideo(video: RawVideo, source: Source): CatalogVideo {
  const candidate = toFlatCatalogCandidate(video);
  if (!candidate) throw new Error("Identifiant YouTube absent ou invalide dans les métadonnées.");
  const duration =
    typeof video.duration === "number" && Number.isFinite(video.duration) && video.duration >= 0
      ? Math.ceil(video.duration)
      : null;
  return CatalogVideoSchema.parse({
    ...candidate,
    title: stringField(video, "title") ?? "Sans titre",
    channel: stringField(video, "channel") ?? stringField(video, "uploader"),
    channelId: stringField(video, "channel_id"),
    channelUrl: channelPage(video),
    language: stringField(video, "language"),
    durationSeconds: duration,
    description: stringField(video, "description") ?? "",
    publicStats: {
      views: count(video.view_count),
      likes: count(video.like_count),
      comments: count(video.comment_count),
    },
    source,
  });
}

function channelPage(video: RawVideo): string | null {
  for (const key of ["uploader_url", "channel_url"]) {
    const value = stringField(video, key);
    if (!value) continue;
    try {
      return normalizeChannelUrl(value);
    } catch {
      /* Not a channel page (a playlist uploader, for example). */
    }
  }
  return null;
}

export function runProcess(
  binary: string,
  args: string[],
  timeoutMs: number,
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { shell: false, windowsHide: true });
    let stdout = "";
    let stderr = "";
    let bytes = 0;
    let settled = false;
    const fail = (message: string, cause?: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      // yt-dlp may have spawned ffmpeg or a JS runtime: a timeout must not leave them running.
      killProcessTree(child);
      reject(new YtDlpError(message, { command: binary, args, stderr: stderr.slice(-4000), cause }));
    };
    const timer = setTimeout(
      () => fail("Le délai de réponse YouTube est dépassé. Réessayez ou réduisez la sélection."),
      timeoutMs,
    );
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      bytes += Buffer.byteLength(chunk);
      if (bytes > 32 * 1024 * 1024) fail("Le catalogue dépasse la limite de mémoire autorisée.");
      else stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr = (stderr + chunk).slice(-8000);
    });
    child.on("error", (error) =>
      fail("Impossible de démarrer yt-dlp. Vérifiez YT_DLP_BINARY et son installation.", error),
    );
    child.on("close", (exitCode) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (exitCode === 0) resolve({ stdout, stderr });
      else if (isYoutubeBlocked(stderr))
        reject(new YoutubeBlockedError({ command: binary, args, stderr: stderr.slice(-4000), exitCode }));
      else
        reject(
          new YtDlpError(
            "YouTube n’a pas fourni les métadonnées. Vérifiez l’URL, l’accès public et la version de yt-dlp.",
            { command: binary, args, stderr: stderr.slice(-4000), exitCode },
          ),
        );
    });
  });
}

function stringField(video: RawVideo, key: string): string | null {
  const value = video[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
function count(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}
function publicationDate(video: RawVideo): string | null {
  const date = stringField(video, "upload_date");
  if (date && /^\d{8}$/.test(date)) {
    const iso = `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}`;
    if (isIsoDate(iso)) return iso;
  }
  const timestamp = count(video.timestamp);
  if (timestamp === null) return null;
  const parsed = new Date(timestamp * 1000);
  return Number.isFinite(parsed.getTime()) ? parsed.toISOString().slice(0, 10) : null;
}

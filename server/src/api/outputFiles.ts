import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { Response } from "express";
import { pipeline } from "node:stream/promises";
import type { Readable } from "node:stream";
import yazl from "yazl";
import { HttpError } from "../shared/errors.js";
import { CatalogVideoSchema, type CatalogVideo } from "../shared/schemas.js";

export const exportFilePattern = /^video--[A-Za-z0-9_-]{11}\.(?:md|json|local\.json)$/;
export function videoIdFromFilename(name: string): string | null {
  return /^video--([A-Za-z0-9_-]{11})\.(?:md|json|local\.json)$/.exec(name)?.[1] ?? null;
}
export async function resolveOutputPath(root: string, filename: string): Promise<string | null> {
  if (!exportFilePattern.test(filename)) return null;
  try {
    const target = await fs.realpath(path.join(root, filename));
    return path.dirname(target) === root && (await fs.stat(target)).isFile() ? target : null;
  } catch {
    return null;
  }
}
export type ExportIndexEntry = {
  videoId: string;
  markdown: string | null;
  json: string | null;
  local: string | null;
};
export async function readExportIndex(outputDir: string): Promise<ExportIndexEntry[]> {
  let root: string;
  try {
    root = await fs.realpath(outputDir);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  const entries = new Map<string, ExportIndexEntry>();
  for (const file of await fs.readdir(root, { withFileTypes: true })) {
    if (!file.isFile()) continue;
    const id = videoIdFromFilename(file.name);
    if (!id) continue;
    const entry = entries.get(id) ?? { videoId: id, markdown: null, json: null, local: null };
    if (file.name.endsWith(".local.json")) entry.local = file.name;
    else if (file.name.endsWith(".md")) entry.markdown = file.name;
    else entry.json = file.name;
    entries.set(id, entry);
  }
  return [...entries.values()];
}
export type LibraryVideo = { video: CatalogVideo; generatedAt: string | null };
// Parsing a 200 KB export per video at every page load would not scale to a whole channel.
const libraryCache = new Map<string, { mtimeMs: number; size: number; value: LibraryVideo | null }>();

/** The channel identity is not in the original JSON contract: it comes from the metadata cache. */
async function channelIdentity(cacheDir: string | undefined, videoId: string) {
  if (!cacheDir) return {};
  try {
    const cached = JSON.parse(await fs.readFile(path.join(cacheDir, videoId, "metadata.json"), "utf8")) as {
      channelId?: unknown;
      channelUrl?: unknown;
    };
    return {
      channelId: typeof cached.channelId === "string" ? cached.channelId : null,
      channelUrl: typeof cached.channelUrl === "string" ? cached.channelUrl : null,
    };
  } catch {
    return {};
  }
}

async function libraryVideo(
  root: string,
  entry: ExportIndexEntry,
  cacheDir?: string,
): Promise<LibraryVideo | null> {
  if (!entry.json) return null;
  const filename = path.join(root, entry.json);
  const stat = await fs.stat(filename).catch(() => null);
  if (!stat) return null;
  const cached = libraryCache.get(filename);
  if (cached && cached.mtimeMs === stat.mtimeMs && cached.size === stat.size) return cached.value;
  let value: LibraryVideo | null = null;
  try {
    const document = JSON.parse(await fs.readFile(filename, "utf8")) as {
      generatedAt?: string;
      source?: {
        title?: string;
        channel?: string | null;
        publicationDate?: string | null;
        durationSeconds?: number | null;
      };
      publicStats?: CatalogVideo["publicStats"];
      youtubeDescription?: string;
    };
    const parsed = CatalogVideoSchema.safeParse({
      id: entry.videoId,
      title: document.source?.title,
      url: `https://www.youtube.com/watch?v=${entry.videoId}`,
      channel: document.source?.channel ?? null,
      ...(await channelIdentity(cacheDir, entry.videoId)),
      publishedAt: document.source?.publicationDate ?? null,
      durationSeconds: document.source?.durationSeconds ?? null,
      description: (document.youtubeDescription ?? "").slice(0, 50000),
      publicStats: document.publicStats ?? { views: null, likes: null, comments: null },
      metadataStatus: "complete",
    });
    if (parsed.success) value = { video: parsed.data, generatedAt: document.generatedAt ?? null };
  } catch {
    /* An unreadable export is left out of the library; its files stay downloadable. */
  }
  libraryCache.set(filename, { mtimeMs: stat.mtimeMs, size: stat.size, value });
  return value;
}

/** Every exported video, most recently processed first: the library the interface always shows. */
export async function readLibrary(
  outputDir: string,
  cacheDir?: string,
): Promise<{ videos: CatalogVideo[]; exports: ExportIndexEntry[]; processedAt: Record<string, string> }> {
  const exports = await readExportIndex(outputDir);
  if (!exports.length) return { videos: [], exports, processedAt: {} };
  const root = await fs.realpath(outputDir);
  const items = (await Promise.all(exports.map((entry) => libraryVideo(root, entry, cacheDir)))).filter(
    (item): item is LibraryVideo => item !== null,
  );
  items.sort((a, b) => (b.generatedAt ?? "").localeCompare(a.generatedAt ?? ""));
  const processedAt: Record<string, string> = {};
  for (const item of items) if (item.generatedAt) processedAt[item.video.id] = item.generatedAt;
  return { videos: items.map((item) => item.video), exports, processedAt };
}

export async function streamZip(root: string, filenames: string[], res: Response): Promise<void> {
  const checked: Array<{ target: string; name: string }> = [];
  for (const name of new Set(filenames)) {
    const target = await resolveOutputPath(root, name);
    if (!target) throw new HttpError(404, "Export introuvable : " + name);
    checked.push({ target, name });
  }
  if (!checked.length) throw new HttpError(404, "Aucun fichier exporté correspondant.");
  const zip = new yazl.ZipFile();
  const output = zip.outputStream as Readable;
  const readers = checked.map(({ target, name }) => {
    const reader = createReadStream(target);
    reader.on("error", (error) => output.destroy(error));
    zip.addReadStream(reader, name);
    return reader;
  });
  zip.on("error", (error) => output.destroy(error));
  res.attachment("yt2md.zip").type("application/zip");
  zip.end();
  try {
    await pipeline(output, res);
  } finally {
    for (const reader of readers) reader.destroy();
  }
}

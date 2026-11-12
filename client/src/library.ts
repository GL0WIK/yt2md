import type { AnalysisRun, AnalyzeVideoResult, CatalogVideo } from "./types";
import type { CollectionSummary } from "../../server/src/shared/schemas";
import type { ExportIndexEntry } from "./api";
import { reattachExports } from "./catalogState";

export type Library = {
  videos: CatalogVideo[];
  exports: ExportIndexEntry[];
  processedAt: Record<string, string>;
  collections: CollectionSummary[];
};
export const emptyLibrary: Library = { videos: [], exports: [], processedAt: {}, collections: [] };

export type VideoState = "todo" | "queued" | "running" | "done" | "error";

/** Files on disk, then the latest run: a failure stays visible until the video is exported. */
export function buildResults(
  videos: CatalogVideo[],
  exports: ExportIndexEntry[],
  run: AnalysisRun | null,
): Record<string, AnalyzeVideoResult> {
  const fromRun: Record<string, AnalyzeVideoResult> = {};
  for (const result of run?.results ?? []) fromRun[result.videoId] = result;
  return reattachExports(videos, exports, fromRun);
}

export function videoState(
  id: string,
  results: Record<string, AnalyzeVideoResult>,
  run: AnalysisRun | null,
): VideoState {
  if (run?.status === "running") {
    if (run.progress?.videoId === id) return "running";
    if (run.videos.some((video) => video.id === id) && !run.results.some((r) => r.videoId === id))
      return "queued";
  }
  const result = results[id];
  if (result?.status === "success") return "done";
  if (result?.status === "error") return "error";
  return "todo";
}

export type CollectionView = Omit<CollectionSummary, "url"> & {
  url: string | null;
  /** Saved by the server (the channel was opened), or only known from processed videos. */
  stored: boolean;
  processed: number;
  errors: number;
  lastActivity: string;
};

const nameKey = (name: string) => name.trim().toLocaleLowerCase();

/**
 * YouTube sometimes gives a longer display name on a video ("Les Freres Poulain, French DIY…")
 * than on its channel page ("Les Freres Poulain").
 */
function sameChannel(a: string, b: string) {
  const [short, long] = a.length <= b.length ? [a, b] : [b, a];
  return long === short || (long.startsWith(short) && /^[\s,|:·-]/.test(long.slice(short.length)));
}

/**
 * The collection a processed video belongs to: the channel that lists it, then its channel id,
 * then its channel name (videos processed before channel ids were recorded).
 */
export function collectionKey(
  video: CatalogVideo,
  channels: Array<{ id: string; name: string }>,
  listedBy: Map<string, string>,
) {
  const listed = listedBy.get(video.id);
  if (listed) return listed;
  if (video.channelId && channels.some((channel) => channel.id === video.channelId)) return video.channelId;
  const name = video.channel && nameKey(video.channel);
  const named = name && channels.find((channel) => sameChannel(nameKey(channel.name), name));
  if (named) return named.id;
  if (video.channelId) return video.channelId;
  return video.channel ? "name:" + video.channel : "name:";
}

/**
 * One card per channel or playlist: those opened in the interface, plus the channels of the
 * processed videos. Most recent activity first.
 */
export function buildCollections(
  library: Library,
  results: Record<string, AnalyzeVideoResult>,
): CollectionView[] {
  const views = new Map<string, CollectionView>();
  const channels: Array<{ id: string; name: string }> = [];
  const listedBy = new Map<string, string>();
  for (const item of library.collections) {
    views.set(item.id, { ...item, stored: true, processed: 0, errors: 0, lastActivity: item.updatedAt });
    if (item.kind !== "channel") continue;
    channels.push({ id: item.id, name: item.name });
    for (const id of item.videoIds) listedBy.set(id, item.id);
  }
  for (const video of library.videos) {
    const key = collectionKey(video, channels, listedBy);
    let view = views.get(key);
    if (!view) {
      view = {
        id: key,
        kind: "channel",
        name: video.channel ?? "Chaîne inconnue",
        url: video.channelUrl ?? null,
        avatarUrl: null,
        bannerUrl: null,
        followers: null,
        videoIds: [],
        nextOffset: null,
        complete: false,
        updatedAt: "",
        stored: false,
        processed: 0,
        errors: 0,
        lastActivity: "",
      };
      views.set(key, view);
      if (video.channel) channels.push({ id: key, name: video.channel });
    }
    if (!view.videoIds.includes(video.id)) view.videoIds = [...view.videoIds, video.id];
    if (!view.url && video.channelUrl) view.url = video.channelUrl;
  }
  for (const view of views.values()) {
    for (const id of view.videoIds) {
      if (results[id]?.status === "success") view.processed++;
      else if (results[id]?.status === "error") view.errors++;
      const at = library.processedAt[id];
      if (at && at > view.lastActivity) view.lastActivity = at;
    }
  }
  return [...views.values()].sort((a, b) => b.lastActivity.localeCompare(a.lastActivity));
}

/** A channel's page lists YouTube's order first, then processed videos it did not list (yet). */
export function collectionVideos(listed: CatalogVideo[], view: CollectionView | undefined, library: Library) {
  const seen = new Set(listed.map((video) => video.id));
  const processed = library.videos.filter(
    (video) => view?.videoIds.includes(video.id) && !seen.has(video.id),
  );
  return [...listed, ...processed];
}

export function formatTotalDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return hours ? `${hours} h ${String(minutes).padStart(2, "0")}` : `${minutes} min`;
}

export function formatFollowers(count: number | null | undefined): string | null {
  if (!count) return null;
  if (count >= 1_000_000)
    return (count / 1_000_000).toFixed(1).replace(".0", "").replace(".", ",") + " M d’abonnés";
  if (count >= 1000) return Math.round(count / 1000) + " k abonnés";
  return count + " abonnés";
}

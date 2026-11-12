import type { AnalyzeVideoResult, AnalyzeVideoSuccess, CatalogVideo } from "./types";
import type { ExportIndexEntry } from "./api";

export function addSelection(previous: Set<string>, ids: string[], limit = 1000): Set<string> {
  const next = new Set(previous);
  for (const id of ids) {
    if (next.size >= limit) break;
    next.add(id);
  }
  return next;
}

// Rebuilds a lightweight result from files already on disk so a freshly loaded catalog
// can relink to exports generated in earlier sessions. These entries carry no usage.
export function reattachExports(
  videos: CatalogVideo[],
  entries: ExportIndexEntry[],
  existing: Record<string, AnalyzeVideoResult>,
): Record<string, AnalyzeVideoResult> {
  const byId = new Map(entries.map((entry) => [entry.videoId, entry]));
  const next = { ...existing };
  for (const video of videos) {
    // A file on disk outranks a stale failure, but never a live success (it carries usage and warnings).
    if (existing[video.id]?.status === "success") continue;
    const entry = byId.get(video.id);
    if (!entry || (!entry.markdown && !entry.json)) continue;
    const restored: AnalyzeVideoSuccess = {
      status: "success",
      videoId: video.id,
      title: video.title,
      markdownPath: entry.markdown ?? "",
      ...(entry.markdown ? { downloadUrl: "/api/outputs/" + entry.markdown } : {}),
      ...(entry.json ? { jsonDownloadUrl: "/api/outputs/" + entry.json } : {}),
      // Only link a provenance file that actually exists on disk.
      ...(entry.local ? { provenanceUrl: "/api/outputs/" + entry.local } : {}),
      restored: true,
      warnings: [],
      usage: {
        model: "",
        promptTokenCount: null,
        candidatesTokenCount: null,
        totalTokenCount: null,
        elapsedMs: 0,
        cacheHits: 0,
        visionGroups: 0,
      },
    };
    next[video.id] = restored;
  }
  return next;
}

import type { AnalysisRun, CatalogPage, CatalogVideo } from "./types";
import type { Collection } from "../../server/src/shared/schemas";
import type { Library } from "./library";
import {
  normalizeChannelUrl,
  normalizePlaylistUrl,
  normalizeVideoUrl,
} from "../../server/src/shared/youtube";
import type { DeleteFailure } from "./exports";
import type { ExportIndexEntry } from "../../server/src/api/outputFiles";
export type { ExportIndexEntry };
export type CatalogSource =
  | { kind: "direct"; videoUrls: string[] }
  | { kind: "channel" | "playlist"; url: string; pageSize: number; startDate?: string; endDate?: string };
export type Health = {
  local: boolean;
  model: string;
  busy: boolean;
  fps: number;
  groupFrames: number;
  models: { asr: string; vision: string };
  asrLanguage: string;
  runtime: {
    ready: boolean;
    issues: string[];
    gpu?: { name: string; totalVramBytes: number; freeVramBytes: number } | null;
  };
};
export async function request<T>(url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, {
    method: body === undefined ? "GET" : "POST",
    ...(body === undefined
      ? {}
      : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as T;
}
/** A stopped API behind the Vite proxy answers with non-JSON 5xx pages: never surface a JSON parse error. */
export async function errorMessage(response: Response): Promise<string> {
  const data = (await response.json().catch(() => ({}))) as { error?: string };
  if (data.error) return data.error;
  return response.status >= 500
    ? `Serveur local indisponible (HTTP ${response.status}).`
    : `Erreur HTTP ${response.status}`;
}
export async function fetchCatalogPage(
  source: CatalogSource,
  offset = 0,
  signal?: AbortSignal,
): Promise<CatalogPage> {
  if (source.kind === "direct")
    return request("/api/catalog/videos", { videoUrls: source.videoUrls }, signal);
  return request(
    `/api/catalog/${source.kind}`,
    {
      [source.kind === "channel" ? "channelUrl" : "playlistUrl"]: source.url,
      offset,
      pageSize: source.pageSize,
      startDate: source.startDate,
      endDate: source.endDate,
    },
    signal,
  );
}
export async function startRun(videos: CatalogVideo[]) {
  return request<AnalysisRun>("/api/runs", { videos, mode: "rag", requestId: crypto.randomUUID() });
}
export type DeleteExportsResponse = {
  deleted: string[];
  prunedRuns?: unknown;
  /** Files the server could not remove (it then answers 500 with the same body). */
  failed?: DeleteFailure[];
};
/** A partial failure still returns what was deleted, so the UI can drop those results. */
export async function deleteExports(videoIds: string[]): Promise<DeleteExportsResponse> {
  const response = await fetch("/api/outputs/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ videoIds }),
  });
  if (response.ok) return (await response.json()) as DeleteExportsResponse;
  const data = (await response
    .clone()
    .json()
    .catch(() => null)) as DeleteExportsResponse | null;
  if (data && Array.isArray(data.deleted) && data.failed?.length) return data;
  throw new Error(await errorMessage(response));
}
export async function downloadExportsZip(filenames: string[], name = "yt2md.zip") {
  const response = await fetch("/api/outputs/export-zip", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filenames }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
/** Every video already exported (most recently processed first) and the saved channels. */
export function fetchLibrary() {
  return request<Library>("/api/library");
}
export function fetchCollection(id: string, signal?: AbortSignal) {
  return request<Collection>(`/api/collections/${encodeURIComponent(id)}`, undefined, signal);
}
export function removeCollection(id: string) {
  return request<{ removed: boolean }>("/api/collections/remove", { id });
}

export type LinkKind = { kind: "video" | "channel" | "playlist"; url: string };
/** What a pasted link points to; a video shared from a playlist stays a video. */
export function detectLink(text: string): LinkKind | null {
  const value = text.trim();
  if (!value) return null;
  const attempts: Array<[LinkKind["kind"], (value: string) => string]> = [
    ["video", normalizeVideoUrl],
    ["playlist", normalizePlaylistUrl],
    ["channel", normalizeChannelUrl],
  ];
  for (const [kind, normalize] of attempts) {
    try {
      return { kind, url: normalize(/^https?:\/\//i.test(value) ? value : "https://" + value) };
    } catch {
      /* Not this kind of link. */
    }
  }
  return null;
}
export function fetchRun(runId: string, signal?: AbortSignal) {
  return request<AnalysisRun>(`/api/runs/${runId}`, undefined, signal);
}

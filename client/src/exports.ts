import type { AnalyzeVideoResult, CatalogVideo } from "./types";

export type DeleteFailure = { filename: string; videoId: string; error: string };

function basename(url: string | undefined): string | null {
  return url ? (url.split("/").at(-1) ?? null) : null;
}

/** MD and JSON exports a result links to (the files a ZIP should carry). */
export function exportFilenames(result: AnalyzeVideoResult | undefined): string[] {
  if (result?.status !== "success") return [];
  return [result.downloadUrl, result.jsonDownloadUrl].map(basename).filter((v): v is string => Boolean(v));
}

/** The selected videos that actually have exports, with their files: delete and ZIP act on these only. */
export function selectionExports(
  chosen: CatalogVideo[],
  results: Record<string, AnalyzeVideoResult>,
): { videoIds: string[]; filenames: string[] } {
  const videoIds: string[] = [];
  const filenames: string[] = [];
  for (const video of chosen) {
    const files = exportFilenames(results[video.id]);
    if (!files.length) continue;
    videoIds.push(video.id);
    filenames.push(...files);
  }
  return { videoIds, filenames };
}

/** Drops the deleted videos' results, keeping any whose files the server failed to remove. */
export function pruneDeletedResults(
  results: Record<string, AnalyzeVideoResult>,
  videoIds: string[],
  failed: DeleteFailure[] = [],
): Record<string, AnalyzeVideoResult> {
  const incomplete = new Set(failed.map((item) => item.videoId));
  const next = { ...results };
  for (const id of videoIds) if (!incomplete.has(id)) delete next[id];
  return next;
}

export function deleteConfirmation(count: number): string {
  return count === 1
    ? "Supprimer les exports de 1 vidéo ? Cette action est définitive."
    : `Supprimer les exports de ${count} vidéos ? Cette action est définitive.`;
}

export function deleteFailureMessage(failed: DeleteFailure[] | undefined): string {
  if (!failed?.length) return "";
  return `Certains fichiers n’ont pas pu être supprimés : ${failed
    .map((item) => `${item.filename} (${item.error})`)
    .join(", ")}`;
}

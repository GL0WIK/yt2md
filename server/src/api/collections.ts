import fs from "node:fs/promises";
import path from "node:path";
import { atomicWrite } from "../shared/files.js";
import {
  CollectionSchema,
  type CatalogVideo,
  type Collection,
  type CollectionInfo,
  type CollectionSummary,
} from "../shared/schemas.js";

/**
 * YouTube order: videos listed before the first known one were published since (they go first),
 * the others come from further down the list (they go last).
 */
export function mergeIds(existing: string[], page: string[]): string[] {
  const known = new Set(existing);
  const firstKnown = page.findIndex((id) => known.has(id));
  const newer = firstKnown > 0 ? page.slice(0, firstKnown).filter((id) => !known.has(id)) : [];
  const older = page.filter((id) => !known.has(id) && !newer.includes(id));
  return [...newer, ...existing, ...older].slice(0, 10000);
}

/**
 * Channels and playlists opened in the interface, with the videos listed so far: the library
 * shows one page per collection, and reopening it does not need YouTube.
 */
export function createCollectionStore(outputDir: string) {
  const file = path.join(outputDir, ".collections.json");
  let writes: Promise<unknown> = Promise.resolve();

  async function read(): Promise<Collection[]> {
    try {
      const raw = JSON.parse(await fs.readFile(file, "utf8")) as unknown;
      if (!Array.isArray(raw)) return [];
      return raw.flatMap((item) => {
        const parsed = CollectionSchema.safeParse(item);
        return parsed.success ? [parsed.data] : [];
      });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT" || error instanceof SyntaxError) return [];
      throw error;
    }
  }
  /** Writes are serialized: two pages loaded at once must not drop each other's videos. */
  function change<T>(work: (items: Collection[]) => { items: Collection[]; value: T }): Promise<T> {
    const next = writes.then(async () => {
      const { items, value } = work(await read());
      await fs.mkdir(outputDir, { recursive: true });
      await atomicWrite(file, JSON.stringify(items) + "\n");
      return value;
    });
    writes = next.catch(() => {});
    return next;
  }

  return {
    /** Without their videos: the library page only needs identities and counts. */
    async list(): Promise<CollectionSummary[]> {
      return (await read()).map(({ videos, ...item }) => ({
        ...item,
        videoIds: videos.map((video) => video.id),
      }));
    },
    async get(id: string): Promise<Collection | null> {
      return (await read()).find((item) => item.id === id) ?? null;
    },
    /**
     * Records a loaded page. Reloading the first page only adds the newest videos: the cursor
     * further down the list is kept.
     */
    record(info: CollectionInfo, videos: CatalogVideo[], offset: number, nextOffset: number | null) {
      return change((items) => {
        const existing = items.find((item) => item.id === info.id);
        const byId = new Map((existing?.videos ?? []).map((video) => [video.id, video]));
        for (const video of videos) {
          const old = byId.get(video.id);
          // A lightweight listing must not erase details already verified for a video.
          if (!old || old.metadataStatus !== "complete" || video.metadataStatus === "complete")
            byId.set(video.id, video);
        }
        const order = mergeIds(
          (existing?.videos ?? []).map((video) => video.id),
          videos.map((video) => video.id),
        );
        const continues = !existing || offset > 0;
        const merged: Collection = {
          ...info,
          // Pictures and follower counts are kept when a later listing omits them.
          avatarUrl: info.avatarUrl ?? existing?.avatarUrl ?? null,
          bannerUrl: info.bannerUrl ?? existing?.bannerUrl ?? null,
          followers: info.followers ?? existing?.followers ?? null,
          videos: order.map((id) => byId.get(id)!),
          nextOffset: continues ? nextOffset : (existing?.nextOffset ?? null),
          complete: continues ? nextOffset === null : (existing?.complete ?? false),
          updatedAt: new Date().toISOString(),
        };
        return { items: [merged, ...items.filter((item) => item.id !== info.id)], value: merged };
      });
    },
    remove(id: string) {
      return change((items) => ({
        items: items.filter((item) => item.id !== id),
        value: items.some((item) => item.id === id),
      }));
    },
  };
}
export type CollectionStore = ReturnType<typeof createCollectionStore>;

import { describe, expect, it } from "vitest";
import path from "node:path";
import { createCollectionStore, mergeIds } from "../src/api/collections.js";
import { collectionInfo } from "../src/catalog/ytDlp.js";
import { fixtureVideo } from "./fixtures.js";
import { temporaryDirectory } from "./helpers.js";

const video = (id: string, metadataStatus: "preview" | "complete" = "preview") => ({
  ...fixtureVideo,
  id,
  url: `https://www.youtube.com/watch?v=${id}`,
  metadataStatus,
});
const info = {
  id: "UCsci",
  kind: "channel" as const,
  name: "Scilabus",
  url: "https://www.youtube.com/@scilabus/videos",
  avatarUrl: "https://yt3.googleusercontent.com/a=s176",
  bannerUrl: null,
  followers: null,
};

describe("channel collections", () => {
  it("keeps YouTube order: newer videos first, further pages last", () => {
    expect(mergeIds(["b", "c"], ["a", "b", "c"])).toEqual(["a", "b", "c"]);
    expect(mergeIds(["a", "b"], ["c", "d"])).toEqual(["a", "b", "c", "d"]);
    expect(mergeIds([], ["a", "b"])).toEqual(["a", "b"]);
  });

  it("records pages, keeps the cursor on a refresh and verified details", async () => {
    const store = createCollectionStore(path.join(await temporaryDirectory(), "outputs"));
    await store.record(info, [video("bbbbbbbbbbb", "complete"), video("ccccccccccc")], 0, 2);
    await store.record(info, [video("ddddddddddd")], 2, null);
    // A later refresh of the first page finds one new video and must not forget the end of the list.
    const refreshed = await store.record(
      { ...info, avatarUrl: null },
      [video("aaaaaaaaaaa"), video("bbbbbbbbbbb")],
      0,
      2,
    );
    expect(refreshed.videos.map((item) => item.id)).toEqual([
      "aaaaaaaaaaa",
      "bbbbbbbbbbb",
      "ccccccccccc",
      "ddddddddddd",
    ]);
    expect(refreshed).toMatchObject({ complete: true, nextOffset: null, avatarUrl: info.avatarUrl });
    expect(refreshed.videos[1].metadataStatus).toBe("complete");
    expect(await store.list()).toMatchObject([{ id: "UCsci", videoIds: expect.any(Array) }]);
    expect(await store.remove("UCsci")).toBe(true);
    expect(await store.list()).toEqual([]);
  });

  it("reads the channel identity and resized pictures from a listing", () => {
    const channel = collectionInfo(
      {
        channel: "Scilabus",
        channel_id: "UCeR8BYZS7IHYjk_9Mh5JgkA",
        uploader_url: "https://www.youtube.com/@scilabus",
        thumbnails: [
          { id: "avatar_uncropped", url: "https://yt3.googleusercontent.com/abc=s0" },
          { id: "banner_uncropped", url: "https://yt3.googleusercontent.com/def=s0" },
        ],
      },
      "channel",
      "https://www.youtube.com/@scilabus/videos",
    );
    expect(channel).toMatchObject({
      id: "UCeR8BYZS7IHYjk_9Mh5JgkA",
      name: "Scilabus",
      url: "https://www.youtube.com/@scilabus/videos",
      avatarUrl: "https://yt3.googleusercontent.com/abc=s176",
      bannerUrl: "https://yt3.googleusercontent.com/def=w1280",
    });
    expect(
      collectionInfo(
        { id: "PL123", title: "Ma liste" },
        "playlist",
        "https://www.youtube.com/playlist?list=PL123",
      ),
    ).toMatchObject({ id: "playlist:PL123", kind: "playlist", name: "Ma liste" });
  });
});

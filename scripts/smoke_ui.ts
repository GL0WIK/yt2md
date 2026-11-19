import fs from "node:fs/promises";
import path from "node:path";
import { once } from "node:events";
import assert from "node:assert/strict";
import { chromium, expect } from "@playwright/test";
import { createApp } from "../server/src/app.js";
import { getConfig, modelLabel, PROJECT_ROOT } from "../server/src/config.js";
import type { LocalWorker } from "../server/src/local/worker.js";
import { fixtureVideo, fixtureAnalysis } from "../server/test/fixtures.js";
import { YtDlpError } from "../server/src/catalog/ytDlp.js";

const base = path.join(PROJECT_ROOT, ".smoke-tmp/ui");
await fs.mkdir(base, { recursive: true });
const session = await fs.mkdtemp(path.join(base, "run-"));
const config = getConfig({
  OUTPUT_DIR: path.join(session, "outputs"),
  CACHE_DIR: path.join(session, "cache"),
  MODELS_DIR: path.join(session, "models"),
  LOGS_DIR: path.join(session, "logs"),
});
let attempts = 0;
const worker: LocalWorker = {
  busy: false,
  health: async () => ({
    ready: true,
    issues: [],
    gpu: {
      name: "RTX 5090 (test)",
      totalVramBytes: 32 * 1024 ** 3,
      freeVramBytes: 30 * 1024 ** 3,
    },
  }),
  async analyze(_video, signal, progress) {
    attempts++;
    progress({ phase: "audio", chunk: 0, totalChunks: 1, detail: "Transcription de test" });
    if (attempts === 1) {
      await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
      signal.throwIfAborted();
    }
    return {
      durationSeconds: 120,
      model: modelLabel(config),
      promptVersion: "local-observed-video-v1",
      chunkSeconds: 7,
      transcript: [
        {
          speaker: "Indéterminé",
          timestamp: "00:01",
          endTimestamp: "00:03",
          text: "bonjour qwen",
          chunkIndex: 1,
        },
      ],
      visualSegments: fixtureAnalysis.visualSegments,
      warnings: [],
      localProvenance: { transcriptionSource: "qwen3_asr_forced_alignment" },
      usage: {
        model: config.visionModel,
        promptTokenCount: 5,
        candidatesTokenCount: 5,
        totalTokenCount: 10,
        elapsedMs: 100,
        cacheHits: 0,
        visionGroups: 8,
      },
    };
  },
  cleanup: async () => {},
  close: async () => {},
};
const channel = {
  id: "UCtest",
  kind: "channel" as const,
  name: "Chaine test",
  url: "https://www.youtube.com/@chaine-test/videos",
  avatarUrl: null,
  bannerUrl: null,
  followers: null,
};
const video = {
  ...fixtureVideo,
  title: "Démonstration locale : Qwen3-ASR et Qwen",
  channelId: channel.id,
  channelUrl: channel.url,
};
const channelVideos = [
  video,
  {
    ...video,
    id: "new123def45",
    url: "https://www.youtube.com/watch?v=new123def45",
    title: "Nouvelle vidéo de la chaîne",
  },
  {
    ...video,
    id: "old123def45",
    url: "https://www.youtube.com/watch?v=old123def45",
    title: "Autre vidéo de la chaîne",
  },
];
const pageOfVideos = async () => ({
  videos: [video],
  nextOffset: null,
  scannedCount: 1,
  limitReached: true,
  warnings: [],
});
const app = createApp({
  config,
  worker,
  catalog: {
    getDirectVideos: pageOfVideos,
    getChannelVideos: async (input) => {
      if (input.channelUrl.includes("@chaine-inexistante")) {
        throw new YtDlpError("YouTube renvoie une erreur 404 : chaîne introuvable.", {
          command: "test",
          args: [],
          exitCode: 1,
        });
      }
      const offset = input.offset ?? 0;
      return {
        videos: channelVideos.slice(offset, offset + 2),
        nextOffset: offset === 0 ? 2 : null,
        scannedCount: offset === 0 ? 2 : 3,
        limitReached: false,
        warnings: [],
        collection: channel,
      };
    },
    getPlaylistVideos: pageOfVideos,
    getVideoMetadata: async () => video,
  },
});
const server = app.listen(0, "127.0.0.1");
await once(server, "listening");
const address = server.address();
if (!address || typeof address === "string") throw new Error("Server address unavailable");
const browser = await chromium.launch({
  channel: "msedge",
  headless: true,
  downloadsPath: path.join(session, "downloads"),
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, acceptDownloads: true });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${address.port}`);
  await page.getByText("Aucune chaîne pour l’instant.", { exact: false }).waitFor();

  // One video: preview, then transcription with pause and resume.
  await page.getByRole("textbox", { name: "Lien YouTube" }).fill(video.url);
  await page.getByRole("button", { name: "Afficher la vidéo", exact: true }).click();
  await page.locator(".preview-card").getByText(video.title).waitFor();
  await page.getByRole("button", { name: "Transcrire cette vidéo", exact: true }).click();
  await page.locator(".run-strip").click();
  await page.locator(".run-panel").getByText("Transcription de test").waitFor();
  await page.locator(".journal-content").getByText("Vidéo 1/1", { exact: false }).first().waitFor();
  await page.getByRole("button", { name: "Mettre en pause", exact: true }).click();
  await page.getByText("Traitement interrompu", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Reprendre", exact: true }).click();
  await page.getByText("Traitement terminé", { exact: true }).waitFor();
  const [journal] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name: "Télécharger le journal", exact: true }).click(),
  ]);
  const journalBuffers: Buffer[] = [];
  for await (const chunk of await journal.createReadStream()) journalBuffers.push(Buffer.from(chunk));
  const log = Buffer.concat(journalBuffers).toString("utf8");
  assert.match(log, /local\.run\.interrupted/);
  assert.match(log, /local\.run\.completed/);
  const runIds = log
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line).runId);
  assert.equal(new Set(runIds).size, 1);

  // The library groups the transcription under its channel; the channel page lists the rest.
  await page.getByRole("link", { name: "Bibliothèque", exact: true }).click();
  await page.locator(".collection-card", { hasText: "Chaine test" }).click();
  await page.locator(".video-row", { hasText: video.title }).getByText("Traitée").waitFor();
  const [json] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("link", { name: "JSON", exact: true }).click(),
  ]);
  const jsonBuffers: Buffer[] = [];
  for await (const chunk of await json.createReadStream()) jsonBuffers.push(Buffer.from(chunk));
  assert.equal(JSON.parse(Buffer.concat(jsonBuffers).toString("utf8")).schemaVersion, "2.0");
  await page.getByRole("button", { name: "Charger les vidéos de la chaîne", exact: true }).click();
  await expect(page.locator(".video-row", { hasText: channelVideos[1].title })).toBeVisible();
  // The end of a short list is in view: the next page loads by itself.
  await expect(page.locator(".video-row", { hasText: channelVideos[2].title })).toBeVisible();
  await expect(page.getByRole("button", { name: "Charger toute la chaîne", exact: true })).toHaveCount(0);
  await expect(page.locator(".video-row")).toHaveCount(3);
  await page.getByRole("tab", { name: /À traiter/ }).click();
  await expect(page.locator(".video-row")).toHaveCount(2);
  await page.getByRole("checkbox", { name: "Sélectionner toutes les vidéos affichées" }).check();
  await expect(page.locator(".selection-dock")).toContainText("2 vidéos sélectionnées");
  await page.screenshot({ path: path.join(PROJECT_ROOT, ".smoke-tmp/local-ui.png"), fullPage: true });

  // A wrong channel link explains itself and keeps the library.
  await page.locator(".back-link").click();
  await page
    .getByRole("textbox", { name: "Lien YouTube" })
    .fill("https://www.youtube.com/@chaine-inexistante");
  await page.getByRole("button", { name: "Ouvrir", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("chaîne introuvable");
  await expect(page.locator(".collection-card")).toHaveCount(1);
  assert.deepEqual(errors, []);
  console.log(
    "Interface OK : vidéo seule, pause/reprise, journal, bibliothèque par chaîne, chargement de la chaîne, filtres, sélection et lien invalide (moteurs simulés).",
  );
} finally {
  await browser.close();
  await app.locals.close();
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

/**
 * Regenerates the README screenshots (docs/images/screenshot-*.png).
 *
 * The real server and interface run with a simulated worker and catalog: fictional channels and
 * videos, generated thumbnails, no network access and no model. Requires Microsoft Edge, like smoke:ui.
 *   npm run build && npm run docs:screenshots
 */
import fs from "node:fs/promises";
import path from "node:path";
import { once } from "node:events";
import { chromium, expect, type Page } from "@playwright/test";
import { createApp } from "../server/src/app.js";
import { getConfig, modelLabel, PROJECT_ROOT } from "../server/src/config.js";
import type { LocalWorker } from "../server/src/local/worker.js";
import { logEvent } from "../server/src/shared/logging.js";
import type { CatalogVideo } from "../server/src/shared/schemas.js";

const outDir = path.join(PROJECT_ROOT, "docs/images");
const base = path.join(PROJECT_ROOT, ".smoke-tmp/screenshots");
await fs.rm(base, { recursive: true, force: true });
await fs.mkdir(base, { recursive: true });
await fs.mkdir(outDir, { recursive: true });
const config = getConfig({
  OUTPUT_DIR: path.join(base, "outputs"),
  CACHE_DIR: path.join(base, "cache"),
  MODELS_DIR: path.join(base, "models"),
  LOGS_DIR: path.join(base, "logs"),
});

type Channel = { id: string; name: string; handle: string; palette: [string, string] };
const channels: Channel[] = [
  {
    id: "UCatelier000000000000001",
    name: "L’Atelier Numérique",
    handle: "atelier-numerique",
    palette: ["#23563d", "#7fb08f"],
  },
  {
    id: "UCcuisine000000000000002",
    name: "Cuisine de saison",
    handle: "cuisine-de-saison",
    palette: ["#9a4a1f", "#e8b36a"],
  },
  {
    id: "UCcosmos0000000000000003",
    name: "Cosmos en 10 minutes",
    handle: "cosmos-10min",
    palette: ["#1f2b5c", "#7b8fe0"],
  },
];

const titles: Record<string, string[]> = {
  "atelier-numerique": [
    "Monter un serveur domestique silencieux pour moins de 300 €",
    "Home Assistant : 10 automatisations vraiment utiles",
    "J’ai remplacé le cloud par un NAS fait maison",
    "Raspberry Pi 5 : le test complet après six mois",
    "Domotique : capteurs Zigbee ou Wi-Fi, que choisir ?",
    "Réseau maison : VLAN, Wi-Fi invités et pare-feu expliqués",
    "Faire tourner une IA en local sur sa carte graphique",
    "Imprimante 3D : réglages de base pour débuter",
    "Sauvegardes : la règle 3-2-1 appliquée chez soi",
    "Mon bureau connecté, visite complète",
  ],
  "cuisine-de-saison": [
    "Velouté de potimarron en 20 minutes",
    "Tarte fine aux pommes, pâte maison",
    "Le pot-au-feu de grand-mère",
    "Pain au levain : la méthode pas à pas",
    "Gratin dauphinois, la vraie recette",
    "Risotto aux champignons des bois",
  ],
  "cosmos-10min": [
    "Pourquoi le ciel est-il noir la nuit ?",
    "Les trous noirs expliqués simplement",
    "James Webb : trois ans d’images",
    "La vie sur Europe, vraiment ?",
    "Comment mesure-t-on la distance des étoiles ?",
  ],
};

function videoId(channel: number, index: number) {
  return `demo${channel}${String(index).padStart(2, "0")}${"abcdefghijk".slice(0, 4)}`.slice(0, 11);
}

const videosByChannel = new Map<string, CatalogVideo[]>();
const videoChannel = new Map<string, Channel>();
channels.forEach((channel, c) => {
  const list = titles[channel.handle].map((title, i): CatalogVideo => {
    const id = videoId(c, i);
    const day = new Date(Date.UTC(2026, 9, 28) - i * 9 * 86400000).toISOString().slice(0, 10);
    const video: CatalogVideo = {
      id,
      title,
      url: `https://www.youtube.com/watch?v=${id}`,
      channel: channel.name,
      channelId: channel.id,
      channelUrl: `https://www.youtube.com/@${channel.handle}/videos`,
      publishedAt: day,
      durationSeconds: 380 + ((i * 397 + c * 211) % 1500),
      description: "Vidéo de démonstration.",
      publicStats: { views: 4200 + ((i * 7919 + c * 3571) % 180000), likes: null, comments: 20 + i * 7 },
      source: "channel",
    };
    videoChannel.set(id, channel);
    return video;
  });
  videosByChannel.set(channel.handle, list);
});

function escapeXml(text: string) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** A generated 16:9 thumbnail: channel colours, a few words of the title. */
function thumbnail(id: string) {
  const channel = videoChannel.get(id) ?? channels[0];
  const video = videosByChannel.get(channel.handle)!.find((v) => v.id === id);
  const index = Number(id.slice(5, 7)) || 0;
  const words = (video?.title ?? "").replace(/[:?,]/g, "").split(" ");
  const lines: string[] = [];
  for (const word of words) {
    const last = lines[lines.length - 1];
    if (last !== undefined && (last + " " + word).length <= 16) lines[lines.length - 1] = last + " " + word;
    else if (lines.length < 3) lines.push(word);
  }
  const [a, b] = channel.palette;
  const angle = (index * 37) % 360;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
  <defs><linearGradient id="g" gradientTransform="rotate(${angle} .5 .5)"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient></defs>
  <rect width="320" height="180" fill="url(#g)"/>
  <circle cx="${250 + (index % 3) * 15}" cy="${50 + (index % 4) * 20}" r="${60 + (index % 5) * 8}" fill="#fff" opacity=".12"/>
  <circle cx="${40 + (index % 4) * 10}" cy="160" r="45" fill="#000" opacity=".10"/>
  ${lines
    .map(
      (line, i) =>
        `<text x="18" y="${48 + i * 34}" font-family="Segoe UI, Arial, sans-serif" font-weight="800" font-size="27" fill="#fff">${escapeXml(line.toUpperCase())}</text>`,
    )
    .join("\n  ")}
</svg>`;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
let holdRun = false;

const worker: LocalWorker = {
  busy: false,
  health: async () => ({
    ready: true,
    issues: [],
    gpu: {
      name: "NVIDIA GeForce RTX 4090",
      totalVramBytes: 24 * 1024 ** 3,
      freeVramBytes: 22 * 1024 ** 3,
      torchCuda: "12.8",
    },
  }),
  async analyze(video, signal, progress) {
    const duration = video.durationSeconds ?? 600;
    const durationMs = duration * 1000;
    const audioChunks = Math.ceil(duration / 60);
    const visionGroups = Math.ceil(duration / 7);
    const hold = holdRun;
    const pace = hold ? 400 : 0;
    logEvent("media.download.start", { message: "Téléchargement." });
    await sleep(pace);
    logEvent("media.download.verified", { message: "ok", sourceBytes: 147 * 1024 ** 2 });
    logEvent("media.download.complete", { message: "ok", durationMs: 4100 });
    logEvent("media.extract.start", { message: "Extraction." });
    logEvent("media.prepared", { message: "ok", durationMs });
    logEvent("media.extract.complete", { message: "ok", durationMs: 9800 });
    logEvent("model.load.complete", { message: "ok", modelId: config.asrModel, durationMs: 6200 });
    for (let chunk = 1; chunk <= audioChunks; chunk++) {
      progress({
        phase: "audio",
        chunk,
        totalChunks: audioChunks,
        detail: `Transcription ${chunk}/${audioChunks}`,
      });
      logEvent("audio.chunk.start", { message: "ok", chunk, totalChunks: audioChunks });
      if (hold) await sleep(pace);
      logEvent("audio.chunk.complete", {
        message: "ok",
        chunk,
        totalChunks: audioChunks,
        endMs: Math.min(chunk * 60000, durationMs),
      });
    }
    logEvent("audio.complete", {
      message: "ok",
      words: Math.round(duration * 2.6),
      utterances: Math.round(duration / 9),
    });
    logEvent("model.load.complete", {
      message: "ok",
      modelId: config.visionModel,
      backend: "llama_cpp",
      quantization: "Q4_K_M",
      durationMs: 7400,
    });
    const stopAt = hold ? Math.round(visionGroups * 0.58) : visionGroups;
    for (let chunk = 1; chunk <= stopAt; chunk++) {
      progress({
        phase: "vision",
        chunk,
        totalChunks: visionGroups,
        detail: `Analyse visuelle ${chunk}/${visionGroups}`,
      });
      logEvent("vision.group.start", { message: "ok", chunk, totalChunks: visionGroups, frameCount: 7 });
      if (hold && chunk % 6 === 0) await sleep(pace);
      logEvent("vision.group.complete", {
        message: "ok",
        chunk,
        totalChunks: visionGroups,
        endMs: Math.min(chunk * 7000, durationMs),
      });
    }
    if (hold) {
      await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
      signal.throwIfAborted();
    }
    return {
      durationSeconds: duration,
      model: modelLabel(config),
      promptVersion: "local-observed-video-v1",
      chunkSeconds: 7,
      transcript: [
        {
          speaker: "Indéterminé",
          timestamp: "00:01",
          endTimestamp: "00:06",
          text: "Bonjour à tous et bienvenue.",
          chunkIndex: 1,
        },
      ],
      visualSegments: [
        {
          timestamp: "00:00",
          endTimestamp: "00:07",
          title: "Introduction",
          visualDescription: "Plan d’ouverture face caméra.",
          screenText: [video.title],
        },
      ],
      warnings: [],
      localProvenance: { transcriptionSource: "qwen3_asr_forced_alignment" },
      usage: {
        model: config.visionModel,
        promptTokenCount: 5,
        candidatesTokenCount: 5,
        totalTokenCount: 10,
        elapsedMs: 100,
        cacheHits: 0,
        visionGroups,
      },
    };
  },
  cleanup: async () => {},
  close: async () => {},
};

function handleOf(url: string) {
  return /@([\w-]+)/.exec(url)?.[1] ?? "";
}

const app = createApp({
  config,
  worker,
  catalog: {
    getDirectVideos: async () => ({
      videos: [],
      nextOffset: null,
      scannedCount: 0,
      limitReached: true,
      warnings: [],
    }),
    getChannelVideos: async (input) => {
      const handle = handleOf(input.channelUrl);
      const channel = channels.find((c) => c.handle === handle)!;
      const all = videosByChannel.get(handle) ?? [];
      const offset = input.offset ?? 0;
      return {
        videos: all.slice(offset, offset + 30),
        nextOffset: null,
        scannedCount: all.length,
        limitReached: false,
        warnings: [],
        collection: {
          id: channel.id,
          kind: "channel" as const,
          name: channel.name,
          url: `https://www.youtube.com/@${channel.handle}/videos`,
          avatarUrl: null,
          bannerUrl: null,
          followers: 12000 + channel.name.length * 1777,
        },
      };
    },
    getPlaylistVideos: async () => ({
      videos: [],
      nextOffset: null,
      scannedCount: 0,
      limitReached: true,
      warnings: [],
    }),
    getVideoMetadata: async (video) =>
      [...videosByChannel.values()].flat().find((v) => v.id === video.id) ?? video,
  },
});

const server = app.listen(0, "127.0.0.1");
await once(server, "listening");
const address = server.address();
if (!address || typeof address === "string") throw new Error("Server address unavailable");
const origin = `http://127.0.0.1:${address.port}`;
const browser = await chromium.launch({ channel: "msedge", headless: true });

async function openChannel(page: Page, channel: Channel) {
  await page.goto(origin);
  await page
    .getByRole("textbox", { name: "Lien YouTube" })
    .fill(`https://www.youtube.com/@${channel.handle}`);
  await page.getByRole("button", { name: "Ouvrir", exact: true }).click();
  await page.locator(".video-row").first().waitFor();
}

async function transcribe(page: Page, count: number) {
  const boxes = page.locator(".video-row input[type=checkbox]");
  for (let i = 0; i < count; i++) await boxes.nth(i).check();
  await page
    .locator(".selection-dock")
    .getByRole("button", { name: /Transcrire/ })
    .click();
}

async function waitIdle(page: Page) {
  await page.goto(origin);
  await expect(page.locator(".run-strip")).toHaveCount(0, { timeout: 30000 });
}

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1.5 });
  await page.route("https://i.ytimg.com/**", (route) => {
    const id = /\/vi\/([^/]+)\//.exec(route.request().url())?.[1] ?? "";
    return route.fulfill({ contentType: "image/svg+xml", body: thumbnail(id) });
  });

  // Fill the library: every channel opened, some videos already transcribed.
  const done = [6, 4, 3];
  for (const [i, channel] of channels.entries()) {
    await openChannel(page, channel);
    await transcribe(page, done[i]);
    await page.waitForTimeout(500);
    await waitIdle(page);
  }

  await page.goto(origin);
  await expect(page.locator(".collection-card")).toHaveCount(channels.length);
  await page.waitForTimeout(400);
  await page.screenshot({
    path: path.join(outDir, "screenshot-library.png"),
    clip: { x: 0, y: 0, width: 1440, height: 640 },
  });

  // Channel page: mixed states, a filter and a selection.
  await page.locator(".collection-card", { hasText: channels[0].name }).click();
  await page.locator(".video-row").first().waitFor();
  const pending = page.locator(".video-row", { hasNotText: "Traitée" }).locator("input[type=checkbox]");
  await pending.nth(0).check();
  await pending.nth(1).check();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, "screenshot-channel.png") });

  // Processing page, caught in the middle of the visual analysis.
  holdRun = true;
  await page
    .locator(".selection-dock")
    .getByRole("button", { name: /Transcrire/ })
    .click();
  await page.waitForTimeout(400);
  await page.goto(origin + "/#/traitement");
  await page.waitForTimeout(15000);
  await page.screenshot({
    path: path.join(outDir, "screenshot-run.png"),
    fullPage: true,
    clip: { x: 0, y: 0, width: 1440, height: 1010 },
  });
  console.log(`Captures écrites dans ${path.relative(PROJECT_ROOT, outDir)}`);
} finally {
  await browser.close();
  await app.locals.close();
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

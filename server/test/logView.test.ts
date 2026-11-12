import { describe, expect, it } from "vitest";
import { Terminal, fitWidth } from "../src/shared/terminal.js";
import {
  LogPresenter,
  collapseProgress,
  formatClock,
  formatDuration,
  formatTerminalLine,
  supportsColor,
  type ViewEntry,
} from "../src/shared/logView.js";

const base = Date.parse("2026-10-14T15:00:00.000Z");
function at(seconds: number, event: string, fields: Partial<ViewEntry> = {}): ViewEntry {
  return {
    timestamp: new Date(base + seconds * 1000).toISOString(),
    event,
    level: "info",
    message: event,
    ...fields,
  } as ViewEntry;
}

function replay(entries: ViewEntry[]) {
  const presenter = new LogPresenter();
  return entries.map((entry) => ({ entry, view: presenter.present(entry) }));
}

describe("readable journal", () => {
  it("formats durations and media positions in French", () => {
    expect(formatDuration(4132)).toBe("4,1 s");
    expect(formatDuration(42_400)).toBe("42 s");
    expect(formatDuration(86_713)).toBe("1 min 27 s");
    expect(formatDuration(3_725_000)).toBe("1 h 02 min");
    expect(formatClock(1_208_188)).toBe("20:08");
    expect(formatClock(3_725_000)).toBe("1:02:05");
  });

  it("tells a run as headings, throttled progress and a timed summary", () => {
    const groups = Array.from({ length: 100 }, (_, index) =>
      at(200 + index * 2, "vision.group.complete", {
        chunk: index + 1,
        totalChunks: 100,
        endMs: (index + 1) * 7000,
      }),
    );
    const lines = replay([
      at(0, "local.run.start", { totalVideos: 2, pendingVideos: 2, completedVideos: 0 }),
      at(0, "local.video.start", { videoId: "gt8MQLpW6fY", title: "Clé de 13" }),
      at(0, "worker.request.start", { action: "analyze" }),
      at(4, "media.download.complete", { durationMs: 4132 }),
      at(5, "media.extract.start"),
      at(10, "media.frames.progress", { mediaTimeMs: 300_000, mediaDurationMs: 700_000 }),
      at(90, "media.extract.complete", { durationMs: 86_000 }),
      at(91, "audio.chunk.start", { chunk: 1, totalChunks: 2 }),
      at(150, "audio.generate.waiting", {
        message: "Transcription Qwen3-ASR. — traitement toujours actif.",
        elapsedMs: 60_000,
      }),
      at(160, "audio.chunk.complete", { chunk: 1, totalChunks: 2, endMs: 61_000 }),
      at(170, "audio.chunk.complete", { chunk: 2, totalChunks: 2, endMs: 700_000 }),
      at(170, "audio.complete", { words: 3153, utterances: 303 }),
      at(198, "vision.group.start", { chunk: 1, totalChunks: 100, frameCount: 7 }),
      ...groups,
      at(400, "local.video.complete", { durationMs: 400_000 }),
    ]);
    const shown = lines.filter(({ view }) => view.essential).map(({ view }) => [view.text, view.note]);
    expect(shown[0]).toEqual(["Traitement démarré", "2 vidéos à traiter"]);
    expect(shown[1]).toEqual(["Vidéo 1/2 · Clé de 13", "gt8MQLpW6fY"]);
    expect(shown).toContainEqual(["Images", "05:00 / 11:40 · reste ~6,7 s"]);
    expect(shown).toContainEqual(["Transcription Qwen3-ASR : toujours en cours", "1 min 00 s"]);
    expect(shown).toContainEqual(["Transcription terminée", "3 153 mots · 303 répliques · 1 min 19 s"]);
    expect(shown.some(([text]) => text === "worker.request.start")).toBe(false);
    // About one progress line per 10 % rather than one per group.
    const vision = lines.filter(({ view }) => view.essential && view.text === "Analyse visuelle");
    expect(vision.length).toBeGreaterThanOrEqual(10);
    expect(vision.length).toBeLessThanOrEqual(14);
    expect(vision.at(-1)?.view.progress).toEqual({ done: 100, total: 100 });
    // Remaining time ignores the warm-up of the first group: 2 s per group.
    expect(vision.find(({ view }) => view.progress?.done === 50)?.view.note).toContain("reste ~1 min 40 s");
    const summary = shown.at(-1)!;
    expect(summary[0]).toBe("Vidéo terminée en 6 min 40 s");
    expect(summary[1]).toContain("0,57 × la durée de la vidéo");
    expect(summary[1]).toContain(
      "téléchargement 4,1 s · extraction 1 min 26 s · audio 1 min 19 s · vision 3 min 20 s",
    );
  });

  it("never hides problems or unknown events, and condenses multi-line PyTorch warnings", () => {
    const prefix = "W1014 17:08:02.305000 11516 torch\\_dynamo\\variables\\tensor.py:1379] [0/0]";
    const lines = replay([
      at(0, "media.native_video.start", { level: "warn", message: "ffmpeg lent" }),
      at(1, "brand.new.event", { message: "Nouvel événement" }),
      at(2, "worker.output", { message: `${prefix} Graph break from \`Tensor.item()\`` }),
      at(2, "worker.output", { message: `${prefix}     torch._dynamo.config.capture_scalar_outputs = True` }),
      at(3, "worker.output", {
        message: "Loading weights:  45%|████▌     | 320/707 [00:01<00:01, 250.00it/s]",
      }),
    ]).map(({ view }) => view);
    expect(lines.map((view) => [view.tone, view.essential])).toEqual([
      ["warn", true],
      ["info", true],
      ["warn", true],
      ["detail", false],
      ["detail", false],
    ]);
    expect(lines[2].text).toBe("PyTorch : Graph break from `Tensor.item()`");
  });

  it("renders plain terminal lines unless colors are supported", () => {
    const [{ entry, view }] = replay([
      at(0, "audio.chunk.complete", { chunk: 5, totalChunks: 10, endMs: 300_000 }),
    ]);
    const line = formatTerminalLine(entry, view, false);
    expect(line).toMatch(/^\d\d:\d\d:\d\d {4}Transcription +█{10}░{10} +50 %/);
    expect(line).not.toContain("\x1b[");
    expect(formatTerminalLine(entry, view, true)).toContain("\x1b[");
    expect(supportsColor({ isTTY: false }, { FORCE_COLOR: "1" }, "linux")).toBe(true);
    expect(supportsColor({ isTTY: true }, { NO_COLOR: "1" }, "win32")).toBe(false);
    expect(supportsColor({ isTTY: false }, {}, "linux")).toBe(false);
    expect(supportsColor({ isTTY: false }, {}, "win32")).toBe(true);
  });

  it("shows one bar per stage in the web journal, updated in place", () => {
    const lines = replay([
      at(0, "local.run.start", { totalVideos: 1 }),
      at(0, "local.video.start", { videoId: "gt8MQLpW6fY", title: "Clé" }),
      ...[1, 2, 3].map((chunk) => at(chunk, "audio.chunk.complete", { chunk, totalChunks: 3 })),
      at(4, "audio.complete", { words: 10, utterances: 2 }),
      ...[1, 2].map((chunk) => at(4 + chunk, "vision.group.complete", { chunk, totalChunks: 4 })),
    ]);
    const shown = collapseProgress(lines).map(({ view }) => [view.text, view.progress?.done]);
    expect(shown).toEqual([
      ["Traitement démarré", undefined],
      ["Vidéo 1/1 · Clé", undefined],
      ["Transcription", 3],
      ["Transcription terminée", undefined],
      ["Analyse visuelle", 2],
    ]);
  });

  it("redraws the live bar in place and prints other lines above it", () => {
    const written: string[] = [];
    const out = { isTTY: true, columns: 40, write: (text: string) => written.push(text) };
    const live = new Terminal(out, {});
    live.progress("1:audio", "Transcription 10 %", false);
    live.print("⚠ avertissement");
    live.progress("1:audio", "Transcription 50 %", false);
    live.progress("1:audio", "Transcription 100 %", true);
    expect(written.join("")).toBe(
      "Transcription 10 %" +
        "\r\x1b[2K⚠ avertissement\nTranscription 10 %" +
        "\r\x1b[2KTranscription 50 %" +
        "\r\x1b[2KTranscription 100 %\n",
    );
    expect(new Terminal({ isTTY: false, write: () => true }, {}).interactive).toBe(false);
    expect(fitWidth("\x1b[36mabcdef\x1b[0m", 3)).toBe("\x1b[36mabc\x1b[0m");
  });
});

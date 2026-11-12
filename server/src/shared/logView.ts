/**
 * Human view of the structured journal, shared by the terminal and the web journal.
 *
 * The JSONL journals keep every event unchanged; this module only decides what a person reads:
 * stage headings, throttled progress bars with a remaining-time estimate, results and every
 * warning or error. Technical events stay available as "detail" lines (LOG_LEVEL=debug, or the
 * "Tous les messages" filter of the web journal). It has no Node dependency so the client can use it.
 */

export type ViewLevel = "debug" | "info" | "warn" | "error";
export type ViewEntry = {
  timestamp: string;
  event: string;
  level: ViewLevel;
  message: string;
  [key: string]: string | number | boolean | null | undefined;
};
export type Tone = "run" | "video" | "stage" | "done" | "progress" | "info" | "detail" | "warn" | "error";
export type ViewLine = {
  tone: Tone;
  text: string;
  /** Secondary facts, already formatted: durations, counts, remaining time. */
  note?: string;
  progress?: { done: number; total: number };
  /** Same value for every update of one progress bar: a live view redraws it in place. */
  key?: string;
  /** Shown by default. Detail lines only appear in verbose views. */
  essential: boolean;
};

type Tracker = {
  startedAt: number;
  firstDoneAt?: number;
  firstDone?: number;
  shownAt?: number;
  shownStep?: number;
};

const MODEL_WORK = new Set(["audio.generate.waiting", "audio.align.waiting", "vision.generate.waiting"]);
// Heartbeats of outer stages repeat the innermost one.
const QUIET = new Set([
  "local.progress",
  "local.metadata.start",
  "local.metadata.complete",
  "local.result.saved",
  "local.validation.complete",
  "local.export.start",
  "worker.request.start",
  "worker.request.complete",
  "worker.request.waiting",
  "worker.analysis.start",
  "worker.started",
  "pipeline.start",
  "pipeline.complete",
  "media.ytdlp.start",
  "media.ytdlp.complete",
  "media.probe.start",
  "media.probe.complete",
  "media.audio.extract.start",
  "media.audio.extract.complete",
  "media.frames.extract.start",
  "media.frames.extract.complete",
  "media.download.verified",
  "media.download.output",
  "media.prepared",
  "media.cleanup.start",
  "media.cleanup.complete",
  "media.native_video.start",
  "media.native_video.complete",
  "model.load.start",
  "model.processor.load.start",
  "model.processor.load.complete",
  "model.ready",
  "model.unload.start",
  "model.unload.complete",
  "vision.memory.check",
  "native.started",
  "catalog.ytdlp.start",
  "catalog.ytdlp.complete",
  "http.request.complete",
]);

export function formatDuration(ms: number): string {
  const seconds = Math.max(0, ms) / 1000;
  if (seconds < 10) return seconds.toFixed(1).replace(".", ",") + " s";
  if (seconds < 60) return Math.round(seconds) + " s";
  const total = Math.round(seconds);
  if (total < 3600) return `${Math.floor(total / 60)} min ${String(total % 60).padStart(2, "0")} s`;
  return `${Math.floor(total / 3600)} h ${String(Math.floor((total % 3600) / 60)).padStart(2, "0")} min`;
}

export function formatClock(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return total >= 3600 ? `${Math.floor(total / 3600)}:${minutes}:${seconds}` : `${minutes}:${seconds}`;
}

export function formatCount(value: number): string {
  return String(Math.round(value)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(1).replace(".", ",") + " Go";
  return Math.round(bytes / 1024 ** 2) + " Mo";
}

function plural(count: number, word: string) {
  return `${formatCount(count)} ${word}${count > 1 ? "s" : ""}`;
}

function shortModel(id: unknown) {
  return typeof id === "string" ? id.split("/").pop() || id : "modèle";
}

function number(entry: ViewEntry, key: string): number | undefined {
  const value = entry[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function joined(...parts: Array<string | number | false | undefined | null>) {
  return parts.filter(Boolean).join(" · ") || undefined;
}

export class LogPresenter {
  private totalVideos = 0;
  private videoIndex = 0;
  private videoStartedAt?: number;
  private videoDurationMs?: number;
  private sourceBytes?: number;
  private stageStarts = new Map<string, number>();
  private stageTimes: Array<[string, number]> = [];
  private lastOutputSource = "";
  private downloadStream = 1;
  private lastDownload = 0;
  private trackers = new Map<string, Tracker>();

  present(entry: ViewEntry): ViewLine {
    const line = this.describe(entry, Date.parse(entry.timestamp) || 0);
    // Problems are never hidden, whatever the event.
    if (entry.level === "error") return { ...line, tone: "error", essential: true };
    if (entry.level === "warn") return { ...line, tone: "warn", essential: true };
    return line;
  }

  private describe(entry: ViewEntry, at: number): ViewLine {
    const duration = number(entry, "durationMs");
    const took = duration === undefined ? undefined : formatDuration(duration);
    const chunk = number(entry, "chunk");
    const totalChunks = number(entry, "totalChunks");
    switch (entry.event) {
      case "server.started":
        return {
          tone: "run",
          text: entry.message,
          note: joined(
            entry.asrModel &&
              entry.visionModel &&
              `${shortModel(entry.asrModel)} + ${shortModel(entry.visionModel)}`,
            entry.asrLanguage && `langue ${entry.asrLanguage}`,
            typeof entry.device === "string" && entry.device,
          ),
          essential: true,
        };
      case "catalog.direct.complete":
      case "catalog.page.complete": {
        const videos = number(entry, "loadedVideos") ?? number(entry, "videos") ?? 0;
        return { tone: "info", text: "Catalogue chargé", note: plural(videos, "vidéo"), essential: true };
      }
      case "local.run.start": {
        this.reset();
        this.totalVideos = number(entry, "totalVideos") ?? 0;
        this.videoIndex = number(entry, "completedVideos") ?? 0;
        const pending = number(entry, "pendingVideos") ?? this.totalVideos;
        return {
          tone: "run",
          text: "Traitement démarré",
          note: joined(
            plural(pending, "vidéo") + " à traiter",
            this.videoIndex > 0 && `reprise après ${plural(this.videoIndex, "vidéo")}`,
          ),
          essential: true,
        };
      }
      case "local.video.start": {
        this.videoIndex++;
        this.videoStartedAt = at;
        this.videoDurationMs = this.sourceBytes = undefined;
        this.trackers.clear();
        this.stageStarts.clear();
        this.stageTimes = [];
        this.downloadStream = 1;
        this.lastDownload = 0;
        const position = this.totalVideos ? `Vidéo ${this.videoIndex}/${this.totalVideos}` : "Vidéo";
        return {
          tone: "video",
          text: `${position} · ${entry.title ?? entry.videoId ?? ""}`.trim(),
          note: typeof entry.videoId === "string" ? entry.videoId : undefined,
          essential: true,
        };
      }
      case "media.download.start":
        return { tone: "stage", text: "Téléchargement de la vidéo", essential: true };
      case "media.download.progress": {
        const percent = Number.parseFloat(String(entry.percent ?? ""));
        if (!Number.isFinite(percent)) return this.detail(entry);
        const eta = typeof entry.eta === "string" && /^\d/.test(entry.eta) ? "reste " + entry.eta : undefined;
        const speed = typeof entry.speed === "string" && /^\d/.test(entry.speed) ? entry.speed : undefined;
        // yt-dlp downloads the video stream, then the audio stream: a restart is the next stream.
        const done = Math.round(percent * 10);
        if (done < this.lastDownload) this.downloadStream++;
        this.lastDownload = done;
        return this.progress(
          "download" + this.downloadStream,
          at,
          `Téléchargement (flux ${this.downloadStream})`,
          done,
          1000,
          joined(speed, eta),
          false,
        );
      }
      case "media.download.verified":
        this.sourceBytes = number(entry, "sourceBytes");
        return this.detail(entry);
      case "media.download.complete":
        if (duration !== undefined) this.stageTimes.push(["téléchargement", duration]);
        return {
          tone: "done",
          text: "Téléchargement terminé",
          note: joined(took, this.sourceBytes && formatBytes(this.sourceBytes)),
          essential: true,
        };
      case "media.extract.start":
        this.stageStarts.set("extract", at);
        return {
          tone: "stage",
          text: "Extraction de l’audio et des images (1 par seconde)",
          essential: true,
        };
      case "media.audio.progress":
      case "media.frames.progress": {
        const done = number(entry, "mediaTimeMs");
        const total = number(entry, "mediaDurationMs");
        if (done === undefined || !total) return this.detail(entry);
        this.videoDurationMs = total;
        const label = entry.event === "media.audio.progress" ? "Audio" : "Images";
        return this.progress(
          entry.event,
          at,
          label,
          Math.min(done, total),
          total,
          `${formatClock(done)} / ${formatClock(total)}`,
          true,
          this.stageStarts.get("extract"),
        );
      }
      case "media.extract.cache_hit":
        return { tone: "done", text: "Audio et images retrouvés dans le cache", essential: true };
      case "media.extract.complete":
        if (duration !== undefined) this.stageTimes.push(["extraction", duration]);
        return { tone: "done", text: "Audio et images extraits", note: took, essential: true };
      case "media.prepared":
        this.videoDurationMs = number(entry, "durationMs") ?? this.videoDurationMs;
        return this.detail(entry);
      case "model.load.complete":
        return {
          tone: "info",
          text: "Modèle chargé : " + shortModel(entry.modelId),
          note: joined(
            took,
            typeof entry.quantization === "string" && entry.backend === "llama_cpp" && entry.quantization,
          ),
          essential: true,
        };
      case "model.compile":
        return {
          tone: "info",
          text: "torch.compile actif",
          note: "les deux premiers morceaux incluent la compilation des kernels (environ 1 min 30 s avec le cache)",
          essential: true,
        };
      case "audio.chunk.start":
        if (chunk === 1 && !this.stageStarts.has("audio")) this.stageStarts.set("audio", at);
        if (chunk === 1)
          return {
            tone: "stage",
            text: "Transcription Qwen3-ASR",
            note: totalChunks ? plural(totalChunks, "morceau").replace(/morceaus$/, "morceaux") : undefined,
            essential: true,
          };
        return this.detail(entry);
      case "audio.chunk.complete": {
        if (chunk === undefined || !totalChunks) return this.detail(entry);
        const end = number(entry, "endMs");
        return this.progress(
          "audio",
          at,
          "Transcription",
          chunk,
          totalChunks,
          joined(
            `${chunk}/${totalChunks}`,
            end !== undefined &&
              (this.videoDurationMs
                ? `${formatClock(end)} / ${formatClock(this.videoDurationMs)}`
                : formatClock(end)),
          ),
          false,
        );
      }
      case "audio.segment.unpunctuated":
        return {
          tone: "warn",
          text: "Segment sans ponctuation décodé de nouveau en deux parties",
          note:
            entry.restored === true
              ? "ponctuation rétablie"
              : entry.restored === false
                ? "non rétablie"
                : undefined,
          essential: true,
        };
      case "audio.complete": {
        const started = this.stageStarts.get("audio");
        if (started !== undefined) this.stageTimes.push(["audio", at - started]);
        return {
          tone: "done",
          text: "Transcription terminée",
          note: joined(
            number(entry, "words") !== undefined && plural(number(entry, "words")!, "mot"),
            number(entry, "utterances") !== undefined && plural(number(entry, "utterances")!, "réplique"),
            started !== undefined && formatDuration(at - started),
          ),
          essential: true,
        };
      }
      case "vision.group.start":
        if (chunk === 1 && !this.stageStarts.has("vision")) this.stageStarts.set("vision", at);
        if (chunk === 1)
          return {
            tone: "stage",
            text: "Analyse visuelle Qwen3.5",
            note: joined(
              totalChunks && plural(totalChunks, "groupe"),
              number(entry, "frameCount") && `${entry.frameCount} images par groupe`,
            ),
            essential: true,
          };
        return this.detail(entry);
      case "vision.group.complete": {
        if (chunk === undefined || !totalChunks) return this.detail(entry);
        const end = number(entry, "endMs");
        const started = this.stageStarts.get("vision");
        if (chunk === totalChunks && started !== undefined) this.stageTimes.push(["vision", at - started]);
        return this.progress(
          "vision",
          at,
          "Analyse visuelle",
          chunk,
          totalChunks,
          joined(
            `${chunk}/${totalChunks}`,
            end !== undefined &&
              (this.videoDurationMs
                ? `${formatClock(end)} / ${formatClock(this.videoDurationMs)}`
                : formatClock(end)),
          ),
          false,
        );
      }
      case "local.export.complete":
        return {
          tone: "done",
          text: "Exports écrits",
          note: joined(
            typeof entry.jsonFile === "string" && entry.jsonFile,
            typeof entry.markdownFile === "string" && entry.markdownFile,
            typeof entry.provenanceFile === "string" && entry.provenanceFile,
          ),
          essential: true,
        };
      case "local.video.complete": {
        const elapsed =
          duration ?? (this.videoStartedAt !== undefined ? at - this.videoStartedAt : undefined);
        const ratio = elapsed && this.videoDurationMs ? elapsed / this.videoDurationMs : undefined;
        return {
          tone: "video",
          text: "Vidéo terminée" + (elapsed !== undefined ? " en " + formatDuration(elapsed) : ""),
          note: joined(
            ratio !== undefined && `${ratio.toFixed(2).replace(".", ",")} × la durée de la vidéo`,
            this.videoDurationMs && `vidéo de ${formatClock(this.videoDurationMs)}`,
            ...this.stageTimes.map(([name, ms]) => `${name} ${formatDuration(ms)}`),
            number(entry, "cacheHits") ? `${entry.cacheHits} étape(s) reprise(s) du cache` : undefined,
          ),
          essential: true,
        };
      }
      case "local.run.completed":
      case "local.run.interrupted": {
        const successes = number(entry, "successCount");
        const errors = number(entry, "errorCount");
        return {
          tone: "run",
          text:
            entry.event === "local.run.completed"
              ? "Traitement terminé"
              : "Traitement en pause, morceaux terminés conservés",
          note: joined(
            successes !== undefined && plural(successes, "réussite"),
            errors !== undefined && plural(errors, "erreur"),
            took,
          ),
          essential: true,
        };
      }
    }
    if (MODEL_WORK.has(entry.event)) {
      const elapsed = number(entry, "elapsedMs");
      return {
        tone: "info",
        text:
          entry.message.replace(/\s*—\s*traitement toujours actif\.?$/, "").replace(/\.$/, "") +
          " : toujours en cours",
        note: elapsed !== undefined ? formatDuration(elapsed) : undefined,
        essential: elapsed !== undefined && elapsed >= 60000,
      };
    }
    if (QUIET.has(entry.event) || entry.event.endsWith(".waiting") || entry.level === "debug")
      return this.detail(entry);
    if (entry.event === "worker.output") return this.workerOutput(entry);
    if (/^(audio|vision|media)\./.test(entry.event) && entry.level === "info") return this.detail(entry);
    // Unknown events stay visible: nothing new is silently swallowed.
    return { tone: "info", text: entry.message, note: took, essential: true };
  }

  /**
   * Python stderr: progress bars are details; a multi-line PyTorch warning (glog format) shows its
   * first line only, the continuation lines sharing its source prefix become details.
   */
  private workerOutput(entry: ViewEntry): ViewLine {
    if (/\d+%\||it\/s\]|s\/it\]/.test(entry.message)) return this.detail(entry);
    const glog = /^([IWEF])\d{4} [\d:.]+\s+\d+ (\S+)\]\s*(?:\[[^\]]*\])?\s*(.*)$/.exec(entry.message);
    if (!glog) {
      this.lastOutputSource = "";
      return { tone: "info", text: entry.message, essential: true };
    }
    const [, severity, source, text] = glog;
    if (source === this.lastOutputSource || !text.trim()) return this.detail(entry);
    this.lastOutputSource = source;
    return {
      tone: severity === "E" || severity === "F" ? "error" : severity === "W" ? "warn" : "info",
      text: "PyTorch : " + text.trim(),
      note: "détails complets dans le journal téléchargeable",
      essential: true,
    };
  }

  private detail(entry: ViewEntry): ViewLine {
    const facts: string[] = [];
    const chunk = number(entry, "chunk");
    const total = number(entry, "totalChunks");
    if (chunk !== undefined && total) facts.push(`${chunk}/${total}`);
    const start = number(entry, "startMs");
    const end = number(entry, "endMs");
    if (start !== undefined && end !== undefined) facts.push(`${formatClock(start)}–${formatClock(end)}`);
    const duration = number(entry, "durationMs") ?? number(entry, "elapsedMs");
    if (duration !== undefined) facts.push(formatDuration(duration));
    const speed = number(entry, "tokensPerSecond");
    if (speed !== undefined) facts.push(`${Math.round(speed)} tokens/s`);
    const free = number(entry, "vramFreeBytes");
    if (free !== undefined) facts.push(`VRAM libre ${formatBytes(free)}`);
    return { tone: "detail", text: entry.message, note: facts.join(" · ") || undefined, essential: false };
  }

  /**
   * One line per step of 10 % (or every 30 s, plus the first and last ones); the others are details.
   * The remaining time ignores the first step, which includes warm-up (kernel compilation, first prompt).
   */
  private progress(
    key: string,
    at: number,
    label: string,
    done: number,
    total: number,
    note: string | undefined,
    steady: boolean,
    since?: number,
  ): ViewLine {
    let tracker = this.trackers.get(key);
    if (!tracker || done < (tracker.firstDone ?? 0)) {
      tracker = { startedAt: since ?? at };
      this.trackers.set(key, tracker);
    }
    if (tracker.firstDoneAt === undefined) {
      tracker.firstDoneAt = at;
      tracker.firstDone = done;
    }
    let remaining: string | undefined;
    const counted = done - (steady ? 0 : tracker.firstDone!);
    const spent = at - (steady ? tracker.startedAt : tracker.firstDoneAt);
    if (done < total && counted > 0 && spent > 0)
      remaining = "reste ~" + formatDuration((spent / counted) * (total - done));
    const step = Math.floor((done / total) * 10);
    const essential =
      done > 0 &&
      (done >= total ||
        tracker.shownStep === undefined ||
        step > tracker.shownStep ||
        at - (tracker.shownAt ?? 0) >= 30000);
    if (essential) {
      tracker.shownStep = step;
      tracker.shownAt = at;
    }
    return {
      tone: "progress",
      text: label,
      note: joined(note, remaining),
      progress: { done, total },
      key: `${this.videoIndex}:${key}`,
      essential,
    };
  }

  private reset() {
    this.totalVideos = this.videoIndex = 0;
    this.videoStartedAt = this.videoDurationMs = this.sourceBytes = undefined;
    this.trackers.clear();
    this.stageStarts.clear();
    this.stageTimes = [];
  }
}

/**
 * Live view of a journal: every update of a progress bar replaces the previous one, in place, so
 * each stage reads as one bar that fills up instead of a list of percentages.
 */
export function collapseProgress<T extends { view: ViewLine }>(lines: T[]): T[] {
  const result: T[] = [];
  const slots = new Map<string, number>();
  for (const line of lines) {
    const key = line.view.tone === "progress" ? line.view.key : undefined;
    const slot = key === undefined ? undefined : slots.get(key);
    if (slot !== undefined) result[slot] = line;
    else {
      if (key !== undefined) slots.set(key, result.length);
      result.push(line);
    }
  }
  return result;
}

export function progressBar(done: number, total: number, width = 20): string {
  const ratio = total > 0 ? Math.min(1, Math.max(0, done / total)) : 0;
  const filled = Math.round(ratio * width);
  return (
    "█".repeat(filled) + "░".repeat(width - filled) + " " + String(Math.floor(ratio * 100)).padStart(3) + " %"
  );
}

const ICONS: Record<Tone, string> = {
  run: "■",
  video: "▶",
  stage: "◆",
  done: "✓",
  progress: " ",
  info: "•",
  detail: "·",
  warn: "⚠",
  error: "✖",
};
const COLORS: Record<Tone, string> = {
  run: "1;35",
  video: "1;36",
  stage: "1;34",
  done: "32",
  progress: "36",
  info: "0",
  detail: "2",
  warn: "33",
  error: "1;31",
};

/** One terminal line; colors only when the output supports them. */
export function formatTerminalLine(entry: ViewEntry, line: ViewLine, color: boolean): string {
  const paint = (code: string, text: string) =>
    color && code !== "0" ? `\x1b[${code}m${text}\x1b[0m` : text;
  const time = new Date(entry.timestamp);
  const clock = Number.isNaN(time.getTime())
    ? "--:--:--"
    : [time.getHours(), time.getMinutes(), time.getSeconds()]
        .map((part) => String(part).padStart(2, "0"))
        .join(":");
  const head = paint("2", clock) + "  " + paint(COLORS[line.tone], ICONS[line.tone]);
  const body = line.progress
    ? line.text.padEnd(24) + paint(COLORS.progress, progressBar(line.progress.done, line.progress.total))
    : paint(line.tone === "done" || line.tone === "info" ? "0" : COLORS[line.tone], line.text);
  const note = line.note ? paint("2", "  " + line.note) : "";
  const separator =
    line.tone === "video" && line.text.startsWith("Vidéo ") && !line.text.startsWith("Vidéo terminée")
      ? paint("2", "─".repeat(72)) + "\n"
      : "";
  return separator + head + " " + body + note;
}

/**
 * Same rule as Vite's output: under `npm run dev`, concurrently pipes the server output, so Windows
 * consoles (which all understand ANSI colors) get them even without a TTY.
 */
export function supportsColor(
  stream: { isTTY?: boolean } | undefined,
  env: Record<string, string | undefined>,
  platform: string,
): boolean {
  if (env.NO_COLOR !== undefined && env.NO_COLOR !== "") return false;
  if (env.FORCE_COLOR !== undefined) return env.FORCE_COLOR !== "0" && env.FORCE_COLOR !== "false";
  return Boolean(stream?.isTTY) || platform === "win32";
}

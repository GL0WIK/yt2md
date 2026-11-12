import { AsyncLocalStorage } from "node:async_hooks";
import fs from "node:fs/promises";
import path from "node:path";
import { LogPresenter, formatTerminalLine, supportsColor } from "./logView.js";
import { terminal } from "./terminal.js";

export type LogLevel = "debug" | "info" | "warn" | "error";
export type Fields = Record<string, string | number | boolean | null | undefined>;
export type LogEntry = Fields & { timestamp: string; event: string; level: LogLevel; message: string };
type Context = { logger?: FileLogger; fields: Fields };
const context = new AsyncLocalStorage<Context>();
const ranks = { debug: 0, info: 1, warn: 2, error: 3 };
const runPattern = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/;

export function sanitizeLogText(value: string, limit = 4000): string {
  return value
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
    .replace(/[\x00-\x08\x0b-\x1f\x7f]/g, "")
    .replace(/\b(?:hf_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,})\b/g, "[redacted]")
    .replace(/\bauthorization\s*:\s*(?:Bearer|Basic)\s+[^\s,;]+/gi, "authorization=[redacted]")
    .replace(
      /\b(authorization|api[_-]?key|access[_-]?token|password|cookie|secret)\s*[:=]\s*[^\s,;]+/gi,
      "$1=[redacted]",
    )
    .replace(/https?:\/\/[^\s<>"']+/g, (value) => {
      try {
        const url = new URL(value);
        url.username = url.password = url.search = url.hash = "";
        return url.toString();
      } catch {
        return "[url]";
      }
    })
    .slice(0, limit);
}

export function captureLogContext(): Context {
  return context.getStore() ?? { fields: {} };
}
export function withLogContext<T>(fields: Fields, work: () => T, logger?: FileLogger): T {
  const current = captureLogContext();
  return context.run({ logger: logger ?? current.logger, fields: { ...current.fields, ...fields } }, work);
}
export function withCapturedLogContext<T>(value: Context, work: () => T): T {
  return context.run(value, work);
}

/** One append-only journal per batch; the server journal rotates independently. */
export class FileLogger {
  private writes: Promise<void> = Promise.resolve();
  private queued = 0;
  private warned = false;
  private readonly presenter = new LogPresenter();
  private readonly color = supportsColor(process.stdout, process.env, process.platform);
  constructor(
    readonly directory: string,
    private readonly level: LogLevel = "info",
    private readonly maxBytes = 20 * 1024 * 1024,
  ) {}
  write(entry: LogEntry) {
    if (ranks[entry.level] < ranks[this.level]) return;
    this.print(entry);
    // A slow/full disk must not grow an unbounded queue or stop inference.
    if (this.queued >= 2000) {
      this.warnOnce("File de journal saturée : les nouveaux messages restent visibles dans le terminal.");
      return;
    }
    this.queued++;
    const record = JSON.stringify(entry) + "\n";
    this.writes = this.writes.then(async () => {
      try {
        await fs.mkdir(this.directory, { recursive: true });
        if (typeof entry.runId === "string" && runPattern.test(entry.runId)) {
          const directory = path.join(this.directory, "runs");
          await fs.mkdir(directory, { recursive: true });
          await fs.appendFile(path.join(directory, entry.runId + ".jsonl"), record, "utf8");
        }
        const server = path.join(this.directory, "server.jsonl");
        const size = await fs
          .stat(server)
          .then((stat) => stat.size)
          .catch((error: NodeJS.ErrnoException) => {
            if (error.code === "ENOENT") return 0;
            throw error;
          });
        if (size + Buffer.byteLength(record) > this.maxBytes) {
          for (let i = 3; i >= 1; i--) {
            const from = i === 1 ? server : `${server}.${i - 1}`;
            await fs.rm(`${server}.${i}`, { force: true });
            await fs.rename(from, `${server}.${i}`).catch((error: NodeJS.ErrnoException) => {
              if (error.code !== "ENOENT") throw error;
            });
          }
        }
        await fs.appendFile(server, record, "utf8");
      } catch (error) {
        this.warnOnce(
          "Écriture du journal impossible : " + (error instanceof Error ? error.message : "erreur disque"),
        );
      } finally {
        this.queued--;
      }
    });
  }
  /**
   * The terminal shows the readable view; LOG_LEVEL=debug adds every technical event, dimmed.
   * In a real terminal, each progress bar fills up in place on the last line.
   */
  private print(entry: LogEntry) {
    const view = this.presenter.present(entry);
    if (terminal.interactive && view.tone === "progress" && view.key && view.progress) {
      const { done, total } = view.progress;
      terminal.progress(view.key, formatTerminalLine(entry, view, this.color), done >= total);
      return;
    }
    if (!view.essential && this.level !== "debug") return;
    // A stage result replaces its bar; a new stage keeps the previous bar where it stopped (pause, error).
    if (view.tone === "done") terminal.drop();
    else if (["run", "video", "stage"].includes(view.tone)) terminal.commit();
    terminal.print(formatTerminalLine(entry, view, this.color));
  }
  private warnOnce(message: string) {
    if (this.warned) return;
    this.warned = true;
    console.warn("[logs] " + sanitizeLogText(message));
  }
  async flush() {
    await this.writes;
  }
}

export function logEvent(event: string, fields: Fields = {}, level: LogLevel = "info") {
  const current = captureLogContext();
  const safe: Fields = {};
  for (const [key, value] of Object.entries({ ...current.fields, ...fields })) {
    if (["timestamp", "event", "level", "message"].includes(key) || value === undefined) continue;
    safe[key] = /^(authorization|cookie|password|secret|apiKey|accessToken)$/i.test(key)
      ? "[redacted]"
      : typeof value === "string"
        ? sanitizeLogText(value)
        : value;
  }
  const entry: LogEntry = {
    ...safe,
    timestamp: new Date().toISOString(),
    event: sanitizeLogText(event, 100),
    level,
    message: sanitizeLogText(typeof fields.message === "string" ? fields.message : event),
  };
  if (current.logger) current.logger.write(entry);
  else {
    const line = JSON.stringify(entry);
    if (level === "error") console.error(line);
    else if (level === "warn") console.warn(line);
    else console.log(line);
  }
}

// Error objects can contain large request bodies; only select diagnostic metadata.
export function errorFields(error: unknown): Fields {
  const record = error && typeof error === "object" ? (error as Record<string, unknown>) : {};
  const status = record.status ?? record.statusCode;
  const code = record.code;
  return {
    errorType: error instanceof Error ? error.name : "UnknownError",
    httpStatus: typeof status === "number" ? status : undefined,
    code: typeof code === "string" && /^[A-Z0-9_]{1,40}$/.test(code) ? code : undefined,
  };
}

export async function resolveRunLog(directory: string, runId: string): Promise<string | null> {
  if (!runPattern.test(runId)) throw new Error("Identifiant de journal invalide.");
  try {
    const root = await fs.realpath(directory);
    const filename = await fs.realpath(path.join(root, "runs", runId + ".jsonl"));
    const relative = path.relative(root, filename);
    if (relative.startsWith("..") || path.isAbsolute(relative))
      throw new Error("Chemin de journal invalide.");
    if (!(await fs.stat(filename)).isFile()) throw new Error("Journal invalide.");
    return filename;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

const MAX_LOG_READ = 512 * 1024;
async function findNewline(handle: fs.FileHandle, from: number, size: number): Promise<number> {
  const buffer = Buffer.alloc(64 * 1024);
  for (let offset = from; offset < size;) {
    const { bytesRead } = await handle.read(buffer, 0, Math.min(buffer.length, size - offset), offset);
    if (!bytesRead) break;
    const index = buffer.subarray(0, bytesRead).indexOf(10);
    if (index >= 0) return offset + index;
    offset += bytesRead;
  }
  return -1;
}

/** Byte cursors avoid rereading a complete 400-video journal every two seconds. */
export async function readRunLog(directory: string, runId: string, cursor?: number, limit = 250) {
  if (
    (cursor !== undefined && (!Number.isSafeInteger(cursor) || cursor < 0)) ||
    !Number.isSafeInteger(limit) ||
    limit < 1 ||
    limit > 500
  )
    throw new Error("Pagination du journal invalide.");
  const filename = await resolveRunLog(directory, runId);
  if (!filename) return { entries: [] as LogEntry[], nextCursor: 0, hasMore: false };
  const handle = await fs.open(filename, "r");
  try {
    const size = (await handle.stat()).size;
    const tail = cursor === undefined;
    const start = tail ? Math.max(0, size - 256 * 1024) : Math.min(cursor, size);
    const buffer = Buffer.alloc(Math.min(size - start, MAX_LOG_READ));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, start);
    const data = buffer.subarray(0, bytesRead);
    const entries: LogEntry[] = [];
    let position = tail && start > 0 ? data.indexOf(10) + 1 : 0;
    let nextCursor = start + position;
    while (position < data.length) {
      const end = data.indexOf(10, position);
      if (end < 0) break;
      try {
        entries.push(JSON.parse(data.subarray(position, end).toString("utf8")) as LogEntry);
      } catch {
        /* A truncated old line does not hide the rest of the journal. */
      }
      position = end + 1;
      nextCursor = start + position;
      if (!tail && entries.length >= limit) break;
    }
    // A single line longer than the read window would otherwise pin the cursor forever (endless re-poll).
    if (position === 0 && data.length === MAX_LOG_READ && data.indexOf(10) < 0) {
      const end = await findNewline(handle, start + data.length, size);
      if (end >= 0) {
        nextCursor = end + 1;
        entries.push({
          timestamp: new Date().toISOString(),
          event: "log.line.skipped",
          level: "warn",
          message: `Ligne de journal trop volumineuse ignorée (${end - start} octets).`,
        });
      }
    }
    return { entries: tail ? entries.slice(-limit) : entries, nextCursor, hasMore: nextCursor < size };
  } finally {
    await handle.close();
  }
}

export async function observeStage<T>(event: string, fields: Fields, work: () => Promise<T>): Promise<T> {
  const started = performance.now();
  const elapsed = () => Math.round(performance.now() - started);
  logEvent(event + ".start", fields);
  const timer = setInterval(() => logEvent(event + ".waiting", { ...fields, elapsedMs: elapsed() }), 30000);
  timer.unref();
  try {
    const result = await work();
    logEvent(event + ".complete", { ...fields, durationMs: elapsed() });
    return result;
  } catch (error) {
    logEvent(event + ".failed", { ...fields, durationMs: elapsed(), ...errorFields(error) }, "error");
    throw error;
  } finally {
    clearInterval(timer);
  }
}

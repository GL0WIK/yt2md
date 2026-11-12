/**
 * Terminal output with one live progress bar: it is redrawn in place on the last line, and the
 * other messages are printed above it. Without a real terminal (pipe, file, CI), every line is
 * printed as is and the journal view keeps its throttled progress lines.
 */
type Stream = { isTTY?: boolean; columns?: number; write(text: string): unknown };

const ANSI = /\x1b\[[0-9;]*m/y;

/** Cuts a colored line to the terminal width: a wrapped live line could not be erased. */
export function fitWidth(text: string, width: number): string {
  if (width <= 0) return text;
  let visible = 0;
  let index = 0;
  let out = "";
  while (index < text.length) {
    ANSI.lastIndex = index;
    const code = ANSI.exec(text);
    if (code) {
      out += code[0];
      index += code[0].length;
      continue;
    }
    if (visible >= width) {
      out += "\x1b[0m";
      break;
    }
    const char = String.fromCodePoint(text.codePointAt(index)!);
    out += char;
    index += char.length;
    visible++;
  }
  return out;
}

export class Terminal {
  private live: { key: string; text: string } | null = null;
  readonly interactive: boolean;
  constructor(
    private readonly out: Stream = process.stdout,
    env: Record<string, string | undefined> = process.env,
  ) {
    this.interactive = Boolean(out.isTTY) && !env.CI;
  }

  /** A permanent line, printed above the live bar. */
  print(text: string) {
    if (!this.interactive) {
      this.out.write(text + "\n");
      return;
    }
    this.erase();
    this.out.write(text + "\n");
    this.draw();
  }

  /** Updates the live bar `key`; a finished bar becomes a permanent line. */
  progress(key: string, text: string, finished: boolean) {
    if (this.live && this.live.key !== key) this.commit();
    if (finished) {
      this.erase();
      this.live = null;
      this.out.write(text + "\n");
      return;
    }
    this.erase();
    this.live = { key, text };
    this.draw();
  }

  /** Leaves the current bar as it is, as a permanent line (stage change, pause, exit). */
  commit() {
    if (!this.live) return;
    this.erase();
    this.out.write(this.live.text + "\n");
    this.live = null;
  }

  /** Removes the current bar: the stage reported its own result line (extraction done…). */
  drop() {
    this.erase();
    this.live = null;
  }

  private erase() {
    if (this.live) this.out.write("\r\x1b[2K");
  }
  private draw() {
    if (this.live) this.out.write(fitWidth(this.live.text, (this.out.columns ?? 120) - 1));
  }
}

export const terminal = new Terminal();
process.once("exit", () => terminal.commit());

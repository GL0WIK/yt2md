/**
 * `npm run dev`: Vite and the API server in one process, so the server writes to the real terminal
 * and can redraw its progress bars in place (a process manager would pipe its output).
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createLogger, createServer, type Logger } from "vite";
import { terminal } from "../server/src/shared/terminal.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const base = createLogger("info", { allowClearScreen: false });
const print = (message: string) => terminal.print(message);
const logger: Logger = {
  ...base,
  info: print,
  warn: print,
  warnOnce: print,
  error: print,
  clearScreen: () => {},
};
const vite = await createServer({ configFile: path.join(root, "vite.config.ts"), customLogger: logger });
await vite.listen();
vite.printUrls();
await import("../server/src/index.js");

import fs from "node:fs/promises";
import path from "node:path";
import { afterEach } from "vitest";
import { PROJECT_ROOT } from "../src/config.js";
const root = path.join(PROJECT_ROOT, ".test-tmp");
const directories: string[] = [];
export async function temporaryDirectory() {
  await fs.mkdir(root, { recursive: true });
  const directory = await fs.mkdtemp(path.join(root, "yt-rag-test-"));
  directories.push(directory);
  return directory;
}
afterEach(async () => {
  for (const directory of directories.splice(0)) {
    const resolved = path.resolve(directory);
    if (path.dirname(resolved) !== path.resolve(root) || !path.basename(resolved).startsWith("yt-rag-test-"))
      throw new Error("Unsafe test cleanup path");
    await fs.rm(resolved, { recursive: true, force: true });
  }
});

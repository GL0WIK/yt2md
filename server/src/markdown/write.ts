import fs from "node:fs/promises";
import { atomicWrite } from "../shared/files.js";
import path from "node:path";
import { CatalogVideo } from "../shared/schemas.js";

export async function writeMarkdownOutput(input: {
  outputDir: string;
  video: CatalogVideo;
  markdown: string;
}): Promise<string> {
  await fs.mkdir(input.outputDir, { recursive: true });

  if (!/^[A-Za-z0-9_-]{11}$/.test(input.video.id)) throw new Error("Identifiant vidéo invalide.");
  const fileName = `video--${input.video.id}.md`;
  const filePath = path.join(input.outputDir, fileName);
  await atomicWrite(filePath, input.markdown);
  return filePath;
}

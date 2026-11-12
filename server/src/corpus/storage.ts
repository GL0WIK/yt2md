import fs from "node:fs/promises";
import path from "node:path";
import { atomicWrite } from "../shared/files.js";
import { CorpusDocument, CorpusDocumentSchema, buildRetrievalChunks } from "./document.js";

export async function writeCorpusDocument(outputDir: string, document: CorpusDocument): Promise<string> {
  const checked = CorpusDocumentSchema.parse(document);
  // One canonical file per YouTube ID, even if the video's title changes.
  const target = path.join(outputDir, "video--" + checked.videoId + ".json");
  await atomicWrite(target, JSON.stringify(checked, null, 2) + "\n");
  return target;
}

async function readCorpusDocument(root: string, name: string): Promise<CorpusDocument> {
  let document: CorpusDocument;
  try {
    document = CorpusDocumentSchema.parse(JSON.parse(await fs.readFile(path.join(root, name), "utf8")));
  } catch (error) {
    throw new Error(
      `${name} invalide (${error instanceof Error ? error.message.slice(0, 200) : "lecture impossible"}).`,
      { cause: error },
    );
  }
  if (name !== "video--" + document.videoId + ".json")
    throw new Error("Identifiant de corpus incohérent : " + name);
  return document;
}

/**
 * Reads and validates every export once, then returns the complete JSONL (empty if no segment).
 * Throws on the first invalid document. Each file is read a single time, so a document replaced or
 * deleted during the export cannot desynchronise validation and content. Bounded by the corpus size
 * (about 1000 videos), which comfortably fits in memory.
 */
export async function buildCorpusExport(outputDir: string): Promise<string> {
  const root = await fs.realpath(outputDir);
  const files = (await fs.readdir(root, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && /^video--[A-Za-z0-9_-]{11}\.json$/.test(entry.name))
    .map((entry) => entry.name)
    .sort();
  const lines: string[] = [];
  for (const name of files) {
    let document: CorpusDocument;
    try {
      document = await readCorpusDocument(root, name);
    } catch (error) {
      // Deleted between readdir and read (concurrent deletion): it is simply no longer in the corpus.
      if ((error as { cause?: NodeJS.ErrnoException }).cause?.code === "ENOENT") continue;
      throw error;
    }
    for (const chunk of buildRetrievalChunks(document)) lines.push(JSON.stringify(chunk) + "\n");
  }
  return lines.join("");
}

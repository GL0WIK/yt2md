import fs from "node:fs/promises";
import path from "node:path";
import { zodToJsonSchema } from "zod-to-json-schema";
import { format, resolveConfig } from "prettier";
import { CorpusDocumentSchema } from "../server/src/corpus/document.js";
import { PROJECT_ROOT } from "../server/src/config.js";

const schema = zodToJsonSchema(CorpusDocumentSchema, { $refStrategy: "none" });
const original = JSON.parse(
  await fs.readFile(path.join(PROJECT_ROOT, "docs/original.corpus.schema.json"), "utf8"),
);
if (JSON.stringify(schema) !== JSON.stringify(original))
  throw new Error("Le schéma local diffère du contrat original. Aucun fichier écrit.");
const target = path.join(PROJECT_ROOT, "docs/corpus.schema.json");
await fs.writeFile(
  target,
  await format(JSON.stringify(schema), { ...(await resolveConfig(target)), parser: "json" }),
);
console.log("Schéma JSON identique à celui du projet d’origine.");

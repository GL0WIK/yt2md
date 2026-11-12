import { describe, expect, it } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import { Ajv } from "ajv";
import { zodToJsonSchema } from "zod-to-json-schema";
import { PROJECT_ROOT } from "../src/config.js";
import { buildCorpusDocument, buildRetrievalChunks, CorpusDocumentSchema } from "../src/corpus/document.js";
import { fixtureAnalysis, fixtureVideo } from "./fixtures.js";

describe("strict original JSON compatibility", () => {
  it("generates exactly the original schema, including every literal, field and optional field", async () => {
    const original = JSON.parse(
      await fs.readFile(path.join(PROJECT_ROOT, "docs/original.corpus.schema.json"), "utf8"),
    );
    expect(zodToJsonSchema(CorpusDocumentSchema, { $refStrategy: "none" })).toEqual(original);
  });
  it("validates a local transcription against the untouched remote JSON schema", async () => {
    const schema = JSON.parse(
      await fs.readFile(path.join(PROJECT_ROOT, "docs/original.corpus.schema.json"), "utf8"),
    );
    const ajv = new Ajv({ allErrors: true, formats: { "date-time": true, uri: true } });
    const validate = ajv.compile(schema);
    const document = buildCorpusDocument(
      {
        ...fixtureAnalysis,
        analysisSettings: {
          model: "Qwen/Qwen3-ASR-1.7B-hf + Qwen/Qwen3.5-9B",
          fps: 1,
          mediaResolution: "video_1fps_7frames",
        },
        transcript: [
          {
            speaker: "Indéterminé",
            timestamp: "00:01",
            endTimestamp: "00:03",
            text: "voici le texte original",
            chunkIndex: 1,
          },
        ],
      },
      {
        videoId: fixtureVideo.id,
        runId: "local-test",
        promptVersion: "local-observed-video-v1",
        chunkSeconds: 7,
      },
    );
    expect(validate(document), JSON.stringify(validate.errors)).toBe(true);
    expect(document.schemaVersion).toBe("2.0");
    expect(document.transcript[0].speakerId).toBeNull();
    expect(document.transcript[0].text).toBe("voici le texte original");
    expect(document).not.toHaveProperty("localProvenance");
    expect(document).not.toHaveProperty("words");
    const rows = [...buildRetrievalChunks(document)];
    expect(rows[0].schemaVersion).toBe("2.0");
    expect(rows[0].text).toContain("voici le texte original");
    expect(rows[0].transcriptIds).toEqual([document.transcript[0].id]);
  });
});

import { describe, expect, it } from "vitest";
import { fixtureAnalysis } from "./fixtures.js";
import { renderVideoAnalysisMarkdown } from "../src/markdown/render.js";

describe("renderVideoAnalysisMarkdown", () => {
  it("renders separated transcript, visual, and keyword sections", () => {
    const markdown = renderVideoAnalysisMarkdown(fixtureAnalysis);

    expect(markdown).toContain("# Video de test");
    expect(markdown).toContain("## Fiche source");
    expect(markdown).not.toContain("## Statistiques publiques");
    expect(markdown).not.toContain("Parametres d'analyse Gemini");
    expect(markdown).toContain("## Transcription");
    expect(markdown).toContain("**Narrateur** · 00:01\n\nBonjour");
    expect(markdown).toContain("## Segments visuels");
    expect(markdown).toContain("### 00:00 - 00:15 | Introduction");
    expect(markdown).toContain("Texte à l’écran :");
    expect(markdown).not.toContain("## Segments mots-cles");
    expect(markdown).not.toContain("## Timeline RAG");
    expect(markdown).not.toContain("#### Transcription audio");
    expect(markdown).not.toContain("Notes de validation");
  });

  it("renders empty screen text without failing", () => {
    const markdown = renderVideoAnalysisMarkdown({
      ...fixtureAnalysis,
      visualSegments: [
        {
          timestamp: "00:00",
          endTimestamp: "02:00",
          title: "Plan fixe",
          visualDescription: "Plan fixe",
          screenText: [],
        },
      ],
    });

    expect(markdown).not.toContain("Texte à l’écran :");
  });
});

it("keeps diagnostics and per-segment YouTube links out of the readable export", () => {
  const markdown = renderVideoAnalysisMarkdown({
    ...fixtureAnalysis,
    transcript: [{ ...fixtureAnalysis.transcript[0], speaker: "Artus", chunkIndex: 1 }],
    visualSegments: [{ ...fixtureAnalysis.visualSegments[0], uncertainties: ["DIAGNOSTIC_TEST"] }],
  });
  expect(markdown).not.toContain("DIAGNOSTIC_TEST");
  expect(markdown).not.toContain("Qualité et limites");
  expect(markdown).not.toContain("&t=");
  expect(markdown).not.toContain("E1 /");
  expect(markdown).toContain("## Intervenants\n\n- Artus");
  expect(markdown).toContain("**Artus**");
});

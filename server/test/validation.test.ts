import { describe, expect, it } from "vitest";
import { fixtureAnalysis, fixtureVideo } from "./fixtures.js";
import { validateAndCanonicalizeAnalysis } from "../src/validation/analysis.js";

const settings = fixtureAnalysis.analysisSettings;

describe("validateAndCanonicalizeAnalysis", () => {
  it("accepts a complete structured JSON payload", () => {
    const result = validateAndCanonicalizeAnalysis(fixtureAnalysis, fixtureVideo, settings);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.analysis.metadata.title).toBe(fixtureVideo.title);
      expect(result.analysis.publicStats.likes).toBeNull();
      expect(result.analysis.transcript[0].text).toBe("Bonjour");
    }
  });

  it("drops 'no visible text' from observation limits but keeps real ones", () => {
    const [segment] = fixtureAnalysis.visualSegments;
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        visualSegments: [
          {
            ...segment,
            uncertainties: [
              "Aucun texte visible à l'écran.",
              "Aucune inscription n'est visible sur les objets.",
              "Certaines inscriptions sont partiellement coupées.",
              "Aucun texte visible au-delà du logo partiellement masqué.",
            ],
          },
        ],
      },
      fixtureVideo,
      settings,
    );
    expect(result.ok).toBe(true);
    if (result.ok)
      expect(result.analysis.visualSegments[0].uncertainties).toEqual([
        "Certaines inscriptions sont partiellement coupées.",
        "Aucun texte visible au-delà du logo partiellement masqué.",
      ]);
  });

  it("rejects missing required fields", () => {
    const result = validateAndCanonicalizeAnalysis(
      { metadata: fixtureAnalysis.metadata },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.issues.join("\n")).toContain("transcript");
    }
  });

  it("canonicalizes trusted yt-dlp metadata while keeping the strict analysis shape", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        metadata: {
          title: "Titre modele different",
          url: "https://www.youtube.com/watch?v=different",
          channel: "Chaine modele",
          publicationDate: "2020-01-01",
          durationSeconds: 999,
        },
        publicStats: fixtureVideo.publicStats,
        youtubeDescription: fixtureVideo.description,
        analysisSettings: {
          model: settings.model,
          fps: settings.fps,
          mediaResolution: settings.mediaResolution,
        },
        transcript: fixtureAnalysis.transcript,
        visualSegments: fixtureAnalysis.visualSegments,
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.analysis.metadata.title).toBe(fixtureVideo.title);
      expect(result.analysis.metadata.url).toBe(fixtureVideo.url);
      expect(result.analysis.transcript).toEqual(fixtureAnalysis.transcript);
    }
  });

  it("rejects invalid public stats", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        publicStats: {
          views: -1,
          likes: null,
          comments: 0,
        },
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(false);
  });

  it("accepts a silent video with a quality warning", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        transcript: [],
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(true);
    if (result.ok) expect(result.warnings.join(" ")).toContain("Aucune parole");
  });

  it("rejects unsorted transcript timestamps", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        transcript: [
          { ...fixtureAnalysis.transcript[0], timestamp: "00:30" },
          { ...fixtureAnalysis.transcript[1], timestamp: "00:10" },
        ],
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.issues.join("\n")).toContain("earlier than the previous transcript entry");
    }
  });

  it("rejects timestamps beyond the known duration", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        visualSegments: [{ ...fixtureAnalysis.visualSegments[0], timestamp: "03:00" }],
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.issues.join("\n")).toContain("exceeds the known video duration");
    }
  });

  it("keeps visual coverage gaps as warnings instead of blocking Markdown generation", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        visualSegments: [{ ...fixtureAnalysis.visualSegments[0], timestamp: "00:00", endTimestamp: "00:20" }],
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.warnings.join("\n")).toContain("visualSegments coverage is incomplete");
    }
  });

  it("preserves long observations without duration-only warnings or fabricated detail", () => {
    const segment = { ...fixtureAnalysis.visualSegments[0], endTimestamp: "00:30" };
    const result = validateAndCanonicalizeAnalysis(
      { ...fixtureAnalysis, visualSegments: [segment, ...fixtureAnalysis.visualSegments.slice(2)] },
      fixtureVideo,
      settings,
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.analysis.visualSegments[0]).toEqual(segment);
      expect(result.analysis.visualSegments).toHaveLength(7);
      expect(result.warnings).toEqual([]);
    }
  });

  it("keeps large gaps between visual segments as warnings", () => {
    const result = validateAndCanonicalizeAnalysis(
      {
        ...fixtureAnalysis,
        visualSegments: [
          fixtureAnalysis.visualSegments[0],
          { ...fixtureAnalysis.visualSegments[1], timestamp: "00:40", endTimestamp: "00:55" },
        ],
      },
      fixtureVideo,
      settings,
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.warnings.join("\n")).toContain("gap exceeds 15s");
    }
  });
});

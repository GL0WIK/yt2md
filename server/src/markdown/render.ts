import { VideoAnalysis } from "../shared/schemas.js";
import { formatSecondsAsTimestamp } from "../shared/time.js";

/** Readable export: diagnostics and quality warnings stay in the JSON files, never in the Markdown. */
export function renderVideoAnalysisMarkdown(analysis: VideoAnalysis): string {
  const id = new URL(analysis.metadata.url).searchParams.get("v");
  const lines = [
    "---",
    'schema_version: "2.0"',
    "video_id: " + JSON.stringify(id),
    "source_url: " + JSON.stringify(analysis.metadata.url),
    "title: " + JSON.stringify(analysis.metadata.title),
    "---",
    "",
    "# " + inline(analysis.metadata.title),
    "",
    "## Fiche source",
    "",
    "- **Titre**: " + inline(analysis.metadata.title),
    "- **URL**: " + analysis.metadata.url,
    "- **Chaîne**: " + inline(analysis.metadata.channel ?? "Non disponible"),
    "- **Date de publication**: " + (analysis.metadata.publicationDate ?? "Non disponible"),
    "- **Durée**: " + (formatSecondsAsTimestamp(analysis.metadata.durationSeconds) || "Non disponible"),
    "",
    "## Description YouTube",
    "",
    ...(analysis.youtubeDescription || "Description non disponible.")
      .split(/\r?\n/)
      .map((line) => "> " + inline(line)),
    "",
    "## Intervenants",
    "",
    ...[...new Set(analysis.transcript.map((entry) => entry.speaker))].map(
      (speaker) => "- " + inline(speaker),
    ),
    "",
    "## Transcription",
    "",
  ];
  if (!analysis.transcript.length) lines.push("- Aucune parole détectée");
  for (const entry of analysis.transcript) {
    const time = entry.timestamp + (entry.endTimestamp ? "–" + entry.endTimestamp : "");
    lines.push("**" + inline(entry.speaker) + "** · " + time, "", inline(entry.text), "");
  }
  lines.push("", "## Segments visuels", "");
  for (const segment of analysis.visualSegments) {
    lines.push("### " + segment.timestamp + " - " + segment.endTimestamp + " | " + inline(segment.title), "");
    lines.push(inline(segment.visualDescription), "");
    if (segment.screenText.length) {
      lines.push("Texte à l’écran :", "", ...segment.screenText.map((text) => "- " + inline(text)));
    }
    lines.push("");
  }
  return lines.join("\n").trimEnd() + "\n";
}
function inline(value: string): string {
  return value
    .replace(/[\r\n]+/g, " ")
    .replace(/[\\*_[\]<>#|]/g, "\\$&")
    .replaceAll(String.fromCharCode(96), "\\" + String.fromCharCode(96));
}

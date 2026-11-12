import { Pause, Play } from "lucide-react";
import type { AnalysisProgress, AnalysisRun } from "../types";

// Typed by the server phases: a new phase cannot be forgotten here.
export const phases: Record<AnalysisProgress["phase"], string> = {
  metadata: "Métadonnées",
  download: "Téléchargement",
  extract: "Extraction du média",
  audio: "Transcription Qwen3-ASR",
  vision: "Analyse vidéo Qwen",
  export: "Export JSON et Markdown",
};

type Props = {
  run: AnalysisRun;
  working: boolean;
  busy: boolean;
  ready: boolean;
  onPause(): void;
  onResume(): void;
};

export function RunPanel({ run, working, busy, ready, onPause, onResume }: Props) {
  const running = run.status === "running";
  return (
    <div className="run-panel">
      <div className="run-heading">
        <strong>
          {running
            ? "Traitement en cours"
            : run.status === "completed"
              ? run.results.some((r) => r.status === "error")
                ? "Traitement terminé avec des erreurs"
                : "Traitement terminé"
              : "Traitement interrompu"}
        </strong>
        <span>
          {run.completed} / {run.total} vidéos traitées
        </span>
        {running ? (
          <button disabled={working} onClick={onPause}>
            <Pause size={16} /> Mettre en pause
          </button>
        ) : (
          (run.status !== "completed" || run.results.some((r) => r.status === "error")) && (
            <button disabled={busy || !ready} onClick={onResume}>
              <Play size={16} /> Reprendre
            </button>
          )
        )}
      </div>
      <progress max={run.total || 1} value={run.completed} />
      {run.progress && (
        <p>
          {phases[run.progress.phase]} · {run.progress.videoTitle}
          <br />
          <span>
            {run.progress.detail} {run.progress.chunk}/{run.progress.totalChunks}
          </span>
        </p>
      )}
      {run.error && <p>{run.error}</p>}
    </div>
  );
}

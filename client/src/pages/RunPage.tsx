import React from "react";
import { RunPanel } from "../components/RunPanel";
import { VideoList } from "../components/VideoList";
import { RunLogs } from "../RunLogs";
import { buildResults, videoState, type Library } from "../library";
import type { AnalysisRun } from "../types";

type Props = {
  run: AnalysisRun | null;
  library: Library;
  working: boolean;
  ready: boolean;
  onPause(): void;
  onResume(): void;
};

/** The current (or last) processing: its videos one by one, pause/resume, and the journal. */
export function RunPage({ run, library, working, ready, onPause, onResume }: Props) {
  const results = React.useMemo(
    () => buildResults(run?.videos ?? [], library.exports, run),
    [run, library.exports],
  );
  if (!run)
    return (
      <section className="workbench">
        <div className="empty">
          Aucun traitement pour l’instant. Choisissez des vidéos dans une chaîne de la bibliothèque, ou collez
          le lien d’une vidéo.
        </div>
      </section>
    );
  return (
    <>
      <section className="workbench">
        <RunPanel
          run={run}
          working={working}
          busy={working}
          ready={ready}
          onPause={onPause}
          onResume={onResume}
        />
        <VideoList
          videos={run.videos}
          results={results}
          stateOf={(id) => videoState(id, results, run)}
          selected={new Set()}
          onToggle={() => {}}
          hideSelection
        />
      </section>
      <section className="workbench">
        <RunLogs runId={run.runId} status={run.status} open />
      </section>
    </>
  );
}

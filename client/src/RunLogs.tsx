import React from "react";
import { Download } from "lucide-react";
import { request } from "./api";
import {
  LogPresenter,
  collapseProgress,
  type ViewEntry,
  type ViewLine,
} from "../../server/src/shared/logView";

type Page = { entries: ViewEntry[]; nextCursor: number; hasMore: boolean; size: number };
type Filter = "essential" | "all" | "issues" | "error";
const icons: Record<ViewLine["tone"], string> = {
  run: "■",
  video: "▶",
  stage: "◆",
  done: "✓",
  progress: "",
  info: "•",
  detail: "·",
  warn: "⚠",
  error: "✖",
};
// A 20-minute video produces about 2,000 events (600 KB); older ones are dropped beyond this.
const KEPT = 6000;
// Up to this size the journal is replayed from its start, so headings and stage timings are complete.
const REPLAY_BYTES = 4 * 1024 * 1024;
const SHOWN = 400;

function visible(view: ViewLine, entry: ViewEntry, filter: Filter) {
  if (filter === "all") return true;
  // Every update counts here: each bar is then shown once, at its latest value.
  if (filter === "essential") return view.essential || view.tone === "progress";
  if (filter === "error") return entry.level === "error";
  return entry.level === "warn" || entry.level === "error";
}

export function RunLogs({ runId, status, open }: { runId: string; status: string; open?: boolean }) {
  const [entries, setEntries] = React.useState<ViewEntry[]>([]);
  const [filter, setFilter] = React.useState<Filter>("essential");
  const [follow, setFollow] = React.useState(true);
  const [error, setError] = React.useState("");
  const cursor = React.useRef<number | undefined>(undefined);
  const viewport = React.useRef<HTMLDivElement | null>(null);
  React.useEffect(() => {
    cursor.current = undefined;
    setEntries([]);
    setError("");
  }, [runId]);
  React.useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const previous = cursor.current;
        const page = await request<Page>(
          `/api/runs/${runId}/logs?limit=500${previous === undefined ? "" : `&cursor=${previous}`}`,
        );
        if (stopped) return;
        if (previous === undefined && page.size <= REPLAY_BYTES && page.nextCursor > 0) {
          // First read: the tail is enough for a large batch; a small journal is replayed from its start.
          cursor.current = 0;
          timer = setTimeout(() => void poll(), 0);
          return;
        }
        cursor.current = page.nextCursor;
        setEntries((old) => [...old, ...page.entries].slice(-KEPT));
        setError("");
        if (page.hasMore && page.nextCursor !== previous) {
          timer = setTimeout(() => void poll(), 50);
          return;
        }
        if (status !== "running") return;
      } catch (error) {
        if (stopped) return;
        setError(error instanceof Error ? error.message : "Journal indisponible.");
        // A finished run's journal does not change: retrying forever would only repeat the same error.
        if (status !== "running") return;
      }
      timer = setTimeout(() => void poll(), 2000);
    }
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [runId, status]);
  const lines = React.useMemo(() => {
    const presenter = new LogPresenter();
    return entries.map((entry) => ({ entry, view: presenter.present(entry) }));
  }, [entries]);
  const filtered = lines.filter(({ entry, view }) => visible(view, entry, filter));
  const shown = filter === "essential" ? collapseProgress(filtered) : filtered;
  React.useEffect(() => {
    if (follow && viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight;
  }, [entries, follow, filter]);
  return (
    <details className="journal" open={open || status === "running" || undefined}>
      <summary>Journal du dernier traitement</summary>
      <div className="journal-toolbar">
        <label>
          Afficher
          <select
            aria-label="Filtrer le journal"
            value={filter}
            onChange={(event) => setFilter(event.target.value as Filter)}
          >
            <option value="essential">Étapes et progression</option>
            <option value="all">Tous les messages techniques</option>
            <option value="issues">Avertissements et erreurs</option>
            <option value="error">Erreurs</option>
          </select>
        </label>
        <label className="checkline">
          <input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />
          Suivre les nouveaux logs
        </label>
        <a className="button" href={`/api/runs/${runId}/logs/download`}>
          <Download size={15} />
          Télécharger le journal
        </a>
      </div>
      <p className="journal-hint">
        {shown.length > SHOWN ? `Les ${SHOWN} derniers messages sur ${shown.length} sont affichés. ` : ""}
        Le téléchargement contient tout le traitement, avec chaque événement technique.
      </p>
      {error && (
        <p role="alert" className="failure">
          {error}
        </p>
      )}
      <div className="journal-lines" ref={viewport} tabIndex={0} aria-label="Messages du journal">
        {!shown.length && (
          <p className="muted">
            {entries.length ? "Aucun message pour ce filtre." : "En attente des premiers messages…"}
          </p>
        )}
        {shown.slice(-SHOWN).map(({ entry, view }, index) => {
          const { message: _message, timestamp, level: _level, event, ...fields } = entry;
          const percent = view.progress
            ? Math.min(100, Math.floor((view.progress.done / view.progress.total) * 100))
            : null;
          return (
            <div
              className={`journal-line journal-${view.tone}`}
              key={view.key ?? `${timestamp}-${event}-${index}`}
            >
              <time className="journal-time" dateTime={timestamp}>
                {new Date(timestamp).toLocaleTimeString("fr-FR")}
              </time>
              <span className="journal-icon" aria-hidden="true">
                {icons[view.tone]}
              </span>
              <div className="journal-content">
                <div className="journal-head">
                  <strong>{view.text}</strong>
                  {percent !== null && (
                    <span
                      className="journal-bar"
                      role="progressbar"
                      aria-valuenow={percent}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    >
                      <span style={{ width: `${percent}%` }} />
                    </span>
                  )}
                  {percent !== null && <span className="journal-percent">{percent} %</span>}
                </div>
                {view.note && <span className="journal-context">{view.note}</span>}
                {filter !== "essential" && (
                  <details>
                    <summary>{event} · détails</summary>
                    <pre>{JSON.stringify(fields, null, 2)}</pre>
                  </details>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </details>
  );
}

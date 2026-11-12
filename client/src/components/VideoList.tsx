import React from "react";
import { AlertTriangle, CheckCircle2, Clock, Loader2, Play } from "lucide-react";
import { formatDuration } from "../format";
import type { AnalyzeVideoResult, CatalogVideo } from "../types";
import type { VideoState } from "../library";

type Props = {
  videos: CatalogVideo[];
  results: Record<string, AnalyzeVideoResult>;
  stateOf(id: string): VideoState;
  selected: Set<string>;
  onToggle(id: string, checked: boolean): void;
  /** Read-only list (the processing page). */
  hideSelection?: boolean;
  /** Starts this video alone; absent while a processing is running or the engine is not ready. */
  onTranscribe?(video: CatalogVideo): void;
  hasMore?: boolean;
  loadingMore?: boolean;
  onEndReached?(): void;
};

const badges: Record<VideoState, { label: string; icon: React.ReactNode }> = {
  todo: { label: "À traiter", icon: null },
  queued: { label: "En file", icon: <Clock size={14} /> },
  running: { label: "En cours", icon: <Loader2 size={14} className="spin" /> },
  done: { label: "Traitée", icon: <CheckCircle2 size={14} /> },
  error: { label: "Erreur", icon: <AlertTriangle size={14} /> },
};

function details(video: CatalogVideo) {
  const parts: string[] = [];
  if (video.publishedAt) parts.push(new Date(video.publishedAt).toLocaleDateString("fr-FR"));
  if (video.durationSeconds !== null) parts.push(formatDuration(video.durationSeconds));
  if (video.publicStats.views !== null)
    parts.push(
      new Intl.NumberFormat("fr-FR", { notation: "compact" }).format(video.publicStats.views) + " vues",
    );
  return parts.join(" · ");
}

export function VideoList(props: Props) {
  const { videos, results, stateOf, selected, onToggle, onTranscribe, hasMore, loadingMore, onEndReached } =
    props;
  const selectable = !props.hideSelection;
  const sentinel = React.useRef<HTMLDivElement | null>(null);
  const reached = React.useRef(onEndReached);
  reached.current = onEndReached;
  React.useEffect(() => {
    if (!hasMore || loadingMore || !sentinel.current) return;
    // Recreated after each page: a sentinel still in view loads the next one.
    const observer = new IntersectionObserver(
      (items) => {
        if (items.some((item) => item.isIntersecting)) reached.current?.();
      },
      { rootMargin: "600px 0px" },
    );
    observer.observe(sentinel.current);
    return () => observer.disconnect();
  }, [hasMore, loadingMore, videos.length]);

  return (
    <div className={selectable ? "video-list" : "video-list read-only"}>
      {videos.map((video) => {
        const state = stateOf(video.id);
        const result = results[video.id];
        return (
          <article key={video.id} className={`video-row state-${state}`}>
            {selectable && (
              <input
                type="checkbox"
                aria-label={`Sélectionner ${video.title}`}
                checked={selected.has(video.id)}
                onChange={(event) => onToggle(video.id, event.target.checked)}
              />
            )}
            <a className="thumb" href={video.url} target="_blank" rel="noreferrer" tabIndex={-1}>
              <img src={`https://i.ytimg.com/vi/${video.id}/mqdefault.jpg`} alt="" loading="lazy" />
              {video.durationSeconds !== null && <span>{formatDuration(video.durationSeconds)}</span>}
            </a>
            <div className="video-main">
              <a className="video-title" href={video.url} target="_blank" rel="noreferrer">
                {video.title}
              </a>
              <span className="video-meta">{details(video)}</span>
              {result?.status === "error" && <span className="failure">{result.error}</span>}
            </div>
            <div className="video-side">
              <span className={`badge badge-${state}`}>
                {badges[state].icon}
                {badges[state].label}
              </span>
              {result?.status === "success" ? (
                <span className="file-links">
                  {result.downloadUrl && <a href={result.downloadUrl}>MD</a>}
                  {result.jsonDownloadUrl && <a href={result.jsonDownloadUrl}>JSON</a>}
                </span>
              ) : (
                onTranscribe &&
                (state === "todo" || state === "error") && (
                  <button className="ghost small" onClick={() => onTranscribe(video)}>
                    <Play size={14} /> {state === "error" ? "Relancer" : "Transcrire"}
                  </button>
                )
              )}
            </div>
          </article>
        );
      })}
      {(hasMore || loadingMore) && (
        <div className="load-more" ref={sentinel} role="status">
          {loadingMore ? (
            <>
              <Loader2 className="spin" size={16} /> Chargement des vidéos suivantes…
            </>
          ) : (
            <button type="button" onClick={() => onEndReached?.()}>
              Charger les vidéos suivantes
            </button>
          )}
        </div>
      )}
    </div>
  );
}

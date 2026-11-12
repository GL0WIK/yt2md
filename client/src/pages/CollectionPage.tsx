import React from "react";
import { ArrowLeft, Download, ExternalLink, Loader2, Play, RefreshCw, Trash2, X } from "lucide-react";
import {
  deleteExports,
  downloadExportsZip,
  fetchCatalogPage,
  fetchCollection,
  removeCollection,
  type CatalogSource,
} from "../api";
import { VideoList } from "../components/VideoList";
import { Avatar } from "../components/Avatar";
import { addSelection } from "../catalogState";
import { deleteConfirmation, deleteFailureMessage, selectionExports } from "../exports";
import {
  buildResults,
  collectionVideos,
  formatFollowers,
  formatTotalDuration,
  videoState,
  type CollectionView,
  type Library,
  type VideoState,
} from "../library";
import { navigate, routeHref } from "../router";
import type { AnalysisRun, CatalogPage, CatalogVideo } from "../types";
import type { Collection } from "../../../server/src/shared/schemas";

type Filter = "all" | "todo" | "active" | "done" | "error";
type Props = {
  id: string;
  view: CollectionView | undefined;
  library: Library;
  run: AnalysisRun | null;
  blocked: string | null;
  onStart(videos: CatalogVideo[]): Promise<void>;
  onLibraryChanged(): Promise<void>;
};

// Each opened channel is checked once per session for videos published since.
const refreshed = new Set<string>();
const LIMIT = 1000;

export function CollectionPage({ id, view, library, run, blocked, onStart, onLibraryChanged }: Props) {
  const [collection, setCollection] = React.useState<Collection | null>(null);
  const [opening, setOpening] = React.useState(false);
  const [syncing, setSyncing] = React.useState<"refresh" | "more" | "all" | null>(null);
  const [filter, setFilter] = React.useState<Filter>("all");
  const [query, setQuery] = React.useState("");
  const [selected, setSelected] = React.useState(new Set<string>());
  const [working, setWorking] = React.useState(false);
  const [error, setError] = React.useState("");
  const abort = React.useRef<AbortController | null>(null);
  const stored = Boolean(view?.stored);

  React.useEffect(() => {
    setSelected(new Set());
    setFilter("all");
    setQuery("");
    setError("");
    setCollection(null);
    if (!stored) return;
    const controller = new AbortController();
    setOpening(true);
    fetchCollection(id, controller.signal)
      .then((value) => {
        setCollection(value);
        if (!refreshed.has(id)) {
          refreshed.add(id);
          void sync("refresh", value);
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Chaîne illisible.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setOpening(false);
      });
    return () => {
      controller.abort();
      abort.current?.abort();
    };
  }, [id, stored]);

  const videos = React.useMemo(
    () => collectionVideos(collection?.videos ?? [], view, library),
    [collection, view, library],
  );
  const results = React.useMemo(
    () => buildResults(videos, library.exports, run),
    [videos, library.exports, run],
  );
  const stateOf = (videoId: string): VideoState => videoState(videoId, results, run);
  const counts = { all: videos.length, todo: 0, active: 0, done: 0, error: 0 };
  for (const video of videos) {
    const state = stateOf(video.id);
    counts[state === "queued" || state === "running" ? "active" : state]++;
  }
  const shown = videos.filter((video) => {
    const state = stateOf(video.id);
    const group = state === "queued" || state === "running" ? "active" : state;
    return (
      (filter === "all" || group === filter) &&
      video.title.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
    );
  });
  const chosen = videos.filter((video) => selected.has(video.id));
  const toTranscribe = chosen.filter((video) => ["todo", "error"].includes(stateOf(video.id)));
  const exports = selectionExports(chosen, results);
  const duration = toTranscribe.reduce((total, video) => total + (video.durationSeconds ?? 0), 0);

  /** Where to read this collection on YouTube; a channel known only from its videos is looked up once. */
  async function source(current: Collection | null): Promise<CatalogSource> {
    if (current) return { kind: current.kind, url: current.url, pageSize: 50 };
    let url = view?.url ?? null;
    if (!url) {
      const sample = videos[0];
      if (!sample) throw new Error("Aucune vidéo pour retrouver cette chaîne.");
      const page = await fetchCatalogPage({ kind: "direct", videoUrls: [sample.url] });
      url = page.videos[0]?.channelUrl ?? null;
      if (!url) throw new Error("YouTube n’indique pas la chaîne de cette vidéo.");
    }
    return { kind: "channel", url, pageSize: 50 };
  }
  /** Reads YouTube: the newest videos, the next page, or every remaining page. */
  async function sync(mode: "refresh" | "more" | "all", current = collection) {
    if (syncing) return;
    const controller = new AbortController();
    abort.current = controller;
    setSyncing(mode);
    setError("");
    try {
      const base = await source(current);
      let offset: number | null = mode === "refresh" ? 0 : (current?.nextOffset ?? 0);
      let page: CatalogPage | null = null;
      do {
        page = await fetchCatalogPage(
          { ...base, pageSize: mode === "all" ? 100 : 50 } as CatalogSource,
          offset ?? 0,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const newId = page.collection?.id;
        if (newId && newId !== id) {
          // A channel known by name only now has its YouTube identity: its page moves there.
          await onLibraryChanged();
          navigate({ page: "collection", id: newId }, true);
          return;
        }
        setCollection(await fetchCollection(id));
        offset = page.nextOffset;
      } while (mode === "all" && offset !== null && !controller.signal.aborted);
      await onLibraryChanged();
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "Lecture de YouTube impossible.");
    } finally {
      if (abort.current === controller) setSyncing(null);
    }
  }
  async function act(work: () => Promise<void>) {
    setWorking(true);
    setError("");
    try {
      await work();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Action impossible.");
    } finally {
      setWorking(false);
    }
  }
  function toggle(videoId: string, checked: boolean) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (checked) return addSelection(next, [videoId], LIMIT);
      next.delete(videoId);
      return next;
    });
  }
  const allShownSelected = shown.length > 0 && shown.every((video) => selected.has(video.id));
  const name = collection?.name ?? view?.name ?? "Chaîne";
  const followers = formatFollowers(collection?.followers ?? view?.followers);
  const banner = collection?.bannerUrl ?? view?.bannerUrl;
  const avatar = collection?.avatarUrl ?? view?.avatarUrl;
  const complete = collection?.complete ?? false;
  const youtube = (collection?.url ?? view?.url ?? "").replace(/\/videos$/, "");
  const chips: Array<[Filter, string]> = [
    ["all", "Toutes"],
    ["todo", "À traiter"],
    ["active", "En cours"],
    ["done", "Transcrites"],
    ["error", "En erreur"],
  ];

  return (
    <>
      <a className="back-link" href={routeHref({ page: "library" })}>
        <ArrowLeft size={16} /> Bibliothèque
      </a>
      <header className="collection-hero" style={banner ? { backgroundImage: `url(${banner})` } : undefined}>
        <div className="hero-content">
          <Avatar name={name} url={avatar} playlist={(collection?.kind ?? view?.kind) === "playlist"} large />
          <div className="hero-text">
            <span className="eyebrow">
              {(collection?.kind ?? view?.kind) === "playlist" ? "PLAYLIST" : "CHAÎNE"}
            </span>
            <h1>{name}</h1>
            <span className="muted">
              {[
                followers,
                `${videos.length} vidéo${videos.length > 1 ? "s" : ""}${stored && !complete ? " chargées" : ""}`,
                `${counts.done} transcrite${counts.done > 1 ? "s" : ""}`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </div>
          <div className="hero-actions">
            {youtube && (
              <a className="button" href={youtube} target="_blank" rel="noreferrer">
                <ExternalLink size={16} /> YouTube
              </a>
            )}
            {stored ? (
              <>
                <button disabled={Boolean(syncing)} onClick={() => void sync("refresh")}>
                  {syncing === "refresh" ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                  Actualiser
                </button>
                {!complete && (
                  <button disabled={Boolean(syncing)} onClick={() => void sync("all")}>
                    {syncing === "all" ? <Loader2 className="spin" size={16} /> : <Download size={16} />}
                    {syncing === "all" ? `Chargement… ${videos.length}` : "Charger toute la chaîne"}
                  </button>
                )}
              </>
            ) : (
              <button
                className="primary-action"
                disabled={Boolean(syncing)}
                onClick={() => void sync("refresh")}
              >
                {syncing ? <Loader2 className="spin" size={16} /> : <Download size={16} />}
                Charger les vidéos de la chaîne
              </button>
            )}
          </div>
        </div>
      </header>
      {error && (
        <div role="alert" className="error">
          {error}
        </div>
      )}
      <section className="workbench">
        <div className="list-toolbar">
          <label className="checkline select-all">
            <input
              type="checkbox"
              aria-label="Sélectionner toutes les vidéos affichées"
              checked={allShownSelected}
              disabled={!shown.length}
              onChange={(event) =>
                setSelected((previous) => {
                  if (!event.target.checked) {
                    const next = new Set(previous);
                    for (const video of shown) next.delete(video.id);
                    return next;
                  }
                  return addSelection(
                    previous,
                    shown.map((video) => video.id),
                    LIMIT,
                  );
                })
              }
            />
          </label>
          <div className="chips" role="tablist" aria-label="Filtrer par état">
            {chips
              .filter(([key]) => key === "all" || counts[key] > 0 || filter === key)
              .map(([key, label]) => (
                <button
                  key={key}
                  role="tab"
                  aria-selected={filter === key}
                  className={filter === key ? "chip active" : "chip"}
                  onClick={() => setFilter(key)}
                >
                  {label} <span>{counts[key]}</span>
                </button>
              ))}
          </div>
          <input
            className="search"
            aria-label="Rechercher une vidéo"
            placeholder="Rechercher un titre…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        {opening ? (
          <div className="empty">
            <Loader2 className="spin" size={18} /> Ouverture de la chaîne…
          </div>
        ) : !videos.length ? (
          <div className="empty">
            {syncing ? "Lecture de la chaîne sur YouTube…" : "Aucune vidéo chargée pour cette chaîne."}
          </div>
        ) : !shown.length ? (
          <div className="empty">Aucune vidéo pour ce filtre.</div>
        ) : (
          <VideoList
            videos={shown}
            results={results}
            stateOf={stateOf}
            selected={selected}
            onToggle={toggle}
            onTranscribe={blocked ? undefined : (video) => void act(() => onStart([video]))}
            hasMore={stored && !complete && filter === "all" && !query && !syncing}
            loadingMore={syncing === "more"}
            onEndReached={() => void sync("more")}
          />
        )}
      </section>
      {stored && !complete && !syncing && (
        <p className="muted center">
          La suite de la chaîne se charge en faisant défiler la liste, ou d’un coup avec « Charger toute la
          chaîne ».
        </p>
      )}
      {selected.size > 0 && (
        <div className="selection-dock" role="region" aria-label="Sélection">
          <div>
            <strong>
              {selected.size} vidéo{selected.size > 1 ? "s" : ""} sélectionnée{selected.size > 1 ? "s" : ""}
            </strong>
            <span className="muted">
              {toTranscribe.length
                ? `${toTranscribe.length} à transcrire · ${formatTotalDuration(duration)} de vidéo`
                : "toutes déjà transcrites ou en cours"}
              {blocked && toTranscribe.length ? ` · ${blocked}` : ""}
            </span>
          </div>
          <div className="dock-actions">
            {exports.filenames.length > 0 && (
              <>
                <button
                  disabled={working}
                  onClick={() => void act(() => downloadExportsZip(exports.filenames, `${name}.zip`))}
                >
                  <Download size={16} /> ZIP
                </button>
                <button
                  disabled={working || Boolean(run && run.status === "running")}
                  aria-label="Supprimer les transcriptions sélectionnées"
                  title="Supprimer les transcriptions (MD et JSON) sélectionnées"
                  onClick={() => {
                    if (!window.confirm(deleteConfirmation(exports.videoIds.length))) return;
                    void act(async () => {
                      const response = await deleteExports(exports.videoIds);
                      await onLibraryChanged();
                      const failures = deleteFailureMessage(response.failed);
                      if (failures) throw new Error(failures);
                    });
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </>
            )}
            <button disabled={working} onClick={() => setSelected(new Set())}>
              <X size={16} /> Désélectionner
            </button>
            <button
              className="primary-action"
              disabled={working || !toTranscribe.length || Boolean(blocked)}
              onClick={() =>
                void act(async () => {
                  await onStart(toTranscribe);
                  setSelected(new Set());
                })
              }
            >
              <Play size={16} /> Transcrire {toTranscribe.length || ""}
            </button>
          </div>
        </div>
      )}
      {stored && (
        <p className="center">
          <button
            className="link-button"
            disabled={working}
            onClick={() => {
              if (
                !window.confirm(
                  `Retirer « ${name} » de la bibliothèque ? Les transcriptions sont conservées.`,
                )
              )
                return;
              void act(async () => {
                await removeCollection(id);
                await onLibraryChanged();
                navigate({ page: "library" });
              });
            }}
          >
            Retirer cette {(collection?.kind ?? view?.kind) === "playlist" ? "playlist" : "chaîne"} de la
            bibliothèque
          </button>
        </p>
      )}
    </>
  );
}

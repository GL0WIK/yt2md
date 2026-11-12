import React from "react";
import {
  Activity,
  CheckCircle2,
  HardDrive,
  Library as LibraryIcon,
  RefreshCw,
  TriangleAlert,
} from "lucide-react";
import { fetchCatalogPage, fetchLibrary, fetchRun, request, startRun, type Health } from "./api";
import { useRunPolling } from "./hooks/useRunPolling";
import { buildCollections, buildResults, emptyLibrary, videoState, type Library } from "./library";
import { navigate, routeHref, useRoute } from "./router";
import { phases } from "./components/RunPanel";
import { LibraryPage } from "./pages/LibraryPage";
import { CollectionPage } from "./pages/CollectionPage";
import { RunPage } from "./pages/RunPage";
import type { AnalysisRun, CatalogPage, CatalogVideo, RunSummary } from "./types";

const languageNames: Record<string, string> = { fr: "français", en: "anglais", auto: "langue détectée" };

export default function App() {
  const route = useRoute();
  const [health, setHealth] = React.useState<Health | null>(null);
  const [library, setLibrary] = React.useState<Library>(emptyLibrary);
  const [loaded, setLoaded] = React.useState(false);
  const [run, setRun] = React.useState<AnalysisRun | null>(null);
  const [working, setWorking] = React.useState(false);
  const [error, setError] = React.useState("");
  const completed = React.useRef(-1);

  const refreshLibrary = React.useCallback(async () => setLibrary(await fetchLibrary()), []);
  const invalidatePolls = useRunPolling(run, {
    onUpdate(next) {
      setRun(next);
      // Each finished video adds its files: the library counts follow the processing.
      if (next.completed !== completed.current) {
        completed.current = next.completed;
        void refreshLibrary().catch(() => {});
      }
    },
    onSettled: () => refreshLibrary(),
    onError: setError,
  });
  React.useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        const [state, books, runs] = await Promise.all([
          request<Health>("/api/health"),
          fetchLibrary(),
          request<{ runs: RunSummary[] }>("/api/runs"),
        ]);
        if (!mounted) return;
        setHealth(state);
        setLibrary(books);
        const latest = runs.runs.find((item) => item.status === "running") ?? runs.runs[0];
        if (latest) {
          const detail = await fetchRun(latest.runId);
          if (mounted) {
            completed.current = detail.completed;
            setRun(detail);
          }
        }
      } catch (e) {
        if (mounted) setError(e instanceof Error ? e.message : "Serveur local indisponible.");
      } finally {
        if (mounted) setLoaded(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  async function runAction(action: () => Promise<AnalysisRun>) {
    setWorking(true);
    setError("");
    try {
      invalidatePolls();
      const value = await action();
      invalidatePolls();
      completed.current = value.completed;
      setRun(value);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Opération impossible.");
    } finally {
      setWorking(false);
    }
  }
  /** Starts a processing; errors go back to the page that asked, next to its button. */
  async function start(videos: CatalogVideo[]) {
    invalidatePolls();
    const value = await startRun(videos);
    invalidatePolls();
    completed.current = value.completed;
    setRun(value);
  }
  async function openCollection(page: CatalogPage) {
    if (!page.collection) throw new Error("YouTube n’a pas indiqué de chaîne pour ce lien.");
    await refreshLibrary();
    navigate({ page: "collection", id: page.collection.id });
  }

  const results = React.useMemo(() => buildResults(library.videos, library.exports, run), [library, run]);
  const collections = React.useMemo(() => buildCollections(library, results), [library, results]);
  const running = run?.status === "running";
  const ready = Boolean(health?.runtime.ready);
  const blocked = running
    ? "Un traitement est déjà en cours."
    : health && !ready
      ? "Installation à terminer avant de transcrire."
      : null;
  const stateOf = (id: string) => videoState(id, results, run);
  const view = route.page === "collection" ? collections.find((item) => item.id === route.id) : undefined;

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href={routeHref({ page: "library" })}>
          <span className="brand-mark">YT</span>
          <span>
            <strong>YT-RAG</strong>
            <small>transcription locale</small>
          </span>
        </a>
        <nav aria-label="Navigation principale">
          <a
            href={routeHref({ page: "library" })}
            aria-current={route.page !== "run" ? "page" : undefined}
            className={route.page !== "run" ? "active" : ""}
          >
            <LibraryIcon size={16} /> Bibliothèque
          </a>
          <a
            href={routeHref({ page: "run" })}
            aria-current={route.page === "run" ? "page" : undefined}
            className={route.page === "run" ? "active" : ""}
          >
            <Activity size={16} /> Traitement
            {running && run && (
              <span className="nav-badge">
                {run.completed}/{run.total}
              </span>
            )}
          </a>
        </nav>
        <button
          className={ready ? "status-chip ok" : "status-chip"}
          title={health?.runtime.issues.join(" ") || "Vérifier l’installation"}
          onClick={() =>
            void request<Health>("/api/health")
              .then(setHealth)
              .catch((e: Error) => setError(e.message))
          }
        >
          {ready ? (
            <CheckCircle2 size={15} />
          ) : health ? (
            <TriangleAlert size={15} />
          ) : (
            <RefreshCw size={15} />
          )}
          <span>
            {health
              ? ready
                ? `${health.runtime.gpu?.name ?? "GPU"} · ${languageNames[health.asrLanguage] ?? health.asrLanguage}`
                : "Installation à terminer"
              : "Connexion…"}
          </span>
        </button>
      </header>
      {running && run && route.page !== "run" && (
        <a className="run-strip" href={routeHref({ page: "run" })}>
          <span className="run-strip-text">
            <strong>
              Traitement en cours · {run.completed}/{run.total}
            </strong>
            {run.progress && (
              <span>
                {phases[run.progress.phase]} · {run.progress.videoTitle}
              </span>
            )}
          </span>
          <span className="run-strip-bar">
            <span style={{ width: `${(run.completed / Math.max(1, run.total)) * 100}%` }} />
          </span>
          <span className="run-strip-link">Voir le traitement →</span>
        </a>
      )}
      <main>
        {health && !ready && (
          <aside className="notice">
            <strong>Préparation du moteur local</strong>
            <p>Exécute les étapes d’installation du README, puis vérifie de nouveau.</p>
            <ul>
              {health.runtime.issues.map((issue, i) => (
                <li key={i}>{issue}</li>
              ))}
            </ul>
          </aside>
        )}
        {error && (
          <div role="alert" className="error">
            {error}
          </div>
        )}
        {route.page === "library" && (
          <LibraryPage
            collections={collections}
            stateOf={stateOf}
            blocked={blocked}
            onTranscribe={(video) => start([video])}
            onCollection={(page) => void openCollection(page).catch((e: Error) => setError(e.message))}
            onOpenChannel={async (video) =>
              openCollection(
                await fetchCatalogPage({ kind: "channel", url: video.channelUrl!, pageSize: 50 }),
              )
            }
          />
        )}
        {route.page === "collection" &&
          (view || !loaded ? (
            <CollectionPage
              id={route.id}
              view={view}
              library={library}
              run={run}
              blocked={blocked}
              onStart={start}
              onLibraryChanged={refreshLibrary}
            />
          ) : (
            <div className="empty">
              Cette chaîne n’est plus dans la bibliothèque.{" "}
              <a href={routeHref({ page: "library" })}>Retour</a>
            </div>
          ))}
        {route.page === "run" && (
          <RunPage
            run={run}
            library={library}
            working={working}
            ready={ready}
            onPause={() =>
              run && void runAction(() => request<AnalysisRun>(`/api/runs/${run.runId}/pause`, {}))
            }
            onResume={() =>
              run && void runAction(() => request<AnalysisRun>(`/api/runs/${run.runId}/resume`, {}))
            }
          />
        )}
      </main>
      <footer>
        <HardDrive size={13} /> Tout reste sur cette machine : Qwen3-ASR pour les paroles, Qwen3.5 pour les
        images, exports JSON et Markdown au format d’origine.
      </footer>
    </div>
  );
}

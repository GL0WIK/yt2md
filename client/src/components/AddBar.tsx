import React from "react";
import { ArrowRight, Loader2, Play, Search, Tv } from "lucide-react";
import { detectLink, fetchCatalogPage } from "../api";
import { formatDuration } from "../format";
import type { CatalogPage, CatalogVideo } from "../types";
import type { VideoState } from "../library";

type Props = {
  stateOf(id: string): VideoState;
  /** Why a transcription cannot start now, if so. */
  blocked: string | null;
  onTranscribe(video: CatalogVideo): Promise<void>;
  onCollection(page: CatalogPage): void;
  onOpenChannel(video: CatalogVideo): Promise<void>;
};

const kindLabels = { video: "Vidéo", channel: "Chaîne", playlist: "Playlist" };

/** One field for every link: a video is previewed, a channel or playlist opens its page. */
export function AddBar({ stateOf, blocked, onTranscribe, onCollection, onOpenChannel }: Props) {
  const [text, setText] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [video, setVideo] = React.useState<CatalogVideo | null>(null);
  const link = detectLink(text);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!link) {
      setError(
        "Lien non reconnu : collez l’adresse d’une vidéo, d’une chaîne (@nom) ou d’une playlist YouTube.",
      );
      return;
    }
    setBusy(true);
    setError("");
    setVideo(null);
    try {
      if (link.kind === "video") {
        const page = await fetchCatalogPage({ kind: "direct", videoUrls: [link.url] });
        if (!page.videos[0]) throw new Error(page.warnings[0] ?? "Vidéo introuvable.");
        setVideo(page.videos[0]);
      } else {
        onCollection(await fetchCatalogPage({ kind: link.kind, url: link.url, pageSize: 50 }));
        setText("");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lecture de YouTube impossible.");
    } finally {
      setBusy(false);
    }
  }
  async function act(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await work();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Action impossible.");
    } finally {
      setBusy(false);
    }
  }
  const state = video ? stateOf(video.id) : null;

  return (
    <section className="add-bar">
      <form onSubmit={(event) => void submit(event)}>
        <label className="add-field">
          <Search size={18} aria-hidden="true" />
          <input
            aria-label="Lien YouTube"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setError("");
            }}
            placeholder="Collez le lien d’une vidéo, d’une chaîne ou d’une playlist YouTube"
          />
          {link && <span className="kind-chip">{kindLabels[link.kind]}</span>}
        </label>
        <button className="primary-action" type="submit" disabled={busy || !text.trim()}>
          {busy ? <Loader2 className="spin" size={16} /> : <ArrowRight size={16} />}
          {link?.kind === "video" ? "Afficher la vidéo" : link ? "Ouvrir" : "Valider"}
        </button>
      </form>
      <p className="add-hint">
        Une vidéo s’affiche ici pour être transcrite. Une chaîne ou une playlist ouvre sa page, où vous
        choisissez les vidéos.
      </p>
      {error && (
        <p role="alert" className="failure">
          {error}
        </p>
      )}
      {video && state && (
        <div className="preview-card">
          <img src={`https://i.ytimg.com/vi/${video.id}/mqdefault.jpg`} alt="" />
          <div className="preview-body">
            <strong>{video.title}</strong>
            <span className="video-meta">
              {[
                video.channel,
                video.publishedAt && new Date(video.publishedAt).toLocaleDateString("fr-FR"),
                video.durationSeconds !== null && formatDuration(video.durationSeconds),
              ]
                .filter(Boolean)
                .join(" · ")}
            </span>
            <div className="preview-actions">
              {state === "done" ? (
                <span className="badge badge-done">Déjà transcrite</span>
              ) : state === "running" || state === "queued" ? (
                <span className="badge badge-running">En cours de traitement</span>
              ) : (
                <button
                  className="primary-action"
                  disabled={busy || Boolean(blocked)}
                  title={blocked ?? undefined}
                  onClick={() => void act(() => onTranscribe(video))}
                >
                  <Play size={16} />{" "}
                  {state === "error" ? "Relancer la transcription" : "Transcrire cette vidéo"}
                </button>
              )}
              {video.channelUrl && (
                <button disabled={busy} onClick={() => void act(() => onOpenChannel(video))}>
                  <Tv size={16} /> Voir la chaîne {video.channel ?? ""}
                </button>
              )}
            </div>
            {blocked && state !== "done" && <span className="muted">{blocked}</span>}
          </div>
        </div>
      )}
    </section>
  );
}

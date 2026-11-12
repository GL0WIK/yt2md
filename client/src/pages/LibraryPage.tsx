import React from "react";
import { AddBar } from "../components/AddBar";
import { Avatar } from "../components/Avatar";
import { routeHref } from "../router";
import { formatFollowers, type CollectionView, type VideoState } from "../library";
import type { CatalogPage, CatalogVideo } from "../types";

type Props = {
  collections: CollectionView[];
  stateOf(id: string): VideoState;
  blocked: string | null;
  onTranscribe(video: CatalogVideo): Promise<void>;
  onCollection(page: CatalogPage): void;
  onOpenChannel(video: CatalogVideo): Promise<void>;
};

export function LibraryPage({ collections, ...add }: Props) {
  const [query, setQuery] = React.useState("");
  const shown = collections.filter((view) =>
    view.name.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()),
  );
  return (
    <>
      <AddBar {...add} />
      <section className="collections">
        <div className="section-head">
          <div>
            <h2>Bibliothèque</h2>
            <span className="muted">
              {collections.length} chaîne{collections.length > 1 ? "s" : ""} ou playlist
              {collections.length > 1 ? "s" : ""}
            </span>
          </div>
          {collections.length > 6 && (
            <input
              className="search"
              aria-label="Rechercher une chaîne"
              placeholder="Rechercher une chaîne…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          )}
        </div>
        {!collections.length ? (
          <div className="empty">
            Aucune chaîne pour l’instant. Collez ci-dessus le lien d’une chaîne YouTube pour parcourir ses
            vidéos, ou celui d’une vidéo pour la transcrire.
          </div>
        ) : (
          <div className="collection-grid">
            {shown.map((view) => {
              const listed = view.videoIds.length;
              const followers = formatFollowers(view.followers);
              return (
                <a
                  key={view.id}
                  className="collection-card"
                  href={routeHref({ page: "collection", id: view.id })}
                >
                  <Avatar name={view.name} url={view.avatarUrl} playlist={view.kind === "playlist"} />
                  <div>
                    <strong>{view.name}</strong>
                    <span className="muted">
                      {view.kind === "playlist" ? "Playlist" : (followers ?? "Chaîne")}
                    </span>
                  </div>
                  <div className="card-stats">
                    <span>
                      <b>{view.processed}</b> transcrite{view.processed > 1 ? "s" : ""}
                    </span>
                    {view.stored && (
                      <span>
                        <b>{listed}</b> vidéo{listed > 1 ? "s" : ""}
                        {view.complete ? "" : " chargées"}
                      </span>
                    )}
                    {view.errors > 0 && (
                      <span className="stat-error">
                        <b>{view.errors}</b> en erreur
                      </span>
                    )}
                  </div>
                  {view.stored && listed > 0 && (
                    <span className="card-progress" aria-hidden="true">
                      <span style={{ width: `${Math.min(100, (view.processed / listed) * 100)}%` }} />
                    </span>
                  )}
                </a>
              );
            })}
          </div>
        )}
      </section>
    </>
  );
}

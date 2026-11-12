const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
const YOUTUBE_HOSTS = new Set(["youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"]);

function parseYouTubeUrl(value: string): URL {
  const url = new URL(value);
  if (!["https:", "http:"].includes(url.protocol) || url.username || url.password || url.port) {
    throw new Error("URL YouTube invalide.");
  }
  return url;
}

export function normalizeVideoUrl(value: string): string {
  const url = parseYouTubeUrl(value);
  let id: string | null = null;
  if (url.hostname === "youtu.be") id = url.pathname.slice(1);
  else if (YOUTUBE_HOSTS.has(url.hostname)) {
    if (url.pathname === "/watch") id = url.searchParams.get("v");
    else id = /^\/(?:embed)\/([^/]+)\/?$/.exec(url.pathname)?.[1] ?? null;
  }
  if (!id || !VIDEO_ID.test(id)) throw new Error("Une URL de vidéo YouTube valide est requise.");
  return `https://www.youtube.com/watch?v=${id}`;
}

export function normalizeChannelUrl(value: string): string {
  const url = parseYouTubeUrl(value);
  if (!YOUTUBE_HOSTS.has(url.hostname)) throw new Error("Une URL de chaîne YouTube est requise.");
  const match = /^(\/@[^/]+|\/(?:channel|c|user)\/[^/]+)(?:\/(?:videos|featured|shorts|streams))?\/?$/.exec(
    url.pathname,
  );
  if (!match) throw new Error("Une URL de chaîne YouTube est requise.");
  return `https://www.youtube.com${match[1]}/videos`;
}

export function normalizePlaylistUrl(value: string): string {
  const url = parseYouTubeUrl(value);
  const id = url.searchParams.get("list");
  const validHost = YOUTUBE_HOSTS.has(url.hostname) || url.hostname === "youtu.be";
  const validPath = ["/playlist", "/watch"].includes(url.pathname) || url.hostname === "youtu.be";
  if (!validHost || !validPath || !id || !/^[A-Za-z0-9_-]{10,100}$/.test(id)) {
    throw new Error("Une URL de playlist YouTube publique est requise.");
  }
  // Mixes and account-specific collections cannot be paginated predictably.
  if (/^(RD|WL|LL)/.test(id))
    throw new Error(
      "Les mixes et listes personnelles ne sont pas pris en charge. Utilisez une playlist publique.",
    );
  return "https://www.youtube.com/playlist?list=" + id;
}

export function isIsoDate(value: string): boolean {
  return (
    /^\d{4}-\d{2}-\d{2}$/.test(value) &&
    Number.isFinite(Date.parse(value)) &&
    new Date(value).toISOString().slice(0, 10) === value
  );
}

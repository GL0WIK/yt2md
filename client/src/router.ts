import React from "react";

export type Route = { page: "library" } | { page: "collection"; id: string } | { page: "run" };

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#/, "");
  const collection = /^\/c\/(.+)$/.exec(path);
  if (collection) {
    try {
      return { page: "collection", id: decodeURIComponent(collection[1]) };
    } catch {
      return { page: "library" };
    }
  }
  if (path === "/traitement") return { page: "run" };
  return { page: "library" };
}

export function routeHref(route: Route): string {
  if (route.page === "collection") return "#/c/" + encodeURIComponent(route.id);
  if (route.page === "run") return "#/traitement";
  return "#/";
}

export function navigate(route: Route, replace = false) {
  const href = routeHref(route);
  if (replace) history.replaceState(null, "", href);
  else history.pushState(null, "", href);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

/** Hash routes keep the local server free of client-side routing: links and back button just work. */
export function useRoute(): Route {
  const [route, setRoute] = React.useState(() => parseRoute(location.hash));
  React.useEffect(() => {
    const update = () => {
      setRoute(parseRoute(location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", update);
    window.addEventListener("popstate", update);
    return () => {
      window.removeEventListener("hashchange", update);
      window.removeEventListener("popstate", update);
    };
  }, []);
  return route;
}

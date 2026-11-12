import express, { type ErrorRequestHandler } from "express";
import fs from "node:fs";
import path from "node:path";
import { ZodError } from "zod";
import { getConfig, type AppConfig } from "./config.js";
import { createYtDlpCatalogService, YtDlpError, type YtDlpCatalogService } from "./catalog/ytDlp.js";
import { PythonWorker, type LocalWorker } from "./local/worker.js";
import { createAnalysisService, type AnalysisService } from "./api/analysisService.js";
import { createRunStore } from "./api/runStore.js";
import { createApiRouter } from "./api/routes.js";
import { HttpError } from "./shared/errors.js";
import { FileLogger, logEvent, withLogContext, errorFields } from "./shared/logging.js";

const LOCAL_HOST = /^(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?$/i;

export function createApp(
  deps: {
    config?: AppConfig;
    catalog?: YtDlpCatalogService;
    worker?: LocalWorker;
    service?: AnalysisService;
  } = {},
) {
  const config = deps.config ?? getConfig();
  const catalog =
    deps.catalog ??
    createYtDlpCatalogService(
      config.ytDlpBinary,
      config.ytDlpTimeoutMs,
      config.catalogMaxVideos,
      undefined,
      undefined,
      config.ytDlpAuthArgs,
    );
  const worker = deps.worker ?? new PythonWorker(config);
  const service = deps.service ?? createAnalysisService(config, catalog, worker);
  const runs = createRunStore(config, service);
  const logger = new FileLogger(config.logsDir, config.logLevel);
  const app = express();
  app.locals.runs = runs;
  app.locals.logger = logger;
  app.locals.close = async () => {
    try {
      await withLogContext(
        {},
        async () => {
          await runs.close();
          await worker.close();
        },
        logger,
      );
    } finally {
      await logger.flush();
    }
  };
  app.disable("x-powered-by");
  app.use((req, res, next) =>
    withLogContext(
      { source: "server" },
      () => {
        const started = performance.now();
        res.on("finish", () =>
          logEvent(
            "http.request.complete",
            {
              method: req.method,
              path: req.path,
              status: res.statusCode,
              durationMs: Math.round(performance.now() - started),
            },
            "debug",
          ),
        );
        next();
      },
      logger,
    ),
  );
  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Referrer-Policy", "no-referrer");
    res.setHeader(
      "Content-Security-Policy",
      // Thumbnails, channel avatars and banners load straight from YouTube's image hosts.
      "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https://i.ytimg.com https://yt3.googleusercontent.com https://yt3.ggpht.com; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    );
    next();
  });
  app.use("/api", (req, res, next) => {
    res.setHeader("Cache-Control", "no-store");
    // DNS rebinding: a hostile domain resolved to 127.0.0.1 keeps its own name in Host.
    if (!LOCAL_HOST.test(req.get("host") ?? "")) {
      next(new HttpError(403, "Hôte non autorisé : utilisez 127.0.0.1 ou localhost."));
      return;
    }
    const origin = req.get("origin");
    if (origin && req.get("sec-fetch-site") !== "same-origin") {
      try {
        if (new URL(origin).host !== req.get("host")) throw new Error("origin");
      } catch {
        next(new HttpError(403, "Origine non autorisée."));
        return;
      }
    }
    next();
  });
  app.use(express.json({ limit: "60mb" }));
  app.use("/api", createApiRouter({ config, catalog, worker, runs, logger }));
  app.use("/api", (_req, res) => res.status(404).json({ error: "Route API introuvable." }));
  const client = path.join(config.projectRoot, "dist/client");
  if (fs.existsSync(path.join(client, "index.html"))) {
    app.use(express.static(client));
    app.use((req, res, next) => {
      if (!["GET", "HEAD"].includes(req.method) || path.extname(req.path) || !req.accepts("html"))
        return next();
      res.sendFile(path.join(client, "index.html"), (error) => {
        if (error) next(error);
      });
    });
  }
  app.use(errorHandler);
  return app;
}
const errorHandler: ErrorRequestHandler = (error, _req, res, next) => {
  if (res.headersSent) return next(error);
  logEvent(
    "http.request.failed",
    {
      message: error instanceof Error ? error.message : "Erreur API",
      method: _req.method,
      path: _req.path,
      ...errorFields(error),
    },
    "error",
  );
  if (error instanceof ZodError)
    res.status(400).json({ error: "Vérifiez les champs saisis.", issues: error.issues });
  else if (error instanceof HttpError) res.status(error.status).json({ error: error.message });
  else if (error instanceof YtDlpError) res.status(502).json({ error: error.message });
  else if (error?.type === "entity.parse.failed") res.status(400).json({ error: "JSON invalide." });
  else if (error?.type === "entity.too.large") res.status(413).json({ error: "Sélection trop volumineuse." });
  else {
    res.status(500).json({ error: "Erreur du serveur local. Consultez le terminal." });
  }
};

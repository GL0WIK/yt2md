import { createApp } from "./app.js";
import { getConfig } from "./config.js";
import { logEvent, withLogContext } from "./shared/logging.js";

const config = getConfig();
const app = createApp({ config });
const server = app.listen(config.port, config.host, () =>
  withLogContext(
    {},
    () =>
      logEvent("server.started", {
        message: `yt-rag local : http://${config.host}:${config.port}`,
        logsDir: config.logsDir,
        asrModel: config.asrModel,
        asrLanguage: config.asrLanguage,
        visionModel: config.visionModel,
        visionBackend: config.visionBackend,
        quantization: config.visionBackend === "llama_cpp" ? "q4_k_m" : config.visionQuantization,
        device: config.device,
        groupFrames: config.groupFrames,
        visionMaxPixels: config.visionMaxPixels,
      }),
    app.locals.logger,
  ),
);
let stopping = false;
async function shutdown() {
  if (stopping) {
    // A second Ctrl+C must always leave, even if a native library blocks the graceful path.
    process.exit(1);
  }
  stopping = true;
  // Last-resort bound: the worker close is itself bounded, but no lingering handle may keep the process alive.
  setTimeout(() => {
    console.error("Arrêt forcé : le serveur ne s’est pas arrêté à temps.");
    process.exit(1);
  }, 30000).unref();
  withLogContext(
    {},
    () =>
      logEvent("server.stopping", {
        message: "Arrêt du serveur ; sauvegarde du traitement et libération du GPU.",
      }),
    app.locals.logger,
  );
  server.close();
  try {
    await app.locals.close();
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Échec de l’arrêt");
    process.exitCode = 1;
  }
  server.closeAllConnections();
  process.exit();
}
process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());
server.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 1;
  void app.locals.close();
});

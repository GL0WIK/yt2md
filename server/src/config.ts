import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnv } from "dotenv";

// Both server/src and dist/server are two levels below this standalone project.
export const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
loadEnv({ path: path.join(PROJECT_ROOT, ".env"), quiet: true });

export function insideProject(root: string, value: string): string {
  const target = path.resolve(root, value);
  const relative = path.relative(root, target);
  if (!relative || relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative))
    throw new Error("Le chemin de données doit rester dans le dossier du projet : " + value);
  let ancestor = target;
  while (!fs.existsSync(ancestor)) ancestor = path.dirname(ancestor);
  const realRoot = fs.realpathSync(root);
  const realAncestor = fs.realpathSync(ancestor);
  const realRelative = path.relative(realRoot, realAncestor);
  if (realRelative === ".." || realRelative.startsWith(".." + path.sep) || path.isAbsolute(realRelative))
    throw new Error("Un lien symbolique fait sortir le chemin du dossier du projet.");
  return target;
}

/** Languages both Qwen3-ASR and its forced aligner support without extra tokenizers; auto detects. */
export const ASR_LANGUAGES = ["fr", "en", "de", "es", "it", "pt", "ru", "zh", "yue", "auto"] as const;

export function getConfig(env: NodeJS.ProcessEnv = process.env, projectRoot = PROJECT_ROOT) {
  const integer = (name: string, fallback: number, min: number, max: number) => {
    const value = Number(env[name] ?? fallback);
    if (!Number.isSafeInteger(value) || value < min || value > max)
      throw new Error(`${name} doit être un entier entre ${min} et ${max}.`);
    return value;
  };
  const boolean = (name: string, fallback: boolean) => {
    const value = env[name] ?? String(fallback);
    if (value !== "true" && value !== "false") throw new Error(name + " doit valoir true ou false.");
    return value === "true";
  };
  const model = (name: string, fallback: string) => {
    const value = env[name] ?? fallback;
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value))
      throw new Error(name + " : identifiant HF invalide.");
    return value;
  };
  const quantization = env.VISION_QUANTIZATION ?? "bf16";
  // Qwen3-ASR (native Transformers) transcribes; Qwen3-ForcedAligner times each word.
  const asrModel = model("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf");
  if (!/qwen3-asr/i.test(asrModel))
    throw new Error(
      "ASR_MODEL doit être un modèle Qwen3-ASR Transformers, par exemple Qwen/Qwen3-ASR-1.7B-hf.",
    );
  const asrAlignerModel = model("ASR_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B-hf");
  if (!/qwen3-forcedaligner/i.test(asrAlignerModel))
    throw new Error(
      "ASR_ALIGNER_MODEL doit être un modèle Qwen3-ForcedAligner, par exemple Qwen/Qwen3-ForcedAligner-0.6B-hf.",
    );
  const asrLanguage = env.ASR_LANGUAGE ?? "fr";
  if (!(ASR_LANGUAGES as readonly string[]).includes(asrLanguage))
    throw new Error(`ASR_LANGUAGE : ${ASR_LANGUAGES.join(", ")}.`);
  const asrContext = env.ASR_CONTEXT ?? "title";
  if (!["title", "none"].includes(asrContext)) throw new Error("ASR_CONTEXT : title ou none.");
  // NF4 needs bitsandbytes (CUDA), absent on Apple Silicon.
  if (quantization !== "bf16") throw new Error("VISION_QUANTIZATION : bf16 uniquement sur Apple Silicon.");
  const visionBackend = env.VISION_BACKEND ?? "llama_cpp";
  if (!["llama_cpp", "transformers"].includes(visionBackend))
    throw new Error("VISION_BACKEND : llama_cpp ou transformers.");
  const visionModel = model("VISION_MODEL", "Qwen/Qwen3.5-9B");
  if (visionBackend === "llama_cpp" && visionModel !== "Qwen/Qwen3.5-9B")
    throw new Error(
      "Le moteur natif validé utilise Qwen/Qwen3.5-9B. Choisir transformers pour un autre modèle.",
    );
  const logLevel = env.LOG_LEVEL ?? "info";
  if (!["debug", "info", "warn", "error"].includes(logLevel))
    throw new Error("LOG_LEVEL : debug, info, warn ou error.");
  const host = env.HOST ?? "127.0.0.1";
  if (!["127.0.0.1", "localhost", "::1"].includes(host))
    throw new Error("HOST doit être une adresse locale.");
  const fps = integer("VIDEO_FPS", 1, 1, 1);
  const groupFrames = integer("VIDEO_GROUP_FRAMES", 7, 2, 32);
  const venvPython = path.join(
    projectRoot,
    ".venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
  const pythonBinary = env.PYTHON_BINARY
    ? env.PYTHON_BINARY.includes("/") || env.PYTHON_BINARY.includes("\\")
      ? path.resolve(projectRoot, env.PYTHON_BINARY)
      : env.PYTHON_BINARY
    : fs.existsSync(venvPython)
      ? venvPython
      : "python";
  // Cookies let yt-dlp read YouTube as a signed-in browser once the bot check starts refusing it.
  const cookiesFile = env.YT_DLP_COOKIES?.trim() ? path.resolve(projectRoot, env.YT_DLP_COOKIES.trim()) : "";
  const cookiesBrowser = env.YT_DLP_COOKIES_FROM_BROWSER?.trim() ?? "";
  if (cookiesFile && cookiesBrowser)
    throw new Error("Choisir YT_DLP_COOKIES ou YT_DLP_COOKIES_FROM_BROWSER, pas les deux.");
  if (cookiesFile && !fs.existsSync(cookiesFile))
    throw new Error("YT_DLP_COOKIES : fichier introuvable : " + cookiesFile);
  if (
    cookiesBrowser &&
    !/^(brave|chrome|chromium|edge|firefox|opera|safari|vivaldi|whale)(\+[\w-]+)?(:[^:]+)?(::.+)?$/i.test(
      cookiesBrowser,
    )
  )
    throw new Error("YT_DLP_COOKIES_FROM_BROWSER : navigateur inconnu (par exemple firefox ou edge).");
  const venvYtDlp = path.join(
    projectRoot,
    ".venv",
    process.platform === "win32" ? "Scripts/yt-dlp.exe" : "bin/yt-dlp",
  );
  return {
    projectRoot,
    host,
    port: integer("PORT", 3002, 1, 65535),
    outputDir: insideProject(projectRoot, env.OUTPUT_DIR ?? "outputs"),
    cacheDir: insideProject(projectRoot, env.CACHE_DIR ?? "cache"),
    modelsDir: insideProject(projectRoot, env.MODELS_DIR ?? "models"),
    logsDir: insideProject(projectRoot, env.LOGS_DIR ?? "logs"),
    logLevel: logLevel as "debug" | "info" | "warn" | "error",
    pythonBinary,
    workerScript: path.join(projectRoot, "worker/main.py"),
    ytDlpBinary:
      env.YT_DLP_BINARY && env.YT_DLP_BINARY !== "yt-dlp"
        ? env.YT_DLP_BINARY
        : fs.existsSync(venvYtDlp)
          ? venvYtDlp
          : "yt-dlp",
    ytDlpAuthArgs: cookiesFile
      ? ["--cookies", cookiesFile]
      : cookiesBrowser
        ? ["--cookies-from-browser", cookiesBrowser]
        : ([] as string[]),
    ffmpegBinary: env.FFMPEG_BINARY ?? "ffmpeg",
    ffprobeBinary: env.FFPROBE_BINARY ?? "ffprobe",
    ytDlpTimeoutMs: integer("YT_DLP_TIMEOUT_MS", 120000, 1000, 1800000),
    downloadTimeoutMs: integer("DOWNLOAD_TIMEOUT_MS", 1800000, 1000, 14400000),
    catalogMaxVideos: integer("CATALOG_MAX_VIDEOS", 10000, 1, 10000),
    workerTimeoutMs: integer("WORKER_TIMEOUT_MS", 14400000, 1000, 86400000),
    asrModel,
    asrAlignerModel,
    asrLanguage,
    asrContext: asrContext as "title" | "none",
    // torch.compile needs Triton, which has no macOS build: opt-in only.
    asrCompile: boolean("ASR_COMPILE", false),
    visionModel,
    visionBackend: visionBackend as "llama_cpp" | "transformers",
    nativeVisionDir: insideProject(
      projectRoot,
      env.NATIVE_VISION_DIR ?? path.join(env.MODELS_DIR ?? "models", "native-vision"),
    ),
    audioChunkSeconds: integer("AUDIO_CHUNK_SECONDS", 60, 10, 120),
    audioOverlapSeconds: integer("AUDIO_OVERLAP_SECONDS", 1, 0, 4),
    nodeBinary: process.execPath,
    fps,
    groupFrames,
    frameMaxSide: integer("FRAME_MAX_SIDE", 1280, 320, 1920),
    visionMaxPixels: integer("VISION_MAX_PIXELS", 393216, 4096, 1048576),
    visionMaxNewTokens: integer("VISION_MAX_NEW_TOKENS", 1024, 128, 4096),
    visionQuantization: quantization as "bf16",
    device: "mps" as const,
    keepMedia: boolean("KEEP_MEDIA", false),
  };
}
export type AppConfig = ReturnType<typeof getConfig>;

/** Combined model label stored in runs and the original JSON contract. */
export function modelLabel(config: Pick<AppConfig, "asrModel" | "visionModel">): string {
  return `${config.asrModel} + ${config.visionModel}`;
}

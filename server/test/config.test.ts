import { describe, expect, it } from "vitest";
import path from "node:path";
import fs from "node:fs/promises";
import { getConfig, insideProject, PROJECT_ROOT } from "../src/config.js";
import { temporaryDirectory } from "./helpers.js";

describe("isolated local configuration", () => {
  it("uses French Qwen3-ASR with its forced aligner, native video defaults and local ports without any Gemini settings", () => {
    const config = getConfig({});
    expect(config.asrModel).toBe("Qwen/Qwen3-ASR-1.7B-hf");
    expect(config.asrAlignerModel).toBe("Qwen/Qwen3-ForcedAligner-0.6B-hf");
    expect(config.asrContext).toBe("title");
    expect(config.asrCompile).toBe(true);
    expect(getConfig({ ASR_COMPILE: "false" }).asrCompile).toBe(false);
    expect(config.asrLanguage).toBe("fr");
    expect(getConfig({ ASR_LANGUAGE: "auto" }).asrLanguage).toBe("auto");
    expect(config.visionModel).toBe("Qwen/Qwen3.5-9B");
    expect(config.visionBackend).toBe("llama_cpp");
    expect(config.fps).toBe(1);
    expect(config.groupFrames).toBe(7);
    expect(config.port).toBe(3002);
    expect(config.host).toBe("127.0.0.1");
    expect(config.outputDir).toBe(path.join(PROJECT_ROOT, "outputs"));
    expect(config.logsDir).toBe(path.join(PROJECT_ROOT, "logs"));
    expect(config.logLevel).toBe("info");
    expect(Object.keys(config).some((key) => /gemini/i.test(key))).toBe(false);
    expect(config.audioChunkSeconds).toBe(60);
    expect(config.audioOverlapSeconds).toBe(1);
    expect(Object.keys(config)).not.toContain("audioMaxNewTokens");
  });
  it("passes YouTube cookies to yt-dlp only when configured", async () => {
    const directory = await temporaryDirectory();
    expect(getConfig({}, directory).ytDlpAuthArgs).toEqual([]);
    expect(getConfig({ YT_DLP_COOKIES_FROM_BROWSER: "firefox" }, directory).ytDlpAuthArgs).toEqual([
      "--cookies-from-browser",
      "firefox",
    ]);
    await fs.writeFile(path.join(directory, "cookies.txt"), "# Netscape HTTP Cookie File\n");
    expect(getConfig({ YT_DLP_COOKIES: "cookies.txt" }, directory).ytDlpAuthArgs).toEqual([
      "--cookies",
      path.join(directory, "cookies.txt"),
    ]);
    expect(() => getConfig({ YT_DLP_COOKIES: "absent.txt" }, directory)).toThrow(/introuvable/);
    expect(() => getConfig({ YT_DLP_COOKIES_FROM_BROWSER: "netscape" }, directory)).toThrow(/navigateur/);
  });
  it("rejects paths outside the project folder and invalid runtime settings", () => {
    for (const name of ["OUTPUT_DIR", "CACHE_DIR", "MODELS_DIR", "LOGS_DIR"])
      expect(() => getConfig({ [name]: "../outputs" })).toThrow("dossier du projet");
    expect(() => getConfig({ HOST: "0.0.0.0" })).toThrow();
    expect(() => getConfig({ VIDEO_FPS: "2" })).toThrow();
    expect(() => getConfig({ AUDIO_CHUNK_SECONDS: "211" })).toThrow();
    expect(() => getConfig({ ASR_MODEL: "not a model id" })).toThrow("identifiant HF invalide");
    expect(() => getConfig({ ASR_MODEL: "openai/whisper-large-v3" })).toThrow("Qwen3-ASR");
    expect(() => getConfig({ ASR_ALIGNER_MODEL: "nvidia/parakeet-ctc-1.1b" })).toThrow("Qwen3-ForcedAligner");
    expect(() => getConfig({ ASR_LANGUAGE: "français" })).toThrow("ASR_LANGUAGE");
    // Japanese and Korean alignment would need extra tokenizers that are not installed.
    expect(() => getConfig({ ASR_LANGUAGE: "ja" })).toThrow("ASR_LANGUAGE");
    expect(() => getConfig({ ASR_CONTEXT: "description" })).toThrow("ASR_CONTEXT");
    expect(() => getConfig({ ASR_COMPILE: "yes" })).toThrow("ASR_COMPILE");
    expect(() => getConfig({ LOG_LEVEL: "verbose" })).toThrow();
    expect(() => getConfig({ VISION_BACKEND: "unknown" })).toThrow();
    expect(() => getConfig({ NATIVE_VISION_DIR: "../models" })).toThrow("dossier du projet");
  });
  it("rejects a symlink or junction escaping the project", async () => {
    const root = await temporaryDirectory();
    const outside = await temporaryDirectory();
    await fs.symlink(outside, path.join(root, "escape"), "junction");
    expect(() => insideProject(root, "escape/output")).toThrow("symbolique");
  });
});

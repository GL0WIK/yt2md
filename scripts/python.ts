import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { createInterface } from "node:readline";
import { getConfig } from "../server/src/config.js";
import { workerConfig } from "../server/src/local/worker.js";
import {
  FileLogger,
  captureLogContext,
  logEvent,
  withCapturedLogContext,
  withLogContext,
} from "../server/src/shared/logging.js";

const config = getConfig();
const action = process.argv[2];
if (action !== "download" && action !== "processors")
  throw new Error("Action attendue : download ou processors.");
const roles = process.argv.slice(3).filter((a) => a !== "--refresh");
if (roles.some((role) => !["asr", "aligner", "vision"].includes(role)))
  throw new Error("Modèles possibles : asr, aligner, vision.");
const logger = new FileLogger(config.logsDir, config.logLevel);
withLogContext(
  { runId: randomUUID(), source: "cli" },
  () => {
    const logContext = captureLogContext();
    const started = performance.now();
    logEvent("models.command.start", {
      message:
        action === "download"
          ? "Préparation du téléchargement des modèles."
          : "Vérification des processeurs locaux.",
      action,
      journal: path.join(config.logsDir, "runs", String(logContext.fields.runId) + ".jsonl"),
    });
    const child = spawn(
      config.pythonBinary,
      action === "processors"
        ? [config.projectRoot + "/scripts/smoke_processors.py"]
        : [config.workerScript, "--download"],
      {
        cwd: config.projectRoot,
        shell: false,
        windowsHide: true,
        env: { ...process.env, PYTHONUTF8: "1", PYTHONUNBUFFERED: "1" },
        stdio: ["pipe", "pipe", "pipe"],
      },
    );
    const readers = [child.stdout, child.stderr].map((input) =>
      createInterface({ input, crlfDelay: Infinity }),
    );
    for (const reader of readers)
      reader.on("line", (line) => {
        if (line.trim())
          withCapturedLogContext(logContext, () =>
            logEvent(
              "models.output",
              {
                message: line,
                source: "python",
              },
              /traceback|error|exception/i.test(line) ? "warn" : "info",
            ),
          );
      });
    child.stdin.end(
      JSON.stringify({
        config: workerConfig(config),
        ...(roles.length ? { roles } : {}),
        refresh: process.argv.includes("--refresh"),
      }),
    );
    child.on("error", (error) => {
      withCapturedLogContext(logContext, () =>
        logEvent("models.command.failed", { message: "Python indisponible : " + error.message }, "error"),
      );
      process.exitCode = 1;
    });
    child.stdin.on("error", (error) => {
      withCapturedLogContext(logContext, () =>
        logEvent("models.input.failed", { message: error.message }, "error"),
      );
      process.exitCode = 1;
    });
    child.on("close", async (code) => {
      if (code !== 0) process.exitCode = code ?? 1;
      for (const reader of readers) reader.close();
      withCapturedLogContext(logContext, () =>
        logEvent(
          "models.command." + (code === 0 ? "complete" : "failed"),
          {
            message: code === 0 ? "Commande terminée." : "Commande échouée ; consulter le journal.",
            action,
            exitCode: code,
            durationMs: Math.round(performance.now() - started),
          },
          code === 0 ? "info" : "error",
        ),
      );
      await logger.flush();
    });
  },
  logger,
);

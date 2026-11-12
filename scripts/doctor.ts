import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { getConfig } from "../server/src/config.js";
import { PythonWorker } from "../server/src/local/worker.js";

const config = getConfig();
const worker = new PythonWorker(config);
const execute = promisify(execFile);
try {
  const commands = [
    ["yt-dlp", config.ytDlpBinary, ["--version"]],
    ["ffmpeg", config.ffmpegBinary, ["-version"]],
    ["ffprobe", config.ffprobeBinary, ["-version"]],
  ] as const;
  const checks = await Promise.allSettled(
    commands.map(async ([name, binary, args]) => {
      const { stdout } = await execute(binary, [...args], {
        cwd: config.projectRoot,
        windowsHide: true,
        timeout: 10000,
      });
      return { name, version: stdout.split(/\r?\n/)[0] };
    }),
  );
  const tools = checks.map((check, i) =>
    check.status === "fulfilled" ? check.value : { name: commands[i][0], error: String(check.reason) },
  );
  const health = await worker.health();
  console.log(
    JSON.stringify({ projectRoot: config.projectRoot, port: config.port, tools, ...health }, null, 2),
  );
  if (!health.ready || checks.some((check) => check.status === "rejected")) process.exitCode = 1;
} finally {
  await worker.close();
}

import { execFile, type ChildProcess } from "node:child_process";

/**
 * Terminates a child and its descendants. On Windows `child.kill()` only stops the direct child
 * (yt-dlp/ffmpeg or Python native workers survive), so the whole tree goes through taskkill.
 * On POSIX, `group` targets the process group of a child spawned with `detached: true`.
 */
export function killProcessTree(
  child: ChildProcess,
  signal: NodeJS.Signals = "SIGTERM",
  options: { group?: boolean } = {},
): void {
  const pid = child.pid;
  if (!pid) {
    child.kill(signal);
    return;
  }
  if (process.platform === "win32") {
    execFile("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true }, () => {
      // Fallback if taskkill is unavailable or the tree was already gone; a no-op after exit.
      child.kill();
    });
    return;
  }
  if (options.group) {
    try {
      process.kill(-pid, signal);
      return;
    } catch {
      /* Not a group leader or already gone: fall back to the direct child. */
    }
  }
  child.kill(signal);
}

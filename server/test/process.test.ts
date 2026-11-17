import { describe, expect, it } from "vitest";
import { runProcess } from "../src/catalog/ytDlp.js";
describe("external process bounds", () => {
  it("terminates a hanging child", async () => {
    await expect(runProcess(process.execPath, ["-e", "setInterval(() => {}, 1000)"], 50)).rejects.toThrow(
      "délai",
    );
  });
  it("handles missing executables", async () => {
    await expect(runProcess("yt2md-nonexistent-executable", [], 1000)).rejects.toThrow(
      "Impossible de démarrer",
    );
  });
  it("bounds stderr on failures", async () => {
    await expect(
      runProcess(process.execPath, ["-e", "process.stderr.write('x'.repeat(10000)); process.exit(1)"], 3000),
    ).rejects.toMatchObject({ details: { exitCode: 1, stderr: "x".repeat(4000) } });
  });
});

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    sequence: { hooks: "stack" },
    include: ["server/test/**/*.test.ts", "client/src/**/*.test.ts"],
    coverage: {
      reporter: ["text"],
    },
  },
});

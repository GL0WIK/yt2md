import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { config } from "dotenv";

const root = path.dirname(fileURLToPath(import.meta.url));
config({ path: path.join(root, ".env"), quiet: true });
export default defineConfig({
  plugins: [react()],
  root: path.join(root, "client"),
  server: {
    host: "127.0.0.1",
    port: Number(process.env.CLIENT_PORT ?? 5174),
    strictPort: true,
    proxy: { "/api": `http://127.0.0.1:${process.env.PORT ?? 3002}` },
  },
  build: { outDir: path.join(root, "dist/client"), emptyOutDir: true },
});

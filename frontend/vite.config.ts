import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  test: {
    environment: "jsdom",
    exclude: ["node_modules", "dist", "tests/e2e"],
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});

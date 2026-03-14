import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_GITHUB_PAGES ? "/SVG-gen/" : "/",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8100",
        changeOrigin: true,
      },
    },
  },
});

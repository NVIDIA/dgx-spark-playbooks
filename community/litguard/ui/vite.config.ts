import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/health": "http://localhost:8234",
      "/models": "http://localhost:8234",
      "/metrics": "http://localhost:8234",
      "/api": "http://localhost:8234",
      "/v1": "http://localhost:8234",
    },
  },
});

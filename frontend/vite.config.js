import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: "../dist", // builda no root/dist
    emptyOutDir: true,
  },
  server: {
    host: "0.0.0.0", // Permite acesso externo no Docker
    port: 5173,
    proxy: {
      "/api": "http://localhost:7860",
    },
  },
  publicDir: "public", // Pasta com arquivos estáticos
});
